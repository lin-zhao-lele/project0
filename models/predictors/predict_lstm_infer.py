import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import joblib
from utils import create_sequences, build_model
from sklearn.metrics import mean_squared_error, r2_score

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed")

def resolve_path(path_str, base="project"):
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))
    elif base == "script":
        return os.path.normpath(os.path.join(BASE_DIR, path_str))
    elif base == "data":
        return os.path.normpath(os.path.join(DATA_DIR, path_str))

# ========== 加载配置 ==========
config_path = os.path.join(PROJECT_ROOT, "predictors", "resource", "FinalModel_LSTM_args.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

predict_path = resolve_path(config["predict"], base="data")
predict_length = config.get("predict_length", 5)
window_size = config["params"]["window_size"]
hidden_units = config["params"]["hidden_units"]

# ========== 加载Scaler和模型 ==========
x_scaler = joblib.load(os.path.join(PROJECT_ROOT, "predictors", "resource", "scalerX.joblib"))   # 12维特征归一化器
y_scaler = joblib.load(os.path.join(PROJECT_ROOT, "predictors", "resource", "scalerY.joblib"))   # close归一化器

model_path = resolve_path(os.path.join(PROJECT_ROOT, "predictors", "resource", "FinalModel_LSTM.pt"), base="script")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据预处理 ==========
def preprocess_predict_data(path):
    df = pd.read_csv(path).sort_values("trade_date")

    # 计算技术指标等
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['open', 'high', 'low', 'vol', 'amount',
                    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                    'ma5', 'ma10', 'return_1d', 'vol_ma5']

    X_raw = df[feature_cols].values  # 12维特征
    y_raw = df[["close"]].values     # 1维close

    X_scaled = x_scaler.transform(X_raw)
    y_scaled = y_scaler.transform(y_raw)

    # 拼接成13维数组，跟训练时输入格式一致
    full_input = np.hstack([X_scaled, y_scaled])

    df["scaled_close"] = y_scaled.flatten()

    return full_input, df

# ========== 滚动预测 ==========
def rolling_forecast(last_window, predict_len):
    """
    last_window: numpy数组，shape=(window_size, 13)，最后一列是close归一化值
    predict_len: int，预测天数
    """
    predictions = []
    current_seq = last_window.copy()

    for _ in range(predict_len):
        inp = torch.tensor(current_seq[np.newaxis, :, :], dtype=torch.float32).to(device)  # shape (1, window_size, 13)
        with torch.no_grad():
            pred = model(inp).cpu().numpy().flatten()[0]

        predictions.append(pred)

        # 生成下一步输入序列：窗口滑动1步，最后一行close用预测值替换
        next_input = current_seq[-1].copy()
        current_seq = np.vstack([current_seq[1:], next_input])
        current_seq[-1, -1] = pred  # 替换close列

    return np.array(predictions)

# ========== 主流程 ==========
full_input, df_all = preprocess_predict_data(predict_path)  # 13维输入

input_size = full_input.shape[1]  # 应该是13
model = build_model(input_shape=(window_size, input_size), hidden_units=hidden_units).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 创建序列用于已有数据的预测和评估
X_seq, _ = create_sequences(full_input, window_size)
X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_scaled = model(X_seq_t).cpu().numpy().flatten()

# 反归一化预测结果
y_pred_inv = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 真实close序列（从第window_size开始对应预测的y）
y_true_scaled = df_all["scaled_close"].values[window_size:]
y_true_inv = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

dates = df_all["trade_date"].values[window_size:]

# ========== 计算评估指标 ==========
mse = mean_squared_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)
print(f"[已有数据] MSE: {mse:.4f}")
print(f"[已有数据] R²: {r2:.4f}")

df_out = pd.DataFrame({
    "trade_date": dates,
    "true_close": y_true_inv,
    "predicted_close": y_pred_inv
})

# 追加预测未来N天
print(f"🔄 追加预测未来 {predict_length} 天")
last_window = full_input[-window_size:]  # 取最后一个窗口，包含13维
pred_scaled_future = rolling_forecast(last_window, predict_length)
pred_inv_future = y_scaler.inverse_transform(pred_scaled_future.reshape(-1, 1)).flatten()

last_date = pd.to_datetime(df_all["trade_date"].iloc[-1], format="%Y%m%d")
future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime("%Y%m%d") for i in range(predict_length)]

df_future = pd.DataFrame({
    "trade_date": future_dates,
    "predicted_close": pred_inv_future
})

df_all_out = pd.concat([df_out, df_future], ignore_index=True)

# ========== 保存结果 ==========
os.makedirs(resolve_path("results", base="script"), exist_ok=True)
save_path = resolve_path("results/lstm_inference_output.csv", base="script")
df_all_out.to_csv(save_path, index=False)
print(f"✅ 推理结果保存至 {save_path}")

# ========== 绘图 ==========
df_all_out["trade_date"] = pd.to_datetime(df_all_out["trade_date"], format="%Y%m%d")
plt.figure(figsize=(10, 4))

if "true_close" in df_all_out.columns:
    plt.plot(df_all_out["trade_date"], df_all_out["true_close"], label="True Close")
plt.plot(df_all_out["trade_date"], df_all_out["predicted_close"], label="Predicted Close")

plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("LSTM Inference Prediction with Future Forecast")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plot_path = resolve_path("results/lstm_inference_plot.png", base="script")
plt.savefig(plot_path)
print(f"📊 推理图保存至 {plot_path}")
