import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from utils import create_sequences, build_model, tune_lstm_model
import sys
import joblib  # 导入joblib以保存scaler对象


# 字体配置
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")

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
config_path = resolve_path("Model_args.json", base="script")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

training_path = resolve_path(config["training"], base="data")
predict_path = resolve_path(config["predict"], base="data")
load_only = config["model"]
auto_tune = config["auto_tune"]
params = config["params"]

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据加载函数（x 和 y 分开归一化） ==========
def load_and_preprocess_separate_scalers(path, x_scaler=None, y_scaler=None):
    df_raw = pd.read_csv(path)
    df_raw = df_raw.sort_values("trade_date")

    df_raw["ma5"] = df_raw["close"].rolling(window=5).mean()
    df_raw["ma10"] = df_raw["close"].rolling(window=10).mean()
    df_raw["return_1d"] = df_raw["close"].pct_change(1)
    df_raw["vol_ma5"] = df_raw["vol"].rolling(window=5).mean()

    df_raw = df_raw.dropna().reset_index(drop=True)

    feature_cols = ['open', 'high', 'low', 'vol', 'amount',
                    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                    'ma5', 'ma10', 'return_1d', 'vol_ma5']
    target_col = ['close']

    X_raw = df_raw[feature_cols].values
    y_raw = df_raw[target_col].values

    if x_scaler is None:
        x_scaler = MinMaxScaler()
        X_scaled = x_scaler.fit_transform(X_raw)
    else:
        X_scaled = x_scaler.transform(X_raw)

    if y_scaler is None:
        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y_raw)
    else:
        y_scaled = y_scaler.transform(y_raw)

    # 将目标y拼接到最后一列，作为create_sequences用
    df_scaled = np.hstack((X_scaled, y_scaled))
    return df_scaled, x_scaler, y_scaler, df_raw

# ========== 加载训练数据 ==========
train_data, x_scaler, y_scaler, _ = load_and_preprocess_separate_scalers(training_path)
X, y = create_sequences(train_data, params["window_size"], target_col="close")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

# ========== 模型训练 / 加载 ==========
model_path = resolve_path("FinalModel_LSTM.pt", base="script")

if load_only and os.path.exists(model_path):
    input_size = X_train.shape[2]
    model = build_model(input_shape=(params["window_size"], input_size),
                        hidden_units=params["hidden_units"]).to(device)
    model.load_state_dict(torch.load(model_path))
else:
    if auto_tune:
        model, best_params = tune_lstm_model(X_train, y_train, X_val, y_val, device=device)
        config["params"].update(best_params)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        print(f"Updated training parameters saved to {config_path}")
    else:
        model = build_model(
            input_shape=(params["window_size"], X_train.shape[2]),
            hidden_units=params["hidden_units"]
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batch_size = params["batch_size"]

        for epoch in range(params["epochs"]):
            model.train()
            for i in range(0, len(X_train_t), batch_size):
                xb = X_train_t[i:i+batch_size]
                yb = y_train_t[i:i+batch_size]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

    torch.save(model.state_dict(), model_path)

# ========== 保存scaler ==========
joblib.dump(x_scaler, "scalerX.joblib")
joblib.dump(y_scaler, "scalerY.joblib")
print("Scalers saved to scaler.joblib and scalerY.pkl")

# ========== 验证集预测 ==========
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_t).cpu().numpy()

mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print(f"[测试集] MSE: {mse_val:.4f}")
print(f"[测试集] R²: {r2_val:.4f}")

# ========== 未来数据预测 ==========
future_data, _, _, raw_df = load_and_preprocess_separate_scalers(predict_path, x_scaler, y_scaler)
X_future, y_future = create_sequences(future_data, params["window_size"])
X_future_t = torch.tensor(X_future, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_future = model(X_future_t).cpu().numpy()

mse_future = mean_squared_error(y_future, y_pred_future)
r2_future = r2_score(y_future, y_pred_future)
print(f"[预测集] MSE: {mse_future:.4f}")
print(f"[预测集] R²: {r2_future:.4f}")

# ========== 保存结果 ==========
trade_dates = raw_df["trade_date"].values[params["window_size"]:]

# 保存归一化结果
os.makedirs(resolve_path("results", base="script"), exist_ok=True)
df_scaled = pd.DataFrame({
    "trade_date": trade_dates,
    "true_close_scaled": y_future.flatten(),
    "predicted_close_scaled": y_pred_future.flatten()
})
df_scaled.to_csv(resolve_path("results/future_predictions_scaled.csv", base="script"), index=False)
print("归一化结果保存至 results/future_predictions_scaled.csv")

# 保存反归一化结果
y_true_inv = y_scaler.inverse_transform(y_future.reshape(-1, 1)).flatten()
y_pred_inv = y_scaler.inverse_transform(y_pred_future.reshape(-1, 1)).flatten()

df_final = pd.DataFrame({
    "trade_date": trade_dates,
    "true_close": y_true_inv,
    "predicted_close": y_pred_inv
})
df_final.to_csv(resolve_path("results/future_predictions.csv", base="script"), index=False)
print("反归一化结果保存至 results/future_predictions.csv")

# ========== 绘图 ==========
df_final["trade_date"] = pd.to_datetime(df_final["trade_date"], format="%Y%m%d")

plt.figure(figsize=(10, 4))
plt.plot(df_final["trade_date"], df_final["true_close"], label="True Close")
plt.plot(df_final["trade_date"], df_final["predicted_close"], label="Predicted Close")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.title("Future Stock Price Prediction")
plt.legend()
plt.tight_layout()
plt.savefig(resolve_path("results/future_predictions_plot.png", base="script"))
print("Plot saved to \\results\\future_predictions_plot.png")
