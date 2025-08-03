import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import create_sequences, build_model  # 保证 utils.py 可导入
import sys

# 字体配置（跨平台）
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # predictors 目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 上级目录
MODEL_DIR = os.path.join(PROJECT_ROOT, "LSTM_GRU")
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "predictors", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def resolve_path(relative_path, base="project"):
    if os.path.isabs(relative_path):
        return os.path.normpath(relative_path)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, relative_path))
    elif base == "model":
        return os.path.normpath(os.path.join(MODEL_DIR, relative_path))
    elif base == "data":
        return os.path.normpath(os.path.join(DATA_DIR, relative_path))

# ========== 加载配置 ==========
config_path = resolve_path("Model_args.json", base="model")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

training_path = resolve_path(config["training"], base="data")
predict_path = resolve_path(config["predict"], base="data")
params = config["params"]
model_path = resolve_path("FinalModel_LSTM.pt", base="model")

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据预处理函数 ==========
def load_and_preprocess(path, scaler):
    df_raw = pd.read_csv(path)
    df_raw = df_raw.sort_values("trade_date")

    df_raw["ma5"] = df_raw["close"].rolling(window=5).mean()
    df_raw["ma10"] = df_raw["close"].rolling(window=10).mean()
    df_raw["return_1d"] = df_raw["close"].pct_change(1)
    df_raw["vol_ma5"] = df_raw["vol"].rolling(window=5).mean()

    df_raw = df_raw.dropna().reset_index(drop=True)

    df = df_raw[['open', 'high', 'low', "close", 'vol', 'amount',
                 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                 'ma5', 'ma10', 'return_1d', 'vol_ma5']]

    if scaler is None:
        raise ValueError("Scaler must be provided for prediction")
    scaled = scaler.transform(df)

    return scaled, df_raw

# ========== 用训练集构建 scaler ==========
def create_scaler_from_training(path):
    df_raw = pd.read_csv(path)
    df_raw = df_raw.sort_values("trade_date")
    df_raw["ma5"] = df_raw["close"].rolling(window=5).mean()
    df_raw["ma10"] = df_raw["close"].rolling(window=10).mean()
    df_raw["return_1d"] = df_raw["close"].pct_change(1)
    df_raw["vol_ma5"] = df_raw["vol"].rolling(window=5).mean()
    df_raw = df_raw.dropna().reset_index(drop=True)
    df = df_raw[['open', 'high', 'low', "close", 'vol', 'amount',
                 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                 'ma5', 'ma10', 'return_1d', 'vol_ma5']]
    scaler = MinMaxScaler()
    scaler.fit(df)
    return scaler

# ========== 初始化 scaler 并加载预测数据 ==========
scaler = create_scaler_from_training(training_path)
predict_data, raw_df = load_and_preprocess(predict_path, scaler=scaler)
X_future, y_future = create_sequences(predict_data, params["window_size"])

X_future_t = torch.tensor(X_future, dtype=torch.float32).to(device)

# ========== 加载模型 ==========
input_size = X_future.shape[2]
model = build_model(input_shape=(params["window_size"], input_size),
                    hidden_units=params["hidden_units"]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== 执行预测 ==========
with torch.no_grad():
    y_pred_future = model(X_future_t).cpu().numpy()

# ========== 输出评估 ==========
mse_future = mean_squared_error(y_future, y_pred_future)
r2_future = r2_score(y_future, y_pred_future)
print(f"[预测集] MSE: {mse_future:.4f}")
print(f"[预测集] R²: {r2_future:.4f}")

# ========== 保存结果 ==========
trade_dates = raw_df["trade_date"].values[params["window_size"]:]
pred_df = pd.DataFrame({
    "trade_date": trade_dates,
    "true_close": y_future,
    "predicted_close": y_pred_future.flatten()
})
pred_csv = os.path.join(RESULTS_DIR, "future_predictions_LSTM.csv")
pred_df.to_csv(pred_csv, index=False)
print(f"Predictions saved to {pred_csv}")

# ========== 绘图保存 ==========
plt.figure(figsize=(10, 4))
plt.plot(pred_df["trade_date"], pred_df["true_close"], label="True Close")
plt.plot(pred_df["trade_date"], pred_df["predicted_close"], label="Predicted Close")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.title("Future Stock Price Prediction")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "future_predictions_plot_LSTM.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
