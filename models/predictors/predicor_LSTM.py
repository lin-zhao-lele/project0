import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import sys
from utils import create_sequences, build_model

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
config_path = resolve_path("resource/predicor_LSTM_args.json", base="script")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
predict_path = resolve_path(config["predict"], base="data")

params = config["params"]

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据加载函数 ==========
def load_and_preprocess(path, scaler=None):
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
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    return scaled, scaler, df_raw

# ========== 加载训练数据 ==========
predict_data, scaler, _ = load_and_preprocess(predict_path)
X, y = create_sequences(predict_data, params["window_size"], target_col="close")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

# ========== 模型训练 / 加载 ==========
model_path = resolve_path("resource/FinalModel_LSTM.pt", base="script")

if os.path.exists(model_path):
    input_size = X_train.shape[2]
    model = build_model(input_shape=(params["window_size"], input_size),
                        hidden_units=params["hidden_units"]).to(device)
    model.load_state_dict(torch.load(model_path))



# ========== 未来数据预测 ==========
future_data, _, raw_df = load_and_preprocess(predict_path, scaler)
X_future, y_future = create_sequences(future_data, params["window_size"])
X_future_t = torch.tensor(X_future, dtype=torch.float32).to(device)

with torch.no_grad():
    y_pred_future = model(X_future_t).cpu().numpy()

mse_future = mean_squared_error(y_future, y_pred_future)
r2_future = r2_score(y_future, y_pred_future)
print(f"[预测集] MSE: {mse_future:.4f})")
print(f"[预测集] R²: {r2_future:.4f})")


# ========== 保存结果 ==========
trade_dates = raw_df["trade_date"].values[params["window_size"]:]
pred_df = pd.DataFrame({
    "trade_date": trade_dates,
    "true_close": y_future,
    "predicted_close": y_pred_future.flatten()
})
os.makedirs(resolve_path("results", base="script"), exist_ok=True)
pred_df.to_csv(resolve_path("results/future_predictions_LSTM.csv", base="script"), index=False)
print("Predictions saved to \\results\\future_predictions.csv")

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
plt.savefig(resolve_path("results/future_predictions_plot.png", base="script"))
print("Plot saved to \\results\\future_predictions_plot_LSTM.png")
