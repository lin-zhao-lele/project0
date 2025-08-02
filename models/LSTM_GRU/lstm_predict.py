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
from utils import create_sequences, build_model, tune_lstm_model
import sys

# 字体配置
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "raw")

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

# ========== 数据加载函数 ==========
def load_and_preprocess(path, scaler=None):
    df_raw = pd.read_csv(path)
    df_raw = df_raw.sort_values("trade_date")

    # 排除指定列
    exclude_cols = {"ts_code", "trade_date"}
    df = df_raw[[col for col in df_raw.columns if col not in exclude_cols]]
    # df = df_raw[["open", "high", "low", "close", "vol", "amount"]]

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    return scaled, scaler, df_raw

# ========== 加载训练数据 ==========
train_data, scaler, _ = load_and_preprocess(training_path)
X, y = create_sequences(train_data, params["window_size"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

# ========== 模型训练 / 加载 ==========
model_path = resolve_path("FinalModel.pt", base="script")

if load_only and os.path.exists(model_path):
    input_size = X_train.shape[2]
    model = build_model(input_shape=(params["window_size"], input_size),
                        hidden_units=params["hidden_units"]).to(device)
    model.load_state_dict(torch.load(model_path))
else:
    if auto_tune:
        model, best_params = tune_lstm_model(X_train, y_train, X_val, y_val, device=device)
        # 更新参数并写回 JSON
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

# ========== 验证集预测 ==========
model.eval()
with torch.no_grad():
    y_pred_val = model(X_val_t).cpu().numpy()

mse_val = mean_squared_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)

print(f"[测试集] MSE: {mse_val:.4f}")
print(f"[测试集] R²: {r2_val:.4f}")


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
pred_df.to_csv(resolve_path("results/future_predictions.csv", base="script"), index=False)
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
print("Plot saved to \\results\\future_predictions_plot.png")
