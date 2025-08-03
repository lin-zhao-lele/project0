import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import sys

# 调用 RandomForest 和 XGBoost 两个模型 自动混合权重

# 判断操作系统并设置字体
# 解决中文和负号显示问题
if sys.platform == 'darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS'] # 首选苹方，备选Arial Unicode MS（更通用，但可能需要系统安装）
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
elif sys.platform == 'win32':  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Windows 上使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
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


# 读取配置文件
XGBoostModeJsonlDIR = resolve_path("EnsemblePrediction_args.json", base="script")
with open(XGBoostModeJsonlDIR, "r", encoding="utf-8") as f:
    config = json.load(f)
predict_file = resolve_path(config["predict"], base="data")

# 加载预测数据（和之前一样的处理逻辑）
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date")
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)

    # 动态生成特征列：排除 close, ts_code, trade_date
    # exclude_cols = { "ts_code", "trade_date"}
    # feature_cols = [col for col in df.columns if col not in exclude_cols]
    # 手动构建feature_cols
    feature_cols = ["open", "high", "low", "close", "vol", "amount", "ma5", "ma10", "return_1d", "vol_ma5"]


    X = df[feature_cols]
    y = df["close"]
    dates = df["trade_date"]
    return X, y, dates

X_pred, y_true, dates_pred = load_and_process_data(predict_file)

# 加载模型
xgb_model_path = resolve_path("models/XGBoost/FinalModel_XGBoost", base="project")
rf_model_path = resolve_path("models/RandomForest/FinalModel_RF", base="project")

xgb_model = joblib.load(xgb_model_path)
rf_model = joblib.load(rf_model_path)

# 预测
y_pred_xgb = xgb_model.predict(X_pred)
y_pred_rf = rf_model.predict(X_pred)

# 权重搜索区间和步长
weights = np.arange(0, 1.05, 0.05)

best_r2 = -np.inf
best_weight_xgb = 0.5
best_weight_rf = 0.5
best_pred = None

for w_xgb in weights:
    w_rf = 1 - w_xgb
    ensemble_pred = w_xgb * y_pred_xgb + w_rf * y_pred_rf
    r2 = r2_score(y_true, ensemble_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_weight_xgb = w_xgb
        best_weight_rf = w_rf
        best_pred = ensemble_pred

print(f"最佳权重 - XGBoost: {best_weight_xgb:.2f}, RandomForest: {best_weight_rf:.2f}")
print(f"集成模型最佳 R²: {best_r2:.4f}")
print(f"集成模型最佳 MSE: {mean_squared_error(y_true, best_pred):.4f}")

# 画图用最佳预测结果
plt.figure(figsize=(14, 7))
plt.plot(dates_pred, y_true, label="真实收盘价", color="black")
plt.plot(dates_pred, y_pred_xgb, label="XGBoost预测", linestyle="--")
plt.plot(dates_pred, y_pred_rf, label="RandomForest预测", linestyle="--")
plt.plot(dates_pred, best_pred, label="集成预测 (最佳权重)", linestyle="-.", linewidth=2)
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.title("三种模型预测对比（自动调权重）")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resolve_path("ensemble_prediction_best_weight.png"))
plt.close()
