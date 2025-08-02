import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys


# 判断操作系统并设置字体
# 解决中文和负号显示问题
if sys.platform == 'darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS'] # 首选苹方，备选Arial Unicode MS（更通用，但可能需要系统安装）
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
elif sys.platform == 'win32':  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Windows 上使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 路径配置
# ===============================
# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "raw")

def resolve_path(path_str, base="project"):
    """
    解析路径：
    base="project"  -> 相对工程根目录解析（适合数据文件）
    base="script"   -> 相对脚本目录解析（适合模型文件、配置文件）
    """
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))
    elif base == "script":
        return os.path.normpath(os.path.join(BASE_DIR, path_str))
    elif base == "data":
        return os.path.normpath(os.path.join(DATA_DIR, path_str))

# ===============================
# 读取配置文件
# ===============================
config_path = resolve_path("XGBoostModel_args.json", base="script")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# 数据路径（相对工程根目录）
training_file = resolve_path(config["training"], base="data")
predict_file = resolve_path(config["predict"], base="data")
# 模型路径（相对脚本目录）
model_filename = resolve_path("FinalModel", base="script")

model_only = config["model"]
auto_tune = config.get("auto_tune", False)
xgb_params = config["params"]

# ===============================
# 数据处理函数
# ===============================
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date")

    # 简单特征工程
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()

    df = df.dropna().reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "vol", "amount",
                    "ma5", "ma10", "return_1d", "vol_ma5"]
    X = df[feature_cols]
    y = df["close"]

    return X, y, df["trade_date"], df

# ===============================
# 训练阶段
# ===============================
if not model_only:
    print("=== 开始加载训练数据 ===")
    X, y, dates, df_full = load_and_process_data(training_file)

    split_index = int(len(X) * 0.9)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = dates.iloc[split_index:]

    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    if auto_tune:
        print("=== 启动 Optuna 自动调参 ===")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": xgb_params["random_state"]
            }
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return r2_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print("最佳参数:", study.best_params)
        print("最佳 R²:", study.best_value)

        # 更新并保存最佳参数
        xgb_params.update(study.best_params)
        config["params"] = xgb_params
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("已将最佳参数写回 XGBoostModel_args.json")

    print("=== 开始训练 XGBoost 模型 ===")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)
    print(f"模型已保存到 {model_filename}")

    # 测试集评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="真实收盘价", color="blue")
    plt.plot(dates_test, y_pred, label="预测收盘价", color="red", linestyle="--")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title("XGBoost 模型预测效果")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(resolve_path("train_test_prediction.png", base="script"), dpi=300)
    plt.close()

else:
    print("=== 只加载模型进行预测 ===")
    if not os.path.exists(model_filename):
        raise FileNotFoundError("找不到已保存的模型 FinalModel，请先训练一次模型。")
    model = joblib.load(model_filename)

# ===============================
# 最终预测阶段
# ===============================
print("=== 加载预测数据 ===")
X_pred, y_true, dates_pred, df_pred_full = load_and_process_data(predict_file)
y_pred_final = model.predict(X_pred)

if y_true is not None and len(y_true) > 0:
    mse_pred = mean_squared_error(y_true, y_pred_final)
    r2_pred = r2_score(y_true, y_pred_final)
    print(f"预测集 MSE: {mse_pred:.4f}")
    print(f"预测集 R²: {r2_pred:.4f}")
else:
    print("预测数据集中未提供真实值，无法计算 MSE 和 R²")

plt.figure(figsize=(12, 6))
plt.plot(dates_pred, y_true, label="真实收盘价", color="blue")
plt.plot(dates_pred, y_pred_final, label="预测收盘价", color="orange", linestyle="--")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.title("最终预测结果可视化")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resolve_path("final_prediction.png", base="script"), dpi=300)
plt.close()
