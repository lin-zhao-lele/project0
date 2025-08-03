import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import RandomForestRegressor
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # 工程根目录
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

# ===============================
# 读取配置文件
# ===============================
config_path = resolve_path("RandomForestModel_args.json", base="script")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

training_file = resolve_path(config["training"], base="data")
predict_file = resolve_path(config["predict"], base="data")
model_filename = resolve_path("FinalModel_RF", base="script")

model_only = config["model"]
auto_tune = config.get("auto_tune", False)
rf_params = config["params"]

# ===============================
# 数据处理函数
# ===============================
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date")

    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()

    df = df.dropna().reset_index(drop=True)

    # 动态生成特征列：排除 close, ts_code, trade_date
    # exclude_cols = {"close", "ts_code", "trade_date"}
    # feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 手动构建feature_cols
    # feature_cols = ["open", "high", "low", "vol", "amount",
    #                 "ma5", "ma10", "return_1d", "vol_ma5"]

    feature_cols = ['open', 'high', 'low', 'vol', 'amount',
     'turnover_rate', 'turnover_rate_f', 'volume_ratio',
     'ma5', 'ma10', 'return_1d', 'vol_ma5']

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
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": rf_params["random_state"],
                "n_jobs": -1
            }
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return r2_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        print("最佳参数:", study.best_params)
        print("最佳 R²:", study.best_value)

        rf_params.update(study.best_params)
        config["params"] = rf_params
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("已将最佳参数写回 RandomForestModel_args.json")

    print("=== 开始训练 RandomForest 模型 ===")
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)

    joblib.dump(model, model_filename)
    print(f"模型已保存到 {model_filename}")

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
    plt.title("RandomForest 模型预测效果")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(resolve_path("train_test_prediction_RF.png", base="script"), dpi=300)
    plt.close()

else:
    print("=== 只加载模型进行预测 ===")
    if not os.path.exists(model_filename):
        raise FileNotFoundError("找不到已保存的模型 FinalModel_RF，请先训练一次模型。")
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

# ========== 转换日期列为 datetime 类型 ==========
dates_pred = pd.to_datetime(dates_pred, format="%Y%m%d")

plt.figure(figsize=(12, 6))
plt.plot(dates_pred, y_true, label="真实收盘价", color="blue")
plt.plot(dates_pred, y_pred_final, label="预测收盘价", color="orange", linestyle="--")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.title("最终预测结果可视化 - RandomForest")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resolve_path("final_prediction_RF.png", base="script"), dpi=300)
plt.close()
