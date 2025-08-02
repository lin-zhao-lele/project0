import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# 字体设置
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

# 路径设置
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

# ===============================
# 读取配置文件
# ===============================
config_path = resolve_path("XGBoostModelSlidingWindow_args.json", base="script")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

training_file = resolve_path(config["training"], base="data")
predict_file = resolve_path(config["predict"], base="data")
model_filename = resolve_path("FinalModel_XGBoost_sliding.pkl", base="script")
model_only = config["model"]
auto_tune = config.get("auto_tune", False)
xgb_params = config["params"]

# 从 params 中取出 window_size，并移除该键，避免 XGBRegressor 报错
window_size = xgb_params.pop("window_size", 5)

# 安全移除不属于 XGBoost 的 sklearn 风格参数
xgb_params.pop("min_samples_split", None)
xgb_params.pop("min_samples_leaf", None)

# ===============================
# 滑动窗口数据函数
# ===============================
def load_and_process_data(csv_path, window_size=5):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date")

    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "vol", "amount",
                    "ma5", "ma10", "return_1d", "vol_ma5"]
    features = df[feature_cols].values
    labels = df["close"].values
    dates = df["trade_date"].values

    X, y, date_list = [], [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i].flatten())
        y.append(labels[i])
        date_list.append(dates[i])
    return np.array(X), np.array(y), pd.Series(date_list), df

# ===============================
# 训练阶段
# ===============================
if not model_only:
    print("=== 开始加载训练数据 ===")
    X, y, dates, df_full = load_and_process_data(training_file, window_size=window_size)

    split_index = int(len(X) * 0.9)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
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
                "random_state": xgb_params.get("random_state", 42)
            }
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            return r2_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        print("最佳参数:", study.best_params)
        print("最佳 R²:", study.best_value)

        xgb_params.update(study.best_params)
        # 写入配置文件时单独附加 window_size，不影响模型用参数
        params_for_config = xgb_params.copy()
        params_for_config["window_size"] = window_size
        config["params"] = params_for_config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("已将最佳参数写回配置文件")

    print("=== 开始训练 XGBoost 模型 ===")
    # 排除非 XGBRegressor 支持的参数
    xgb_model_params = {k: v for k, v in xgb_params.items() if k in XGBRegressor().get_params()}
    model = XGBRegressor(**xgb_model_params)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    print(f"模型已保存到 {model_filename}")

    y_pred = model.predict(X_test)
    print(f"测试集 MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"测试集 R²: {r2_score(y_test, y_pred):.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="真实收盘价", color="blue")
    plt.plot(dates_test, y_pred, label="预测收盘价", color="red", linestyle="--")
    plt.title("XGBoost 模型预测效果")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(resolve_path("train_test_prediction.png", base="script"), dpi=300)
    plt.close()

else:
    print("=== 只加载模型进行预测 ===")
    if not os.path.exists(model_filename):
        raise FileNotFoundError("找不到模型，请先训练一次。")
    model = joblib.load(model_filename)

# ===============================
# 最终预测阶段
# ===============================
print("=== 加载预测数据 ===")
X_pred, y_true, dates_pred, _ = load_and_process_data(predict_file, window_size=window_size)
y_pred_final = model.predict(X_pred)

if y_true is not None and len(y_true) > 0:
    print(f"预测集 MSE: {mean_squared_error(y_true, y_pred_final):.4f}")
    print(f"预测集 R²: {r2_score(y_true, y_pred_final):.4f}")
else:
    print("预测数据集中无真实值，无法评估")

plt.figure(figsize=(12, 6))
plt.plot(dates_pred, y_true, label="真实收盘价", color="blue")
plt.plot(dates_pred, y_pred_final, label="预测收盘价", color="orange", linestyle="--")
plt.title("最终预测结果可视化")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(resolve_path("final_prediction.png", base="script"), dpi=300)
plt.close()