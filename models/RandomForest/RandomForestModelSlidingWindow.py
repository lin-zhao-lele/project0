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

# 字体配置，支持中文显示和负号
if sys.platform == 'darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
elif sys.platform == 'win32':  # Windows
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

# 路径相关
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

def resolve_path(path_str, base="project"):
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))
    elif base == "script":
        return os.path.normpath(os.path.join(BASE_DIR, path_str))

# 滑动窗口构造序列数据
def create_sequences_rf(df, window_size, feature_cols, target_col='close'):
    X, y = [], []
    for i in range(len(df) - window_size):
        window_df = df.iloc[i:i+window_size]
        features = window_df[feature_cols].values.flatten()  # 展平多个时间步的特征到一维
        X.append(features)
        y.append(df.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

# 读取数据并生成滑动窗口特征和标签
def load_and_process_data_sliding(csv_path, window_size):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date").reset_index(drop=True)

    # 计算一些技术指标
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()

    df = df.dropna().reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "vol", "amount",
                    "ma5", "ma10", "return_1d", "vol_ma5"]

    X, y = create_sequences_rf(df, window_size, feature_cols)
    trade_dates = df["trade_date"].iloc[window_size:].values

    return X, y, trade_dates, df

def main():
    # 读取配置文件
    config_path = resolve_path("RandomForestModel_args.json", base="script")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    training_file = resolve_path(config["training"], base="project")
    predict_file = resolve_path(config["predict"], base="project")
    model_filename = resolve_path("FinalModel_RF_sliding.pkl", base="script")

    model_only = config.get("model", False)
    auto_tune = config.get("auto_tune", False)
    rf_params = config.get("params", {})

    # 提取并剥离 window_size，防止传入模型
    window_size = rf_params.get("window_size", 5)
    rf_params_model = {k: v for k, v in rf_params.items() if k != "window_size"}

    if not model_only:
        print("=== 加载训练数据，使用滑动窗口处理 ===")
        X, y, dates, df_full = load_and_process_data_sliding(training_file, window_size)

        split_index = int(len(X) * 0.9)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        dates_test = dates[split_index:]

        print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

        if auto_tune:
            print("=== 启动 Optuna 自动调参 ===")

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "random_state": rf_params_model.get("random_state", 42),
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

            rf_params_model.update(study.best_params)
            config["params"].update(rf_params_model)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print(f"已将最佳参数写回 {config_path}")

        print("=== 训练随机森林模型 ===")
        model = RandomForestRegressor(**rf_params_model)
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
        plt.title("RandomForest 滑动窗口模型预测效果")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(resolve_path("train_test_prediction_RF_sliding.png", base="script"), dpi=300)
        plt.close()

    else:
        print("=== 只加载模型进行预测 ===")
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"找不到已保存的模型 {model_filename}，请先训练一次模型。")
        model = joblib.load(model_filename)

    # 预测阶段
    print("=== 加载预测数据，使用滑动窗口处理 ===")
    X_pred, y_true, dates_pred, df_pred_full = load_and_process_data_sliding(predict_file, window_size)
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
    plt.title("最终预测结果可视化 - RandomForest 滑动窗口")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(resolve_path("final_prediction_RF_sliding.png", base="script"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
