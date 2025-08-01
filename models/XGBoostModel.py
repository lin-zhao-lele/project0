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

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# ===============================
# 1. 读取配置文件
# ===============================
with open("XGBoostModel_args.json", "r", encoding="utf-8") as f:
    config = json.load(f)

training_file = config["training"]
predict_file = config["predict"]
model_only = config["model"]
auto_tune = config.get("auto_tune", False)
xgb_params = config["params"]

model_filename = "FinalModel"

# ===============================
# 2. 数据处理函数
# ===============================
def load_and_process_data(csv_path):
    """
    读取股票数据并构造特征
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date")  # 按日期排序

    # 生成简单的技术指标特征
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()

    # 删除缺失值
    df = df.dropna().reset_index(drop=True)

    # 特征与目标
    feature_cols = ["open", "high", "low", "close", "vol", "amount", "ma5", "ma10", "return_1d", "vol_ma5"]
    X = df[feature_cols]
    y = df["close"]

    return X, y, df["trade_date"], df

# ===============================
# 3. 模型训练 & 保存
# ===============================
if not model_only:
    print("=== 开始加载训练数据 ===")
    X, y, dates, df_full = load_and_process_data(training_file)

    # 划分 90% 训练集 10% 测试集
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
        study.optimize(objective, n_trials=30, show_progress_bar=True)  # 30次搜索

        print("最佳参数:", study.best_params)
        print("最佳 R²:", study.best_value)

        # 更新配置
        xgb_params.update(study.best_params)
        config["params"] = xgb_params
        with open("XGBoostModel_args.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print("已将最佳参数写回 XGBoostModel_args.json")

    # 训练模型
    print("=== 开始训练 XGBoost 模型 ===")
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, model_filename)
    print(f"模型已保存到 {model_filename}")

    # 测试集预测
    y_pred = model.predict(X_test)

    # 评价指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"测试集 MSE: {mse:.4f}")
    print(f"测试集 R²: {r2:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="真实收盘价", color="blue")
    plt.plot(dates_test, y_pred, label="预测收盘价", color="red", linestyle="--")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title("XGBoost 模型预测效果")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("train_test_prediction.png", dpi=300)  # 保存训练预测对比图
    plt.close()  # 关闭图，释放内存

else:
    print("=== 只加载模型进行预测 ===")
    if not os.path.exists(model_filename):
        raise FileNotFoundError("找不到已保存的模型 FinalModel，请先训练一次模型。")
    model = joblib.load(model_filename)

# ===============================
# 4. 最终预测阶段
# ===============================
print("=== 加载预测数据 ===")
X_pred, y_true, dates_pred, df_pred_full = load_and_process_data(predict_file)
y_pred_final = model.predict(X_pred)

# 如果预测文件包含真实值，则计算 MSE 和 R²
if y_true is not None and len(y_true) > 0:
    mse_pred = mean_squared_error(y_true, y_pred_final)
    r2_pred = r2_score(y_true, y_pred_final)
    print(f"预测集 MSE: {mse_pred:.4f}")
    print(f"预测集 R²: {r2_pred:.4f}")
else:
    print("预测数据集中未提供真实值，无法计算 MSE 和 R²")

# 可视化最终预测效果
plt.figure(figsize=(12, 6))
plt.plot(dates_pred, y_true, label="真实收盘价", color="blue")
plt.plot(dates_pred, y_pred_final, label="预测收盘价", color="orange", linestyle="--")
plt.xlabel("日期")
plt.ylabel("收盘价")
plt.title("最终预测结果可视化")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("final_prediction.png", dpi=300)  # 保存最终预测图
plt.close()
