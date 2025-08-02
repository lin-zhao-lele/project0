import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# 字体配置
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

def resolve_path(path_str, base="project"):
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    base_path = {"project": PROJECT_ROOT, "script": BASE_DIR, "data": DATA_DIR}[base]
    return os.path.normpath(os.path.join(base_path, path_str))

def create_sequences_rf(df, window_size, feature_cols, target_col='close'):
    X, y = [], []
    for i in range(len(df) - window_size):
        window_df = df.iloc[i:i+window_size]
        features = window_df[feature_cols].values.flatten()
        X.append(features)
        y.append(df.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

def load_and_process_data_sliding(csv_path, window_size):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="trade_date").reset_index(drop=True)

    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    df["return_1d"] = df["close"].pct_change(1)
    df["vol_ma5"] = df["vol"].rolling(window=5).mean()
    df = df.dropna().reset_index(drop=True)

    # 动态生成特征列：排除 close, ts_code, trade_date
    exclude_cols = { "ts_code", "trade_date"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # 手动构建feature_cols
    # feature_cols = ["open", "high", "low", "close", "vol", "amount",
    #                 "ma5", "ma10", "return_1d", "vol_ma5"]

    X, y = create_sequences_rf(df, window_size, feature_cols)
    trade_dates = df["trade_date"].iloc[window_size:].astype(str).values
    return X, y, trade_dates, df, feature_cols

def get_next_trade_date(last_date_str, existing_dates_set):
    next_date = pd.to_datetime(str(last_date_str), format="%Y%m%d")
    while True:
        next_date += pd.Timedelta(days=1)
        next_date_str = next_date.strftime("%Y%m%d")
        if next_date_str not in existing_dates_set and next_date.weekday() < 5:
            return next_date_str

def main():
    config_path = resolve_path("resource/predicor_RF_Sliding_args.json", base="script")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    predict_file = resolve_path(config["predict"], base="data")
    model_filename = resolve_path("resource/FinalModel_RF_sliding.pkl", base="script")

    rf_params = config.get("params", {})
    window_size = rf_params.pop("window_size", 5)
    predict_length = config.get("predict_length", 5)

    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"模型文件 {model_filename} 不存在，请先训练模型。")
    model = joblib.load(model_filename)

    X, y_true, dates, df_full, feature_cols = load_and_process_data_sliding(predict_file, window_size)

    # feature_cols = ["open", "high", "low", "close", "vol", "amount",
    #                 "ma5", "ma10", "return_1d", "vol_ma5"]

    # 模型评估（使用真实历史数据，不含未来N天）
    y_pred_eval = model.predict(X)
    mse_pred = mean_squared_error(y_true, y_pred_eval)
    r2_pred = r2_score(y_true, y_pred_eval)
    print(f"评估集 MSE: {mse_pred:.4f}")
    print(f"评估集 R²: {r2_pred:.4f}")

    # 未来 N 天滚动预测
    latest_df = df_full.copy()
    current_window_df = latest_df.iloc[-window_size:].copy()
    existing_dates_set = set(latest_df["trade_date"].astype(str))

    rolling_preds = []

    for _ in range(predict_length):
        if len(current_window_df) < window_size:
            break

        features = current_window_df[feature_cols].values.flatten().reshape(1, -1)
        pred = model.predict(features)[0]

        if rolling_preds:
            last_date = rolling_preds[-1]["trade_date"]
        else:
            last_date = current_window_df.iloc[-1]["trade_date"]

        next_date_str = get_next_trade_date(last_date, existing_dates_set)
        existing_dates_set.add(next_date_str)

        new_row = {
            "ts_code": latest_df.iloc[-1]["ts_code"],
            "trade_date": next_date_str,
            "open": pred,
            "high": pred,
            "low": pred,
            "close": pred,
            "vol": latest_df.iloc[-1]["vol"],
            "amount": latest_df.iloc[-1]["amount"]
        }

        latest_df = pd.concat([latest_df, pd.DataFrame([new_row])], ignore_index=True)
        latest_df["ma5"] = latest_df["close"].rolling(window=5).mean()
        latest_df["ma10"] = latest_df["close"].rolling(window=10).mean()
        latest_df["return_1d"] = latest_df["close"].pct_change(1)
        latest_df["vol_ma5"] = latest_df["vol"].rolling(window=5).mean()
        latest_df = latest_df.dropna().reset_index(drop=True)
        current_window_df = latest_df.iloc[-window_size:].copy()

        rolling_preds.append({
            "trade_date": next_date_str,
            "predicted_close": pred
        })

    # 合并结果保存
    df_eval = pd.DataFrame({
        "trade_date": dates,
        "real_close": y_true,
        "predicted_close": y_pred_eval
    })

    df_roll = pd.DataFrame(rolling_preds)
    final_df = pd.concat([df_eval, df_roll], ignore_index=True)
    csv_out_path = os.path.join("results", "rolling_prediction_RF_sliding.csv")
    final_df.to_csv(csv_out_path, index=False, encoding="utf-8-sig")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df_eval["trade_date"], df_eval["real_close"], label="真实收盘价", color="blue")
    plt.plot(df_eval["trade_date"], df_eval["predicted_close"], label="预测收盘价", color="orange", linestyle="--")
    if not df_roll.empty:
        plt.plot(df_roll["trade_date"], df_roll["predicted_close"], label="未来预测", color="green", linestyle=":")
    plt.xlabel("日期")
    plt.ylabel("收盘价")
    plt.title("预测结果与未来滚动预测 - RF 滑动窗口")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolve_path(os.path.join("results", "final_prediction_RF_sliding_rollN.png"), base="script"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
