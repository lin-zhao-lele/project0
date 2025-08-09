import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import create_sequences, build_model  # 保证 utils.py 可导入
import sys
from module.visualization.pltTrend import plot_trend_signals_from_csv

# 字体配置（跨平台）
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # predictors 目录
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 上级目录
MODEL_DIR = os.path.join(PROJECT_ROOT, "predictors", "resource")
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
config_path = os.path.join(PROJECT_ROOT, "predictors", "resource", "FinalModel_LSTM0_args.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

training_path = resolve_path(config["training"], base="data")
predict_path = resolve_path(config["predict"], base="data")
params = config["params"]
predict_length = config.get("predict_length", 5)       # 读取未来预测天数
model_path = resolve_path("FinalModel_LSTM0.pt", base="model")
print("model_path..." + model_path)
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
    scaler = MinMaxScaler()                                     # scaler 的第 4 列（索引 3）对应 "close" 做股价还原时要用
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


# ========== 滚动预测未来 N 天 ==========
last_window = X_future[-1]        # shape: (window_size, feature_dim)
future_preds = []
input_seq = last_window.copy()

for _ in range(predict_length):
    input_tensor = torch.tensor(input_seq[np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        next_pred = model(input_tensor).cpu().numpy().flatten()[0]
    future_preds.append(next_pred)

    # 构造新一日的特征输入（只更新 close）
    new_input = input_seq[-1].copy()
    close_idx = 3  # 'close'列在特征中是第4列
    new_input[close_idx] = next_pred

    # 滚动窗口更新
    input_seq = np.vstack([input_seq[1:], new_input])

# 构造日期
from datetime import datetime, timedelta

# ========== 构造日期 ==========
last_date_str = raw_df["trade_date"].values[-1]
last_date = datetime.strptime(str(last_date_str), "%Y%m%d")
future_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y%m%d") for i in range(predict_length)]

# ========== 汇总结果 + 保存 CSV ==========
trade_dates = raw_df["trade_date"].values[params["window_size"]:]
pred_df = pd.DataFrame({
    "trade_date": trade_dates,
    "true_close": y_future,
    "predicted_close": y_pred_future.flatten()
})

future_df = pd.DataFrame({
    "trade_date": future_dates,
    "true_close": [np.nan] * predict_length,
    "predicted_close": future_preds
})

full_df = pd.concat([pred_df, future_df], ignore_index=True)
pred_csv = os.path.join(RESULTS_DIR, "future_predictions_LSTM.csv")
full_df.to_csv(pred_csv, index=False)
print(f"Predictions saved to {pred_csv}")

# ========== 转换日期列为 datetime 类型 ==========
pred_df["trade_date"] = pd.to_datetime(pred_df["trade_date"], format="%Y%m%d")
full_df["trade_date"] = pd.to_datetime(full_df["trade_date"], format="%Y%m%d")

# ========== 绘图 ==========
plt.figure(figsize=(15, 8))
plt.plot(pred_df["trade_date"], pred_df["true_close"], label="True Close")
plt.plot(full_df["trade_date"], full_df["predicted_close"], label="Predicted Close")
plt.xlabel("Date")
plt.ylabel("Close Price")

# ========== 设置 x 轴刻度 ==========
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))  # 自动选择刻度位置，限制最大刻度数
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')  # 旋转刻度标签并右对齐

plt.title("Stock Price Prediction with Future Forecast")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "future_predictions_plot_LSTM.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")


# 延迟分析
# 读取预测文件
pred_df = pd.read_csv(os.path.join(RESULTS_DIR, "future_predictions_LSTM.csv"))

true_vals = pred_df["true_close"].values
pred_vals = pred_df["predicted_close"].values

# 计算最大滞后为 ±20 天的相关性
max_lag = 20
correlations = []
lags = range(-max_lag, max_lag+1)

for lag in lags:
    if lag < 0:
        corr = np.corrcoef(true_vals[:lag], pred_vals[-lag:])[0, 1]
    elif lag > 0:
        corr = np.corrcoef(true_vals[lag:], pred_vals[:-lag])[0, 1]
    else:
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
    correlations.append(corr)

# 找到相关性最大的延迟
# 做预测延迟分析。思路是：
# 把预测值和真实值做时间序列相关性分析
# 用 np.correlate 或 pandas.Series.corr 计算两个序列在不同时间偏移量（lag）下的相关系数。
# 找到相关系数最高时的偏移量，就是模型预测的平均延迟天数。
# 分析你的 140 条预测数据
# 如果延迟是正的（比如 lag=3），说明预测曲线整体比真实曲线晚 3 天；
# 如果延迟是负的（比如 lag=-2），说明预测曲线提前 2 天反应变化。

best_lag = lags[np.argmax(correlations)]
best_corr = max(correlations)

print(f"模型预测延迟约为 {best_lag} 天，对应最大相关系数 {best_corr:.4f}")


# ========== 生成趋势信号并回测准确率 ==========

# 阈值：预测涨跌幅小于该值时忽略信号（单位：百分比）
threshold_pct = 0.01  # 0.5%

# 计算预测涨跌幅
pred_df["predicted_change"] = pred_df["predicted_close"].diff() / pred_df["predicted_close"].shift(1)
pred_df["true_change"] = pred_df["true_close"].diff() / pred_df["true_close"].shift(1)

# 生成预测趋势信号（1 = 上涨，-1 = 下跌，0 = 无操作）
pred_df["trend_signal"] = 0
pred_df.loc[pred_df["predicted_change"] > threshold_pct, "trend_signal"] = 1
pred_df.loc[pred_df["predicted_change"] < -threshold_pct, "trend_signal"] = -1

# 生成真实趋势方向（用于验证）
pred_df["true_trend"] = 0
pred_df.loc[pred_df["true_change"] > 0, "true_trend"] = 1
pred_df.loc[pred_df["true_change"] < 0, "true_trend"] = -1

# 计算趋势方向准确率
valid_mask = pred_df["trend_signal"] != 0
accuracy = (pred_df.loc[valid_mask, "trend_signal"] == pred_df.loc[valid_mask, "true_trend"]).mean()

print(f"趋势信号准确率（过滤小波动后）: {accuracy:.2%}")
print(f"总信号数: {valid_mask.sum()} 条")

# 保存信号文件
trend_path = os.path.join(RESULTS_DIR, "trend_signals.csv")
pred_df.to_csv(trend_path, index=False)
print(f"趋势信号已保存到 {trend_path}")

plot_trend_signals_from_csv(trend_path, os.path.join(RESULTS_DIR, "trend_signals_plot.png"))
