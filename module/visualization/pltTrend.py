import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互环境下绘图
import matplotlib.pyplot as plt

# 跨平台字体设置
def set_chinese_font():
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    elif sys.platform == 'win32':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

# 主绘图函数
def plot_trend_signals_from_csv(csv_path: str, save_path: str):
    """
    从趋势信号 CSV 文件绘制价格走势图和买卖信号图
    :param csv_path: CSV 文件路径
    :param save_path: 图片保存路径（包含文件名）
    """
    set_chinese_font()

    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    trend_df = pd.read_csv(csv_path)
    trend_df["trade_date"] = pd.to_datetime(trend_df["trade_date"])

    plt.figure(figsize=(12, 6))
    plt.plot(trend_df["trade_date"], trend_df["true_close"], label="True Close", color="blue")

    # Buy signals
    buy_signals = trend_df[trend_df["trend_signal"] == 1]
    plt.scatter(buy_signals["trade_date"], buy_signals["true_close"],
                marker="^", color="green", label="Buy Signal", alpha=0.8)

    # Sell signals
    sell_signals = trend_df[trend_df["trend_signal"] == -1]
    plt.scatter(sell_signals["trade_date"], sell_signals["true_close"],
                marker="v", color="red", label="Sell Signal", alpha=0.8)

    plt.title("Trend Signals Visualization")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"✅ 趋势信号图已保存到 {save_path}")

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # visualization 目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # project0 目录
DATA_DIR = os.path.join(PROJECT_ROOT, "models", "predictors", "results")
PIC_DIR = os.path.join(BASE_DIR, "pic")

plot_trend_signals_from_csv(os.path.join(DATA_DIR, "lstm_inference_trend_signals.csv"), os.path.join(PIC_DIR, "trend_signals_plot.png"))
