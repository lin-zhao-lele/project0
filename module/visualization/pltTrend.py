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



def plot_stick_chart_from_csv(csv_path, features, output_path=None):
    """
    从 CSV 文件绘制指定列的折线图

    参数：
        csv_path (str): CSV 文件路径
        features (list): 要绘制的列名列表，例如 ["true_close", "predicted_close"]
        output_path (str): 保存图片路径（默认与 CSV 同目录，文件名为 stick_chart.png）
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到 CSV 文件: {csv_path}")

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 确保 trade_date 存在
    if "trade_date" not in df.columns:
        raise ValueError("❌ CSV 文件缺少 trade_date 列")

    # 日期格式转换
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")

    # 创建画布
    plt.figure(figsize=(10, 4))

    # 遍历需要绘制的列
    for col in features:
        if col in df.columns:
            plt.plot(df["trade_date"], df[col], label=col)
        else:
            print(f"⚠️ 列 {col} 不存在，已跳过。")

    # 坐标轴与标题
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("LSTM Inference Prediction with Future Forecast")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 默认输出路径
    if output_path is None:
        output_path = os.path.join(os.path.dirname(csv_path), "stick_chart.png")

    # 保存图片
    plt.savefig(output_path)
    print(f"📊 图表保存至 {output_path}")


# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "resource")
PIC_DIR = os.path.join(BASE_DIR, "pic")

plot_stick_chart_from_csv(
    csv_path=os.path.join(DATA_DIR, "trend_signals.csv"),
    features=["true_close", "predicted_close"],
    output_path=os.path.join(PIC_DIR, "lstm_inference_plot2.png")
)

plot_trend_signals_from_csv(os.path.join(DATA_DIR, "trend_signals.csv"),
                            os.path.join(PIC_DIR, "trend_signals_plot.png"))
