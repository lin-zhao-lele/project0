import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os


# 蓝色线条 = 真实收盘价走势
# 绿色三角 ▲ = 模型预测的买入信号
# 红色三角 ▼ = 模型预测的卖出信号


# 字体配置（跨平台）
if sys.platform == 'darwin':
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
elif sys.platform == 'win32':
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # visualization 目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # project0 目录
DATA_DIR = os.path.join(PROJECT_ROOT, "models", "predictors", "results")
PIC_DIR = os.path.join(BASE_DIR, "pic")
os.makedirs(PIC_DIR, exist_ok=True)



# 读取趋势信号文件
trend_df = pd.read_csv(os.path.join(DATA_DIR, "trend_signals.csv"))

# 转换日期
trend_df["trade_date"] = pd.to_datetime(trend_df["trade_date"])

# 创建图形
plt.figure(figsize=(12, 6))
plt.plot(trend_df["trade_date"], trend_df["true_close"], label="True Close", color="blue")

# 绘制买入信号（绿色向上三角）
buy_signals = trend_df[trend_df["trend_signal"] == 1]
plt.scatter(buy_signals["trade_date"], buy_signals["true_close"], marker="^", color="green", label="Buy Signal", alpha=0.8)

# 绘制卖出信号（红色向下三角）
sell_signals = trend_df[trend_df["trend_signal"] == -1]
plt.scatter(sell_signals["trade_date"], sell_signals["true_close"], marker="v", color="red", label="Sell Signal", alpha=0.8)

# 图表设置
plt.title("Trend Signals Visualization")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图片
plt.savefig(os.path.join(PIC_DIR, "trend_signals_plot.png"), dpi=300)


print("趋势信号图已保存到 results/trend_signals_plot.png")
