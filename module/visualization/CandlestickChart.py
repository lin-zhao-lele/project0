import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_chinese_font():
    if sys.platform == 'darwin':
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
    elif sys.platform == 'win32':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 路径配置
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # 脚本所在目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # 工程根目录
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "raw")
PIC_DIR = os.path.join(BASE_DIR, "pic")



def resolve_path(path_str, base="project"):
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    if base == "project":
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))
    elif base == "script":
        return os.path.normpath(os.path.join(BASE_DIR, path_str))
    elif base == "data":
        return os.path.normpath(os.path.join(DATA_DIR, path_str))


# 美化 Matplotlib / Seaborn 样式
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="whitegrid", palette="muted")



# ===== 1. 读取数据 =====
file_path = resolve_path("600519.SH_20250101_20250730_1day_A.csv", base="data")  # 你的股票CSV文件
df = pd.read_csv(file_path)

# 日期格式化
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
df = df.sort_values("trade_date").reset_index(drop=True)

# ===== 2. 技术指标计算 =====
# 移动均线
df['MA5'] = df['close'].rolling(5).mean()
df['MA20'] = df['close'].rolling(20).mean()
df['MA60'] = df['close'].rolling(60).mean()

# MACD
short_ema = df['close'].ewm(span=12, adjust=False).mean()
long_ema = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Hist'] = df['MACD'] - df['Signal']

# RSI (相对强弱指数)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# 布林带
df['BB_Mid'] = df['close'].rolling(20).mean()
df['BB_Upper'] = df['BB_Mid'] + 2*df['close'].rolling(20).std()
df['BB_Lower'] = df['BB_Mid'] - 2*df['close'].rolling(20).std()

# 涨跌幅
df['pct_change'] = df['close'].pct_change() * 100

# ===== 3. 图表绘制 =====
# 交互式 K线图（Plotly 深色主题）
#
# 成交量变化柱状图
#
# 均线分析（MA5、MA20、MA60）
#
# MACD 指标
#
# RSI（相对强弱指数）
#
# 布林带（Bollinger Bands）
#
# 收盘价分布图
#
# 涨跌幅趋势
#
# 成交量与价格关系散点图


# --- 3.1 K线 + 均线 ---
fig_candle = go.Figure(data=[go.Candlestick(
    x=df['trade_date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='red',
    decreasing_line_color='green'
)])
# 添加均线
fig_candle.add_trace(go.Scatter(x=df['trade_date'], y=df['MA5'], mode='lines', name='MA5', line=dict(color='orange')))
fig_candle.add_trace(go.Scatter(x=df['trade_date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
fig_candle.add_trace(go.Scatter(x=df['trade_date'], y=df['MA60'], mode='lines', name='MA60', line=dict(color='purple')))
fig_candle.update_layout(title="Candlestick chart + Moving Averages", template="plotly_dark", xaxis_rangeslider_visible=False)
fig_candle.show()

# --- 3.2 成交量 ---
fig_vol = px.bar(df, x='trade_date', y='vol', title="Volume change", template="plotly_dark")
fig_vol.show()

# --- 3.3 MACD ---
plt.figure(figsize=(12,5))
plt.bar(df['trade_date'], df['Hist'], color=np.where(df['Hist']>=0, 'red', 'green'))
plt.plot(df['trade_date'], df['MACD'], label='MACD', color='blue')
plt.plot(df['trade_date'], df['Signal'], label='Signal', color='orange')
plt.title("MACD index")
plt.legend()
plt.savefig(os.path.join(PIC_DIR, "MACD 指标.jpg"), dpi=300)
# plt.show()

# --- 3.4 RSI ---
plt.figure(figsize=(12,3))
plt.plot(df['trade_date'], df['RSI'], color='purple')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title("RSI (Relative Strength Index)")
plt.savefig(os.path.join(PIC_DIR, "相对强弱指数.jpg"), dpi=300)
# plt.show()

# --- 3.5 布林带 ---
plt.figure(figsize=(12,6))
plt.plot(df['trade_date'], df['close'], label='Close', color='blue')
plt.plot(df['trade_date'], df['BB_Mid'], label='Middle', color='orange')
plt.plot(df['trade_date'], df['BB_Upper'], label='Upper', color='green')
plt.plot(df['trade_date'], df['BB_Lower'], label='Lower', color='red')
plt.fill_between(df['trade_date'], df['BB_Upper'], df['BB_Lower'], color='lightgray', alpha=0.3)
plt.title("Bollinger Bands")
plt.legend()
plt.savefig(os.path.join(PIC_DIR, "布林带.jpg"), dpi=300)
# plt.show()

# --- 3.6 收盘价分布 ---
plt.figure(figsize=(8,5))
sns.histplot(df['close'], kde=True, color="royalblue")
plt.title("Closing price distribution")
plt.savefig(os.path.join(PIC_DIR, "收盘价分布.jpg"), dpi=300)
# plt.show()

# --- 3.7 涨跌幅时间序列 ---
plt.figure(figsize=(12,5))
sns.lineplot(x='trade_date', y='pct_change', data=df, color="crimson")
plt.axhline(0, color='black', linestyle="--")
plt.title("Daily change (%)")
plt.savefig(os.path.join(PIC_DIR, "日涨跌幅.jpg"), dpi=300)
# plt.show()

# --- 3.8 成交量与价格关系 ---
plt.figure(figsize=(8,5))
sns.scatterplot(x='vol', y='close', data=df, color="teal")
plt.title("Volume vs. Closing Price")
plt.savefig(os.path.join(PIC_DIR, "成交量 vs 收盘价.jpg"), dpi=300)
# plt.show()

