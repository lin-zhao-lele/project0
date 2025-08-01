import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# 构造保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
save_dir = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "raw")

# 解决中文乱码和负号问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 抓取最近5个交易日的机构净买入数据，并绘制净买入趋势图
# ===== 1. 配置 Tushare Token =====
# 获取 TUSHARE_TOKEN
load_dotenv()
token = os.getenv('TUSHARE_TOKEN')
ts.set_token(token)
pro = ts.pro_api()

# ===== 2. 获取最近 N 个交易日 =====
def get_last_n_trade_dates(n=5):
    today = datetime.now().strftime('%Y%m%d')
    cal = pro.trade_cal(exchange='SSE', start_date='20250720', end_date=today, is_open='1')
    trade_dates = cal['cal_date'].tolist()
    return trade_dates[-n:]  # 最近 n 天

trade_dates = get_last_n_trade_dates(5)
print(f"最近交易日: {trade_dates}")

# ===== 3. 获取每日机构净买入数据 =====
all_data = []

for trade_date in trade_dates:
    df = pro.top_inst(trade_date=trade_date)
    if df.empty:
        print(f"{trade_date} 没有数据")
        continue

    df['net_buy'] = df['buy'] - df['sell']
    daily_sum = df.groupby('ts_code', as_index=False).agg({'net_buy': 'sum'})
    daily_sum['trade_date'] = trade_date
    all_data.append(daily_sum)

# 合并所有日期数据
if not all_data:
    print("没有数据，退出。")
    exit()

df_all = pd.concat(all_data, ignore_index=True)

# ===== 4. 获取股票名称 =====
stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
df_all = pd.merge(df_all, stock_info, on='ts_code', how='left')

# ===== 5. 计算总净买入排名 =====
top_stocks = df_all.groupby('ts_code', as_index=False).agg({'net_buy': 'sum'})
top_stocks = top_stocks.sort_values(by='net_buy', ascending=False).head(10)
top_codes = top_stocks['ts_code'].tolist()

# ===== 6. 筛选前10股票的趋势数据 =====
trend_data = df_all[df_all['ts_code'].isin(top_codes)]

# ===== 7. 绘制趋势图 =====
plt.figure(figsize=(12, 6))

stock_info = pro.stock_basic(
    exchange='',
    list_status='L,P,D',  # L=上市, P=暂停上市, D=退市
    fields='ts_code,name'
)

for code in top_codes:
    row = stock_info.loc[stock_info['ts_code'] == code]
    if row.empty:
        stock_name = code  # 找不到就用代码代替
    else:
        stock_name = row['name'].values[0]

    df_plot = trend_data[trend_data['ts_code'] == code].sort_values('trade_date')
    plt.plot(df_plot['trade_date'], df_plot['net_buy'], marker='o', label=f"{stock_name} ({code})")


plt.title("近5日机构净买入趋势")
plt.xlabel("交易日期")
plt.ylabel("净买入额（元）")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
# plt.savefig("top_inst_trend.png", dpi=300)
plt.show()

# ===== 8. 保存CSV =====
# ===== 6. 输出到 CSV =====
csv_file_name = os.path.join(save_dir, f"最近5个交易日的机构净买入数据top_inst_{trade_dates[-1]}_{trade_dates[0]}.csv")
df_sorted = df_all.sort_values(by='net_buy', ascending=False)
df_sorted.to_csv(csv_file_name, index=False, encoding='utf-8-sig')
print("✅ 数据已保存到"+csv_file_name)
