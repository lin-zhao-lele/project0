import tushare as ts
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv


# 构造保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
save_dir = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data", "raw")

# 获取机构净买入前 20 股票
# ===== 1. 配置 Tushare Token =====
load_dotenv()
token = os.getenv('TUSHARE_TOKEN')
ts.set_token(token)
pro = ts.pro_api()

# ===== 2. 设置查询日期（默认今天） =====
# trade_date = datetime.now().strftime('%Y%m%d')  # 格式: YYYYMMDD
# 如果要指定日期，比如 '20250731'
trade_date = '20250730'

# ===== 3. 获取机构席位交易数据 =====
df = pro.top_inst(trade_date=trade_date)

if df.empty:
    print(f"{trade_date} 没有数据")
else:
    # ===== 4. 计算净买入额并排序 =====
    df['net_buy'] = df['buy'] - df['sell']
    df_sorted = df.groupby(['ts_code'], as_index=False).agg({
        'buy': 'sum',
        'sell': 'sum',
        'net_buy': 'sum'
    }).sort_values(by='net_buy', ascending=False).head(20)

    # ===== 5. 获取股票名称 =====
    stock_info = pro.stock_basic(exchange='', list_status='L',
                                 fields='ts_code,name')
    df_final = pd.merge(df_sorted, stock_info, on='ts_code', how='left')

    # ===== 6. 输出到 CSV =====
    file_name = os.path.join(save_dir, f"机构净买入前20 top_inst_{trade_date}.csv")

    df_final.to_csv(file_name, index=False, encoding='utf-8-sig')

    print(f"✅ {trade_date} 机构净买入前 20 已保存到 {file_name}")
    print(df_final)
