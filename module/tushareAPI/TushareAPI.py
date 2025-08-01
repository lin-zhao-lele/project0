import os
import tushare as ts
from dotenv import load_dotenv

# 获取 TUSHARE_TOKEN
load_dotenv()
token = os.getenv('TUSHARE_TOKEN')
ts.set_token(token)
pro = ts.pro_api()

df = pro.index_daily(ts_code='002594.SZ')  # 沪深300指数示例
print(df.columns.tolist())  # 打印所有字段名



# 定义接口及其示例参数
interfaces = {
    "index_daily": {"ts_code": "000300.SH", "limit": 1},
    "index_weekly": {"ts_code": "000300.SH", "limit": 1},
    "index_monthly": {"ts_code": "000300.SH", "limit": 1},
    "index_dailybasic": {"trade_date": "20250731", "fields": ""},
    "daily": {"ts_code": "000001.SZ", "limit": 1},
    "daily_basic": {"trade_date": "20250731", "fields": ""},              # 日线基础数据（含总股本、流通股本）
    "weekly": {"ts_code": "000001.SZ", "limit": 1},
    "monthly": {"ts_code": "000001.SZ", "limit": 1},
    "stock_basic": {"exchange": "", "list_status": "L", "fields": ""},    # 股票基础信息
    "balancesheet": {"ts_code": "000001.SZ", "period": "20240331", "fields": ""}, # 财务报表 - 资产负债表

}

# 循环打印字段
for api_name, params in interfaces.items():
    try:
        df = getattr(pro, api_name)(**params)
        print(f"\n=== {api_name} ===")
        print("字段列表:", list(df.columns))
    except Exception as e:
        print(f"{api_name} 调用失败: {e}")

# # 获取最近一天的龙虎榜数据
# df_top_list = pro.top_list(trade_date='20230815')
# print(df_top_list.head())
#
#
# # 获取最近一天的机构席位交易数据
# df_top_inst = pro.top_inst(trade_date='20230815')
# print(df_top_inst)