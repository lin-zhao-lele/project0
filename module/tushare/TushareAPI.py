import tushare as ts
import pandas as pd

# ===== 1. 设置 Token =====
token = "514ab91812f8ed47b3836b320bde3204ead13ef27c6259bef25665d0"
ts.set_token(token)

# 初始化 pro_api
pro = ts.pro_api()

# ===== 2. 定义测试函数 =====
def test_api(api_name, func, **kwargs):
    print(f"\n=== 测试 {api_name} 接口 ===")
    try:
        df = func(**kwargs)
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[成功] {api_name} 返回 {len(df)} 行数据")
            print(df.head(3))
        else:
            print(f"[失败] {api_name} 返回空数据")
    except Exception as e:
        print(f"[错误] {api_name} 调用失败: {e}")

# ===== 3. 测试几个常用 API =====
# 股票基础信息
test_api("stock_basic", pro.stock_basic, exchange='', list_status='L', fields='ts_code,symbol,name,list_date')

# 日线行情
test_api("daily", pro.daily, ts_code='000001.SZ', start_date='20250101', end_date='20250120')

# 日线基础数据（含总股本、流通股本）
test_api("daily_basic", pro.daily_basic, ts_code='000001.SZ', start_date='20250101', end_date='20250120',
         fields='ts_code,trade_date,close,turnover_rate,pe,pe_ttm,pb,total_share,float_share,total_mv,circ_mv')

# 财务报表 - 资产负债表
test_api("balancesheet", pro.balancesheet, ts_code='000001.SZ', period='20240331',
         fields='ts_code,end_date,total_share')

print("\n=== 测试完成 ===")
