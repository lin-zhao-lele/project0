import os
import json
import tushare as ts
import pandas as pd

def load_config(config_path="args.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_fields(field_type):
    # A类指标（交易数据）
    fields_A = ["open", "high", "low", "close", "vol", "amount"]

    # B类指标（交易数据 + 财务指标）
    fields_B = fields_A + ["adj_factor", "turnover_rate", "pe", "pb"]

    if field_type.upper() == "A":
        return fields_A
    elif field_type.upper() == "B":
        return fields_B
    else:
        raise ValueError("field_type 必须是 A 或 B")

def get_data(config):
    token = config["token"]
    ts.set_token(token)
    pro = ts.pro_api()

    stock_code = config["stock_code"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    freq = config["freq"]
    field_type = config["field_type"].upper()

    fields = get_fields(field_type)
    filename = f"{stock_code}_{start_date}_{end_date}_{freq}_{field_type}.csv"

    # 如果文件已存在则提示用户
    if os.path.exists(filename):
        print(f"文件已存在：{filename}，无需再次调用 API。")
        return

    # 根据频率选择API
    if freq.endswith("min"):
        # 分钟级别数据
        df = pro.stk_mins(ts_code=stock_code, start_date=start_date, end_date=end_date,
                          freq=freq, fields="ts_code,trade_time," + ",".join(fields))
    else:
        # 日级别数据
        df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date,
                       fields="ts_code,trade_date," + ",".join(fields))

    if df.empty:
        print("未获取到数据，请检查参数是否正确。")
        return

    # 保存 CSV
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"数据已保存到 {filename}")

if __name__ == "__main__":
    config = load_config("args.json")
    get_data(config)
