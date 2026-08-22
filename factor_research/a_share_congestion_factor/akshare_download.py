"""
A股成交额集中度因子数据获取

数据源：AKShare stock_a_congestion_lg
用途：下载现成历史序列，作为聚宽研究前的数据入口。
"""

import akshare as ak
import pandas as pd


OUTPUT = "a_share_congestion.csv"


def main():
    df = ak.stock_a_congestion_lg()

    rename = {
        "日期": "date",
        "收盘价": "close",
        "拥挤度": "congestion",
    }
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if df["congestion"].median() > 1:
        df["congestion"] = df["congestion"] / 100

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(df.head())
    print(df.tail())
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
