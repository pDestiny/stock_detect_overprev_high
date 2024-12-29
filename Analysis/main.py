import argparse
from pathlib import Path
from utils import load_stock_data, find_pick, generate_figure, is_lte_criteria
import os
from toolz.curried import *
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--krx-etf-excel", type=Path, help="krx etf 데이터 http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC020103010901", default=Path("../Resources/data_0219_20241228.xlsx"), dest="krx_etf_excel")
parser.add_argument("--exclude-keywords", nargs="+", type=str, default=["인버스", "레버리지", "CD", "채", "달러"], dest="exclude_keywords")
parser.add_argument("--period", choices=["1mo", "3mo"], default="3mo")

args = parser.parse_args()

def main():
    # check image is already exist. remove them
    image_ls = pipe(os.listdir("../Results/images"), filter(lambda x: x.endswith(".png")))
    for image_path in image_ls:
        os.remove("../Results/images/" + image_path)

    krx_excel_file = args.krx_etf_excel
    stock_data, stock_info_data = load_stock_data(krx_excel_file, period=args.period,
                                                  namefilter=args.exclude_keywords)
    rows = []
    for col in tqdm(stock_data.columns):
        if is_lte_criteria(stock_data[col]): continue
        comp_row, price_data, dates, prices, new_dates, spline_prices, peak_dates, peak_prices = \
            find_pick(stock_data[col])
        if comp_row.is_overprevhigh:
            rows.append(comp_row)
            generate_figure(comp_row.code, price_data, dates, prices, new_dates, spline_prices, peak_dates, peak_prices)

    filtered_data = pd.DataFrame(rows).merge(stock_info_data, how="inner", left_on="code", right_on="단축코드")
    filtered_data.to_excel("../Results/overprev_high_result.xlsx")
    orig_price_data = stock_data.loc[:, [c + ".KS" for c in filtered_data["단축코드"]]]
    orig_price_data.to_excel("../Results/overprev_high_price_result.xlsx", index=True)

if __name__ == "__main__":
    main()