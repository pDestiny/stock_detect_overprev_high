import yfinance as yf
from pathlib import Path
import pandas as pd
import numpy as np
from toolz.curried import *
import datetime as dt
import re
import warnings
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from ta.trend import SMAIndicator

warnings.filterwarnings("ignore", category=UserWarning)

# 30


def load_stock_data(krx_excel_file:Path, period="3mo", namefilter=["인버스", "레버리지", "CD", "채"]):
    data = pd.read_excel(krx_excel_file, sheet_name="Sheet1", converters={
        "단축코드": str
    })
    data["상장일"] = pd.to_datetime(data["상장일"], format="%Y/%m/%d")
    # exclude symbol 265690.KS - 현재 상장 정지상태
    # etf should be opened before at least a year ago
    regex = re.compile("(" + "|".join(namefilter) + ")")
    stock_trade_stop_symbols = ["265690", '310970', '301400', '310960']
    stop_symbol_regex = re.compile("(" + "|".join(stock_trade_stop_symbols) + ")")

    filt_data = data[
        (data["상장일"] < dt.datetime.now() - dt.timedelta(days=365)) & \
        ~data["한글종목명"].str.contains(regex, regex=True) & \
        ~data["단축코드"].str.contains(stop_symbol_regex, regex=True)
    ]
    # load codes
    codes = pipe(filt_data["단축코드"], map(lambda x: f"{x}.KS"), list)
    # download stock data for default 3 month.
    stock_data = yf.download(",".join(codes), period=period)
    return stock_data["Close"], data[data["단축코드"].isin(filt_data["단축코드"])]

def find_pick(price_data:pd.Series):
    prices = price_data
    dates = np.arange(0, len(price_data))
    # 스플라인 보간
    spline_interp = interp1d(dates, prices, kind="cubic")

    # 보간 후 데이터 생성
    new_dates = np.linspace(0, len(price_data) - 1, 500)  # 연속 시간 생성
    spline_prices = spline_interp(new_dates)

    # 시각화
    # 스플라인 보간된 데이터에서 전고점 찾기
    peaks, _ = find_peaks(spline_prices)
    # 전고점에 해당하는 시간과 가격 데이터
    peak_dates = new_dates[peaks]
    peak_prices = spline_prices[peaks]
    try:
        previous_high = peak_prices[-2]
        current_high = peak_prices[-1]
    except IndexError:
        return pd.Series([
                price_data.name.replace(".KS", ""),
                -1,
                -1,
                False,
            ], index=["code", "prev_high", "current_high", "is_overprevhigh"]), price_data, dates, prices, new_dates, spline_prices, peak_dates, peak_prices

    return pd.Series(
        [
        price_data.name.replace(".KS", ""),
        previous_high,
        current_high,
        previous_high < current_high
    ], index=["code", "prev_high", "current_high", "is_overprevhigh"]), price_data, dates, prices, new_dates, spline_prices, peak_dates, peak_prices

def generate_figure(codename, price_data, dates, prices, new_dates, spline_prices, peak_dates, peak_prices):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, prices, "o", label="Original Data")  # 원래 데이터
    plt.plot(new_dates, spline_prices, label="Cubic Spline Interpolation", linestyle=":")
    plt.scatter(peak_dates, peak_prices, color="red", label="Peaks")  # 전고점 표시
    for i, value in enumerate(peak_prices):
        plt.annotate(f'{round(value)}', (peak_dates[i], peak_prices[i]), textcoords="offset points", xytext=(0, 5),
                     ha='center')
    plt.legend()
    plt.title(f"Stock({price_data.name}) Price Interpolation with Peaks")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.savefig(f"../Results/images/{price_data.name}.png")
    plt.close()

def is_lte_criteria(price_data):
    sma_data = SMAIndicator(price_data, window=20).sma_indicator()
    return price_data.iloc[-1] < sma_data.iloc[-1]

if __name__ == "__main__":
    krx_excel_file = Path("../Resources/data_0219_20241228.xlsx")
    stock_data, stock_info_data = load_stock_data(krx_excel_file, period="3mo", namefilter=["인버스", "레버리지", "CD", "채", "선물"])
    rows = []
    for col in stock_data.columns:
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


