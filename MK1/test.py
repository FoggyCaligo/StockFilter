import pandas as pd
import yfinance as yf

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
table = pd.read_html(url)
tickers = table[0]['Symbol'].tolist()


def select_ticker(ticker):
    
    draw_graph(ticker)



def get_sp500_tickers():
    table = pd.read_html(url)
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers


def draw_graph(ticker):
    data = yf.download(ticker, period='1mo', interval='1d')  # 최근 1개월 일별 데이터
    print(data)
def print_dividend_data(ticker):
    stock = yf.Ticker(ticker)
    dividends = stock.dividends  # Series: 날짜-배당금
    print(dividends)


