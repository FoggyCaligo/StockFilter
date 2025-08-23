import pandas as pd
import yfinance as yf

class StockFilter:
    def __init__(self):
        self.url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.tickers = pd.read_html(self.url)[0]['Symbol'].tolist()
        self.values={
            'drop_percent': 0.6,  # 전일대비 하락 퍼센트
            'dividend_yield': 7.0,  # 연 배당수익률 퍼센트
            'market_cap': 5000000000  # 시가총액 50억
        }
        self.filtered = []
        
        self.pages = {
            'list': 1,
            'detail': 2,
            'setting': 3,
        }

    def main(self):
        self.filter_stocks()

    def get_sp500_tickers(self):
        self.tickers = pd.read_html(self.url)[0]['Symbol'].tolist()
        # return self.tickers

    #-------------------------------------------------------------------------
    #필터링된 조건 반환 및 저장 : 
    def filter_stocks(self):
        self.get_sp500_tickers()
        self.filtered = []
        for ticker in self.tickers:
            if self.is_drop_percent_over(ticker) and self.is_dividend_yield_over(ticker) and self.is_market_cap_over(ticker):
                self.filtered.append(ticker)
    #종목 필터링 조건 1 : 전일대비 0.6% 이상 하락
    def is_drop_percent_over(self, ticker):
        #ticker(문자열) 종목의 전일대비 등락률이 -0.6% 이하(0.6% 이상 하락)면 True, 아니면 False 반환
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2d')
        if len(hist) >= 2:
            prev_close = hist['Close'][-2]
            last_close = hist['Close'][-1]
            change_pct = ((last_close - prev_close) / prev_close) * 100
            return change_pct <= -self.values['drop_percent']
        else:
            return False
    #종목필터링 조건 2 :  연 배당수익률 7% 이상
    def is_dividend_yield_over(self, ticker):
        #ticker(문자열) 종목의 연 배당수익률이 7% 이상이면 True, 아니면 False 반환
        stock = yf.Ticker(ticker)
        info = stock.info
        # yfinance의 'dividendYield'는 소수(0.07=7%)로 제공됨
        dividend_yield = info.get('dividendYield', None)
        if dividend_yield is not None:
            return dividend_yield >= self.values['dividend_yield'] / 100
        else:
            return False
    #종목 필터링 조건 3 : 시가총액 50억 이상
    def is_market_cap_over(self, ticker):
        #ticker(문자열) 종목의 시가총액이 50억(5,000,000,000) 이상이면 True, 아니면 False 반환
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', None)
        if market_cap is not None:
            return market_cap >= self.values['market_cap']
        else:
            return False
        




    #UI---------------------------------------------------------------------------------------------------------------------------
    # 티커 선택
    def select_ticker(self, ticker):
        self.print_graph(ticker)
        self.print_dividend_data(ticker)
    def get_stock_price(self,ticker):
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        if(price):
            return price
        hist = stock.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return '-'

    #종목 정보 출력
    def print_filtered_tickers(self):
        self.filter_stocks()
        for ticker in self.filtered:
            print(str(ticker)+' | '+str(self.get_stock_price(ticker)))

    #종목단위 출력
    def print_graph(self, ticker):
        data = yf.download(ticker, period='1mo', interval='1d')  # 최근 1개월 일별 데이터
        print(data)
    def print_dividend_data(self, ticker):
        stock = yf.Ticker(ticker)
        dividends = stock.dividends  # Series: 날짜-배당금
        print(dividends)
