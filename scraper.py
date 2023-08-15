import yfinance as yf


class Scraper:
    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.ticker = yf.Ticker(symbol)
        self.start_date = start_date
        self.end_date = end_date

    def parse_quotes(self):
        data = self.ticker.history(interval='1d', start=self.start_date, end=self.end_date)
        return data