from talib import RSI, ADX, MACD, CCI


def count_indicators(quote_history):
    rsi = RSI(quote_history.get('Close'), 14)
    cci = CCI(quote_history.get('High'), quote_history.get('Low'), quote_history.get('Close'), 14)
    adx = ADX(quote_history.get('High'), quote_history.get('Low'), quote_history.get('Close'), 14)
    macd = MACD(quote_history.get('Close'), 14)[0]
    coefficient = list(quote_history['Close'] / quote_history['Open'])
    param_close = list(quote_history['Close'])
    indexes = list(quote_history.index)
    indicator = {
        'rsi': list(rsi),
        'cci': list(cci),
        'macd': list(macd),
        'adx': list(adx),
        'coefficient': coefficient,
        'Close': param_close,
        'index': indexes
    }
    return indicator
