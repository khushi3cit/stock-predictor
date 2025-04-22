import pandas as pd
import numpy as np

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 1. Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # 2. Average True Range (ATR) for volatility
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # 3. Stochastic Oscillator
    low_min = df['Low'].rolling(14).min()
    high_max = df['High'].rolling(14).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(3).mean()
    
    # 4. On Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # 5. Ichimoku Cloud
    conversion_line = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
    base_line = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
    df['Ichimoku_Conversion'] = conversion_line
    df['Ichimoku_Base'] = base_line
    df['Ichimoku_SpanA'] = ((conversion_line + base_line) / 2).shift(26)
    df['Ichimoku_SpanB'] = ((df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2).shift(26)
    
    # 6. Fibonacci Retracement Levels
    max_price = df['Close'].rolling(60).max()
    min_price = df['Close'].rolling(60).min()
    diff = max_price - min_price
    df['Fib_23.6'] = max_price - diff * 0.236
    df['Fib_38.2'] = max_price - diff * 0.382
    df['Fib_50.0'] = max_price - diff * 0.5
    df['Fib_61.8'] = max_price - diff * 0.618
    
    return df

