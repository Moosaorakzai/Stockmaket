import pandas as pd
import numpy as np

def calculate_moving_average(data, window=20):
    """Calculate the moving average for the given data."""
    return data.rolling(window=window).mean()

def calculate_exponential_moving_average(data, window=20):
    """Calculate the exponential moving average for the given data."""
    return data.ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for the given data.
    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given data.
    MACD Line = Fast EMA - Slow EMA
    Signal Line = EMA of MACD Line
    Histogram = MACD Line - Signal Line
    """
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'Signal_Line': signal_line,
        'Histogram': histogram
    })

def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands for the given data.
    Upper Band = MA + (Standard Deviation * num_std)
    Lower Band = MA - (Standard Deviation * num_std)
    """
    ma = calculate_moving_average(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    
    return pd.DataFrame({
        'MA': ma,
        'Upper_Band': upper_band,
        'Lower_Band': lower_band
    })

def calculate_atr(data, window=14):
    """
    Calculate the Average True Range (ATR) for the given data.
    True Range (TR) = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = Moving Average of TR over window period
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator for the given data.
    %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K
    """
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    
    k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'K': k,
        'D': d
    })

def calculate_fibonacci_retracement(high, low):
    """
    Calculate Fibonacci Retracement levels.
    Levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    """
    diff = high - low
    levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    
    retracement_levels = {}
    for level in levels:
        retracement_levels[f"{level*100}%"] = high - (diff * level)
    
    return retracement_levels

def calculate_volume_weighted_average_price(data):
    """
    Calculate Volume Weighted Average Price (VWAP).
    VWAP = Sum(Price * Volume) / Sum(Volume)
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    return vwap 