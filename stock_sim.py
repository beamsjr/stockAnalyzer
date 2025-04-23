import pandas as pd
import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict
import pickle
import time
import multiprocessing as mp
from tqdm.auto import tqdm
import threading
import functools

# --- Configuration ---
DATA_FOLDER = 'stock_data'
CACHE_FOLDER = 'indicator_cache'  # Folder to store pre-calculated indicators
INITIAL_CAPITAL = 160000.0

# --- Ranges (Keep small for testing) ---
PERCENTAGE_MIN = 100
PERCENTAGE_MAX = 100
PERCENTAGE_STEP = 1

# --- Rebalancing Configuration ---
# Rebalancing types 'TIME', 'THRESHOLD', 'VOLATILITY', 'COMBINED'
REBALANCE_TYPES = [ 'COMBINED'] 

# Fixed time interval (days)
REBALANCE_FREQ_MIN = 7
REBALANCE_FREQ_MAX = 7

# Threshold-based rebalancing (percentage deviation from target)
THRESHOLD_MIN = 10.0
THRESHOLD_MAX = 20.0
THRESHOLD_STEP = 10.0

# Volatility-based rebalancing (volatility multiplier)
VOLATILITY_LOOKBACK = 20  # Days to calculate volatility
VOLATILITY_THRESHOLD_MIN = 2.0
VOLATILITY_THRESHOLD_MAX = 3.0
VOLATILITY_THRESHOLD_STEP = 1.0

# Combined rebalancing weight (0.0 = all threshold, 1.0 = all volatility)
COMBINED_WEIGHT_MIN = 0.0
COMBINED_WEIGHT_MAX = 1.0
COMBINED_WEIGHT_STEP = 0.25

# Indicator Types to test
INDICATOR_TYPES_TO_TEST = ['SMA_CROSS', 'PRICE_VS_SMA', 'RSI', 'MACD', 'BOLLINGER', 'MULTI', 'NONE']

# Settings for each indicator
INDICATOR_PERIODS = {
    'SMA_CROSS': {'short': 50, 'long': 200},
    'PRICE_VS_SMA': {'short': 0, 'long': 200},
    'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BOLLINGER': {'period': 20, 'std_dev': 2},
    'MULTI': {},  # Will use weighted combination of other indicators
    'NONE': {'short': 0, 'long': 0}
}

# NEW: Weight ranges for each indicator in MULTI strategy
INDICATOR_WEIGHT_RANGES = {
    'SMA_CROSS': {'min': 0.0, 'max': 1.0, 'step': 0.1},
    'PRICE_VS_SMA': {'min': 0.0, 'max': 1.0, 'step': 0.1},
    'RSI': {'min': 0.0, 'max': 1.0, 'step':0.1},
    'MACD': {'min': 0.0, 'max': 1.0, 'step': 0.1},
    'BOLLINGER': {'min': 0.0, 'max': 1.0, 'step': 0.1}
}

# Default weights (will be overridden during testing)
INDICATOR_WEIGHTS = {
    'SMA_CROSS': 0.25,
    'PRICE_VS_SMA': 0.15,
    'RSI': 0.20,
    'MACD': 0.25,
    'BOLLINGER': 0.15
}
# --- Indicator Factor Range ---
INDICATOR_FACTOR_MIN = 0.0
INDICATOR_FACTOR_MAX = 1.0
INDICATOR_FACTOR_STEP = 0.25

# --- Other Config ---
SETTLEMENT_DAYS = 2
TOP_N_RESULTS = 5
FORCE_RECALCULATION = False  # Set to True to force recalculation of indicators

# Dynamically calculate required lookback
_lookbacks = {
    'SMA_CROSS': max(INDICATOR_PERIODS['SMA_CROSS']['short'], INDICATOR_PERIODS['SMA_CROSS']['long']),
    'PRICE_VS_SMA': INDICATOR_PERIODS['PRICE_VS_SMA']['long'],
    'RSI': INDICATOR_PERIODS['RSI']['period'] * 2,  # Need more data for reliable RSI
    'MACD': max(INDICATOR_PERIODS['MACD']['slow'], INDICATOR_PERIODS['MACD']['slow'] + INDICATOR_PERIODS['MACD']['signal']),
    'BOLLINGER': INDICATOR_PERIODS['BOLLINGER']['period'] * 2,
    'MULTI': 0,  # Will be calculated below
    'NONE': 0
}
# Set MULTI lookback to maximum of all included indicators
_lookbacks['MULTI'] = max([_lookbacks[ind] for ind in INDICATOR_WEIGHTS.keys()])
REQUIRED_LOOKBACK = max(_lookbacks.values())
RISK_FREE_RATE = 0.02  # Example: 2% annual risk-free rate


# --- Utility Functions ---
def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# --- Data Loading and Preprocessing ---
def load_stock_data(folder_path):
    """Loads and aligns stock data with vectorized operations where possible"""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"Error: No CSV files found: {folder_path}")
        return None, None
    
    print(f"Loading data from: {folder_path}")
    
    # Load all dataframes initially
    dataframes = {}
    tickers = []
    
    for f in all_files:
        try:
            ticker = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_csv(f, parse_dates=['Date'], index_col='Date')
            
            if 'Close' not in df.columns:
                print(f"Warn: No 'Close' in {f}")
                continue
                
            # Fill NaN values in place
            df['Close'].fillna(method='ffill', inplace=True)
            
            if df['Close'].isnull().any():
                print(f"Warn: Unresolved NaN in {f}")
                continue
                
            # Keep only Close column and rename
            df_close = df[['Close']].rename(columns={'Close': ticker})
            
            # Remove duplicated indices
            df_close = df_close[~df_close.index.duplicated(keep='first')]
            
            dataframes[ticker] = df_close
            tickers.append(ticker)
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dataframes:
        print("Error: No valid data.")
        return None, None
        
    # Find common date range
    print(f"\nAligning data for {len(tickers)} tickers...")
    
    # Get common index with vectorized operations
    common_index = None
    for ticker in tickers:
        if ticker in dataframes:
            if common_index is None:
                common_index = dataframes[ticker].index
            else:
                common_index = common_index.intersection(dataframes[ticker].index)
    
    if common_index is None or common_index.empty:
        print("Error: No common dates.")
        return None, None
        
    # Sort the common index
    common_index = common_index.sort_values()
    
    # Reindex all dataframes to common index
    aligned_data = []
    valid_tickers = []
    
    for ticker in tickers:
        if ticker in dataframes:
            df = dataframes[ticker].sort_index()
            df_reindexed = df.reindex(common_index).fillna(method='ffill').fillna(method='bfill')
            
            if not df_reindexed.loc[common_index].isnull().values.any():
                aligned_data.append(df_reindexed.loc[common_index])
                valid_tickers.append(ticker)
            else:
                print(f"Warn: Cannot align {ticker}. Skipping.")
    
    if not aligned_data:
        print("Error: No tickers aligned.")
        return None, None
        
    # Combine all dataframes into one
    try:
        combined_closes = pd.concat(aligned_data, axis=1)
        combined_closes.columns = valid_tickers
    except Exception as e:
        print(f"Error concat: {e}")
        return None, None
        
    if combined_closes.empty:
        print("Error: Aligned DataFrame empty.")
        return None, None
        
    print(f"Data aligned: {len(combined_closes)} days ({combined_closes.index.min().date()} to {combined_closes.index.max().date()}), Tickers: {', '.join(valid_tickers)}")
    
    # Final check for NaN values
    if combined_closes.isnull().values.any():
        print("Warn: NaNs post-align. Filling.")
        combined_closes = combined_closes.fillna(method='ffill').fillna(method='bfill')
        
        if combined_closes.isnull().values.any():
            print("Error: Unresolvable NaNs.")
            return None, None
            
    return combined_closes, valid_tickers


# --- Indicator Calculation Functions ---
def get_cache_filename(indicator_type, prices_df, tickers):
    """Generate a unique cache filename based on indicator type and data"""
    # Create a hash based on data shape, date range, and tickers
    data_hash = hash((prices_df.shape[0], prices_df.index[0], prices_df.index[-1], 
                     '_'.join(sorted(tickers))))
    return os.path.join(CACHE_FOLDER, f"indicator_{indicator_type}_{data_hash}.pkl")


def load_cached_indicators(indicator_type, prices_df, tickers):
    """Load pre-calculated indicators from cache if available"""
    if FORCE_RECALCULATION:
        return None
        
    cache_file = get_cache_filename(indicator_type, prices_df, tickers)
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                print(f"Loaded cached indicators for {indicator_type}")
                return cached_data
        except Exception as e:
            print(f"Error loading cached indicators: {e}")
            
    return None


def save_indicators_to_cache(indicator_type, indicators_data, prices_df, tickers):
    """Save calculated indicators to cache"""
    ensure_directory_exists(CACHE_FOLDER)
    cache_file = get_cache_filename(indicator_type, prices_df, tickers)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(indicators_data, f)
        print(f"Saved indicators for {indicator_type} to cache")
    except Exception as e:
        print(f"Error saving indicators to cache: {e}")

@functools.lru_cache(maxsize=1024)
def calculate_rsi_vectorized(prices_tuple, period=14):
    """Calculate RSI using vectorized operations with caching"""
    # Convert tuple back to Series for calculation
    prices = pd.Series(prices_tuple)
    
    # Calculate price changes
    delta = prices.diff()
    
    # Create two series: gain (positive changes) and loss (negative changes)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate initial averages
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Calculate subsequent averages using the RSI formula
    for i in range(period, len(delta)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd_vectorized(prices, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD using vectorized operations"""
    # Calculate EMAs
    ema_fast = prices.ewm(span=fast_period, min_periods=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period, min_periods=slow_period).mean()
    
    # Calculate MACD line and signal line
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands_vectorized(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands using vectorized operations"""
    # Calculate rolling mean (middle band)
    middle_band = prices.rolling(window=period, min_periods=period).mean()
    
    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=period, min_periods=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return {
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band
    }


def calculate_indicators(prices_df, tickers, indicator_type, indicator_periods):
    """
    Calculates required indicators upfront with caching.
    Uses vectorized operations where possible.
    """
    # Check cache first
    cached_indicators = load_cached_indicators(indicator_type, prices_df, tickers)
    if cached_indicators is not None:
        return cached_indicators
        
    start_time = time.time()
    indicator_data = {'type': indicator_type}
    
    if indicator_type == 'NONE':
        print("Skipping indicator calculation (Type: NONE)")
        return indicator_data
    
    print(f"Calculating Indicators (Type: {indicator_type})...")
    
    # Calculate all needed indicators for this type
    if indicator_type == 'SMA_CROSS' or indicator_type == 'MULTI':
        short_period = INDICATOR_PERIODS['SMA_CROSS']['short']
        long_period = INDICATOR_PERIODS['SMA_CROSS']['long']
        
        # Use rolling mean (vectorized operation)
        indicator_data['sma_short'] = prices_df[tickers].rolling(
            window=short_period, min_periods=short_period).mean()
        indicator_data['sma_long'] = prices_df[tickers].rolling(
            window=long_period, min_periods=long_period).mean()
    
    if indicator_type == 'PRICE_VS_SMA' or indicator_type == 'MULTI':
        long_period = INDICATOR_PERIODS['PRICE_VS_SMA']['long']
        
        # Use rolling mean (vectorized operation)
        indicator_data['price_sma'] = prices_df[tickers].rolling(
            window=long_period, min_periods=long_period).mean()
    
    if indicator_type == 'RSI' or indicator_type == 'MULTI':
        period = INDICATOR_PERIODS['RSI']['period']
        
        # Modified approach for each ticker
        indicator_data['rsi'] = pd.DataFrame(index=prices_df.index, columns=tickers)
        
        for ticker in tickers:
            # Convert Series to tuple for caching
            prices_tuple = tuple(prices_df[ticker].values)
            # Call cached function with tuple
            indicator_data['rsi'][ticker] = calculate_rsi_vectorized(prices_tuple, period)
            
    
    if indicator_type == 'MACD' or indicator_type == 'MULTI':
        fast = INDICATOR_PERIODS['MACD']['fast']
        slow = INDICATOR_PERIODS['MACD']['slow']
        signal = INDICATOR_PERIODS['MACD']['signal']
        
        # Calculate MACD (vectorized)
        macd_result = calculate_macd_vectorized(prices_df[tickers], fast, slow, signal)
        indicator_data['macd_line'] = macd_result['macd_line']
        indicator_data['macd_signal'] = macd_result['signal_line']
        indicator_data['macd_hist'] = macd_result['histogram']
    
    if indicator_type == 'BOLLINGER' or indicator_type == 'MULTI':
        period = INDICATOR_PERIODS['BOLLINGER']['period']
        std_dev = INDICATOR_PERIODS['BOLLINGER']['std_dev']
        
        # Calculate Bollinger Bands (vectorized)
        bb_result = calculate_bollinger_bands_vectorized(prices_df[tickers], period, std_dev)
        indicator_data['bb_middle'] = bb_result['middle']
        indicator_data['bb_upper'] = bb_result['upper']
        indicator_data['bb_lower'] = bb_result['lower']
    
    # Save to cache
    save_indicators_to_cache(indicator_type, indicator_data, prices_df, tickers)
    
    elapsed_time = time.time() - start_time
    print(f"Indicators calculated in {elapsed_time:.2f} seconds")
    
    return indicator_data


def get_indicator_signals_vectorized(current_date, tickers, current_prices, indicator_data, indicator_type):
    """Get trading signals from indicators for all tickers using vectorized operations where possible"""
    if indicator_type == 'NONE':
        # Equal weight for all tickers - fast return
        return pd.Series(1.0, index=tickers)
    
    # Initialize signals with zeros
    signals = pd.Series(0.0, index=tickers)
    
    # Extract current date data for all indicators at once
    current_data = {}
    
    # Get all relevant indicator values for current date
    for key, df in indicator_data.items():
        if isinstance(df, pd.DataFrame) and key != 'type' and current_date in df.index:
            current_data[key] = df.loc[current_date]
    
    # Current prices as Series
    prices = pd.Series(current_prices)
    
    # Apply indicator logic based on type
    if indicator_type == 'SMA_CROSS':
        if 'sma_short' in current_data and 'sma_long' in current_data:
            sma_short = current_data['sma_short']
            sma_long = current_data['sma_long']
            
            # Calculate signal strength
            valid_mask = (~pd.isna(sma_short)) & (~pd.isna(sma_long)) & (sma_long > 0)
            cross_signal = pd.Series(0.0, index=tickers)
            
            if valid_mask.any():
                # Calculate the ratio for valid tickers
                valid_tickers = valid_mask[valid_mask].index
                ratio = sma_short[valid_tickers] / sma_long[valid_tickers] - 1.0
                
                # Apply signal strength (only where short > long)
                cross_signal[valid_tickers] = np.where(ratio > 0, 
                                                     np.minimum(ratio * 10, 1.0), 
                                                     0.0)
            signals = cross_signal
            
    elif indicator_type == 'PRICE_VS_SMA':
        if 'price_sma' in current_data:
            price_sma = current_data['price_sma']
            
            # Calculate signal strength
            valid_mask = (~pd.isna(price_sma)) & (~pd.isna(prices)) & (price_sma > 0) & (prices > 0)
            price_signal = pd.Series(0.0, index=tickers)
            
            if valid_mask.any():
                # Calculate the ratio for valid tickers
                valid_tickers = valid_mask[valid_mask].index
                ratio = prices[valid_tickers] / price_sma[valid_tickers] - 1.0
                
                # Apply signal strength (only where price > sma)
                price_signal[valid_tickers] = np.where(ratio > 0, 
                                                     np.minimum(ratio * 10, 1.0), 
                                                     0.0)
            signals = price_signal
            
    elif indicator_type == 'RSI':
        if 'rsi' in current_data:
            rsi_val = current_data['rsi']
            oversold = INDICATOR_PERIODS['RSI']['oversold']
            overbought = INDICATOR_PERIODS['RSI']['overbought']
            
            # Calculate signal strength
            valid_mask = ~pd.isna(rsi_val)
            rsi_signal = pd.Series(0.0, index=tickers)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                
                # Below oversold (strong buy)
                below_oversold = rsi_val[valid_tickers] < oversold
                if below_oversold.any():
                    below_tickers = below_oversold[below_oversold].index
                    rsi_signal[below_tickers] = np.minimum((oversold - rsi_val[below_tickers]) / 10, 1.0)
                
                # Middle range
                middle_range = (rsi_val[valid_tickers] >= oversold) & (rsi_val[valid_tickers] <= overbought)
                if middle_range.any():
                    middle_tickers = middle_range[middle_range].index
                    rsi_signal[middle_tickers] = 0.5 - (rsi_val[middle_tickers] - 50) / 40
            
            signals = rsi_signal
            
    elif indicator_type == 'MACD':
        if 'macd_line' in current_data and 'macd_signal' in current_data:
            macd_line = current_data['macd_line']
            signal_line = current_data['macd_signal']
            histogram = current_data.get('macd_hist')
            
            # Calculate signal strength
            valid_mask = (~pd.isna(macd_line)) & (~pd.isna(signal_line))
            macd_signal = pd.Series(0.0, index=tickers)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                
                # MACD above signal line (bullish)
                bullish = macd_line[valid_tickers] > signal_line[valid_tickers]
                
                if bullish.any():
                    bullish_tickers = bullish[bullish].index
                    
                    # If histogram available, use it for strength
                    if histogram is not None and not pd.isna(histogram[bullish_tickers]).all():
                        hist_valid = ~pd.isna(histogram[bullish_tickers])
                        hist_tickers = hist_valid[hist_valid].index
                        
                        if len(hist_tickers) > 0 and not pd.isna(prices[hist_tickers]).all():
                            price_valid = ~pd.isna(prices[hist_tickers])
                            price_tickers = price_valid[price_valid].index
                            
                            if len(price_tickers) > 0:
                                # Use histogram for signal strength
                                hist_ratio = histogram[price_tickers] / prices[price_tickers] * 100
                                macd_signal[price_tickers] = np.minimum(np.maximum(hist_ratio, 0), 1.0)
                    
                    # If histogram not available, use MACD-signal difference
                    else:
                        price_valid = ~pd.isna(prices[bullish_tickers])
                        price_tickers = price_valid[price_valid].index
                        
                        if len(price_tickers) > 0:
                            diff_ratio = (macd_line[price_tickers] - signal_line[price_tickers]) / prices[price_tickers] * 100
                            macd_signal[price_tickers] = np.minimum(np.maximum(diff_ratio, 0), 1.0)
            
            signals = macd_signal
            
    elif indicator_type == 'BOLLINGER':
        if all(k in current_data for k in ['bb_lower', 'bb_middle', 'bb_upper']):
            bb_lower = current_data['bb_lower']
            bb_middle = current_data['bb_middle']
            bb_upper = current_data['bb_upper']
            
            # Calculate signal strength
            valid_mask = (~pd.isna(bb_lower)) & (~pd.isna(bb_middle)) & (~pd.isna(bb_upper)) & (~pd.isna(prices))
            bb_signal = pd.Series(0.0, index=tickers)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                
                # Calculate band width
                band_width = bb_upper[valid_tickers] - bb_lower[valid_tickers]
                valid_width = band_width > 0
                
                if valid_width.any():
                    width_tickers = valid_width[valid_width].index
                    
                    # Below middle band (potentially bullish)
                    below_middle = prices[width_tickers] < bb_middle[width_tickers]
                    
                    if below_middle.any():
                        below_tickers = below_middle[below_middle].index
                        relative_pos = (prices[below_tickers] - bb_lower[below_tickers]) / band_width[below_tickers]
                        bb_signal[below_tickers] = np.maximum(0, 0.5 - relative_pos)
                    
                    # Above middle band (decreasing bullish signal)
                    above_tickers = below_middle[~below_middle].index
                    if len(above_tickers) > 0:
                        relative_pos = (prices[above_tickers] - bb_middle[above_tickers]) / (bb_upper[above_tickers] - bb_middle[above_tickers])
                        bb_signal[above_tickers] = np.maximum(0, 0.5 - relative_pos * 0.5)
            
            signals = bb_signal
            
    elif indicator_type == 'MULTI':
        multi_signal = pd.Series(0.0, index=tickers)
        weight_applied = pd.Series(0.0, index=tickers)
        
        # Calculate weighted signals from each indicator
        # SMA CROSS
        if all(k in current_data for k in ['sma_short', 'sma_long']):
            sma_short = current_data['sma_short']
            sma_long = current_data['sma_long']
            
            valid_mask = (~pd.isna(sma_short)) & (~pd.isna(sma_long)) & (sma_long > 0)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                sma_cross_signal = (sma_short[valid_tickers] > sma_long[valid_tickers]).astype(float)
                
                # Apply weight
                multi_signal[valid_tickers] += INDICATOR_WEIGHTS['SMA_CROSS'] * sma_cross_signal
                weight_applied[valid_tickers] += INDICATOR_WEIGHTS['SMA_CROSS']
        
        # PRICE VS SMA
        if 'price_sma' in current_data:
            price_sma = current_data['price_sma']
            
            valid_mask = (~pd.isna(price_sma)) & (~pd.isna(prices)) & (price_sma > 0)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                price_sma_signal = (prices[valid_tickers] > price_sma[valid_tickers]).astype(float)
                
                # Apply weight
                multi_signal[valid_tickers] += INDICATOR_WEIGHTS['PRICE_VS_SMA'] * price_sma_signal
                weight_applied[valid_tickers] += INDICATOR_WEIGHTS['PRICE_VS_SMA']
        
        # RSI
        if 'rsi' in current_data:
            rsi_val = current_data['rsi']
            
            valid_mask = ~pd.isna(rsi_val)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                # Convert RSI to signal (1.0 at RSI 30, 0.5 at RSI 50, 0.0 at RSI 70)
                rsi_signal = np.maximum(0, np.minimum(1, (50 - rsi_val[valid_tickers]) / 20 + 0.5))
                
                # Apply weight
                multi_signal[valid_tickers] += INDICATOR_WEIGHTS['RSI'] * rsi_signal
                weight_applied[valid_tickers] += INDICATOR_WEIGHTS['RSI']
        
        # MACD
        if 'macd_line' in current_data and 'macd_signal' in current_data:
            macd_line = current_data['macd_line']
            signal_line = current_data['macd_signal']
            
            valid_mask = (~pd.isna(macd_line)) & (~pd.isna(signal_line))
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                macd_signal = (macd_line[valid_tickers] > signal_line[valid_tickers]).astype(float)
                
                # Apply weight
                multi_signal[valid_tickers] += INDICATOR_WEIGHTS['MACD'] * macd_signal
                weight_applied[valid_tickers] += INDICATOR_WEIGHTS['MACD']
        
        # BOLLINGER
        if all(k in current_data for k in ['bb_lower', 'bb_middle', 'bb_upper']):
            bb_lower = current_data['bb_lower']
            bb_middle = current_data['bb_middle']
            bb_upper = current_data['bb_upper']
            
            valid_mask = (~pd.isna(bb_lower)) & (~pd.isna(bb_middle)) & (~pd.isna(bb_upper)) & (~pd.isna(prices))
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                band_width = bb_upper[valid_tickers] - bb_lower[valid_tickers]
                
                valid_width = band_width > 0
                if valid_width.any():
                    width_tickers = valid_width[valid_width].index
                    # Normalize position (0 at lower, 1 at upper)
                    norm_pos = (prices[width_tickers] - bb_lower[width_tickers]) / band_width[width_tickers]
                    # Convert to signal (1 near lower, 0 near upper)
                    bb_signal = np.maximum(0, 1 - norm_pos)
                    
                    # Apply weight
                    multi_signal[width_tickers] += INDICATOR_WEIGHTS['BOLLINGER'] * bb_signal
                    weight_applied[width_tickers] += INDICATOR_WEIGHTS['BOLLINGER']
        
        # Normalize by applied weights
        valid_weight = weight_applied > 0
        if valid_weight.any():
            valid_tickers = valid_weight[valid_weight].index
            signals[valid_tickers] = multi_signal[valid_tickers] / weight_applied[valid_tickers]
    
    return signals


def calculate_sharpe_ratio(daily_values_series, risk_free_rate_annual=0.0):
    """Calculates the annualized Sharpe ratio for a series of portfolio values."""
    if daily_values_series is None or len(daily_values_series) < 20: # Need sufficient data
        return np.nan
    # Calculate daily returns
    daily_returns = daily_values_series.pct_change().dropna()
    if len(daily_returns) < 2: # Need at least 2 returns for std dev
        return np.nan

    # Calculate excess returns
    risk_free_rate_daily = risk_free_rate_annual / 252.0
    excess_returns = daily_returns - risk_free_rate_daily

    # Calculate Sharpe Ratio components
    avg_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()

    # Handle cases with zero volatility
    if std_dev_excess_return is None or pd.isna(std_dev_excess_return) or std_dev_excess_return < 1e-9:
        return 0.0 if abs(avg_excess_return) < 1e-9 else np.nan

    # Calculate annualized Sharpe Ratio
    daily_sharpe = avg_excess_return / std_dev_excess_return
    annualized_sharpe = daily_sharpe * np.sqrt(252)

    return annualized_sharpe


# --- Optimized Rebalancing Decision Functions ---
def should_rebalance_threshold(shares, current_prices, target_allocations, threshold_pct):
    """
    Determines if rebalancing is needed based on allocation drift threshold
    Returns True if any position has drifted by more than threshold_pct
    """
    if not shares or not len(current_prices) or not len(target_allocations):
        return False
    
    # Convert to pandas Series for vectorized operations
    shares_series = pd.Series(shares)
    prices_series = pd.Series(current_prices)
    target_alloc_series = pd.Series(target_allocations)
    
    # Calculate current values and total portfolio value
    valid_mask = (~pd.isna(prices_series)) & (prices_series > 0) & (shares_series > 0)
    
    if not valid_mask.any():
        return False
        
    valid_tickers = valid_mask[valid_mask].index
    
    # Calculate position values
    position_values = shares_series[valid_tickers] * prices_series[valid_tickers]
    total_value = position_values.sum()
    
    if total_value <= 0:
        return False
    
    # Calculate current allocation percentages
    current_alloc_pct = (position_values / total_value * 100)
    
    # Calculate drift for each position
    drift = np.abs(current_alloc_pct - target_alloc_series[valid_tickers])
    
    # Check if any position exceeds threshold
    return (drift > threshold_pct).any()


def should_rebalance_volatility(portfolio_volatility, volatility_threshold):
    """
    Determines if rebalancing is needed based on portfolio volatility
    Returns True if portfolio volatility exceeds threshold
    """
    if pd.isna(portfolio_volatility) or portfolio_volatility <= 0:
        return False
    
    return portfolio_volatility > volatility_threshold


def should_rebalance_combined(shares, current_prices, target_allocations, 
                              threshold_pct, portfolio_volatility, volatility_threshold, 
                              combined_weight):
    """
    Combined rebalancing decision using both threshold and volatility metrics
    combined_weight: 0.0 = all threshold, 1.0 = all volatility
    """
    threshold_signal = should_rebalance_threshold(shares, current_prices, target_allocations, threshold_pct)
    volatility_signal = should_rebalance_volatility(portfolio_volatility, volatility_threshold)
    
    threshold_component = (1 - combined_weight) * (1 if threshold_signal else 0)
    volatility_component = combined_weight * (1 if volatility_signal else 0)
    
    # Rebalance if combined score > 0.5
    return (threshold_component + volatility_component) > 0.5


def simulate_rebalancing(prices_df, tickers, initial_capital, target_stock_percent,
                        rebalance_config, indicator_config, indicator_factor,
                        settlement_days, precalculated_indicators, indicator_weights=None):
    """
    Simulates rebalancing using indicator signals for allocation, with alternative rebalancing triggers.
    Optimized for performance with vectorized operations where possible.
    
    rebalance_config: {
        'type': 'TIME', 'THRESHOLD', 'VOLATILITY', or 'COMBINED',
        'param1': value,  # freq_days, threshold_pct, or volatility_threshold
        'param2': value,  # only for COMBINED (volatility_threshold)
        'param3': value   # only for COMBINED (combined_weight)
    }
    """

    # If indicator weights are provided, use them locally
    global INDICATOR_WEIGHTS
    original_weights = None
    
    if indicator_weights is not None and indicator_config['type'] == 'MULTI':
        original_weights = INDICATOR_WEIGHTS.copy()
        INDICATOR_WEIGHTS = indicator_weights

    if prices_df is None or prices_df.empty or not tickers:
        return None
    n_stocks = len(tickers)
    if n_stocks == 0:
        return None

    indicator_type = indicator_config['type']
    required_lookback = _lookbacks.get(indicator_type, 0)
    rebalance_type = rebalance_config['type']

    target_stock_ratio = target_stock_percent / 100.0

    # Initialize portfolio state
    portfolio_values = pd.Series(index=prices_df.index, dtype=float)
    portfolio_returns = pd.Series(index=prices_df.index, dtype=float)
    total_cash = initial_capital
    shares = {t: 0 for t in tickers}
    cost_basis = {t: {'total_cost': 0.0, 'total_shares': 0} for t in tickers}
    stock_value_history = pd.DataFrame(0.0, index=prices_df.index, columns=tickers, dtype=float)
    rebalance_history = pd.Series(0, index=prices_df.index, dtype=int)  # Track rebalancing events
    unsettled_cash = defaultdict(float)
    transaction_log = []
    
    # Calculate portfolio volatility series for volatility-based rebalancing
    portfolio_volatility = pd.Series(index=prices_df.index, dtype=float)
    days_since_last_rebalance = 0

    # --- Initial Buy (Equal Weight) ---
    if len(prices_df) > 0:
        first_date = prices_df.index[0]
        first_day_prices = prices_df.iloc[0]
        target_total_stock_value = initial_capital * target_stock_ratio
        target_inv_per_stock_init = target_total_stock_value / n_stocks if n_stocks > 0 else 0
        actual_stock_investment = 0.0
        settled_cash_for_init_buy = total_cash
        running_settled_cash_init = total_cash
        
        for ticker in tickers:
            price = first_day_prices.get(ticker)
            if price is not None and not pd.isna(price) and price > 0:
                affordable_shares = math.floor(target_inv_per_stock_init / price)
                cost = affordable_shares * price
                if settled_cash_for_init_buy >= cost and affordable_shares > 0:
                    cash_before = running_settled_cash_init
                    shares[ticker] = affordable_shares
                    total_cash -= cost
                    settled_cash_for_init_buy -= cost
                    running_settled_cash_init -= cost
                    cash_after = running_settled_cash_init
                    actual_stock_investment += cost
                    cost_basis[ticker]['total_shares'] = affordable_shares
                    cost_basis[ticker]['total_cost'] = cost
                    stock_value_history.loc[first_date, ticker] = affordable_shares * price
                    transaction_log.append({'Date': first_date.date(), 'Ticker': ticker, 'Action': 'Buy',
                                           'Shares': affordable_shares, 'Price': price,
                                           'Settled Cash Before': cash_before, 'Settled Cash After': cash_after})
        portfolio_values.iloc[0] = actual_stock_investment + total_cash
        rebalance_history.iloc[0] = 1  # Initial buy counts as a rebalance
        days_since_last_rebalance = 0
    else:
        return None

    # --- Daily Simulation Loop ---
    all_dates = prices_df.index
    
    # Pre-calculate portfolio returns and volatility
    if rebalance_type in ['VOLATILITY', 'COMBINED']:
        # We'll calculate these as we go, since they depend on portfolio_values
        pass
        
    for i in range(1, len(prices_df)):
        current_date = all_dates[i]
        current_prices = prices_df.iloc[i]
        days_since_last_rebalance += 1

        # Settle cash
        settled_today = 0.0
        keys_to_remove = []
        for settle_date, amount in unsettled_cash.items():
            if settle_date <= current_date:
                settled_today += amount
                keys_to_remove.append(settle_date)
        for key in keys_to_remove:
            del unsettled_cash[key]

        # Calculate values
        current_stock_value = 0.0
        valid_prices_today = True
        
        # Vectorized calculation of position values
        shares_series = pd.Series({ticker: shares[ticker] for ticker in tickers})
        price_series = pd.Series({ticker: current_prices.get(ticker, 0) for ticker in tickers})
        
        # Filter valid positions
        valid_mask = (~pd.isna(price_series)) & (price_series > 0) & (shares_series > 0)
        
        if valid_mask.any():
            valid_tickers = valid_mask[valid_mask].index
            position_values = shares_series[valid_tickers] * price_series[valid_tickers]
            
            # Update stock value history
            for ticker in tickers:
                value_today = shares[ticker] * price_series.get(ticker, 0) if pd.notna(price_series.get(ticker, 0)) else 0
                stock_value_history.loc[current_date, ticker] = value_today
                current_stock_value += value_today
        else:
            valid_prices_today = False
            
        current_total_value = current_stock_value + total_cash
        portfolio_values.iloc[i] = current_total_value
        
        # Calculate daily return
        if i > 0 and portfolio_values.iloc[i-1] > 0:
            portfolio_returns.iloc[i] = portfolio_values.iloc[i] / portfolio_values.iloc[i-1] - 1
        
        # Calculate portfolio volatility (rolling 20-day)
        if i >= VOLATILITY_LOOKBACK:
            volatility = portfolio_returns.iloc[i-VOLATILITY_LOOKBACK+1:i+1].std() * np.sqrt(252)  # Annualized
            portfolio_volatility.iloc[i] = volatility

        if current_total_value <= 0:  # Bankruptcy
            portfolio_values.iloc[i:] = 0
            stock_value_history.iloc[i:] = 0.0
            return {'daily_values': portfolio_values, 'final_cash': total_cash, 'final_shares': shares,
                   'final_cost_basis': cost_basis, 'stock_value_history': stock_value_history,
                   'unsettled_cash': unsettled_cash, 'transaction_log': transaction_log,
                   'rebalance_history': rebalance_history}

        # --- Rebalancing Decision Logic ---
        should_rebalance = False
        
        # Only check rebalancing if we have valid prices
        if valid_prices_today:
            # Calculate target allocations for threshold check
            use_indicator_signal = (indicator_type != 'NONE') and (i >= required_lookback)
            base_equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0
            indicator_signal_weights = pd.Series(base_equal_weight, index=tickers)  # Default

            if use_indicator_signal:
                # Get indicator signals with vectorized operations
                raw_signals = get_indicator_signals_vectorized(current_date, tickers, current_prices, 
                                                            precalculated_indicators, indicator_type)
                
                # Normalize signals
                signal_sum = raw_signals.sum()
                if signal_sum > 0:
                    indicator_signal_weights = raw_signals / signal_sum
            
            # Blend weights (vectorized operation)
            final_weights = ((1 - indicator_factor) * base_equal_weight +
                             indicator_factor * indicator_signal_weights)
            
            # Re-normalize (vectorized operation)
            weight_sum = final_weights.sum()
            if weight_sum > 1e-6:
                normalized_weights = final_weights / weight_sum
            else:
                normalized_weights = pd.Series(base_equal_weight, index=tickers)
            
            # Calculate target allocations (% of total stock investment)
            target_allocations = normalized_weights * 100  # Convert to percentages
            
            # Decide whether to rebalance based on selected strategy
            if rebalance_type == 'TIME':
                freq_days = rebalance_config['param1']
                should_rebalance = (days_since_last_rebalance >= freq_days)
            
            elif rebalance_type == 'THRESHOLD':
                threshold_pct = rebalance_config['param1']
                should_rebalance = should_rebalance_threshold(shares, current_prices, target_allocations, threshold_pct)
            
            elif rebalance_type == 'VOLATILITY':
                volatility_threshold = rebalance_config['param1']
                current_volatility = portfolio_volatility.iloc[i] if pd.notna(portfolio_volatility.iloc[i]) else 0
                should_rebalance = should_rebalance_volatility(current_volatility, volatility_threshold)
            
            elif rebalance_type == 'COMBINED':
                threshold_pct = rebalance_config['param1']
                volatility_threshold = rebalance_config['param2']
                combined_weight = rebalance_config['param3']
                current_volatility = portfolio_volatility.iloc[i] if pd.notna(portfolio_volatility.iloc[i]) else 0
                
                should_rebalance = should_rebalance_combined(shares, current_prices, target_allocations,
                                                           threshold_pct, current_volatility, 
                                                           volatility_threshold, combined_weight)
            
            # Override: Always enforce a minimum rebalancing frequency (e.g., 30 days)
            if days_since_last_rebalance >= 30:
                should_rebalance = True

        # --- Execute Rebalancing if Needed ---
        if should_rebalance and valid_prices_today:
            current_unsettled_total_start = sum(unsettled_cash.values())
            settled_cash_at_rebalance_start = total_cash - current_unsettled_total_start

            target_stock_investment = current_total_value * target_stock_ratio
            target_cash_investment = current_total_value * (1.0 - target_stock_ratio)
            
            # Calculate target value per stock based on indicator signals
            use_indicator_signal = (indicator_type != 'NONE') and (i >= required_lookback)
            base_equal_weight = 1.0 / n_stocks if n_stocks > 0 else 0
            indicator_signal_weights = pd.Series(base_equal_weight, index=tickers)  # Default

            if use_indicator_signal:
                # Get indicator signals with vectorized operations
                raw_signals = get_indicator_signals_vectorized(current_date, tickers, current_prices, 
                                                            precalculated_indicators, indicator_type)
                
                # Normalize signals
                signal_sum = raw_signals.sum()
                if signal_sum > 0:
                    indicator_signal_weights = raw_signals / signal_sum

            # Blend weights (vectorized operation)
            final_weights = ((1 - indicator_factor) * base_equal_weight +
                             indicator_factor * indicator_signal_weights)

            # Re-normalize (vectorized operation)
            weight_sum = final_weights.sum()
            if weight_sum > 1e-6:
                normalized_weights = final_weights / weight_sum
            else:
                normalized_weights = pd.Series(base_equal_weight, index=tickers)

            # Final target value per stock
            target_value_per_stock = normalized_weights * target_stock_investment

            # --- Selling Phase ---
            potential_sells = {}
            
            # Vectorized calculation of current position values
            shares_series = pd.Series(shares)
            price_series = pd.Series({ticker: current_prices.get(ticker, 0) for ticker in tickers})
            
            # Filter valid positions
            valid_mask = (~pd.isna(price_series)) & (price_series > 0) & (shares_series > 0)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                current_values = shares_series[valid_tickers] * price_series[valid_tickers]
                
                # Get target values
                target_values = pd.Series({ticker: target_value_per_stock.get(ticker, 0) for ticker in valid_tickers})
                
                # Calculate value differences
                value_diffs = target_values - current_values
                
                # Find positions to sell (negative value diff)
                sell_mask = value_diffs < 0
                
                if sell_mask.any():
                    sell_tickers = sell_mask[sell_mask].index
                    
                    for ticker in sell_tickers:
                        value_diff = value_diffs[ticker]
                        price = price_series[ticker]
                        
                        shares_to_sell_ideal = abs(value_diff) / price
                        shares_to_sell_whole = min(math.floor(shares_to_sell_ideal), shares[ticker])
                        
                        if shares_to_sell_whole > 0:
                            current_dev = abs(current_values[ticker] - target_values[ticker])
                            dev_after_sell = abs((shares[ticker] - shares_to_sell_whole) * price - target_values[ticker])
                            
                            if dev_after_sell <= current_dev + 1e-6:
                                potential_sells[ticker] = shares_to_sell_whole

            # Execute Sells & Log
            for ticker, num_shares_to_sell in potential_sells.items():
                price = price_series.get(ticker, 0)
                if price is None or price <= 0:
                    continue
                proceeds = num_shares_to_sell * price
                cash_before = settled_cash_at_rebalance_start
                cash_after = settled_cash_at_rebalance_start
                transaction_log.append({'Date': current_date.date(), 'Ticker': ticker, 'Action': 'Sell',
                                       'Shares': num_shares_to_sell, 'Price': price,
                                       'Settled Cash Before': cash_before, 'Settled Cash After': cash_after})
                total_cash += proceeds
                settlement_date_index = i + settlement_days
                if settlement_date_index < len(all_dates):
                    unsettled_cash[all_dates[settlement_date_index]] += proceeds
                if cost_basis[ticker]['total_shares'] > 0:
                    avg_cost = cost_basis[ticker]['total_cost'] / cost_basis[ticker]['total_shares']
                    cost_redux = num_shares_to_sell * avg_cost
                    cost_basis[ticker]['total_cost'] = max(0, cost_basis[ticker]['total_cost'] - cost_redux)
                    cost_basis[ticker]['total_shares'] -= num_shares_to_sell
                    if cost_basis[ticker]['total_shares'] == 0:
                        cost_basis[ticker]['total_cost'] = 0.0
                shares[ticker] -= num_shares_to_sell

            # --- Buying Phase (Constrained by Settled Cash) ---
            current_unsettled_total = sum(unsettled_cash.values())
            settled_cash_now = total_cash - current_unsettled_total
            max_buy_spend = min(settled_cash_now, max(0, total_cash - target_cash_investment))
            cash_spent_on_buys_today = 0.0
            running_settled_cash_buy_phase = settled_cash_now

            # Identify buy candidates with vectorized operations
            shares_series = pd.Series(shares)
            price_series = pd.Series({ticker: current_prices.get(ticker, 0) for ticker in tickers})
            
            # Filter valid prices
            valid_mask = (~pd.isna(price_series)) & (price_series > 0)
            
            if valid_mask.any():
                valid_tickers = valid_mask[valid_mask].index
                current_values = pd.Series({ticker: shares.get(ticker, 0) * price_series[ticker] for ticker in valid_tickers})
                
                # Get target values
                target_values = pd.Series({ticker: target_value_per_stock.get(ticker, 0) for ticker in valid_tickers})
                
                # Calculate value differences
                value_diffs = target_values - current_values
                
                # Find positions to buy (positive value diff)
                buy_mask = value_diffs > 0
                
                if buy_mask.any():
                    buy_tickers = buy_mask[buy_mask].index
                    buy_values = value_diffs[buy_tickers]
                    
                    # Sort by value difference (descending)
                    buy_candidates = [(ticker, buy_values[ticker]) for ticker in buy_tickers.sort_values(
                        key=lambda x: buy_values[x], ascending=False)]
                    
                    # Execute Buys respecting max_buy_spend limit & Log
                    for ticker, value_diff in buy_candidates:
                        if cash_spent_on_buys_today >= max_buy_spend - 1e-6:
                            break
                        price = price_series[ticker]
                        
                        shares_to_buy_ideal = value_diff / price
                        shares_to_buy_whole = math.floor(shares_to_buy_ideal)
                        
                        if shares_to_buy_whole > 0:
                            remaining_spendable = max(0, max_buy_spend - cash_spent_on_buys_today)
                            affordable_budget = math.floor(remaining_spendable / price) if price > 0 else 0
                            affordable_shares = min(shares_to_buy_whole, affordable_budget)
                            
                            if affordable_shares > 0:
                                cost = affordable_shares * price
                                if total_cash - cost >= target_cash_investment - 1e-6:
                                    cash_before = running_settled_cash_buy_phase
                                    cash_after = running_settled_cash_buy_phase - cost
                                    transaction_log.append({'Date': current_date.date(), 'Ticker': ticker, 'Action': 'Buy',
                                                           'Shares': affordable_shares, 'Price': price,
                                                           'Settled Cash Before': cash_before, 'Settled Cash After': cash_after})
                                    shares[ticker] = shares.get(ticker, 0) + affordable_shares
                                    total_cash -= cost
                                    cash_spent_on_buys_today += cost
                                    running_settled_cash_buy_phase -= cost
                                    cost_basis[ticker]['total_shares'] = cost_basis[ticker].get('total_shares', 0) + affordable_shares
                                    cost_basis[ticker]['total_cost'] = cost_basis[ticker].get('total_cost', 0) + cost
    
            # Record rebalancing event and reset counter
            rebalance_history.iloc[i] = 1
            days_since_last_rebalance = 0

    # Restore original weights if changed
    if original_weights is not None:
        INDICATOR_WEIGHTS = original_weights

    # --- Simulation End ---
    final_state = {
        'daily_values': portfolio_values,
        'final_cash': total_cash,
        'final_shares': shares,
        'final_cost_basis': cost_basis,
        'stock_value_history': stock_value_history,
        'unsettled_cash': dict(unsettled_cash),
        'transaction_log': transaction_log,
        'rebalance_history': rebalance_history,
        'portfolio_volatility': portfolio_volatility
    }
    
    return final_state


# --- Enhanced Plotting Functions ---
def plot_stock_contributions(stock_value_df, strategy_key):
    if stock_value_df is None or stock_value_df.empty:
        return
    plt.figure(figsize=(14, 8))
    sorted_columns = sorted(stock_value_df.columns)
    y_values = [stock_value_df[ticker] for ticker in sorted_columns]
    labels = sorted_columns
    try:
        plt.stackplot(stock_value_df.index, y_values, labels=labels, alpha=0.8)
    except Exception as e:
        print(f"Stack plot error: {e}")
        plt.close()
        return
    
    # Unpack strategy key based on rebalance type
    rebal_type = strategy_key[3]
    if rebal_type == 'TIME':
        p, freq, ind_f, rtype = strategy_key
        plt.title(f'Stock Value Contribution\n({p}%/{freq}d/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'THRESHOLD':
        p, thresh, ind_f, rtype = strategy_key
        plt.title(f'Stock Value Contribution\n({p}%/Thresh={thresh}%/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'VOLATILITY':
        p, vol_thresh, ind_f, rtype = strategy_key
        plt.title(f'Stock Value Contribution\n({p}%/VolThresh={vol_thresh:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'COMBINED':
        p, thresh, vol_thresh, weight, ind_f, rtype = strategy_key
        plt.title(f'Stock Value Contribution\n({p}%/Thresh={thresh}%/VolThresh={vol_thresh:.2f}/Weight={weight:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    else:
        plt.title(f'Stock Value Contribution\n(Strategy: {strategy_key})')
    
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

def plot_rebalance_events(portfolio_values, rebalance_history, strategy_key):
    """Plot portfolio value with rebalancing events marked"""
    if portfolio_values is None or rebalance_history is None:
        return
    
    plt.figure(figsize=(14, 8))
    plt.plot(portfolio_values.index, portfolio_values.values, label='Portfolio Value', alpha=0.9)
    
    # Mark rebalancing events
    rebalance_dates = rebalance_history[rebalance_history == 1].index
    rebalance_values = portfolio_values.loc[rebalance_dates]
    plt.scatter(rebalance_dates, rebalance_values, color='red', marker='^', 
               s=100, label='Rebalance Events', zorder=5)
    
    # Strategy label
    rebal_type = strategy_key[3]
    if rebal_type == 'TIME':
        p, freq, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Value and Rebalancing Events\n({p}%/{freq}d/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'THRESHOLD':
        p, thresh, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Value and Rebalancing Events\n({p}%/Thresh={thresh}%/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'VOLATILITY':
        p, vol_thresh, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Value and Rebalancing Events\n({p}%/VolThresh={vol_thresh:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'COMBINED':
        p, thresh, vol_thresh, weight, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Value and Rebalancing Events\n({p}%/Thresh={thresh}%/VolThresh={vol_thresh:.2f}/Weight={weight:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    else:
        plt.title(f'Portfolio Value and Rebalancing Events\n(Strategy: {strategy_key})')
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

def plot_volatility(portfolio_volatility, rebalance_history, strategy_key):
    """Plot portfolio volatility with rebalancing events marked"""
    if portfolio_volatility is None or rebalance_history is None:
        return
    
    plt.figure(figsize=(14, 8))
    plt.plot(portfolio_volatility.index, portfolio_volatility.values, label='Portfolio Volatility', alpha=0.9)
    
    # Mark rebalancing events
    rebalance_dates = rebalance_history[rebalance_history == 1].index
    rebalance_volatilities = portfolio_volatility.loc[rebalance_dates]
    plt.scatter(rebalance_dates, rebalance_volatilities, color='red', marker='^', 
               s=100, label='Rebalance Events', zorder=5)
    
    # Strategy label
    rebal_type = strategy_key[3]
    if rebal_type == 'TIME':
        p, freq, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Volatility and Rebalancing Events\n({p}%/{freq}d/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'THRESHOLD':
        p, thresh, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Volatility and Rebalancing Events\n({p}%/Thresh={thresh}%/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'VOLATILITY':
        p, vol_thresh, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Volatility and Rebalancing Events\n({p}%/VolThresh={vol_thresh:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    elif rebal_type == 'COMBINED':
        p, thresh, vol_thresh, weight, ind_f, rtype = strategy_key
        plt.title(f'Portfolio Volatility and Rebalancing Events\n({p}%/Thresh={thresh}%/VolThresh={vol_thresh:.2f}/Weight={weight:.2f}/IndFactor={ind_f:.2f}, Rebal={rtype})')
    else:
        plt.title(f'Portfolio Volatility and Rebalancing Events\n(Strategy: {strategy_key})')
    
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.gca().yaxis
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
def plot_results(history_dict, top_results, initial_capital):
    plt.figure(figsize=(14, 8))
    keys_plotted = set()
    for item in top_results:
        k = item[0]
        r = item[1]
        fv = r.get('final_value')
        if fv is None or not pd.notna(fv):
            continue
        series = history_dict.get(k)
        if isinstance(series, pd.Series) and not series.empty and k not in keys_plotted:
            # Create label based on rebalance type
            try:
                rebal_type = k[-1]
                sharpe = r.get('sharpe_ratio', np.nan)
                
                # Check if this is a MULTI strategy with weights
                is_multi_with_weights = False
                for elem in k:
                    if isinstance(elem, tuple) and len(elem) > 0 and isinstance(elem[0], tuple):
                        is_multi_with_weights = True
                        break
                
                if is_multi_with_weights:
                    # Handle MULTI strategies with weights
                    if rebal_type == 'TIME':
                        p, freq, ind_f = k[0:3]
                        label = f"MULTI {p}%/{freq}d/IndF={ind_f:.2f}/{rebal_type} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'THRESHOLD':
                        p, thresh, ind_f = k[0:3]
                        label = f"MULTI {p}%/Thresh={thresh}%/IndF={ind_f:.2f}/{rebal_type} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'VOLATILITY':
                        p, vol_thresh, ind_f = k[0:3]
                        label = f"MULTI {p}%/VolThresh={vol_thresh:.2f}/IndF={ind_f:.2f}/{rebal_type} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'COMBINED':
                        p, thresh, vol_thresh, weight, ind_f = k[0:5]
                        label = f"MULTI {p}%/Comb[{thresh}%,{vol_thresh:.2f},{weight:.2f}]/IndF={ind_f:.2f}/{rebal_type} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    else:
                        label = f"{k} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                else:
                    # Handle regular strategies
                    if rebal_type == 'TIME':
                        p, freq, ind_f, rtype = k
                        label = f"{p}%/{freq}d/IndF={ind_f:.2f}/{rtype} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'THRESHOLD':
                        p, thresh, ind_f, rtype = k
                        label = f"{p}%/Thresh={thresh}%/IndF={ind_f:.2f}/{rtype} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'VOLATILITY':
                        p, vol_thresh, ind_f, rtype = k
                        label = f"{p}%/VolThresh={vol_thresh:.2f}/IndF={ind_f:.2f}/{rtype} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    elif rebal_type == 'COMBINED':
                        p, thresh, vol_thresh, weight, ind_f, rtype = k
                        label = f"{p}%/Comb[{thresh}%,{vol_thresh:.2f},{weight:.2f}]/IndF={ind_f:.2f}/{rtype} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                    else:
                        label = f"{k} (Shp:{sharpe:.2f}, Val:${fv:,.0f})"
                
                plt.plot(series.index, series.values, label=label, alpha=0.9)
                keys_plotted.add(k)
            except Exception as e:
                print(f"Error plotting result: {e}")
                print(f"Key structure: {k}")
    
    if not keys_plotted:
        print("Warn: No data for overall plot.")
        plt.close()
        return
    
    plt.title(f'Top Performance (Sorted by Sharpe, T+{SETTLEMENT_DAYS})\n(Initial Cap: ${initial_capital:,.0f})')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('$%.0f'))
    plt.legend(fontsize='small')
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()

# *** UPDATED Sorting Helper ***
def get_sort_metric(item):
    """Sorts by Sharpe Ratio (descending)."""
    sim_key, result_data = item
    if isinstance(result_data, dict) and 'sharpe_ratio' in result_data:
        value = result_data['sharpe_ratio']
        return value if pd.notna(value) else -float('inf')
    else:
        return -float('inf')
    
# --- Parallel Processing with Progress Tracking ---
def run_parallel_simulations(param_combinations, prices_df, tickers, initial_capital, 
                           settlement_days, precalculated_indicators_dict):
    """Run simulations in parallel using multiprocessing with progress tracking"""
    # Determine number of CPUs to use
    num_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    print(f"Running simulations in parallel using {num_cpus} CPUs")
    
    # Create a manager for sharing objects between processes
    manager = mp.Manager()
    progress_dict = manager.dict()
    progress_dict['completed'] = 0
    progress_dict['total'] = len(param_combinations)
    
    # Create a lock for synchronizing updates
    progress_lock = manager.Lock()
    
    # Split parameter combinations into chunks
    chunk_size = max(1, len(param_combinations) // (num_cpus * 10))  # Each worker gets multiple chunks
    param_chunks = [param_combinations[i:i+chunk_size] 
                   for i in range(0, len(param_combinations), chunk_size)]
    
    print(f"Split {len(param_combinations)} combinations into {len(param_chunks)} chunks")
    
    # Start progress tracking in a separate thread
    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=_display_progress, 
        args=(progress_dict, stop_event)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        # Create a pool of workers
        with mp.Pool(num_cpus) as pool:
            # Map each chunk to a worker
            results_chunks = pool.starmap(
                _run_simulation_chunk, 
                [(chunk, prices_df, tickers, initial_capital, 
                 settlement_days, precalculated_indicators_dict, 
                 progress_dict, progress_lock) for chunk in param_chunks]
            )
        
        # Signal progress thread to stop
        stop_event.set()
        progress_thread.join(timeout=1.0)
        
        # Combine results from all chunks
        combined_results = {}
        combined_portfolio_history = {}
        
        for results, portfolio_history in results_chunks:
            combined_results.update(results)
            combined_portfolio_history.update(portfolio_history)
        
        return combined_results, combined_portfolio_history
    
    except Exception as e:
        # Signal progress thread to stop in case of error
        stop_event.set()
        progress_thread.join(timeout=1.0)
        raise e

def _display_progress(progress_dict, stop_event):
    """Display progress bar for parallel processing"""
    try:
        from tqdm.auto import tqdm
        
        # Create progress bar
        with tqdm(total=progress_dict['total'], desc="Simulations Progress") as pbar:
            completed_prev = 0
            
            while not stop_event.is_set():
                # Update progress bar
                completed = progress_dict['completed']
                if completed > completed_prev:
                    pbar.update(completed - completed_prev)
                    completed_prev = completed
                
                # Calculate percentage
                if progress_dict['total'] > 0:
                    percentage = (completed / progress_dict['total']) * 100
                    pbar.set_postfix({"Complete": f"{percentage:.1f}%"})
                
                # Sleep briefly to reduce CPU usage
                time.sleep(0.5)
            
            # Final update to ensure 100% is shown
            pbar.update(progress_dict['total'] - completed_prev)
            
    except Exception as e:
        print(f"Error in progress tracking: {e}")
        # Fall back to simple progress reporting
        while not stop_event.is_set():
            completed = progress_dict['completed']
            total = progress_dict['total']
            if total > 0:
                percentage = (completed / total) * 100
                print(f"\rProgress: {completed}/{total} ({percentage:.1f}%)", end="")
            time.sleep(2)
        print()  # Final newline

def _run_simulation_chunk(param_chunk, prices_df, tickers, initial_capital, 
                         settlement_days, precalculated_indicators_dict,
                         progress_dict, progress_lock):
    """Run a chunk of simulations with progress tracking"""
    results = {}
    portfolio_history = {}
    local_progress = 0
    
    try:
        # Create thread-local copies of any shared data if needed
        # (This avoids issues with multiple processes modifying the same data)
        local_indicator_weights = INDICATOR_WEIGHTS.copy()
        
        # Make a single copy of prices_df for the whole chunk
        prices_df_copy = prices_df.copy()
        
        for params in param_chunk:
            # Unpack parameters
            percent, rebalance_config, indicator_type, indicator_factor, weight_combo = params
            
            # Set weights if using MULTI strategy with specific weights
            if indicator_type == 'MULTI' and weight_combo is not None:
                # Apply the current weight combination (to local copy)
                temp_weights = local_indicator_weights.copy()
                for k, v in weight_combo.items():
                    temp_weights[k] = v
            else:
                temp_weights = local_indicator_weights
            
            # Get pre-calculated indicators for this indicator type
            precalculated_indicators = precalculated_indicators_dict.get(indicator_type, {})
            
            # Set up indicator config
            indicator_config = {'type': indicator_type}
            
            # Run simulation (pass temp_weights for MULTI strategy)
            if indicator_type == 'MULTI':
                # For MULTI, we need to pass the weights
                sim_output = simulate_rebalancing(
                    prices_df_copy, tickers, initial_capital, percent, 
                    rebalance_config, indicator_config, indicator_factor,
                    settlement_days, precalculated_indicators,
                    temp_weights if indicator_type == 'MULTI' else None
                )
            else:
                # For other strategies, no need to pass weights
                sim_output = simulate_rebalancing(
                    prices_df_copy, tickers, initial_capital, percent, 
                    rebalance_config, indicator_config, indicator_factor,
                    settlement_days, precalculated_indicators
                )
            
            # Process results
            if isinstance(sim_output, dict) and 'daily_values' in sim_output:
                daily_values = sim_output['daily_values']
                
                # Create sim key based on rebalance type and weight combo
                rebalance_type = rebalance_config['type']
                if indicator_type == 'MULTI' and weight_combo is not None:
                    # Include weight combination in the key
                    weight_key = tuple((k, round(v, 2)) for k, v in sorted(weight_combo.items()))
                    
                    if rebalance_type == 'TIME':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                    elif rebalance_type == 'THRESHOLD':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                    elif rebalance_type == 'VOLATILITY':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                    elif rebalance_type == 'COMBINED':
                        sim_key = (percent, rebalance_config['param1'], rebalance_config['param2'], 
                                  rebalance_config['param3'], indicator_factor, weight_key, rebalance_type)
                else:
                    # Regular strategy without weight combo
                    if rebalance_type == 'TIME':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                    elif rebalance_type == 'THRESHOLD':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                    elif rebalance_type == 'VOLATILITY':
                        sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                    elif rebalance_type == 'COMBINED':
                        sim_key = (percent, rebalance_config['param1'], rebalance_config['param2'], 
                                  rebalance_config['param3'], indicator_factor, rebalance_type)
                
                # Store weight combination in the results
                if indicator_type == 'MULTI' and weight_combo is not None:
                    sim_output['weight_combo'] = weight_combo
                
                try:
                    final_value = daily_values.iloc[-1]
                    if pd.notna(final_value):
                        sim_output['final_value'] = final_value
                        sharpe = calculate_sharpe_ratio(daily_values, RISK_FREE_RATE)
                        sim_output['sharpe_ratio'] = sharpe
                        
                        # Calculate rebalance frequency
                        rebalance_count = sim_output['rebalance_history'].sum()
                        sim_output['rebalance_count'] = rebalance_count
                        sim_output['rebalance_freq'] = len(daily_values) / max(1, rebalance_count)
                        
                        results[sim_key] = sim_output
                        portfolio_history[sim_key] = daily_values
                except IndexError:
                    pass
            
            # Update progress counter
            local_progress += 1
            if local_progress % 10 == 0:  # Update shared counter less frequently to reduce overhead
                with progress_lock:
                    progress_dict['completed'] += local_progress
                    local_progress = 0
        
        # Update any remaining progress
        if local_progress > 0:
            with progress_lock:
                progress_dict['completed'] += local_progress
    
    except Exception as e:
        print(f"Error in worker process: {e}")
        import traceback
        traceback.print_exc()
    
    return results, portfolio_history

# --- Main Execution with Parallel Processing ---
if __name__ == "__main__":
    start_time = time.time()
    print(f"--- Enhanced Stock Rebalancing Simulator (Optimized) ---")
    
    # Create cache directory
    ensure_directory_exists(CACHE_FOLDER)

    # 1. Load Data
    prices_df, tickers = load_stock_data(DATA_FOLDER)
    if prices_df is None or prices_df.empty or not tickers:
        exit("Exiting: Data loading issues.")
    try:
        final_prices = prices_df.iloc[-1]
    except IndexError:
        exit("Error: Cannot get final prices.")

    # 2. Pre-calculate all needed indicators
    print("\nPre-calculating indicators for all types...")
    precalculated_indicators_dict = {}
    for indicator_type in INDICATOR_TYPES_TO_TEST:
        precalculated_indicators_dict[indicator_type] = calculate_indicators(
            prices_df, tickers, indicator_type, 
            INDICATOR_PERIODS.get(indicator_type, {})
        )

    # Initialize Results Dictionaries
    results = {}
    portfolio_history = {}

    # 3. Build parameter combinations for all simulations
    indicator_factors_to_test = np.arange(INDICATOR_FACTOR_MIN, INDICATOR_FACTOR_MAX + INDICATOR_FACTOR_STEP/2, INDICATOR_FACTOR_STEP)
    threshold_values = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP/2, THRESHOLD_STEP)
    volatility_thresholds = np.arange(VOLATILITY_THRESHOLD_MIN, VOLATILITY_THRESHOLD_MAX + VOLATILITY_THRESHOLD_STEP/2, VOLATILITY_THRESHOLD_STEP)
    combined_weights = np.arange(COMBINED_WEIGHT_MIN, COMBINED_WEIGHT_MAX + COMBINED_WEIGHT_STEP/2, COMBINED_WEIGHT_STEP)
    
    # NEW: Generate weight combinations for MULTI strategy
    def generate_weight_combinations():
        """Generate valid combinations of indicator weights"""
        # Start with just one combination - equal weights
        weight_combinations = [{}]
        
        # For each indicator
        for indicator, range_info in INDICATOR_WEIGHT_RANGES.items():
            new_combinations = []
            # For each existing combination
            for combo in weight_combinations:
                # For each possible weight value for this indicator
                for weight in np.arange(range_info['min'], range_info['max'] + range_info['step']/2, range_info['step']):
                    # Create a new combination with this weight
                    new_combo = combo.copy()
                    new_combo[indicator] = weight
                    new_combinations.append(new_combo)
            weight_combinations = new_combinations
        
        # Filter out invalid combinations (sum of weights should be > 0)
        valid_combinations = []
        for combo in weight_combinations:
            weight_sum = sum(combo.values())
            if weight_sum > 0:
                # Normalize weights to sum to 1.0
                normalized_combo = {k: v/weight_sum for k, v in combo.items()}
                valid_combinations.append(normalized_combo)
        
        return valid_combinations
    
    # Generate all valid weight combinations
    multi_weight_combinations = generate_weight_combinations()
    print(f"Generated {len(multi_weight_combinations)} valid weight combinations for MULTI strategy")
    
    param_combinations = []
    
    # Populate parameter combinations for all strategies
    for indicator_type in INDICATOR_TYPES_TO_TEST:
        for percent in range(PERCENTAGE_MIN, PERCENTAGE_MAX + 1, PERCENTAGE_STEP):
            for indicator_factor in indicator_factors_to_test:
                # For MULTI strategy, test all weight combinations
                if indicator_type == 'MULTI':
                    for weight_combo in multi_weight_combinations:
                        # Set the current weight combination
                        for k, v in weight_combo.items():
                            INDICATOR_WEIGHTS[k] = v
                        
                        # Add combinations for all rebalancing strategies
                        # TIME-BASED REBALANCING
                        for freq_days in range(REBALANCE_FREQ_MIN, REBALANCE_FREQ_MAX + 1):
                            rebalance_config = {
                                'type': 'TIME',
                                'param1': freq_days
                            }
                            # Include weight combo as part of the parameters
                            param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, weight_combo))
                        
                        # Add other rebalancing strategies similarly...
                        # THRESHOLD-BASED REBALANCING
                        for threshold_pct in threshold_values:
                            rebalance_config = {
                                'type': 'THRESHOLD',
                                'param1': threshold_pct
                            }
                            param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, weight_combo))
                        
                        # VOLATILITY-BASED REBALANCING
                        for vol_threshold in volatility_thresholds:
                            rebalance_config = {
                                'type': 'VOLATILITY',
                                'param1': vol_threshold
                            }
                            param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, weight_combo))
                        
                        # COMBINED REBALANCING
                        for threshold_pct in threshold_values:
                            for vol_threshold in volatility_thresholds:
                                for combined_weight in combined_weights:
                                    rebalance_config = {
                                        'type': 'COMBINED',
                                        'param1': threshold_pct,
                                        'param2': vol_threshold,
                                        'param3': combined_weight
                                    }
                                    param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, weight_combo))
                else:
                    # For non-MULTI strategies, use normal approach
                    # TIME-BASED REBALANCING
                    for freq_days in range(REBALANCE_FREQ_MIN, REBALANCE_FREQ_MAX + 1):
                        rebalance_config = {
                            'type': 'TIME',
                            'param1': freq_days
                        }
                        param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, None))
                    
                    # Add other rebalancing strategies similarly...
                    # THRESHOLD-BASED REBALANCING
                    for threshold_pct in threshold_values:
                        rebalance_config = {
                            'type': 'THRESHOLD',
                            'param1': threshold_pct
                        }
                        param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, None))
                    
                    # VOLATILITY-BASED REBALANCING
                    for vol_threshold in volatility_thresholds:
                        rebalance_config = {
                            'type': 'VOLATILITY',
                            'param1': vol_threshold
                        }
                        param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, None))
                    
                    # COMBINED REBALANCING
                    for threshold_pct in threshold_values:
                        for vol_threshold in volatility_thresholds:
                            for combined_weight in combined_weights:
                                rebalance_config = {
                                    'type': 'COMBINED',
                                    'param1': threshold_pct,
                                    'param2': vol_threshold,
                                    'param3': combined_weight
                                }
                                param_combinations.append((percent, rebalance_config, indicator_type, indicator_factor, None))

    # 4. Run simulations in parallel
    print("\nRunning simulations...")
    try:
        results, portfolio_history = run_parallel_simulations(
            param_combinations, prices_df, tickers, INITIAL_CAPITAL, 
            SETTLEMENT_DAYS, precalculated_indicators_dict
        )
    except Exception as e:
        print(f"Error in parallel processing: {e}")
        print("Falling back to sequential processing...")
        
        # If parallel processing fails, run sequentially with progress bar
        results = {}
        portfolio_history = {}
        
        with tqdm(total=len(param_combinations), desc="Sequential Processing") as pbar:
            for sim_count, params in enumerate(param_combinations):
                # Unpack parameters
                percent, rebalance_config, indicator_type, indicator_factor, weight_combo = params
                
                # Set weights if using MULTI strategy with specific weights
                if indicator_type == 'MULTI' and weight_combo is not None:
                    # Save original weights
                    original_weights = INDICATOR_WEIGHTS.copy()
                    
                    # Apply the current weight combination
                    for k, v in weight_combo.items():
                        INDICATOR_WEIGHTS[k] = v
                
                # Get pre-calculated indicators for this indicator type
                precalculated_indicators = precalculated_indicators_dict.get(indicator_type, {})
                
                # Set up indicator config
                indicator_config = {'type': indicator_type}
                
                # Run simulation
                sim_output = simulate_rebalancing(
                    prices_df.copy(), tickers, INITIAL_CAPITAL, percent, 
                    rebalance_config, indicator_config, indicator_factor,
                    SETTLEMENT_DAYS, precalculated_indicators
                )
                
                # Restore original weights if changed
                if indicator_type == 'MULTI' and weight_combo is not None:
                    # Restore the original weights
                    for k, v in original_weights.items():
                        INDICATOR_WEIGHTS[k] = v
                
                # Process results
                if isinstance(sim_output, dict) and 'daily_values' in sim_output:
                    daily_values = sim_output['daily_values']
                    
                    # Create sim key based on rebalance type and weight combo
                    rebalance_type = rebalance_config['type']
                    
                    if indicator_type == 'MULTI' and weight_combo is not None:
                        # Include weight combination in the key
                        weight_key = tuple((k, round(v, 2)) for k, v in sorted(weight_combo.items()))
                        
                        if rebalance_type == 'TIME':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                        elif rebalance_type == 'THRESHOLD':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                        elif rebalance_type == 'VOLATILITY':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, weight_key, rebalance_type)
                        elif rebalance_type == 'COMBINED':
                            sim_key = (percent, rebalance_config['param1'], rebalance_config['param2'], 
                                    rebalance_config['param3'], indicator_factor, weight_key, rebalance_type)
                    else:
                        # Regular strategy without weight combo
                        if rebalance_type == 'TIME':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                        elif rebalance_type == 'THRESHOLD':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                        elif rebalance_type == 'VOLATILITY':
                            sim_key = (percent, rebalance_config['param1'], indicator_factor, rebalance_type)
                        elif rebalance_type == 'COMBINED':
                            sim_key = (percent, rebalance_config['param1'], rebalance_config['param2'], 
                                    rebalance_config['param3'], indicator_factor, rebalance_type)
                    
                    # Store weight combination in the results
                    if indicator_type == 'MULTI' and weight_combo is not None:
                        sim_output['weight_combo'] = weight_combo
                        
                    try:
                        final_value = daily_values.iloc[-1]
                        if pd.notna(final_value):
                            sim_output['final_value'] = final_value
                            sharpe = calculate_sharpe_ratio(daily_values, RISK_FREE_RATE)
                            sim_output['sharpe_ratio'] = sharpe
                            
                            # Calculate rebalance frequency
                            rebalance_count = sim_output['rebalance_history'].sum()
                            sim_output['rebalance_count'] = rebalance_count
                            sim_output['rebalance_freq'] = len(daily_values) / max(1, rebalance_count)
                            
                            results[sim_key] = sim_output
                            portfolio_history[sim_key] = daily_values
                    except IndexError:
                        pass
                
                # Update progress bar
                pbar.update(1)
                if sim_count % 100 == 0:  # Show current strategy details periodically
                    pbar.set_postfix({
                        "Type": indicator_type, 
                        "Rebal": rebalance_type, 
                        "Pct": percent,
                        "Factor": f"{indicator_factor:.2f}"
                    })


    # 5. Analyze Results
    print("\nSorting results by Sharpe Ratio...")
    try:
        sorted_results = sorted(list(results.items()), key=get_sort_metric, reverse=True)
    except Exception as e:
        exit(f"Error during sorting: {e}")

    # 6. Report Top Results
    print(f"\n--- Top {TOP_N_RESULTS} Performing Combinations (Sorted by Sharpe Ratio) ---")
    num_results_to_show = min(TOP_N_RESULTS, len(sorted_results))
    top_results_list = sorted_results[:num_results_to_show]
    valid_top_results = [item for item in top_results_list if isinstance(item[1], dict) and pd.notna(item[1].get('sharpe_ratio')) and item[1].get('sharpe_ratio') != -float('inf')]

    if not valid_top_results:
        print("No valid results with calculable Sharpe Ratio to display in the top N.")
    else:
        # Print detailed information for each top result
        print("Rank | Stock % | Rebal Type | Rebal Params | Ind. Factor | Indicator Type | Weights | Sharpe Ratio | Final Value   | Total Profit | Rebal Freq")
        print("-----|---------|------------|--------------|-------------|----------------|---------|--------------|---------------|--------------|----------")
        for i, (sim_key, result_data) in enumerate(valid_top_results):
            try:
                # Get the rebalance type (always the last element)
                rebal_type = sim_key[-1]
                
                # Check if this is a MULTI strategy with weights
                is_multi_with_weights = False
                for elem in sim_key:
                    if isinstance(elem, tuple) and len(elem) > 0 and isinstance(elem[0], tuple):
                        is_multi_with_weights = True
                        weight_key = elem
                        break
                
                # Handle MULTI strategies with weights
                if is_multi_with_weights:
                    if rebal_type == 'TIME':
                        percent, freq, indicator_factor = sim_key[0:3]
                        rebal_params = f"{freq}d"
                    elif rebal_type == 'THRESHOLD':
                        percent, threshold, indicator_factor = sim_key[0:3]
                        rebal_params = f"{threshold}%"
                    elif rebal_type == 'VOLATILITY':
                        percent, vol_thresh, indicator_factor = sim_key[0:3]
                        rebal_params = f"{vol_thresh:.2f}"
                    elif rebal_type == 'COMBINED':
                        percent, threshold, vol_thresh, comb_weight, indicator_factor = sim_key[0:5]
                        rebal_params = f"T={threshold}%,V={vol_thresh:.2f},W={comb_weight:.2f}"
                    
                    indicator_type_res = 'MULTI'
                    
                    # Format weights for display
                    weight_dict = {str(k): float(v) for k, v in weight_key}
                    weight_display = ", ".join([f"{k}: {v:.2f}" for k, v in sorted(weight_dict.items())])
                
                # Handle regular strategies
                else:
                    if rebal_type == 'TIME':
                        percent, freq, indicator_factor, rt = sim_key
                        rebal_params = f"{freq}d"
                    elif rebal_type == 'THRESHOLD':
                        percent, threshold, indicator_factor, rt = sim_key
                        rebal_params = f"{threshold}%"
                    elif rebal_type == 'VOLATILITY':
                        percent, vol_thresh, indicator_factor, rt = sim_key
                        rebal_params = f"{vol_thresh:.2f}"
                    elif rebal_type == 'COMBINED':
                        percent, threshold, vol_thresh, weight, indicator_factor, rt = sim_key
                        rebal_params = f"T={threshold}%,V={vol_thresh:.2f},W={weight:.2f}"
                    else:
                        print(f"Unknown rebalance type: {rebal_type}")
                        continue
                        
                    indicator_type_res = result_data.get('indicator_type', rebal_type)
                    weight_display = "N/A"
                
                # Convert numpy types to Python types for display
                percent = float(percent) if hasattr(percent, 'item') else percent
                indicator_factor = float(indicator_factor) if hasattr(indicator_factor, 'item') else indicator_factor
                
                final_value = result_data.get('final_value', np.nan)
                sharpe_ratio = result_data['sharpe_ratio']
                total_profit = final_value - INITIAL_CAPITAL if pd.notna(final_value) else np.nan
                rebal_freq = result_data.get('rebalance_freq', 0)
                
                print(f"{i+1:<4} | {percent:<7} | {rebal_type:<10} | {rebal_params:<12} | {indicator_factor:<11.2f} | {indicator_type_res:<14} | {weight_display} | {sharpe_ratio:<12.3f} | ${final_value:<13,.2f} | ${total_profit:,.2f} | {rebal_freq:.1f}d")
            
            except Exception as e:
                print(f"Error unpacking result {i+1}: {e}")
                print(f"Key structure: {sim_key}")

        # Print detailed information for the best strategy
        print("\n--- Transaction Log for Top Performing Strategy (by Sharpe Ratio) ---")
        best_sim_key, best_result_data = valid_top_results[0]
        rebal_type = best_sim_key[-1]

        try:
            # Check if this is a MULTI strategy with weights
            is_multi_with_weights = False
            weight_key = None
            for elem in best_sim_key:
                if isinstance(elem, tuple) and len(elem) > 0 and isinstance(elem[0], tuple):
                    is_multi_with_weights = True
                    weight_key = elem
                    break
            
            # Handle MULTI strategies with weights
            if is_multi_with_weights:
                if rebal_type == 'TIME':
                    bp, freq, bif = best_sim_key[0:3]
                    strategy_desc = f"{bp}% Stock / {freq}d Rebalance Freq / IndFactor={bif:.2f} (Type: MULTI {rebal_type})"
                elif rebal_type == 'THRESHOLD':
                    bp, thresh, bif = best_sim_key[0:3]
                    strategy_desc = f"{bp}% Stock / {thresh}% Threshold / IndFactor={bif:.2f} (Type: MULTI {rebal_type})"
                elif rebal_type == 'VOLATILITY':
                    bp, vol_thresh, bif = best_sim_key[0:3]
                    strategy_desc = f"{bp}% Stock / {vol_thresh:.2f} Vol Threshold / IndFactor={bif:.2f} (Type: MULTI {rebal_type})"
                elif rebal_type == 'COMBINED':
                    bp, thresh, vol_thresh, comb_weight, bif = best_sim_key[0:5]
                    strategy_desc = f"{bp}% Stock / Combined[Thresh={thresh}%, Vol={vol_thresh:.2f}, Weight={comb_weight:.2f}] / IndFactor={bif:.2f} (Type: MULTI {rebal_type})"
                
                # Add weight information
                weight_dict = {str(k): float(v) for k, v in weight_key}
                weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in sorted(weight_dict.items())])
                strategy_desc += f"\nWeights: {weights_str}"
            
            # Handle regular strategies
            else:
                if rebal_type == 'TIME':
                    bp, freq, bif, rt = best_sim_key
                    strategy_desc = f"{bp}% Stock / {freq}d Rebalance Freq / IndFactor={bif:.2f} (Type: {rt})"
                elif rebal_type == 'THRESHOLD':
                    bp, thresh, bif, rt = best_sim_key
                    strategy_desc = f"{bp}% Stock / {thresh}% Threshold / IndFactor={bif:.2f} (Type: {rt})"
                elif rebal_type == 'VOLATILITY':
                    bp, vol_thresh, bif, rt = best_sim_key
                    strategy_desc = f"{bp}% Stock / {vol_thresh:.2f} Vol Threshold / IndFactor={bif:.2f} (Type: {rt})"
                elif rebal_type == 'COMBINED':
                    bp, thresh, vol_thresh, weight, bif, rt = best_sim_key
                    strategy_desc = f"{bp}% Stock / Combined[Thresh={thresh}%, Vol={vol_thresh:.2f}, Weight={weight:.2f}] / IndFactor={bif:.2f} (Type: {rt})"
                else:
                    strategy_desc = f"Strategy with key: {best_sim_key}"
                    
        except Exception as e:
            # Fallback if unpacking fails
            strategy_desc = f"Strategy key: {best_sim_key}"
            print(f"Error unpacking best strategy key: {e}")
            
        print(f"Strategy: {strategy_desc}")

    # 7. Plot Results
    if valid_top_results:
        print(f"\nGenerating plots (Top {len(valid_top_results)} by Sharpe Ratio)...")
        plot_results(portfolio_history, valid_top_results, INITIAL_CAPITAL)
        
        # Get best strategy for detailed plots
        best_sim_key, best_result_data = valid_top_results[0]
        
        # Plot stock contribution
        best_stock_history = best_result_data.get('stock_value_history')
        if isinstance(best_stock_history, pd.DataFrame) and not best_stock_history.empty:
            print(f"Generating stock contribution plot for best strategy...")
            plot_stock_contributions(best_stock_history, best_sim_key)
        
        # Plot rebalancing events
        best_portfolio_values = best_result_data.get('daily_values')
        best_rebalance_history = best_result_data.get('rebalance_history')
        if (isinstance(best_portfolio_values, pd.Series) and 
            isinstance(best_rebalance_history, pd.Series) and 
            not best_portfolio_values.empty and 
            not best_rebalance_history.empty):
            print(f"Generating rebalancing events plot...")
            plot_rebalance_events(best_portfolio_values, best_rebalance_history, best_sim_key)
        
        # Plot volatility with rebalancing events (if using volatility-based rebalancing)
        if best_sim_key[-1] in ['VOLATILITY', 'COMBINED']:
            best_volatility = best_result_data.get('portfolio_volatility')
            if isinstance(best_volatility, pd.Series) and not best_volatility.empty:
                print(f"Generating volatility plot with rebalancing events...")
                plot_volatility(best_volatility, best_rebalance_history, best_sim_key)
        
        plt.show()  # This call is required to display the plots
    else:
        print("\nNo valid results to plot.")

    # Calculate and display total execution time
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n--- Simulation Complete ---")
    print(f"Total execution time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
