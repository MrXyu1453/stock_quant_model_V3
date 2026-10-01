import pandas as pd
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 双均线策略
def double_moving_average_strategy(data, short_window=5, long_window=20):
    try:
        if len(data) < max(short_window, long_window, 2):
            logging.warning(f"数据不足，无法计算双均线策略（需要至少 {max(short_window, long_window)} 天数据）")
            return pd.DataFrame(index=data.index, columns=['signal', 'short_mavg', 'long_mavg', 'positions'], data=0.0)
        
        logging.info(f"开始计算 {data.index[0]} 到 {data.index[-1]} 的双均线策略，短期周期: {short_window}，长期周期: {long_window}")
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1, center=False).mean()
        if len(data) > short_window:
            valid_index = data.index[short_window:]
            signals.loc[valid_index, 'signal'] = (signals.loc[valid_index, 'short_mavg'] > signals.loc[valid_index, 'long_mavg']).astype(int)
        signals['positions'] = signals['signal'].diff()
        logging.info("双均线策略计算完成")
        return signals
    except Exception as e:
        logging.error(f"双均线策略计算出错: {e}")
        return pd.DataFrame(index=data.index, columns=['signal', 'short_mavg', 'long_mavg', 'positions'], data=0.0)

# 简单移动平均策略
def simple_moving_average_strategy(data, window=20):
    try:
        if len(data) < max(window, 2):
            logging.warning(f"数据不足，无法计算简单移动平均策略（需要至少 {window} 天数据）")
            return pd.DataFrame(index=data.index, columns=['signal', 'mavg', 'positions'], data=0.0)
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['mavg'] = data['close'].rolling(window=window, min_periods=1, center=False).mean()
        if len(data) > window - 1:
            valid_index = data.index[window - 1:]
            signals.loc[valid_index, 'signal'] = (data.loc[valid_index, 'close'] > signals.loc[valid_index, 'mavg']).astype(int)
        signals['positions'] = signals['signal'].diff()
        return signals
    except Exception as e:
        logging.error(f"简单移动平均策略计算出错: {e}")
        return pd.DataFrame(index=data.index, columns=['signal', 'mavg', 'positions'], data=0.0)

# 指数移动平均策略
def exponential_moving_average_strategy(data, window=20):
    try:
        if len(data) < max(window, 2):
            logging.warning(f"数据不足，无法计算指数移动平均策略（需要至少 {window} 天数据）")
            return pd.DataFrame(index=data.index, columns=['signal', 'emavg', 'positions'], data=0.0)
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['emavg'] = data['close'].ewm(span=window, adjust=False).mean()
        if len(data) > window - 1:
            valid_index = data.index[window - 1:]
            signals.loc[valid_index, 'signal'] = (data.loc[valid_index, 'close'] > signals.loc[valid_index, 'emavg']).astype(int)
        signals['positions'] = signals['signal'].diff()
        return signals
    except Exception as e:
        logging.error(f"指数移动平均策略计算出错: {e}")
        return pd.DataFrame(index=data.index, columns=['signal', 'emavg', 'positions'], data=0.0)