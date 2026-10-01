"""
信号增强模块：减少均线策略的虚假信号（whipsaw）

三重入场过滤（只过滤买入，卖出/离场一律放行以保证风控）：
1. 趋势过滤：收盘价须在 trend_ma 日均线上方才允许买入（只做顺势单）
2. 信号确认：金叉/上穿条件须连续 confirm_days 个交易日成立才确认入场
   （基于历史数据向后确认，无未来函数）
3. ADX 过滤：ADX（平均趋向指数）低于阈值视为震荡市，均线策略在震荡市
   极易反复被打脸，此时不入场
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_adx(data, period=14):
    """Wilder 平均趋向指数 (ADX)。返回 ADX Series；缺少 high/low 时返回 None"""
    try:
        for col in ('high', 'low', 'close'):
            if col not in data.columns or data[col].isna().all():
                return None
        high = data['high'].astype(float)
        low = data['low'].astype(float)
        close = data['close'].astype(float)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)

        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

        # Wilder 平滑（ewm alpha=1/period 等价于 RMA）
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx
    except Exception as e:
        logger.warning(f"计算 ADX 失败: {e}")
        return None


def enhance_signals(data, signals, kind='dma', use_trend=True, trend_ma=60,
                    use_confirm=True, confirm_days=2, use_adx=True, adx_threshold=20):
    """对均线策略信号做三重过滤，重建持仓序列。

    返回 (filtered_signals, stats):
      filtered_signals: 与输入同结构（signal/positions + 原均线列），signal 为过滤后的 0/1 持仓
      stats: {'raw_entries', 'filtered_entries', 'removed', 'adx_available'}
    """
    stats = {'raw_entries': 0, 'filtered_entries': 0, 'removed': 0, 'adx_available': False}
    if signals is None or signals.empty or data is None or data.empty:
        return signals, stats

    close = data['close'].astype(float)
    if kind == 'dma':
        cond = (signals['short_mavg'] > signals['long_mavg']).fillna(False)
    elif kind == 'sma':
        cond = (close > signals['mavg']).fillna(False)
    elif kind == 'ema':
        cond = (close > signals['emavg']).fillna(False)
    else:
        raise ValueError(f'未知策略类型: {kind}')

    raw_signal = cond.astype(int)
    stats['raw_entries'] = int((raw_signal.diff() == 1).sum())

    # 信号确认：条件须连续 confirm_days 日成立（向后看，无未来函数）
    if use_confirm and confirm_days and confirm_days > 1:
        cond = cond.rolling(int(confirm_days)).sum() == int(confirm_days)

    # 趋势过滤：收盘价在趋势均线上方才允许入场
    trend_ok = pd.Series(True, index=data.index)
    if use_trend and trend_ma and int(trend_ma) > 1:
        trend_line = close.rolling(int(trend_ma), min_periods=max(2, int(trend_ma) // 3)).mean()
        trend_ok = (close > trend_line).fillna(False)

    # ADX 过滤：ADX 达到阈值（处于趋势行情）才允许入场
    adx_ok = pd.Series(True, index=data.index)
    if use_adx and adx_threshold is not None:
        adx = calculate_adx(data)
        if adx is not None:
            stats['adx_available'] = True
            adx_ok = (adx >= float(adx_threshold)).fillna(False)

    # 状态机重建持仓：条件成立且过滤放行 → 入场；条件转假 → 无条件离场
    entry_ok = trend_ok & adx_ok
    new_signal = np.zeros(len(signals), dtype=int)
    in_market = False
    idx = signals.index
    for i in range(len(idx)):
        bullish = bool(cond.iloc[i])
        if not in_market:
            if bullish and bool(entry_ok.iloc[i]):
                in_market = True
        elif not bullish:
            in_market = False
        new_signal[i] = 1 if in_market else 0

    filtered = signals.copy()
    filtered['signal'] = new_signal.astype(float)
    filtered['positions'] = filtered['signal'].diff()
    filtered['positions'] = filtered['positions'].fillna(0.0)

    filtered_entries = int((filtered['signal'].diff() == 1).sum())
    stats['filtered_entries'] = filtered_entries
    stats['removed'] = max(stats['raw_entries'] - filtered_entries, 0)
    return filtered, stats


def simulate_trades_detailed(data, signal, initial_capital=100000.0, transaction_fee=0.0005):
    """按信号逐笔模拟全仓交易，返回累计收益/交易次数/胜率/明细。

    signal: 0/1 持仓 Series（1=持股），与 simulate_trades 的语义一致。
    """
    result = {'cumulative_return': 0.0, 'n_trades': 0, 'win_rate': 0.0, 'trades': []}
    try:
        if data is None or signal is None:
            return result
        df = data.dropna(subset=['close'])
        sig = pd.Series(signal).reindex(df.index).fillna(0.0)
        if df.empty:
            return result

        position = 0.0  # 上根K线结束时的持仓状态
        trades = []
        entry_price = None
        entry_idx_pos = None
        for i in range(len(df)):
            s = sig.iloc[i]
            price = float(df['close'].iloc[i])
            if position <= 0 and s > 0:
                entry_price = price * (1 + transaction_fee)
                entry_idx_pos = i
                position = 1.0
            elif position > 0 and s <= 0:
                exit_price = price * (1 - transaction_fee)
                trades.append({'entry_pos': entry_idx_pos, 'exit_pos': i,
                               'return': exit_price / entry_price - 1 if entry_price else 0.0})
                position = 0.0
                entry_price = None

        # 期末仍持股：按最后收盘价强平计入统计（mark-to-market）
        if position > 0 and entry_price:
            last_price = float(df['close'].iloc[-1]) * (1 - transaction_fee)
            trades.append({'entry_pos': entry_idx_pos, 'exit_pos': len(df) - 1,
                           'return': last_price / entry_price - 1})

        cum = 1.0
        for t in trades:
            cum *= (1 + t['return'])
        wins = sum(1 for t in trades if t['return'] > 0)
        result['cumulative_return'] = cum - 1
        result['n_trades'] = len(trades)
        result['win_rate'] = wins / len(trades) if trades else 0.0
        result['trades'] = trades
        return result
    except Exception as e:
        logger.warning(f"逐笔交易模拟失败: {e}")
        return result
