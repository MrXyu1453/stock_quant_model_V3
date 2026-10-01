"""信号增强模块测试：ADX 计算、三重过滤、逐笔胜率统计"""
import numpy as np
import pandas as pd
import pytest

from app.signal_enhancer import (calculate_adx, enhance_signals,
                                 simulate_trades_detailed)
from app.strategy_manager import double_moving_average_strategy, simple_moving_average_strategy


def _make_data(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    return pd.DataFrame({
        'trade_date': idx,
        'open': closes,
        'high': closes * 1.01,
        'low': closes * 0.99,
        'close': closes,
        'vol': 1000.0,
        'amount': 100000.0,
    }, index=idx)


def test_adx_high_in_trending_market():
    closes = list(np.linspace(10, 30, 120))
    adx = calculate_adx(_make_data(closes))
    assert adx is not None
    assert adx.iloc[-1] > 40  # 单边上涨行情 ADX 应该很高


def test_adx_low_in_choppy_market():
    rng = np.random.default_rng(3)
    closes = 20 + rng.normal(0, 0.3, 120).cumsum()
    adx = calculate_adx(_make_data(closes))
    assert adx.iloc[-1] < 35  # 无趋势震荡行情 ADX 偏低


def test_adx_missing_columns_returns_none():
    df = _make_data([10, 11, 12] * 10).drop(columns=['high', 'low'])
    assert calculate_adx(df) is None


def test_confirm_filter_reduces_whipsaw_entries():
    # 单日即反转的毛刺行情：短均线单日穿越长均线后立刻回落，
    # 确认过滤（需连续2日成立）应把这些虚假入场全部剔除
    closes = list(20 + 0.8 * np.tile([1.0, -1.0], 60))   # 120 根：每日交替毛刺
    closes += list(np.linspace(21, 30, 60))              # 60 根：真实上涨趋势
    data = _make_data(closes)
    raw = double_moving_average_strategy(data, short_window=3, long_window=10)
    raw_entries = int((raw['signal'].diff() == 1).sum())
    assert raw_entries >= 5, '毛刺数据应产生较多原始入场'

    filtered, stats = enhance_signals(data, raw, kind='dma',
                                      use_trend=False, use_confirm=True, confirm_days=2,
                                      use_adx=False)
    assert stats['raw_entries'] == raw_entries
    assert stats['filtered_entries'] < raw_entries, '确认过滤应减少入场次数'
    assert stats['filtered_entries'] >= 1, '趋势段的真实入场应保留'
    assert stats['removed'] == raw_entries - stats['filtered_entries']
    # 重建后的 signal 仍为 0/1，positions 与 signal 一致
    assert set(filtered['signal'].unique()) <= {0.0, 1.0}
    assert ((filtered['signal'].diff().fillna(0) == filtered['positions']) |
            (filtered['signal'].isna())).all()

    # 确认天数=1 时等于无确认，不应改变信号
    same, stats1 = enhance_signals(data, raw, kind='dma',
                                   use_trend=False, use_confirm=True, confirm_days=1,
                                   use_adx=False)
    assert stats1['filtered_entries'] == stats1['raw_entries']


def test_trend_filter_blocks_counter_trend_buys():
    # 先跌后涨：下跌段的反弹金叉应被趋势过滤拦截
    closes = list(np.linspace(30, 10, 80)) + list(np.linspace(10, 25, 80))
    data = _make_data(closes)
    raw = double_moving_average_strategy(data, short_window=5, long_window=20)
    filtered, stats = enhance_signals(data, raw, kind='dma',
                                      use_trend=True, trend_ma=40,
                                      use_confirm=False, use_adx=False)
    assert stats['filtered_entries'] <= stats['raw_entries']
    # 过滤后所有入场点的收盘价都应在其 40 日均线上方
    trend_line = data['close'].rolling(40, min_periods=13).mean()
    entries = filtered.index[filtered['signal'].diff() == 1]
    for t in entries:
        assert data['close'].loc[t] > trend_line.loc[t]


def test_no_filter_reproduces_raw_signal():
    rng = np.random.default_rng(5)
    closes = 20 + rng.normal(0, 0.3, 150).cumsum()
    data = _make_data(closes)
    raw = simple_moving_average_strategy(data, window=10)
    filtered, stats = enhance_signals(data, raw, kind='sma',
                                      use_trend=False, use_confirm=False, use_adx=False)
    # 全部过滤关闭时，状态机重建结果与原始信号一致（跳过策略的预热区）
    warmup = 10
    pd.testing.assert_series_equal(filtered['signal'].iloc[warmup:], raw['signal'].iloc[warmup:])


def test_sells_are_never_blocked():
    # 任凭过滤条件多严格，持仓期间的离场不受影响：条件转假必须离场
    rng = np.random.default_rng(9)
    closes = 20 + rng.normal(0, 0.3, 150).cumsum()
    data = _make_data(closes)
    raw = double_moving_average_strategy(data, short_window=3, long_window=10)
    filtered, _ = enhance_signals(data, raw, kind='dma',
                                  use_trend=True, trend_ma=60, use_confirm=True, confirm_days=3,
                                  use_adx=True, adx_threshold=99)  # ADX 阈值极高，几乎不允许入场
    # 不允许入场 → 持仓时间应为 0
    assert filtered['signal'].max() == 0


def test_simulate_trades_detailed_win_rate_and_return():
    # 交易1: idx2 入场(10) → idx5 离场(12)，盈利 20%
    # 交易2: idx7 入场(12) → idx9 离场(9)，亏损 25%
    closes = [10, 10, 10, 12, 12, 12, 12, 12, 9, 9, 9, 9]
    data = _make_data(closes)
    signal = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0], index=data.index, dtype=float)
    result = simulate_trades_detailed(data, signal, transaction_fee=0.0)
    assert result['n_trades'] == 2
    assert result['win_rate'] == pytest.approx(0.5)
    expected = (12 / 10) * (9 / 12) - 1
    assert result['cumulative_return'] == pytest.approx(expected)


def test_simulate_trades_consistent_with_legacy_simulator():
    rng = np.random.default_rng(21)
    closes = 20 + rng.normal(0, 0.4, 200).cumsum()
    data = _make_data(closes)
    sig = double_moving_average_strategy(data, 5, 20)
    from app.metrics_calculator import simulate_trades
    legacy = simulate_trades(data, sig)
    detailed = simulate_trades_detailed(data, sig['signal'])
    # 整数股 vs 小数股的差异应在 2% 以内
    assert abs(legacy - detailed['cumulative_return']) < 0.02
