"""策略与绩效指标单元测试（strategy_manager / metrics_calculator）"""
import numpy as np
import pandas as pd
import pytest

from app.strategy_manager import (
    double_moving_average_strategy,
    simple_moving_average_strategy,
    exponential_moving_average_strategy,
)
from app.metrics_calculator import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    simulate_trades,
)


def make_price_data(n=60, seed=3):
    """构造一段带趋势的价格序列，index 为日期"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    close = pd.Series(100 + np.cumsum(rng.normal(0.2, 1.0, size=n)), index=dates)
    close = close.clip(lower=1)
    return pd.DataFrame({'close': close}, index=dates)


# ==================== 策略 ====================

class TestDoubleMovingAverage:
    def test_signal_only_0_or_1(self):
        data = make_price_data()
        sig = double_moving_average_strategy(data, 5, 20)
        assert set(sig['signal'].dropna().unique()).issubset({0.0, 1.0})

    def test_insufficient_data_returns_zero_signals(self):
        data = make_price_data(10)
        sig = double_moving_average_strategy(data, 5, 20)
        assert (sig['signal'] == 0).all()
        assert list(sig.columns) == ['signal', 'short_mavg', 'long_mavg', 'positions']

    def test_positions_is_diff_of_signal(self):
        data = make_price_data()
        sig = double_moving_average_strategy(data, 3, 10)
        expected = sig['signal'].diff()
        pd.testing.assert_series_equal(sig['positions'], expected, check_names=False)


class TestSimpleMA:
    def test_basic(self):
        data = make_price_data()
        sig = simple_moving_average_strategy(data, window=10)
        assert 'mavg' in sig.columns
        # 第 window-1 个点之前 signal 为 0
        assert (sig['signal'].iloc[:9] == 0).all()

    def test_empty_data(self):
        data = make_price_data(3)
        sig = simple_moving_average_strategy(data, window=20)
        assert (sig['signal'] == 0).all()


class TestEmaMA:
    def test_basic(self):
        data = make_price_data()
        sig = exponential_moving_average_strategy(data, window=10)
        assert 'emavg' in sig.columns
        assert sig['emavg'].notna().all()


# ==================== 指标 ====================

class TestSharpe:
    def test_constant_price_zero_sharpe(self):
        data = pd.DataFrame({'close': [100.0] * 30})
        assert calculate_sharpe_ratio(data) == 0.0

    def test_positive_trend_positive_sharpe(self):
        data = pd.DataFrame({'close': np.linspace(100, 120, 60)})
        assert calculate_sharpe_ratio(data) > 0

    def test_too_short_returns_zero(self):
        assert calculate_sharpe_ratio(pd.DataFrame({'close': [1.0]})) == 0.0


class TestMaxDrawdown:
    def test_monotonic_up_zero_dd(self):
        data = pd.DataFrame({'close': np.linspace(100, 150, 50)})
        assert calculate_max_drawdown(data) == 0.0

    def test_known_drawdown(self):
        # 100 -> 120 -> 60: 回撤 = 1 - 60/120 = -0.5
        data = pd.DataFrame({'close': [100.0, 120.0, 60.0]})
        assert calculate_max_drawdown(data) == pytest.approx(-0.5)


class TestSimulateTrades:
    def test_buy_and_hold_capture_trend(self):
        data = make_price_data(80, seed=11)
        sig = double_moving_average_strategy(data, 5, 20)
        ret = simulate_trades(data, sig, initial_capital=100000)
        assert isinstance(ret, float)
        # 有交易发生（信号出现过 1）
        assert sig['signal'].max() == 1.0

    def test_no_trades_keeps_capital(self):
        data = make_price_data(30)
        sig = pd.DataFrame({'signal': 0.0, 'positions': np.nan},
                           index=data.index)
        ret = simulate_trades(data, sig, initial_capital=100000)
        assert ret == 0.0

    def test_mismatched_index_handled(self):
        data = make_price_data(30)
        sig = pd.DataFrame({'signal': 1.0, 'positions': 1.0},
                           index=pd.date_range('2030-01-01', periods=30, freq='B'))
        # 索引完全不匹配不应抛异常
        ret = simulate_trades(data, sig)
        assert isinstance(ret, float)

    def test_none_signals_returns_zero(self):
        data = make_price_data(10)
        assert simulate_trades(data, None) == 0.0
