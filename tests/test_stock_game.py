"""模拟炒股小游戏核心规则测试: 费用、整手、T+1、FIFO 平仓、资产曲线、评级、战绩持久化、分钟线、随机时间段"""
import numpy as np
import pandas as pd
import pytest

from app import database_manager as dm
from app import stock_game as sg
from app.stock_game import (
    apply_buy, apply_sell, available_shares, avg_cost, buy_fee, equity_curve,
    max_drawdown, portfolio_value, rank_of, sell_fee, trade_stats,
)


def _state(cash=100000.0, lots=None, trades=None, idx=0, initial=100000.0):
    return {'game_id': 'g1', 'started': True, 'finished': False, 'idx': idx,
            'cash': cash, 'lots': lots or [], 'trades': trades or [],
            'initial_cash': initial, 'config': {}}


# ===================== 费用 =====================

def test_buy_fee_minimum_commission():
    # 小额交易佣金取最低 5 元: 100股*10元=1000元 → max(0.25, 5)=5 + 过户费0.01
    assert buy_fee(1000) == pytest.approx(5 + 1000 * 0.00001)
    # 大额交易按万2.5: 100万 → 250 + 10
    assert buy_fee(1_000_000) == pytest.approx(250 + 10)


def test_sell_fee_includes_stamp_tax():
    # 卖出才有印花税: 100股*100元=1万元 → max(2.5,5)=5 + 印花5 + 过户0.1
    assert sell_fee(10_000) == pytest.approx(5 + 10_000 * 0.0005 + 0.1)


# ===================== 买入 =====================

def test_buy_rejects_odd_lot():
    st, err = apply_buy(_state(), 150, 10.0, 0, '2024-01-01')
    assert st is None and '100' in err


def test_buy_rejects_insufficient_cash():
    # 需要 100*10+5.01 = 1005.01，只有 1000
    st, err = apply_buy(_state(cash=1000.0), 100, 10.0, 0, '2024-01-01')
    assert st is None and '资金不足' in err


def test_buy_success_updates_cash_and_lots():
    st, err = apply_buy(_state(cash=10000.0), 300, 10.0, 3, '2024-01-01')
    assert err == ''
    assert st['cash'] == pytest.approx(10000 - 3000 - buy_fee(3000), abs=0.01)
    assert st['lots'] == [{'bar': 3, 'shares': 300, 'price': 10.0}]
    assert st['trades'][0]['action'] == 'buy' and st['trades'][0]['bar'] == 3


def test_buy_does_not_mutate_input():
    st0 = _state(cash=10000.0)
    apply_buy(st0, 100, 10.0, 0, '2024-01-01')
    assert st0['cash'] == 10000.0 and st0['lots'] == []


# ===================== T+1 与卖出 =====================

def test_t_plus_one_lock():
    lots = [{'bar': 5, 'shares': 200, 'price': 10.0}]
    assert available_shares(lots, 5) == 0, '当根买入不可卖'
    assert available_shares(lots, 6) == 200, '下一根可卖'


def test_sell_rejects_over_availability():
    st0 = _state(lots=[{'bar': 1, 'shares': 100, 'price': 10.0}], idx=3)
    st, err = apply_sell(st0, 200, 11.0, 3, '2024-01-02')
    assert st is None and '不足' in err
    # T+1 锁定: 当根(bar=5)买入的不可卖
    st0 = _state(lots=[{'bar': 5, 'shares': 100, 'price': 10.0}], idx=5)
    st, err = apply_sell(st0, 100, 11.0, 5, '2024-01-02')
    assert st is None and 'T+1' in err


def test_sell_odd_lot_only_full_liquidation():
    st0 = _state(lots=[{'bar': 1, 'shares': 150, 'price': 10.0}], idx=3)
    # 卖出零股部分(50股)且不是清仓 → 拒绝（零股只能一次全清）
    st, err = apply_sell(st0, 50, 11.0, 3, '2024-01-02')
    assert st is None and '清仓' in err
    # 卖出整手部分(100股)合法
    st, err = apply_sell(st0, 100, 11.0, 3, '2024-01-02')
    assert err == '' and st['lots'] == [{'bar': 1, 'shares': 50, 'price': 10.0}]
    # 剩余零股50股一次性清掉合法
    st2, err2 = apply_sell(st, 50, 11.0, 3, '2024-01-02')
    assert err2 == '' and st2['lots'] == []


def test_sell_fifo_partial_reduction_and_pnl():
    lots = [{'bar': 1, 'shares': 200, 'price': 10.0},
            {'bar': 2, 'shares': 100, 'price': 12.0}]
    st0 = _state(cash=0.0, lots=lots, idx=5)
    # 拆散整手卖出250股 → 拒绝（非零股场景必须整手）
    st, err = apply_sell(st0, 250, 14.0, 5, '2024-01-02')
    assert st is None and '清仓' in err
    # 整手卖出200股 → FIFO 吃掉第一批 200股@10
    st, err = apply_sell(st0, 200, 14.0, 5, '2024-01-02')
    assert err == ''
    cost_basis = 200 * 10
    proceeds = 200 * 14
    fee = sell_fee(proceeds)
    assert st['trades'][0]['pnl'] == pytest.approx(proceeds - fee - cost_basis, abs=0.01)
    assert st['lots'] == [{'bar': 2, 'shares': 100, 'price': 12.0}]
    assert available_shares(st['lots'], 5) == 100
    # 清仓300股(含第二批)合法
    st2, err2 = apply_sell(st0, 300, 14.0, 5, '2024-01-02')
    assert err2 == '' and st2['lots'] == []


def test_avg_cost():
    lots = [{'bar': 1, 'shares': 200, 'price': 10.0},
            {'bar': 2, 'shares': 100, 'price': 13.0}]
    assert avg_cost(lots) == pytest.approx((2000 + 1300) / 300)
    assert avg_cost([]) == 0.0


# ===================== 资产曲线与统计 =====================

def test_portfolio_value():
    st = _state(cash=5000.0, lots=[{'bar': 0, 'shares': 100, 'price': 10.0}])
    assert portfolio_value(st, [10.0, 12.0], 1) == pytest.approx(5000 + 1200)


def test_equity_curve_and_benchmark():
    st = _state(cash=10000.0, idx=3, initial=10000.0)
    st, _ = apply_buy(st, 500, 10.0, 0, 'd0')   # 全仓 5000元 → 500股@10
    closes = [10.0, 11.0, 12.0, 9.0]
    dates = ['d0', 'd1', 'd2', 'd3']
    curve = equity_curve(st, closes, dates)
    assert len(curve) == 4
    # bar0: 买入后现金=5000-费用, 持仓500股 → 资产≈10000
    assert curve[0][1] == pytest.approx(10000 - buy_fee(5000), abs=0.01)
    assert curve[0][2] == pytest.approx(10000.0)  # 基准首日=初始资金
    assert curve[1][1] == pytest.approx(st['cash'] + 500 * 11.0, abs=0.01)
    assert curve[3][2] == pytest.approx(10000 / 10.0 * 9.0)  # 基准随价格下跌


def test_max_drawdown():
    assert max_drawdown([100, 120, 90, 110, 60]) == pytest.approx(0.5)
    assert max_drawdown([100, 110]) == 0.0
    assert max_drawdown([]) == 0.0


def test_trade_stats_win_rate():
    trades = [
        {'action': 'buy', 'shares': 100, 'fee': 5, 'price': 10, 'cash_after': 0, 'bar': 0},
        {'action': 'sell', 'shares': 100, 'fee': 6, 'price': 12, 'pnl': 194, 'cash_after': 0, 'bar': 2},
        {'action': 'buy', 'shares': 100, 'fee': 5, 'price': 12, 'cash_after': 0, 'bar': 3},
        {'action': 'sell', 'shares': 100, 'fee': 5, 'price': 11, 'pnl': -105, 'cash_after': 0, 'bar': 5},
    ]
    stats = trade_stats(trades)
    assert stats['n_buy'] == 2 and stats['n_sell'] == 2
    assert stats['win_rate'] == pytest.approx(0.5)
    assert stats['realized_pnl'] == pytest.approx(89)


# ===================== 评级 =====================

def test_rank_of_thresholds():
    assert rank_of(20, 4)[0] == 'S'
    assert rank_of(10, 4)[0] == 'A'
    assert rank_of(3, 4)[0] == 'B'
    assert rank_of(-1, 4)[0] == 'C'
    assert rank_of(-10, 4)[0] == 'D'
    assert rank_of(50, 0)[0] == '—', '未交易不给评级'


# ===================== 战绩持久化 / 排行榜 =====================

@pytest.fixture()
def game_db(monkeypatch, tmp_path):
    """临时数据库 + 固定登录用户，隔离真实库"""
    monkeypatch.setattr(dm, 'DB_NAME', str(tmp_path / 'game_test.db'))
    monkeypatch.setattr(dm, 'get_current_user_id', lambda: 7)
    monkeypatch.setattr(dm, 'get_current_username', lambda: 'tester')
    dm.init_db()
    return tmp_path


def _record(excess=5.0, n_buy=2, code='600519.SH'):
    return {'stock_code': code, 'stock_name': '贵州茅台', 'freq': 'daily',
            'play_bars': 120, 'initial_cash': 100000, 'final_value': 105000,
            'return_pct': 5.0, 'bench_return_pct': 2.0, 'excess_pp': excess,
            'max_dd': 0.05, 'n_buy': n_buy, 'n_sell': 1, 'win_rate': 1.0,
            'realized_pnl': 3000, 'fees': 50, 'grade': 'B', 'comment': '稳健盈利'}


def test_game_record_roundtrip(game_db):
    assert dm.add_game_record(_record(excess=3.0)) is not None
    assert dm.add_game_record(_record(excess=9.0, code='000001.SZ')) is not None
    top = dm.get_top_game_records(limit=10)
    assert len(top) == 2 and top[0]['excess_pp'] == 9.0, '按超额降序'
    mine = dm.get_my_game_records(7)
    assert len(mine) == 2
    s = dm.get_game_user_summary(7)
    assert s['games'] == 2 and s['avg_excess'] == 6.0
    assert s['positive_ratio'] == 1.0 and s['best_excess'] == 9.0


def test_game_record_requires_login(game_db, monkeypatch):
    monkeypatch.setattr(dm, 'get_current_user_id', lambda: None)
    assert dm.add_game_record(_record()) is None
    assert dm.get_top_game_records() == []


def test_game_record_excludes_zero_trade_entries(game_db):
    dm.add_game_record(_record(excess=99.0, n_buy=0))  # 未交易不入榜
    assert dm.get_top_game_records() == []


def test_save_record_on_finish(game_db):
    monkey_id = None
    # 构造一个已结算的游戏状态
    closes = [10.0, 11.0, 12.0]
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    g = {'df': pd.DataFrame({'trade_date': dates}), 'view_start': 0, 'ctx_len': 0,
         'play_len': 3, 'code': '600519.SH', 'name': '贵州茅台', 'freq': 'daily'}
    # _game_bars 依赖 df 切片: view_start+ctx 到 +play_len
    g['df'] = pd.DataFrame({'trade_date': pd.date_range('2024-01-01', periods=3),
                            'close': closes})
    state = _state(cash=6000.0, idx=2)
    state, _ = apply_buy(state, 100, 10.0, 0, '2024-01-01')
    state['finished'] = True
    assert sg._save_record(state, g) is True
    top = dm.get_top_game_records()
    assert len(top) == 1 and top[0]['username'] == 'tester'
    assert top[0]['stock_code'] == '600519.SH'


def test_board_panels_render(game_db):
    dm.add_game_record(_record())
    top_panel = sg.build_leaderboard_panel()
    assert '600519.SH' in str(top_panel)
    my_panel = sg.build_my_records_panel()
    assert '600519.SH' in str(my_panel)
    container = sg.build_board_container('my')
    assert 'tester' in str(container) or '600519' in str(container)


# ===================== 分钟线快节奏模式 =====================

def _minute_data(n=400, freq_min=60, start='2025-01-05'):
    idx = pd.date_range(start, periods=n, freq=f'{freq_min * 60}min')
    closes = 10 + np.linspace(0, 3, n) + np.sin(np.arange(n) / 8) * 0.5
    wiggle = 0.008 * closes
    return pd.DataFrame({'trade_date': idx, 'open': closes, 'high': closes + wiggle,
                         'low': closes - wiggle, 'close': closes, 'vol': 1e4,
                         'amount': 1e6})


def test_create_game_minute_freq(monkeypatch):
    monkeypatch.setattr(sg, 'get_and_process_data',
                        lambda *a, **k: _minute_data(400, 60).copy())
    game_id, err = sg._create_game('600519.SH', '60min', 120, 'tushare')
    assert err == '' and game_id in sg._GAMES
    g = sg._GAMES[game_id]
    assert g['freq'] == '60min' and g['play_len'] == 120
    closes, dates = sg._game_bars(g)
    assert len(closes) == 120 and ':' in dates[0], '分钟线日期含时间'


def test_create_game_weekly_freq(monkeypatch):
    idx = pd.date_range('2024-01-01', periods=300, freq='W')
    closes = 10 + np.linspace(0, 5, 300)
    df = pd.DataFrame({'trade_date': idx, 'open': closes, 'high': closes * 1.01,
                       'low': closes * 0.99, 'close': closes, 'vol': 1e4, 'amount': 1e6})
    monkeypatch.setattr(sg, 'get_and_process_data', lambda *a, **k: df.copy())
    game_id, err = sg._create_game('600519.SH', 'weekly', 120, 'tushare')
    assert err == '' and sg._GAMES[game_id]['freq'] == 'weekly'


# ===================== 随机时间段 =====================

def _daily_data(n=600, start='2020-01-05'):
    idx = pd.date_range(start, periods=n, freq='D')
    closes = 10 + np.linspace(0, 5, n)
    return pd.DataFrame({'trade_date': idx, 'open': closes, 'high': closes * 1.01,
                         'low': closes * 0.99, 'close': closes, 'vol': 1e4, 'amount': 1e6})


def test_create_game_random_window(monkeypatch):
    """游戏窗口在拉取的历史中随机抽取，不再固定贴最新数据"""
    monkeypatch.setattr(sg, 'get_and_process_data',
                        lambda *a, **k: _daily_data(600).copy())
    t0 = pd.Timestamp('2020-01-05')
    starts = set()
    for _ in range(10):
        game_id, err = sg._create_game('600519.SH', 'daily', 120, 'tushare')
        assert err == ''
        g = sg._GAMES[game_id]
        closes, dates = sg._game_bars(g)
        assert len(closes) == 120
        # 窗口前保留预热+上下文（≥95根），末尾保留隐藏区（≥40根），不会贴到数据两端
        assert pd.Timestamp(dates[0]) >= t0 + pd.Timedelta(days=95)
        assert pd.Timestamp(dates[-1]) <= t0 + pd.Timedelta(days=560)
        starts.add(pd.Timestamp(dates[0]))
    assert len(starts) >= 3, '多次开局应抽到不同的时间段'


def test_create_game_short_history_falls_back_to_latest(monkeypatch):
    """历史刚够用时退化为贴尾部的固定窗口（与旧行为一致），且不报错"""
    monkeypatch.setattr(sg, 'get_and_process_data',
                        lambda *a, **k: _daily_data(140).copy())
    game_id, err = sg._create_game('600519.SH', 'daily', 120, 'tushare')
    assert err == ''
    g = sg._GAMES[game_id]
    closes, dates = sg._game_bars(g)
    assert len(closes) == 120
    assert pd.Timestamp(dates[-1]) >= pd.Timestamp('2020-01-05') + pd.Timedelta(days=135)
