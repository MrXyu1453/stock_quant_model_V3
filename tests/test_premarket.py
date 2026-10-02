"""盘前提醒模块测试: 缓存、交易日历、复盘、异动计算、聚合与卡片渲染"""
import pandas as pd
import pytest

from app import premarket as pm


class _FakePro:
    """tushare pro 客户端替身，各方法可按测试注入返回值/异常"""

    def __init__(self):
        self.trade_cal_df = None
        self.trade_dates_df = None   # baostock 交易日历替身
        self.index_daily_df = None
        self.new_share_df = None
        self.trade_cal_raises = False

    def trade_cal(self, **kwargs):
        if self.trade_cal_raises:
            raise RuntimeError('api down')
        return self.trade_cal_df

    def index_daily(self, **kwargs):
        if self.index_daily_df is None:
            raise RuntimeError('no perm')
        return self.index_daily_df.get(kwargs.get('ts_code'))

    def new_share(self, **kwargs):
        return self.new_share_df


@pytest.fixture()
def pm_env(monkeypatch):
    """隔离环境: 假 pro、停用限频、强制内存缓存(禁用真实Redis)、baostock 返回空"""
    fake = _FakePro()
    monkeypatch.setattr(pm, 'pro', fake)
    monkeypatch.setattr(pm, '_rate_limit', lambda: None)
    monkeypatch.setattr(pm, 'redis_client', None)
    monkeypatch.setattr(pm, '_memory_cache', {})   # 全新内存缓存，杜绝跨测试串扰
    monkeypatch.setattr(pm, 'fetch_trade_dates', lambda s, e: fake.trade_dates_df)
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda *a, **k: None)
    return fake


def _kdata(closes, start='2025-01-01', amount=4e7):
    idx = pd.date_range(start, periods=len(closes), freq='B')
    closes = [float(c) for c in closes]
    wig = 0.01
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': [c * (1 + wig) for c in closes],
        'low': [c * (1 - wig) for c in closes], 'close': closes,
        'vol': [1000.0] * len(closes), 'amount': [amount] * len(closes),
    })


# ===================== 缓存 =====================

def test_memory_cache_roundtrip(pm_env):
    pm._cache_set('premarket:test:k', {'a': 1})
    assert pm._cache_get('premarket:test:k') == {'a': 1}
    pm._memory_cache['premarket:test:k'] = (0, {'a': 1})  # 置为过期
    assert pm._cache_get('premarket:test:k') is None


# ===================== 交易日历（baostock 主源） =====================

def test_trading_day_open_baostock(pm_env):
    pm_env.trade_dates_df = pd.DataFrame({
        'calendar_date': ['2025-01-02', '2025-01-03'], 'is_trading_day': ['1', '1']})
    info = pm.trading_day_info('20250102')
    assert info['is_open'] is True and info['source'] == 'baostock'


def test_trading_day_closed_with_next_baostock(pm_env):
    # 2025-01-04 周六休市, 下一交易日 01-06
    pm_env.trade_dates_df = pd.DataFrame({
        'calendar_date': ['2025-01-04', '2025-01-05', '2025-01-06'],
        'is_trading_day': ['0', '0', '1']})
    info = pm.trading_day_info('20250104')
    assert info['is_open'] is False and info['next'] == '20250106'


def test_trading_day_fallback_to_tushare(pm_env, monkeypatch):
    monkeypatch.setattr(pm, 'fetch_trade_dates', lambda *a, **k: None)
    pm_env.trade_cal_df = pd.DataFrame({
        'cal_date': ['20250102', '20250103'], 'is_open': [1, 1]})
    info = pm.trading_day_info('20250102')
    assert info['is_open'] is True and info['source'] == 'tushare'


def test_trading_day_api_failure_returns_none(pm_env, monkeypatch):
    monkeypatch.setattr(pm, 'fetch_trade_dates', lambda *a, **k: None)
    pm_env.trade_cal_raises = True
    assert pm.trading_day_info('20250102') is None, '工作日无法确认，返回None不妄断'


def test_trading_day_weekend_fallback(pm_env, monkeypatch):
    # 双源都失败时: 周末必休市(2025-01-04是周六), 工作日不妄断
    monkeypatch.setattr(pm, 'fetch_trade_dates', lambda *a, **k: None)
    pm_env.trade_cal_raises = True
    info = pm.trading_day_info('20250104')
    assert info['is_open'] is False and info['source'] == 'weekend'


# ===================== 隔夜海外 =====================

def test_global_via_tencent(monkeypatch):
    # GBK 编码的腾讯行情响应替身（字段: [3]=最新价 [4]=昨收）；usINX=标普500
    body = ('v_usDJI="200~道琼斯~.DJI~51390.93~51349.92~51424.84~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~2026-09-30~41.01~0.08~0~0~USD";\n'
            'v_usINX="200~标普500~.INX~7706.69~7670.84~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~2026-09-30~35.85~0.47~0~0~USD";\n'
            'v_hkHSI="100~恒生指数~HSI~24613.27~24523.57~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~0~~2026-09-30~89.70~0.37~0~0~HKD";\n'
            'v_jpN225="";\n')

    class _Resp:
        content = body.encode('gbk')

    import types
    import sys
    fake_requests = types.SimpleNamespace(get=lambda *a, **k: _Resp())
    monkeypatch.setitem(sys.modules, 'requests', fake_requests)
    out = pm._global_via_tencent()
    assert len(out) == 3, '空数据(jpN225)应剔除'
    assert out[0]['name'] == '道琼斯' and out[0]['close'] == 51390.93
    assert out[0]['chg'] == pytest.approx(0.08, abs=0.01)


def test_global_overnight_chain_tencent_to_eastmoney(monkeypatch, pm_env):
    # 腾讯失败 → 东财接管
    def tencent_down():
        raise RuntimeError('blocked')

    monkeypatch.setattr(pm, '_global_via_tencent', tencent_down)
    payload = {'data': {'diff': [
        {'f2': 51368.14, 'f3': 0.04, 'f12': 'DJIA', 'f14': '道琼斯'}]}}

    class _Resp:
        def json(self):
            return payload

    import types
    import sys
    fake_requests = types.SimpleNamespace(get=lambda *a, **k: _Resp())
    monkeypatch.setitem(sys.modules, 'requests', fake_requests)
    out = pm.global_overnight()
    assert len(out) == 1 and out[0]['close'] == 51368.14


def test_global_via_eastmoney(monkeypatch):
    payload = {'data': {'diff': [
        {'f2': 51368.14, 'f3': 0.04, 'f12': 'DJIA', 'f14': '道琼斯'},
        {'f2': 7706.69, 'f3': 0.47, 'f12': 'SPX', 'f14': '标普500'},
        {'f2': '-', 'f3': '-', 'f12': 'BAD', 'f14': '坏数据'},
    ]}}

    class _Resp:
        def json(self):
            return payload

    # mock requests 层: _global_via_eastmoney 函数内 import requests 时取到替身
    import types
    import sys
    fake_requests = types.SimpleNamespace(get=lambda *a, **k: _Resp())
    monkeypatch.setitem(sys.modules, 'requests', fake_requests)
    out = pm._global_via_eastmoney()
    assert len(out) == 2, '无效数据(横杠)应被剔除'
    assert out[0]['name'] == '道琼斯' and out[0]['close'] == 51368.14


def test_global_overnight_fallback_to_tushare(monkeypatch, pm_env):
    # 腾讯/东财都失败 → 回退 tushare（仅 DJI 有权限）
    def down():
        raise RuntimeError('blocked')

    monkeypatch.setattr(pm, '_global_via_tencent', down)
    monkeypatch.setattr(pm, '_global_via_eastmoney', down)
    dji = pd.DataFrame({'trade_date': ['20250102', '20250103'],
                        'close': [44000.0, 44300.0]})
    pm_env.index_daily_df = {'DJI': dji}
    out = pm.global_overnight()
    assert len(out) == 1
    assert out[0]['name'] == '道琼斯' and out[0]['chg'] == pytest.approx(0.68, abs=0.01)


# ===================== A股复盘（baostock 主源） =====================

def test_a_share_recap_with_mocked_data(monkeypatch):
    # baostock 的 amount 单位为元: 上证 4e11元 = 4000亿, 深证 3e11元 = 3000亿
    data = {
        '000001.SH': _kdata([3000, 3020, 3040], amount=4e11),
        '399001.SZ': _kdata([10000, 10100, 10200], amount=3e11),
        '399006.SZ': _kdata([2000, 1980, 2020], amount=1e11),
        '000300.SH': _kdata([3900, 3910, 3950], amount=2e11),
    }
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda c, s, e, adjustflag='2', freq='daily': data[c])
    recap = pm.a_share_recap()
    assert len(recap['indices']) == 4
    sh = recap['indices'][0]
    assert sh['name'] == '上证指数' and sh['chg'] == pytest.approx(0.66, abs=0.01)
    assert recap['amount_total'] == 7000.0, '两市成交额 = 4000亿 + 3000亿'
    assert recap['amount_chg'] == 0.0, '前后两日成交额相同(测试数据)应计为0%'


def test_amount_to_yi_unit_detection():
    assert pm._amount_to_yi(4e8) == 4000.0       # 千元 (tushare, 4千亿)
    assert pm._amount_to_yi(4e10) == 400.0       # 元 (baostock, 400亿)
    assert pm._amount_to_yi(5e9) == 50000.0      # 千元极端成交日(5万亿)不被误判为元
    assert pm._amount_to_yi(0) is None
    assert pm._amount_to_yi(None) is None
    assert pm._amount_to_yi(5.0) is None         # 量级异常宁缺毋假


# ===================== 新股 =====================

def test_new_shares_none_vs_empty(pm_env):
    pm_env.new_share_df = None
    assert pm.new_shares_today() == []            # 确认今日无申购
    pm_env.new_share_df = pd.DataFrame([
        {'name': '测试科技', 'subtype_code': '301999'}])
    out = pm.new_shares_today()
    assert out == [{'name': '测试科技', 'code': '301999'}]


# ===================== 自选股异动 =====================

def test_stock_moves_signals(monkeypatch):
    # 构造: 昨日大涨 + 放量 + 创20日新高
    n = 30
    closes = [10.0] * (n - 1) + [10.8]
    df = _kdata(closes)
    df.loc[df.index[-1], 'vol'] = 5000.0  # 放量5倍
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda *a, **k: df)
    m = pm._stock_moves('600519.SH', '20250101', '20250301')
    assert m['chg'] == pytest.approx(8.0)
    assert '大涨' in m['signals'] and '创20日新高' in m['signals']
    assert any(s.startswith('放量') for s in m['signals'])


def test_stock_moves_new_low(monkeypatch):
    closes = [10.0] * 29 + [9.0]
    df = _kdata(closes)
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda *a, **k: df)
    m = pm._stock_moves('600519.SH', '20250101', '20250301')
    assert '大跌' in m['signals'] and '创20日新低' in m['signals']


def test_stock_moves_quiet_when_no_signal(monkeypatch):
    df = _kdata([10.0] * 25)
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda *a, **k: df)
    m = pm._stock_moves('600519.SH', '20250101', '20250301')
    assert m['signals'] == []


def test_watchlist_moves_filters_notable(monkeypatch):
    quiet = _kdata([10.0] * 25)
    mover = _kdata([10.0] * 24 + [10.5])
    data = {'600001.SH': quiet, '600002.SH': mover}
    monkeypatch.setattr(pm, 'fetch_kline_data',
                        lambda c, s, e, adjustflag='2', freq='daily': data[c])
    monkeypatch.setattr(pm, 'get_stock_watchlist_map',
                        lambda: {'600001.SH': ['稳'], '600002.SH': ['动']})
    out = pm.watchlist_moves()
    assert [o['code'] for o in out] == ['600002.SH']
    assert '大涨' in out[0]['signals']


# ===================== 持仓关注 =====================

def test_holdings_watch_pnl(monkeypatch):
    entries = [{'stock_code': '600519.SH', 'stock_name': '贵州茅台', 'direction': '买入',
                'quantity': 100, 'price': 1000.0, 'fee': 5, 'trade_date': '2025-01-01',
                'id': 1}]
    monkeypatch.setattr(pm, 'get_trade_entries', lambda: entries)
    monkeypatch.setattr(pm, 'compute_portfolio',
                        lambda e: {'holdings': [{'stock_code': '600519.SH',
                                                 'stock_name': '贵州茅台',
                                                 'quantity': 100, 'avg_cost': 1000.05}],
                                   'cash_delta': 0, 'total_realized': 0,
                                   'realized_series': []})
    data = _kdata([1000.0, 1020.0], amount=2e11)
    monkeypatch.setattr(pm, 'fetch_kline_data', lambda *a, **k: data)
    out = pm.holdings_watch()
    assert len(out) == 1
    h = out[0]
    assert h['pnl'] == pytest.approx((1020.0 - 1000.05) * 100, abs=1)
    assert h['chg'] == pytest.approx(2.0)


def test_holdings_watch_empty_journal(monkeypatch):
    monkeypatch.setattr(pm, 'get_trade_entries', lambda: [])
    assert pm.holdings_watch() == []


# ===================== 聚合与缓存 =====================

def test_fetch_briefing_aggregates_and_caches(monkeypatch, pm_env):
    calls = {'n': 0}

    def fake_recap():
        calls['n'] += 1
        return {'indices': [{'name': '上证指数', 'close': 3000.0, 'chg': 0.5,
                             'date': '2025-01-02'}]}

    monkeypatch.setattr(pm, 'trading_day_info', lambda: {'is_open': True, 'next': None})
    monkeypatch.setattr(pm, 'a_share_recap', fake_recap)
    monkeypatch.setattr(pm, 'global_overnight', lambda: [])
    monkeypatch.setattr(pm, 'new_shares_today', lambda: [])
    monkeypatch.setattr(pm, 'get_current_user_id', lambda: None)  # 未登录: 无个性化板块

    b1 = pm.fetch_premarket_briefing(use_cache=True)
    b2 = pm.fetch_premarket_briefing(use_cache=True)
    assert b1['from_cache'] is False, '首次应重算'
    assert b2['from_cache'] is True, '第二次应命中缓存'
    assert calls['n'] == 1, '缓存命中不应重算'
    assert 'watchlist' not in b1 and 'holdings' not in b1, '未登录不含个性化板块'

    b3 = pm.fetch_premarket_briefing(use_cache=False)
    assert b3['from_cache'] is False and calls['n'] == 2, '绕过缓存应重算'


def test_fetch_briefing_section_failure_isolated(monkeypatch, pm_env):
    def boom():
        raise RuntimeError('x')

    monkeypatch.setattr(pm, 'get_current_user_id', lambda: 7)
    monkeypatch.setattr(pm, 'global_overnight', boom)
    monkeypatch.setattr(pm, 'a_share_recap', boom)
    monkeypatch.setattr(pm, 'new_shares_today', lambda: [])
    monkeypatch.setattr(pm, 'watchlist_moves', lambda: [])
    monkeypatch.setattr(pm, 'holdings_watch', lambda: [])
    b = pm.fetch_premarket_briefing(use_cache=False)
    assert 'global' not in b and 'recap' not in b   # 失败板块缺失
    assert 'new_shares' in b and 'watchlist' in b   # 成功板块保留


def test_personal_sections_isolated_per_user(monkeypatch, pm_env):
    """个性化板块必须按 user_id 隔离: 不同账号各自计算与缓存，互不可见"""
    calls = {'compute': 0}
    monkeypatch.setattr(pm, 'trading_day_info', lambda: {'is_open': True, 'next': None})
    monkeypatch.setattr(pm, 'global_overnight', lambda: [])
    monkeypatch.setattr(pm, 'a_share_recap', lambda: {'indices': []})
    monkeypatch.setattr(pm, 'new_shares_today', lambda: [])

    def fake_holdings():
        calls['compute'] += 1
        uid = current_uid['v']
        return [{'code': f'60000{uid}.SH', 'name': f'用户{uid}的持仓', 'pnl': uid}]

    current_uid = {'v': 1}
    monkeypatch.setattr(pm, 'holdings_watch', fake_holdings)
    monkeypatch.setattr(pm, 'watchlist_moves', lambda: [])

    current_uid['v'] = 1
    monkeypatch.setattr(pm, 'get_current_user_id', lambda: 1)
    b_u1_a = pm.fetch_premarket_briefing(use_cache=True)
    b_u1_b = pm.fetch_premarket_briefing(use_cache=True)   # 命中用户1缓存
    assert b_u1_a['holdings'][0]['code'] == '600001.SH'
    assert b_u1_b['holdings'][0]['code'] == '600001.SH'
    assert calls['compute'] == 1, '同一用户第二次应命中个人缓存'

    current_uid['v'] = 2
    monkeypatch.setattr(pm, 'get_current_user_id', lambda: 2)
    b_u2 = pm.fetch_premarket_briefing(use_cache=True)
    assert b_u2['holdings'][0]['code'] == '600002.SH', '用户2应看到自己的持仓'
    assert calls['compute'] == 2, '不同用户各自计算'
    assert '600001' not in str(b_u2['holdings']), '用户2不得看到用户1的持仓'

    current_uid['v'] = 1
    monkeypatch.setattr(pm, 'get_current_user_id', lambda: 1)
    b_u1_c = pm.fetch_premarket_briefing(use_cache=True)
    assert b_u1_c['holdings'][0]['code'] == '600001.SH'
    assert calls['compute'] == 2, '切回用户1仍命中其个人缓存'
    assert '600002' not in str(b_u1_c['holdings']), '用户1不得看到用户2的持仓'


# ===================== 卡片渲染 =====================

def test_build_card_renders_all_sections():
    brief = {
        'date': '20250102', 'generated_at': '2025-01-02 08:00', 'is_open': True,
        'global': [{'name': '道琼斯', 'close': 44300.0, 'chg': 0.68, 'date': '20250103'}],
        'recap': {'indices': [{'name': '上证指数', 'close': 3040.0, 'chg': 0.66,
                               'date': '2025-01-03'}],
                  'last_date': '2025-01-03', 'amount_total': 12000, 'amount_chg': 5.0},
        'watchlist': [{'code': '600002.SH', 'chg': 5.0, 'close': 10.5,
                       'signals': ['大涨', '创20日新高'], 'date': '2025-01-03'}],
        'holdings': [{'code': '600519.SH', 'name': '贵州茅台', 'chg': 2.0,
                      'close': 1020.0, 'qty': 100, 'pnl': 1995.0, 'pnl_pct': 2.0}],
        'new_shares': [{'name': '测试科技', 'code': '301999'}],
    }
    card = pm.build_premarket_card(brief)
    s = str(card)
    for token in ('盘前提醒', '道琼斯', '上证指数', '自选股异动', '持仓关注',
                  '测试科技', '今日为交易日', '12,000'):
        assert token in s, f'缺少: {token}'


def test_build_card_closed_day_and_empty():
    brief = {'date': '20250104', 'generated_at': 'x', 'is_open': False,
             'next_open': '20250106', 'new_shares': []}
    s = str(pm.build_premarket_card(brief))
    assert '今日休市' in s and '20250106' in s and '无新股申购' in s


def test_build_card_none_brief():
    s = str(pm.build_premarket_card(None))
    assert '数据加载失败' in s
