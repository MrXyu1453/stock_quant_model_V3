"""多信号共振模块测试：各维度评分方向、权重归一、面板渲染"""
import numpy as np
import pandas as pd
import pytest

from app.confluence import (build_confluence_panel, compute_confluence,
                            _score_clv_flow, _score_trend)


def _make_data(closes, close_near_high=True, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    if close_near_high:
        high, low = closes * 1.001, closes * 0.99   # 收盘贴近最高价 → CLV 为正
    else:
        high, low = closes * 1.01, closes * 0.998   # 收盘贴近最低价 → CLV 为负
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': high, 'low': low,
        'close': closes, 'vol': 10000.0, 'amount': 1e7,
    }, index=idx)


def _top_factor_info():
    return {'covered': True, 'score': 2.1, 'rank': 3, 'total': 90, 'pct': 3.3,
            'model_name': '测试模型'}


def test_uptrend_scores_positive():
    data = _make_data(np.linspace(10, 20, 150))
    r = compute_confluence('600000.SH', data, factor_info=_top_factor_info())
    assert r['ok'] and r['total'] is not None
    dims = {d['key']: d for d in r['dimensions']}
    assert dims['trend']['score'] == pytest.approx(2.0)      # 价格>MA60 且 MA 上行且 ADX 高
    assert dims['factor']['score'] == 2.0                    # 因子前 3.3%
    assert dims['clv_flow']['score'] >= 1.0                  # 收盘贴高 → 资金净流入
    assert r['verdict'] in ('强共振看多', '偏多')
    assert all(d['evidence'] for d in r['dimensions'])


def test_downtrend_scores_negative():
    # 下跌行情：收盘贴近最低价（真实下跌形态），CLV 资金流为负
    data = _make_data(np.linspace(20, 10, 150), close_near_high=False)
    r = compute_confluence('000001.SZ', data, factor_info=None)
    dims = {d['key']: d for d in r['dimensions']}
    assert dims['trend']['score'] == pytest.approx(-2.0)
    assert dims['clv_flow']['score'] <= -1.0
    # 动量维度含均值回归信号（深度超卖+底背离会给出反弹加分），不强制为负；
    # 但综合结论必须由趋势/资金维度压回空头方向
    assert dims['momentum']['score'] <= 1.0
    # 因子维度缺失时按剩余权重归一，总分仍可计算
    assert r['total'] is not None
    assert r['total'] < 0
    assert r['verdict'] in ('偏空', '共振看空')


def test_clv_flow_direction():
    up = _make_data(np.linspace(10, 20, 60), close_near_high=True)
    down = _make_data(np.linspace(10, 20, 60), close_near_high=False)
    s_up, ev_up = _score_clv_flow(up)
    s_dn, ev_dn = _score_clv_flow(down)
    assert s_up >= 1.0 and '净流入' in ev_up
    assert s_dn <= -1.0 and '净流出' in ev_dn


def test_insufficient_data_handled():
    data = _make_data([10, 11, 12])
    r = compute_confluence('600000.SH', data, factor_info=None)
    assert not r['ok'] and r['verdict'] == '数据不足'


def test_factor_missing_redistributes_weights():
    data = _make_data(np.linspace(10, 20, 150))
    r_no = compute_confluence('600000.SH', data, factor_info=None)
    r_yes = compute_confluence('600000.SH', data, factor_info=_top_factor_info())
    assert r_no['total'] is not None and r_yes['total'] is not None
    assert r_yes['total'] > r_no['total']  # 因子头部应抬升总分


def test_high_adx_threshold_blocks_signal_quality():
    data = _make_data(np.linspace(10, 20, 150))
    r = compute_confluence('600000.SH', data, factor_info=None,
                           filter_opts=['trend', 'confirm', 'adx'], adx_threshold=150)
    dims = {d['key']: d for d in r['dimensions']}
    # ADX 阈值高到不可能（合成线性趋势 ADX≈100）→ 原始信号被过滤 → 信号质量应判负
    assert dims['signal_quality']['score'] < 0


def test_panel_renders():
    data = _make_data(np.linspace(10, 20, 150))
    r1 = compute_confluence('600000.SH', data, factor_info=_top_factor_info())
    r2 = compute_confluence('000001.SZ', _make_data([10, 11, 12]), factor_info=None)
    panel = build_confluence_panel([r1, r2], {'600000.SH': '浦发银行'})
    s = str(panel)
    assert '多信号共振' in s and '600000.SH' in s and '浦发银行' in s
    assert '综合' in s and '因子评分' in s
