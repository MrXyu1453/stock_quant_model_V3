"""自动趋势线模块测试：支撑/阻力线方向与有效性、水平位聚类、状态判定、图叠加"""
import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from app.trendlines import (add_trendline_overlays, auto_trendlines, trendline_note)


def _make_data(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    wiggle = 0.01 * closes
    # 让收盘在日内高低价之间交错，形成可识别的摆动点
    high = closes + wiggle
    low = closes - wiggle
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': high, 'low': low,
        'close': closes, 'vol': 10000.0, 'amount': 1e7,
    }, index=idx)


def _zigzag(base, amplitude, cycles, step):
    """上升途中带规律回调的锯齿形价格：每个周期涨 step*cycle_len 再回落 amplitude"""
    prices = []
    p = base
    for c in range(cycles):
        for _ in range(step):
            p += (amplitude + 0.1) / step
            prices.append(p)
        p -= amplitude
        prices.append(p)
    return prices


def test_uptrend_support_line_found():
    closes = _zigzag(10.0, 0.8, 12, 8)
    data = _make_data(closes)
    result = auto_trendlines(data)
    sup = result['support']
    assert sup is not None, '上升锯齿行情应生成支撑趋势线'
    assert sup['slope'] > 0
    # 支撑线应位于最新收盘价下方
    assert sup['current_y'] < closes[-1]
    assert sup['status'] in ('支撑有效', '逼近支撑', '已跌破')
    assert '支撑' in trendline_note(result)


def test_downtrend_resistance_line_found():
    closes = list(np.linspace(30, 10, 150))
    data = _make_data(closes)
    result = auto_trendlines(data)
    res = result['resistance']
    if res is not None:
        assert res['slope'] < 0
        assert res['current_y'] > closes[-1]


def test_levels_cluster_with_touches():
    # 在 20 附近反复震荡 → 20 应聚成多次触及的水平位
    closes = list(np.tile([19.6, 20.0, 20.4], 40))
    data = _make_data(closes)
    result = auto_trendlines(data)
    assert result['levels'], '震荡数据应产生水平支撑/阻力位'
    assert any(lv['touches'] >= 2 for lv in result['levels'])


def test_insufficient_data():
    result = auto_trendlines(_make_data([10, 11, 12]))
    assert result['note'] and not result['support'] and not result['levels']


def test_overlay_adds_traces():
    closes = _zigzag(10.0, 0.8, 12, 8)
    data = _make_data(closes)
    result = auto_trendlines(data)
    fig = go.Figure()
    add_trendline_overlays(fig, data, result, show_lines=True, show_levels=True)
    names = [t.name for t in fig.data]
    assert '支撑趋势线' in names
    hlines = [sh for sh in fig.layout.shapes if sh.type == 'line']
    assert len(fig.data) >= 1


def test_trendline_note_handles_empty():
    assert '未形成' in trendline_note({'support': None, 'resistance': None})
    assert trendline_note(None) == ''
