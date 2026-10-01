"""趋势分析页趋势线模块测试：图表叠加、显示开关、分钟线延伸、信号与去重、收敛形态"""
import numpy as np
import pandas as pd
import pytest

from app.trend_analysis import (
    _dedup_signals, analyze_trendline_convergence, build_trendline_figure,
    generate_trendline_signals,
)
from app.trendlines import auto_trendlines


def _make_data(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    wiggle = 0.01 * closes
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': closes + wiggle,
        'low': closes - wiggle, 'close': closes, 'vol': 10000.0,
    }, index=idx)


def _zigzag(base, amplitude, cycles, step):
    """上升途中带规律回调的锯齿形价格"""
    prices = []
    p = base
    for _ in range(cycles):
        for _ in range(step):
            p += (amplitude + 0.1) / step
            prices.append(p)
        p -= amplitude
        prices.append(p)
    return prices


@pytest.fixture(scope='module')
def tl_setup():
    closes = _zigzag(10.0, 0.8, 25, 8)
    data = _make_data(closes)
    short = auto_trendlines(data, pivot_window=5, lookback=60)
    long_r = auto_trendlines(data, pivot_window=5, lookback=120)
    tl_results = [('短期', short), ('中期', long_r)]
    return data, tl_results, long_r.get('levels', [])


def test_figure_draws_all_trendlines(tl_setup):
    data, tl_results, levels = tl_setup
    fig, info = build_trendline_figure(data, tl_results, levels_result=levels, freq='daily')
    names = [t.name or '' for t in fig.data]
    assert any('短期支撑趋势线' in n for n in names)
    assert any('短期阻力趋势线' in n for n in names)
    assert any('中期支撑趋势线' in n for n in names)
    assert any('中期阻力趋势线' in n for n in names)
    hlines = [s for s in fig.layout.shapes if s.type == 'line']
    assert len(hlines) >= len(levels), '水平支撑/阻力位应绘制为水平线'
    assert '趋势线:' in info and '水平位' in info


def test_figure_respects_show_flags(tl_setup):
    data, tl_results, levels = tl_setup
    fig, _ = build_trendline_figure(data, tl_results, levels_result=levels,
                                    show_lines=False, show_levels=False)
    names = [t.name or '' for t in fig.data]
    assert not any('趋势线' in n for n in names), 'show_lines=False 不应绘制趋势线'
    line_shapes = [s for s in fig.layout.shapes
                   if s.type == 'line' and s.line.dash == 'dot']
    assert not line_shapes, 'show_levels=False 不应绘制水平位虚线'


def test_intraday_no_future_extension(tl_setup):
    data, tl_results, _ = tl_setup
    fig, _ = build_trendline_figure(data, tl_results, freq='60min')
    last_base = pd.Timestamp(data['trade_date'].iloc[-1]).strftime('%Y-%m-%d')
    for trace in fig.data:
        if trace.name and '趋势线' in trace.name:
            assert str(trace.x[-1])[:10] <= last_base, '分钟线趋势线不应向右延伸'


def test_generate_signals_module_tag(tl_setup):
    data, tl_results, levels = tl_setup
    sigs = generate_trendline_signals(data, tl_results, levels)
    assert all(s['module'] == 'trendlines' for s in sigs)
    for s in sigs:
        assert s['action'] in ('buy', 'sell', 'hold')
        assert s['detail']


def test_generate_signals_empty_inputs():
    data = _make_data(_zigzag(10.0, 0.8, 10, 8))
    assert generate_trendline_signals(data, [], []) == []
    empty_result = auto_trendlines(_make_data([10, 11, 12]))
    assert generate_trendline_signals(data, [('短期', empty_result)], []) == []


def test_dedup_signals():
    merged = _dedup_signals([
        {'date': '2024-01-01', 'price': 10.0},
        {'date': '2024-01-01', 'price': 10.0},
        {'date': '2024-01-01', 'price': 11.0},
    ])
    assert len(merged) == 2
    assert merged[0]['price'] == 11.0, '去重后按价格降序'


# ===================== 收敛形态（三角形/楔形） =====================

def _line(slope, intercept, start_idx=0):
    return {'slope': slope, 'intercept': intercept, 'start_idx': start_idx,
            'touches': 2, 'current_y': 0.0, 'status': ''}


def test_convergence_symmetric_triangle():
    sup = _line(0.012, 10.0, start_idx=10)
    res = _line(-0.012, 13.0, start_idx=12)
    conv = analyze_trendline_convergence(sup, res, 120)
    assert conv is not None
    assert conv['pattern'] == '对称三角形'
    assert conv['apex_idx'] == pytest.approx((13.0 - 10.0) / 0.024)
    assert 0 < conv['bars_to_apex'] <= 60
    assert conv['apex_price'] == pytest.approx(10.0 + 0.012 * conv['apex_idx'])


def test_convergence_wedge_classification():
    # 上升楔形: 两线均向上、支撑涨得更快 → 顶点在前方
    conv = analyze_trendline_convergence(_line(0.02, 10.0), _line(0.005, 11.0), 64)
    assert conv is not None and conv['pattern'] == '上升楔形'
    # 下降楔形: 两线均向下、阻力跌得更快
    conv = analyze_trendline_convergence(_line(-0.005, 12.0), _line(-0.02, 13.0), 64)
    assert conv is not None and conv['pattern'] == '下降楔形'
    # 上升三角形: 阻力水平 + 支撑上斜
    conv = analyze_trendline_convergence(_line(0.01, 10.0, start_idx=5), _line(0.0, 13.0), 260)
    assert conv is not None and conv['pattern'] == '上升三角形'
    # 下降三角形: 支撑水平 + 阻力下斜
    conv = analyze_trendline_convergence(_line(0.0, 10.0, start_idx=5), _line(-0.01, 13.0), 260)
    assert conv is not None and conv['pattern'] == '下降三角形'


def test_convergence_rejects_diverging_and_crossed():
    # 发散: 支撑下斜 + 阻力上斜
    assert analyze_trendline_convergence(_line(-0.01, 10.0), _line(0.01, 13.0), 120) is None
    # 已经交叉: 间距在最后一根已为负
    assert analyze_trendline_convergence(_line(0.02, 10.0), _line(0.005, 11.0), 120) is None
    # 顶点太远
    assert analyze_trendline_convergence(_line(0.01, 10.0), _line(0.0, 13.0), 120) is None
    # 缺线
    assert analyze_trendline_convergence(None, _line(0.01, 13.0), 120) is None


def test_convergence_apex_within_horizon():
    # 顶点在 60 根之外 → 不认为有效收敛
    sup = _line(0.001, 10.0)   # 斜率极缓 → 交点极远
    res = _line(-0.001, 13.0)
    assert analyze_trendline_convergence(sup, res, 120) is None


def _triangle_bounce_data(cycles=10, start='2024-01-01'):
    """价格在收敛上下边之间震荡: 上边 13-0.012i, 下边 10+0.012i"""
    rows = []
    for i in range(cycles * 12):
        lo, up = 10.0 + 0.012 * i, 13.0 - 0.012 * i
        w = 1 - abs((i % 12) - 6) / 6.0
        rows.append(lo + (up - lo) * w)
    closes = np.asarray(rows)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    wiggle = 0.01 * closes
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': closes + wiggle,
        'low': closes - wiggle, 'close': closes, 'vol': 1000.0,
    })


def test_convergence_detected_from_real_pipeline():
    data = _triangle_bounce_data()
    result = auto_trendlines(data, pivot_window=5, lookback=120)
    conv = analyze_trendline_convergence(result.get('support'),
                                         result.get('resistance'),
                                         result.get('n_view', 0))
    assert conv is not None, '收敛三角数据应检测出收敛形态'
    assert conv['pattern'] in ('对称三角形', '上升楔形', '下降楔形',
                              '上升三角形', '下降三角形')


def test_figure_renders_convergence():
    data = _triangle_bounce_data()
    result = auto_trendlines(data, pivot_window=5, lookback=120)
    conv = analyze_trendline_convergence(result.get('support'),
                                         result.get('resistance'),
                                         result.get('n_view', 0))
    if conv is None:
        pytest.skip('该组参数下未检出收敛，跳过渲染断言')
    fig, info = build_trendline_figure(data, [('短期', result)], freq='daily')
    names = [t.name or '' for t in fig.data]
    assert any('收敛' in n for n in names), f'缺少收敛区域填充: {names}'
    trace_texts = [''.join(map(str, t.text or [])) for t in fig.data]
    assert any('顶点' in t for t in trace_texts), f'缺少顶点标注: {trace_texts}'
    assert '收敛' in info


def test_convergence_signal_near_apex():
    data = _triangle_bounce_data()
    # 手工构造逼近顶点的线组（顶点约6根后），确保信号路径被执行
    fake_result = {'start': 0, 'n_view': 120,
                   'support': _line(0.012, 10.0, start_idx=10),
                   'resistance': _line(-0.012, 13.0, start_idx=12),
                   'levels': [], 'note': ''}
    sigs = generate_trendline_signals(data, [('短期', fake_result)], [])
    hits = [s for s in sigs if s['signal'] == '⚠️ 收敛变盘']
    assert hits and all(s['module'] == 'trendlines' for s in hits)
    assert '收敛' in hits[0]['detail'] and hits[0]['action'] == 'hold'
    # 远离顶点的线组不给信号
    far_result = dict(fake_result)
    far_result['support'] = _line(0.001, 10.0, start_idx=10)
    far_result['resistance'] = _line(-0.001, 13.0, start_idx=12)
    assert not [s for s in generate_trendline_signals(data, [('短期', far_result)], [])
                if s['signal'] == '⚠️ 收敛变盘']
