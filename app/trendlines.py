"""
自动趋势线生成：供人工判断的技术结构参考

1. 支撑趋势线：连接近期摆动低点，要求全程位于价格下方（跌破即失效重找）
2. 阻力趋势线：连接近期摆动高点，要求全程位于价格上方
3. 水平支撑/阻力位：摆动点价格聚类，≥2 次触及才算有效位
4. 状态提示：支撑有效/逼近支撑/已跌破，压制中/逼近阻力/已突破

趋势线是客观的几何事实（哪两个低点连线、当前距离多远），但"趋势线必被尊重"
并非统计定律——请结合成交量与更大级别趋势判断。
"""
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .trend_analysis import find_swing_points

logger = logging.getLogger(__name__)


def _best_line(pivots, highs, lows, close, m, is_support, tol=0.004):
    """在摆动点两两组合中找最优趋势线。

    有效性：支撑线须位于区间内所有收盘价下方（容差 tol），阻力线在所有最高价上方。
    评分：触点数（线上 ±1% 内的摆动点）为主，连线近期的时效为辅。
    """
    best = None
    pv = pivots[-8:]
    for a in range(len(pv)):
        for b in range(a + 1, len(pv)):
            i1, p1 = pv[a]
            i2, p2 = pv[b]
            if i2 - i1 < 5:
                continue
            slope = (p2 - p1) / (i2 - i1)
            intercept = p1 - slope * i1
            line_at = slope * np.arange(i1, m) + intercept
            if is_support:
                seg = close[i1:]
                if not np.all(seg >= line_at * (1 - tol)):
                    continue
            else:
                seg = highs[i1:]
                if not np.all(seg <= line_at * (1 + tol)):
                    continue
            touches = sum(1 for i, p in pv[a:]
                          if abs(p - (slope * i + intercept)) <= max(p * 0.01, 1e-9))
            score = touches * 10 + i2 / max(m, 1) * 5
            if best is None or score > best['score']:
                best = {'score': score, 'slope': float(slope), 'intercept': float(intercept),
                        'start_idx': int(i1), 'touches': int(max(touches, 2))}
    return best


def _cluster_levels(pivot_prices, cluster_tol=0.015):
    """摆动点价格聚类：相邻价差 ≤ cluster_tol 归为一簇，簇内 ≥2 个点为有效水平位"""
    prices = sorted(p for _, p in pivot_prices)
    clusters = []
    for p in prices:
        if clusters and abs(p - clusters[-1][-1]) <= clusters[-1][-1] * cluster_tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [{'price': float(np.mean(cl)), 'touches': len(cl)} for cl in clusters if len(cl) >= 2]


def _line_status(cur, line_y, is_support):
    dist = (cur - line_y) / line_y if line_y else 0.0
    if is_support:
        if dist < -0.005:
            return '已跌破'
        if dist < 0.02:
            return '逼近支撑'
        return '支撑有效'
    if dist > 0.005:
        return '已突破'
    if dist > -0.02:
        return '逼近阻力'
    return '压制中'


def auto_trendlines(data, pivot_window=5, lookback=120, cluster_tol=0.015):
    """对一段K线数据自动生成趋势线与水平支撑/阻力位。

    返回 dict：start(视图起点在原数据中的偏移)、n_view、support/resistance(线或 None)、
    levels([{'price','touches','type'}])、note
    """
    n = len(data)
    if n < 40:
        return {'start': 0, 'n_view': n, 'support': None, 'resistance': None,
                'levels': [], 'note': 'K线数量不足（需 40 根以上）'}
    start = max(0, n - lookback)
    view = data.iloc[start:].reset_index(drop=True)
    m = len(view)
    swing_highs, swing_lows = find_swing_points(view, pivot_window)
    highs = view['high'].astype(float).values
    lows = view['low'].astype(float).values
    close = view['close'].astype(float).values
    cur = float(close[-1])

    support = _best_line(swing_lows, highs, lows, close, m, is_support=True)
    resistance = _best_line(swing_highs, highs, lows, close, m, is_support=False)

    for line, is_sup in ((support, True), (resistance, False)):
        if line:
            cur_y = line['slope'] * (m - 1) + line['intercept']
            line['current_y'] = float(cur_y)
            line['status'] = _line_status(cur, cur_y, is_sup)

    levels = []
    for lv in _cluster_levels(swing_lows, cluster_tol):
        if lv['price'] < cur:
            lv['type'] = 'support'
            levels.append(lv)
    for lv in _cluster_levels(swing_highs, cluster_tol):
        if lv['price'] >= cur:
            lv['type'] = 'resistance'
            levels.append(lv)
    sup_lv = sorted([l for l in levels if l['type'] == 'support'], key=lambda l: -l['price'])[:2]
    res_lv = sorted([l for l in levels if l['type'] == 'resistance'], key=lambda l: l['price'])[:2]

    return {'start': int(start), 'n_view': int(m), 'support': support, 'resistance': resistance,
            'levels': sup_lv + res_lv, 'note': ''}


def add_trendline_overlays(fig, data, result, show_lines=True, show_levels=True,
                           extend_bars=10, allow_future=True):
    """把趋势线与水平支撑/阻力位叠加到已有K线 fig 上"""
    if not result:
        return
    start = result.get('start', 0)
    n_view = result.get('n_view', 0)
    dates = data['trade_date'].reset_index(drop=True)
    if show_lines:
        for key, color, label in (('support', '#10B981', '支撑趋势线'),
                                  ('resistance', '#EF4444', '阻力趋势线')):
            line = result.get(key)
            if not line:
                continue
            i1 = start + line['start_idx']
            i2 = start + n_view - 1
            if i1 >= len(dates) or i2 >= len(dates):
                continue
            x = [dates.iloc[i1], dates.iloc[i2]]
            y = [line['slope'] * line['start_idx'] + line['intercept'],
                 line['slope'] * (n_view - 1) + line['intercept']]
            if allow_future and extend_bars > 0:
                fut = pd.date_range(dates.iloc[i2], periods=extend_bars + 1, freq='B')[1:]
                x += list(fut)
                y += [line['slope'] * (n_view - 1 + k) + line['intercept'] for k in range(1, extend_bars + 1)]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=label,
                                     line=dict(color=color, width=1.8, dash='dash'),
                                     hovertemplate=f'{label}: %{{y:.2f}}<extra></extra>'))
    if show_levels:
        for lv in result.get('levels', []):
            is_sup = lv['type'] == 'support'
            color = '#10B981' if is_sup else '#EF4444'
            fig.add_hline(y=lv['price'], line_dash='dot', line_color=color, line_width=1,
                          annotation_text=f"{'支撑' if is_sup else '阻力'} {lv['price']:.2f} · {lv['touches']}次触及",
                          annotation_position='right',
                          annotation_font=dict(size=10, color=color))


def trendline_note(result):
    """注释区一行：趋势线状态摘要"""
    if not result:
        return ''
    parts = []
    sup, res = result.get('support'), result.get('resistance')
    if sup:
        parts.append(f"支撑线 {sup['current_y']:.2f}（{sup['status']}）")
    if res:
        parts.append(f"阻力线 {res['current_y']:.2f}（{res['status']}）")
    if not parts:
        return '趋势线: 当前未形成有效趋势线'
    return '趋势线: ' + ' ｜ '.join(parts)
