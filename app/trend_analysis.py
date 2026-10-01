"""
趋势分析模块 - 五浪分析、斐波那契、通道分析、支撑阻力等
"""
import dash
from dash import dcc, html, callback_context, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

from .data_manager import get_and_process_data, validate_code, get_available_sources, set_data_source

# 模块级图表缓存（供全屏弹窗使用）
_figure_cache = {}

# ===================== 交易信号生成 =====================

def generate_trading_signals(data, supports, resistances, fib_levels, channels, bb, patterns, macd_data, wave_segments):
    """
    根据各项分析结果生成交易建议，每条信号带 module 来源标记
    返回: list of dict {date, price, signal, action, detail, module}
    module: 'sr'/'fibonacci'/'channel'/'bollinger'/'elliott'/'pattern'/'macd'
    """
    signals = []
    closes = data['close'].values
    dates = data['trade_date'].dt.strftime('%Y-%m-%d').tolist() if hasattr(data['trade_date'], 'dt') else [str(d)[:10] for d in data['trade_date']]
    last_close = closes[-1]
    last_date = dates[-1]

    # 1. 支撑阻力信号 → module='sr'
    for s in supports[:3]:
        pct = abs(last_close - s['level']) / last_close * 100
        if pct < 2:
            if last_close > s['level']:
                signals.append({'date': last_date, 'price': s['level'],
                    'signal': '🟢 买入', 'action': 'buy',
                    'detail': f"逼近支撑位 {s['level']:.2f}(触及{s['touches']}次)，回调可买入",
                    'module': 'sr'})
            else:
                signals.append({'date': last_date, 'price': s['level'],
                    'signal': '🔴 卖出', 'action': 'sell',
                    'detail': f"跌破支撑位 {s['level']:.2f}，建议止损",
                    'module': 'sr'})

    for r in resistances[:3]:
        pct = abs(last_close - r['level']) / last_close * 100
        if pct < 2:
            if last_close < r['level']:
                signals.append({'date': last_date, 'price': r['level'],
                    'signal': '🔴 卖出', 'action': 'sell',
                    'detail': f"逼近阻力位 {r['level']:.2f}(触及{r['touches']}次)，可止盈",
                    'module': 'sr'})
            else:
                signals.append({'date': last_date, 'price': r['level'],
                    'signal': '🟢 追涨', 'action': 'buy',
                    'detail': f"突破阻力位 {r['level']:.2f}，可追涨买入",
                    'module': 'sr'})

    # 2. 斐波那契信号 → module='fibonacci'
    if fib_levels:
        fib_items = sorted(fib_levels.items(), key=lambda x: x[1])
        for lvl, price in fib_items:
            if abs(last_close - price) / last_close * 100 < 1.5:
                if lvl == 0.0:
                    signals.append({'date': last_date, 'price': price,
                        'signal': '🟢 强支撑', 'action': 'buy',
                        'detail': f"跌至前低 {price:.2f}(0%回撤)，强支撑可买入",
                        'module': 'fibonacci'})
                elif lvl == 0.382:
                    signals.append({'date': last_date, 'price': price,
                        'signal': '🟢 加仓', 'action': 'buy',
                        'detail': f"回调至38.2%回撤位 {price:.2f}，可加仓",
                        'module': 'fibonacci'})
                elif lvl == 0.5:
                    signals.append({'date': last_date, 'price': price,
                        'signal': '🟡 观察', 'action': 'hold',
                        'detail': f"50%回撤位 {price:.2f}，多空分水岭，建议观望",
                        'module': 'fibonacci'})
                elif lvl == 0.618:
                    signals.append({'date': last_date, 'price': price,
                        'signal': '🟢 买入', 'action': 'buy',
                        'detail': f"黄金分割61.8%回撤位 {price:.2f}，最佳买点",
                        'module': 'fibonacci'})

    # 3. 通道信号
    if channels:
        last_ch = channels[-1]
        ch_upper = last_ch['upper_line'][-1] if len(last_ch['upper_line']) > 0 else 0
        ch_lower = last_ch['lower_line'][-1] if len(last_ch['lower_line']) > 0 else 0
        ch_dir = last_ch['direction']

        if ch_lower > 0:
            pct_low = abs(last_close - ch_lower) / last_close * 100
            pct_up = abs(last_close - ch_upper) / last_close * 100
            if pct_low < 2 and last_close > ch_lower:
                action = 'buy' if ch_dir == '上升' else 'hold'
                signals.append({'date': last_date, 'price': ch_lower,
                    'signal': '🟢 买入' if action == 'buy' else '🟡 观察',
                    'action': action,
                    'detail': f"触及{ch_dir}通道下轨 {ch_lower:.2f}，{'上升通道可买入' if action == 'buy' else '建议观望'}",
                    'module': 'channel'})
            if pct_up < 2 and last_close < ch_upper:
                signals.append({'date': last_date, 'price': ch_upper,
                    'signal': '🔴 卖出',
                    'action': 'sell',
                    'detail': f"触及{ch_dir}通道上轨 {ch_upper:.2f}，可减仓止盈",
                    'module': 'channel'})

    # 4. 布林带信号 → module='channel' (与通道共用一个图表)
    if bb and len(bb['upper']) > 0 and not np.isnan(bb['upper'][-1]):
        bb_upper = bb['upper'][-1]
        bb_lower = bb['lower'][-1]
        if abs(last_close - bb_lower) / last_close * 100 < 2:
            signals.append({'date': last_date, 'price': bb_lower,
                'signal': '🟢 超卖', 'action': 'buy',
                'detail': f"触及布林下轨 {bb_lower:.2f}，超卖反弹概率大，可买入",
                'module': 'channel'})
        if abs(last_close - bb_upper) / last_close * 100 < 2:
            signals.append({'date': last_date, 'price': bb_upper,
                'signal': '🔴 超买', 'action': 'sell',
                'detail': f"触及布林上轨 {bb_upper:.2f}，超买回调概率大，可卖出",
                'module': 'channel'})
        if bb['squeeze_idx']:
            signals.append({'date': last_date, 'price': last_close,
                'signal': '⚠️ 变盘', 'action': 'hold',
                'detail': '布林带收窄(挤压)，即将变盘，建议观望等方向确认',
                'module': 'channel'})

    # 5. 波浪信号 → module='elliott'
    if wave_segments:
        last_wave = wave_segments[-1]
        if last_wave['direction'] == 'up':
            w5_price = last_wave['wave_5_price']
            pct_w5 = abs(last_close - w5_price) / last_close * 100
            if pct_w5 < 5:
                signals.append({'date': dates[last_wave['wave_5_idx']] if last_wave['wave_5_idx'] < len(dates) else last_date,
                    'price': w5_price,
                    'signal': '🔴 浪5见顶', 'action': 'sell',
                    'detail': f"接近上升浪5终点 {w5_price:.2f}，建议逐步止盈",
                    'module': 'elliott'})
            w4_price = last_wave['wave_4_price']
            w4_pct = abs(last_close - w4_price) / last_close * 100
            if w4_pct < 3:
                signals.append({'date': dates[last_wave['wave_4_idx']] if last_wave['wave_4_idx'] < len(dates) else last_date,
                    'price': w4_price,
                    'signal': '🟢 浪4回调', 'action': 'buy',
                    'detail': f"回调至浪4区域 {w4_price:.2f}，可买入等浪5",
                    'module': 'elliott'})

    # 6. 形态信号 → module='pattern'
    if patterns:
        recent = patterns[-3:]
        for p in recent:
            if '看涨' in p['name']:
                signals.append({'date': dates[p['idx']] if p['idx'] < len(dates) else last_date,
                    'price': p['close'], 'signal': '🟢 看涨形态', 'action': 'buy',
                    'detail': f"{p['name']}形态，买入信号",
                    'module': 'pattern'})
            elif '看跌' in p['name']:
                signals.append({'date': dates[p['idx']] if p['idx'] < len(dates) else last_date,
                    'price': p['close'], 'signal': '🔴 看跌形态', 'action': 'sell',
                    'detail': f"{p['name']}形态，卖出信号",
                    'module': 'pattern'})

    # 7. MACD背离信号 → module='pattern' (与形态共用一个图表)
    if macd_data and macd_data['divergences']:
        for d in macd_data['divergences'][-3:]:
            if '底背离' in d['type']:
                signals.append({'date': dates[d['idx']] if d['idx'] < len(dates) else last_date,
                    'price': d['price'], 'signal': '🟢 底背离', 'action': 'buy',
                    'detail': f"MACD底背离，价格新低但MACD转强，买入信号",
                    'module': 'pattern'})
            elif '顶背离' in d['type']:
                signals.append({'date': dates[d['idx']] if d['idx'] < len(dates) else last_date,
                    'price': d['price'], 'signal': '🔴 顶背离', 'action': 'sell',
                    'detail': f"MACD顶背离，价格新高但MACD转弱，卖出信号",
                    'module': 'pattern'})

    return _dedup_signals(signals)


def _dedup_signals(signals):
    """按 (日期, 价格) 去重并按价格降序排序"""
    seen = set()
    unique = []
    for s in signals:
        key = (s['date'][:10], round(s['price'], 2))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return sorted(unique, key=lambda x: x['price'], reverse=True)


def build_signal_panel(signals, module_filter=None):
    """根据信号列表构建 HTML 建议面板，可按 module 过滤"""
    if module_filter:
        signals = [s for s in signals if s.get('module') == module_filter]

    if not signals:
        return html.Div("暂无明确交易信号，建议观望", className="text-gray-400 text-xs italic")

    rows = []
    for s in signals[:8]:
        bg = 'bg-red-50 border-red-200' if s['action'] == 'sell' else \
             'bg-green-50 border-green-200' if s['action'] == 'buy' else \
             'bg-yellow-50 border-yellow-200'
        date_str = s.get('date', '')[:10] if s.get('date') else ''
        rows.append(html.Div([
            html.Span(date_str, className="text-[10px] text-gray-400 font-mono mr-1.5 min-w-[72px] inline-block"),
            html.Span(s['signal'], className="text-xs font-bold mr-2"),
            html.Span(s['detail'], className="text-xs text-gray-700"),
        ], className=f"{bg} border rounded px-2 py-1.5 mb-1 flex items-center flex-wrap"))

    return html.Div(rows, className="mt-2")


def generate_trendline_signals(data, tl_results, levels):
    """趋势线模块信号：支撑/阻力趋势线状态 + 水平支撑阻力位 → module='trendlines'

    tl_results: [(标签, auto_trendlines 结果)]，levels: [{'price','touches','type'}]
    """
    signals = []
    closes = data['close'].values
    if len(closes) == 0:
        return signals
    last_close = float(closes[-1])
    if hasattr(data['trade_date'], 'dt'):
        last_date = data['trade_date'].dt.strftime('%Y-%m-%d').tolist()[-1]
    else:
        last_date = str(data['trade_date'].tolist()[-1])[:10]

    for tag, result in tl_results:
        for key, is_sup in (('support', True), ('resistance', False)):
            line = result.get(key)
            if not line or 'current_y' not in line:
                continue
            cur_y = line['current_y']
            status = line.get('status', '')
            label = f"{tag}{'支撑' if is_sup else '阻力'}线"
            if is_sup and status == '逼近支撑':
                signals.append({'date': last_date, 'price': cur_y,
                    'signal': '🟢 买入', 'action': 'buy',
                    'detail': f"{label} {cur_y:.2f} 逼近，回踩企稳可买入",
                    'module': 'trendlines'})
            elif is_sup and status == '已跌破':
                signals.append({'date': last_date, 'price': cur_y,
                    'signal': '🔴 卖出', 'action': 'sell',
                    'detail': f"已跌破{label} {cur_y:.2f}，趋势转弱建议止损",
                    'module': 'trendlines'})
            elif not is_sup and status == '逼近阻力':
                signals.append({'date': last_date, 'price': cur_y,
                    'signal': '🔴 卖出', 'action': 'sell',
                    'detail': f"逼近{label} {cur_y:.2f}，可分批止盈",
                    'module': 'trendlines'})
            elif not is_sup and status == '已突破':
                signals.append({'date': last_date, 'price': cur_y,
                    'signal': '🟢 追涨', 'action': 'buy',
                    'detail': f"已突破{label} {cur_y:.2f}，趋势加速可持有",
                    'module': 'trendlines'})

    for lv in (levels or []):
        pct = abs(last_close - lv['price']) / last_close * 100
        if pct >= 2:
            continue
        is_sup = lv['type'] == 'support'
        if is_sup and last_close > lv['price']:
            signals.append({'date': last_date, 'price': lv['price'],
                'signal': '🟢 买入', 'action': 'buy',
                'detail': f"回调至水平支撑 {lv['price']:.2f}(触及{lv['touches']}次)，可买入",
                'module': 'trendlines'})
        elif not is_sup and last_close < lv['price']:
            signals.append({'date': last_date, 'price': lv['price'],
                'signal': '🔴 卖出', 'action': 'sell',
                'detail': f"逼近水平阻力 {lv['price']:.2f}(触及{lv['touches']}次)，可止盈",
                'module': 'trendlines'})

    # 收敛形态逼近顶点 → 变盘观察信号（方向由突破确认，不给单边建议）
    for tag, result in tl_results:
        conv = analyze_trendline_convergence(result.get('support'),
                                             result.get('resistance'),
                                             result.get('n_view', 0))
        if conv and conv['bars_to_apex'] <= 10:
            signals.append({'date': last_date, 'price': conv['apex_price'],
                'signal': '⚠️ 收敛变盘', 'action': 'hold',
                'detail': (f"{tag}{conv['pattern']}收敛中，约{int(round(conv['bars_to_apex']))}根K线后"
                           f"到达顶点 {conv['apex_price']:.2f}，{conv['tendency']}，等待放量突破确认"),
                'module': 'trendlines'})
    return signals

def find_swing_points(data, window=5):
    """找出局部高点和低点"""
    highs = data['high'].values
    lows = data['low'].values
    n = len(highs)
    swing_highs = []
    swing_lows = []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def elliott_wave_analysis(data, window=5):
    """五浪分析 - 识别潜在的推动浪结构"""
    swing_highs, swing_lows = find_swing_points(data, window)
    waves = []
    all_points = sorted(swing_highs + swing_lows, key=lambda x: x[0])

    for i in range(len(all_points) - 4):
        seg = all_points[i:i + 5]
        prices = [p[1] for p in seg]
        # 推动浪: 浪1、浪3、浪5逐浪升高(上升趋势) 或 逐浪降低(下降趋势)
        is_uptrend = prices[2] > prices[0] and prices[4] > prices[2]  # 浪3>浪1, 浪5>浪3
        is_downtrend = prices[2] < prices[0] and prices[4] < prices[2]
        if is_uptrend or is_downtrend:
            waves.append({
                'wave_1_idx': seg[0][0], 'wave_1_price': seg[0][1],
                'wave_2_idx': seg[1][0], 'wave_2_price': seg[1][1],
                'wave_3_idx': seg[2][0], 'wave_3_price': seg[2][1],
                'wave_4_idx': seg[3][0], 'wave_4_price': seg[3][1],
                'wave_5_idx': seg[4][0], 'wave_5_price': seg[4][1],
                'direction': 'up' if is_uptrend else 'down'
            })
    return waves, swing_highs, swing_lows


def fibonacci_retracement(high, low, levels=None):
    """计算斐波那契回撤位 (0% 在最低点, 100% 在最高点)"""
    if levels is None:
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    hi = max(high, low)
    lo = min(high, low)
    diff = hi - lo
    levels_dict = {}
    for lvl in levels:
        levels_dict[lvl] = lo + lvl * diff
    return levels_dict


def fibonacci_extension(high, low, retrace, levels=None):
    """计算斐波那契扩展位 (基于回撤后延续)"""
    if levels is None:
        levels = [1.0, 1.272, 1.382, 1.5, 1.618, 2.0, 2.618]
    diff = abs(high - low)
    direction = 1 if high > low else -1
    levels_dict = {}
    for lvl in levels:
        levels_dict[lvl] = retrace + direction * lvl * diff
    return levels_dict


def _compute_fib_levels(data, lookback_days=90):
    """预计算斐波那契回撤位，返回 (levels_dict, fib_start_idx, fib_end_idx)"""
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    n = len(closes)
    lookback = min(lookback_days, n)

    recent_high = np.max(highs[-lookback:])
    recent_low = np.min(lows[-lookback:])
    high_idx = np.argmax(highs[-lookback:]) + n - lookback
    low_idx = np.argmin(lows[-lookback:]) + n - lookback

    if high_idx > low_idx:
        fib_start, fib_end = low_idx, high_idx
    else:
        fib_start, fib_end = high_idx, low_idx

    # 无论趋势方向，回撤位始终从近期最低点(0%)画到近期最高点(100%)
    return fibonacci_retracement(recent_high, recent_low), fib_start, fib_end


def channel_analysis(data, lookback=50):
    """上升/下降通道分析 - 线性回归通道"""
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    n = len(closes)
    if n < lookback:
        lookback = n

    channels = []
    for i in range(lookback, n, max(lookback // 3, 5)):
        x = np.arange(i - lookback, i)
        y = closes[i - lookback:i]
        if len(y) < 10:
            continue
        # 线性回归
        coeffs = np.polyfit(x, y, 1)
        mid_line = np.polyval(coeffs, x)
        residuals = y - mid_line
        std_res = np.std(residuals)

        # 通道上轨/下轨
        top_diff = max(highs[i - lookback:i] - mid_line)
        bot_diff = min(lows[i - lookback:i] - mid_line)

        # 通道方向
        slope = coeffs[0]
        direction = '上升' if slope > 0 else '下降'
        width = top_diff - bot_diff

        channels.append({
            'start_idx': i - lookback,
            'end_idx': i,
            'slope': slope,
            'direction': direction,
            'top_offset': top_diff,
            'bot_offset': bot_diff,
            'width': width,
            'x': x,
            'mid_line': mid_line,
            'upper_line': mid_line + top_diff,
            'lower_line': mid_line + bot_diff,
        })

    return channels


def support_resistance_analysis(data, min_touches=2, tolerance=0.02):
    """支撑阻力分析 - 基于历史价格触及次数"""
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values

    all_levels = []
    # 从 swing points 提取可能的支撑阻力位
    window = 5
    for i in range(window, len(closes) - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            all_levels.append(('resistance', highs[i]))
        if lows[i] == min(lows[i - window:i + window + 1]):
            all_levels.append(('support', lows[i]))

    # 聚类相近的水平
    if not all_levels:
        return [], []

    prices = [p for _, p in all_levels]
    price_range = max(prices) - min(prices)
    if price_range == 0:
        price_range = max(prices) * 0.001
    merge_tol = price_range * tolerance

    # 简并聚类
    clusters = []
    used = [False] * len(all_levels)
    for i in range(len(all_levels)):
        if used[i]:
            continue
        cluster = [all_levels[i]]
        used[i] = True
        for j in range(i + 1, len(all_levels)):
            if used[j]:
                continue
            if abs(all_levels[j][1] - all_levels[i][1]) < merge_tol:
                cluster.append(all_levels[j])
                used[j] = True
        avg_price = np.mean([p[1] for p in cluster])
        types = [p[0] for p in cluster]
        dominent_type = 'support' if types.count('support') >= types.count('resistance') else 'resistance'
        if len(cluster) >= min_touches:
            clusters.append({'level': avg_price, 'type': dominent_type, 'touches': len(cluster)})

    supports = sorted([c for c in clusters if c['type'] == 'support'],
                       key=lambda x: x['level'], reverse=True)
    resistances = sorted([c for c in clusters if c['type'] == 'resistance'],
                          key=lambda x: x['level'])

    return supports, resistances


def trendline_breakout(data, lookback=20):
    """趋势线突破检测 - 检测支撑/阻力突破"""
    closes = data['close'].values
    n = len(closes)
    if n < lookback:
        return [], []

    breakouts_up = []
    breakouts_down = []

    for i in range(lookback, n):
        segment = closes[i - lookback:i]
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, 1)
        resistance_line = np.polyval(coeffs, np.array([len(segment)])) + np.std(segment)

        if closes[i] > resistance_line and closes[i - 1] <= resistance_line:
            breakouts_up.append({'idx': i, 'price': closes[i]})

        support_line = np.polyval(coeffs, np.array([len(segment)])) - np.std(segment)
        if closes[i] < support_line and closes[i - 1] >= support_line:
            breakouts_down.append({'idx': i, 'price': closes[i]})

    return breakouts_up, breakouts_down


def candlestick_patterns(data):
    """K线形态识别"""
    patterns = []
    opens = data['open'].values
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values

    for i in range(1, len(closes)):
        body = closes[i] - opens[i]
        upper_shadow = highs[i] - max(opens[i], closes[i])
        lower_shadow = min(opens[i], closes[i]) - lows[i]
        total_range = highs[i] - lows[i] if highs[i] != lows[i] else 0.0001

        name = None

        # 十字星
        if abs(body) < total_range * 0.1 and upper_shadow > 0 and lower_shadow > 0:
            name = '十字星'
        # 锤子线
        elif lower_shadow > abs(body) * 2 and upper_shadow < abs(body) * 0.5:
            name = '锤子线(看涨)' if body > 0 else '吊颈线(看跌)'
        # 射击之星
        elif upper_shadow > abs(body) * 2 and lower_shadow < abs(body) * 0.5:
            name = '倒锤子(看涨)' if body < 0 else '射击之星(看跌)'
        # 大阳线/大阴线
        elif abs(body) > total_range * 0.7:
            name = '大阳线' if body > 0 else '大阴线'
        # 吞没形态
        elif i >= 1:
            prev_body = closes[i - 1] - opens[i - 1]
            if prev_body < 0 and body > 0 and opens[i] < closes[i - 1] and closes[i] > opens[i - 1]:
                name = '看涨吞没'
            elif prev_body > 0 and body < 0 and opens[i] > closes[i - 1] and closes[i] < opens[i - 1]:
                name = '看跌吞没'

        if name:
            patterns.append({'idx': i, 'name': name, 'close': closes[i]})

    return patterns


def moving_average_ribbon(data, windows=None):
    """均线带分析"""
    if windows is None:
        windows = [5, 10, 20, 30, 60, 120]
    ribbons = {}
    for w in windows:
        ribbons[w] = data['close'].rolling(window=w).mean().values
    return ribbons


def macd_divergence(data, fast=12, slow=26, signal=9):
    """MACD 背离检测"""
    closes = data['close'].values
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line

    divergences = []
    # 简化背离检测
    for i in range(20, len(closes) - 1):
        # 顶背离: 价格新高但 MACD 柱更低
        if closes[i] > closes[i - 5] and histogram[i] < histogram[i - 5]:
            divergences.append({'idx': i, 'type': '顶背离(看跌)', 'price': closes[i]})
        # 底背离: 价格新低但 MACD 柱更高
        elif closes[i] < closes[i - 5] and histogram[i] > histogram[i - 5]:
            divergences.append({'idx': i, 'type': '底背离(看涨)', 'price': closes[i]})

    return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram,
            'divergences': divergences[-3:] if divergences else []}


def volume_profile_analysis(data, bins=30):
    """成交量分布分析"""
    closes = data['close'].values
    vols = data['vol'].values
    price_min, price_max = np.min(closes), np.max(closes)
    if price_min == price_max:
        return None, None, None

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    volume_profile = np.zeros(bins)

    for i, close in enumerate(closes):
        for j in range(bins):
            if bin_edges[j] <= close <= bin_edges[j + 1]:
                volume_profile[j] += vols[i]
                break

    # POC (Point of Control)
    poc_idx = np.argmax(volume_profile)
    poc = bin_centers[poc_idx]
    total_vol = np.sum(volume_profile)
    vwap = np.sum(closes * vols) / np.sum(vols) if np.sum(vols) > 0 else 0

    return bin_centers, volume_profile, {'poc': poc, 'vwap': vwap, 'total_vol': total_vol}


def bollinger_bands(data, window=20, num_std=2):
    """布林带分析"""
    closes = data['close'].values
    rolling_mean = pd.Series(closes).rolling(window=window).mean().values
    rolling_std = pd.Series(closes).rolling(window=window).std().values
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    bandwidth = (upper - lower) / rolling_mean  # 带宽 - 衡量波动率

    # 挤压检测: 带宽缩到最小值
    valid_bw = bandwidth[~np.isnan(bandwidth)]
    if len(valid_bw) > 0:
        bw_min = np.nanmin(valid_bw)
        # squeeze when bandwidth < 20th percentile
        squeeze_threshold = np.nanpercentile(valid_bw, 20)
        squeeze_idx = np.where(bandwidth < squeeze_threshold)[0]
    else:
        squeeze_idx = np.array([])

    return {'upper': upper, 'middle': rolling_mean, 'lower': lower,
            'bandwidth': bandwidth, 'squeeze_idx': squeeze_idx.tolist()}


def atr_indicator(data, window=14):
    """ATR (Average True Range) - 真实波动幅度"""
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = pd.Series(tr).rolling(window=window).mean().values
    return atr


def rsi_calculation(data, window=14):
    """RSI 计算"""
    closes = data['close'].values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(window=window).mean().values
    avg_loss = pd.Series(losses).rolling(window=window).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    return np.insert(rsi, 0, np.nan)


# ===================== 图表构建函数 =====================

def build_elliott_wave_figure(data, waves, swing_highs, swing_lows, dates):
    """构建五浪分析图 - 单面板清晰版"""
    n = len(data)
    # 只显示最近 120 天，避免图表过密
    slice_start = max(0, n - 180)
    d_slice = data.iloc[slice_start:].reset_index(drop=True)
    dates_slice = dates[slice_start:]

    fig = go.Figure()

    # K线
    fig.add_trace(go.Candlestick(
        x=dates_slice, open=d_slice['open'], high=d_slice['high'],
        low=d_slice['low'], close=d_slice['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ))

    # 摆动点筛到可见区域
    sh_vis = [(i - slice_start, p) for i, p in swing_highs if i >= slice_start]
    sl_vis = [(i - slice_start, p) for i, p in swing_lows if i >= slice_start]

    if sh_vis:
        fig.add_trace(go.Scatter(
            x=[dates_slice[i] for i, _ in sh_vis],
            y=[p for _, p in sh_vis],
            mode='markers+text',
            text=['H'] * len(sh_vis),
            textposition='top center',
            textfont=dict(color='#ef4444', size=10, family='Arial Black'),
            marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                        line=dict(color='white', width=1)),
            name='摆动高点'
        ))
    if sl_vis:
        fig.add_trace(go.Scatter(
            x=[dates_slice[i] for i, _ in sl_vis],
            y=[p for _, p in sl_vis],
            mode='markers+text',
            text=['L'] * len(sl_vis),
            textposition='bottom center',
            textfont=dict(color='#22c55e', size=10, family='Arial Black'),
            marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                        line=dict(color='white', width=1)),
            name='摆动低点'
        ))

    # 五浪标注 — 筛选可见区域内的完整浪结构
    wave_segments = []
    for w in waves:
        w1_idx_adj = w['wave_1_idx'] - slice_start
        w5_idx_adj = w['wave_5_idx'] - slice_start
        if 0 <= w1_idx_adj < len(dates_slice) and 0 <= w5_idx_adj < len(dates_slice):
            wave_segments.append(w)

    wave_colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

    for wi, w in enumerate(wave_segments[-2:]):  # 至多显示最近2个
        xs, ys, labels = [], [], []
        for j in range(1, 6):
            idx_adj = w[f'wave_{j}_idx'] - slice_start
            if 0 <= idx_adj < len(dates_slice):
                xs.append(dates_slice[idx_adj])
                ys.append(w[f'wave_{j}_price'])
                labels.append(f'浪{j}')

        if len(xs) >= 3:
            for si in range(len(xs) - 1):
                seg_color = wave_colors[si % 5]
                fig.add_trace(go.Scatter(
                    x=xs[si:si + 2], y=ys[si:si + 2],
                    mode='lines+markers',
                    line=dict(color=seg_color, width=3),
                    marker=dict(size=10, color=seg_color,
                                line=dict(color='white', width=1.5)),
                    name=f"浪{si+1}-{si+2}" if wi == 0 and si == 0 else None,
                    showlegend=(wi == 0 and si == 0)
                ))
            # 添加浪标签
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='text',
                text=labels,
                textposition='top center',
                textfont=dict(color='#1e293b', size=13, family='Arial Black'),
                showlegend=False
            ))
            # 浪段背景色带
            for si in range(len(xs) - 1):
                fig.add_vrect(
                    x0=xs[si], x1=xs[si + 1],
                    fillcolor=wave_colors[si % 5], opacity=0.06,
                    line_width=0, layer='below'
                )

    fig.update_layout(
        title=dict(text='五浪结构分析', font=dict(size=16, color='#1e3a5f'), x=0.02),
        template='plotly_white',
        height=520,
        xaxis_rangeslider_visible=True,
        xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='价格', showgrid=True, gridcolor='#f1f5f9'),
        hovermode='x unified',
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='left', x=0)
    )

    visible_waves = len(wave_segments)
    info = (f"检测到 {len(waves)} 个五浪结构 "
            f"(显示最近180天, {len(sh_vis)}个高点/{len(sl_vis)}个低点)")
    if visible_waves > 0:
        last_dir = '↑上升' if wave_segments[-1]['direction'] == 'up' else '↓下降'
        info += f" | 当前: {last_dir}推动"
    return fig, info


def build_fibonacci_figure(data, dates, fib_levels=None, fib_start=None, fib_end=None):
    """构建斐波那契分析图 - K线 + RSI 独立副图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.68, 0.32],
                        vertical_spacing=0.03,
                        subplot_titles=('斐波那契回撤', 'RSI(14) 超买/超卖'))

    closes = data['close'].values

    # ── 上区: K线 + 斐波那契 ──
    fig.add_trace(go.Candlestick(
        x=dates, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ), row=1, col=1)

    # 使用预计算级别或重新计算
    if fib_levels is None or fib_start is None:
        highs = data['high'].values
        lows = data['low'].values
        n = len(closes)
        lookback = min(90, n)
        recent_high = np.max(highs[-lookback:])
        recent_low = np.min(lows[-lookback:])
        if recent_high > recent_low:
            fib_high_val, fib_low_val = recent_high, recent_low
        else:
            fib_high_val, fib_low_val = recent_low, recent_high
        retrace_levels = fibonacci_retracement(fib_high_val, fib_low_val)
    else:
        retrace_levels = fib_levels

    fib_high_val = retrace_levels[1.0]
    fib_low_val = retrace_levels[0.0]

    # 斐波那契黄金分割数列：0% 低点 → 100% 高点，中间为黄金比例分割线
    # 每条线都标注比例，61.8% 为黄金分割点突出显示
    fib_series = [
        (0.0,   '0%',    '#6b7280', 'solid', 1.2, 10),
        (0.236, '23.6%', '#94a3b8', 'dot',   0.8, 9),
        (0.382, '38.2%', '#94a3b8', 'dot',   0.8, 9),
        (0.5,   '50%',   '#eab308', 'dash',  1.1, 10),
        (0.618, '61.8%', '#f59e0b', 'dash',  2.0, 11),
        (0.786, '78.6%', '#94a3b8', 'dot',   0.8, 9),
        (1.0,   '100%',  '#8b5cf6', 'solid', 1.2, 10),
    ]
    for lvl, label, color, dash, width, fontsize in fib_series:
        price = retrace_levels[lvl]
        fig.add_hline(y=price, line_dash=dash, line_color=color, line_width=width,
                       row=1, col=1)
        fig.add_annotation(x=1.0, y=price, xref='x domain', yref='y', row=1, col=1,
                           text=f' {label} {price:.2f}',
                           showarrow=False, xanchor='right', yanchor='bottom',
                           font=dict(size=fontsize, color=color))

    # 当前价格标记
    last_close = closes[-1]
    fig.add_trace(go.Scatter(
        x=[dates[-1]], y=[last_close],
        mode='markers+text',
        text=[f' 现价{last_close:.2f}'],
        textposition='middle left',
        textfont=dict(size=11, color='#1e293b', family='Arial Black'),
        marker=dict(size=10, color='#1e293b', symbol='diamond', line=dict(color='white', width=2)),
        showlegend=False
    ), row=1, col=1)

    pct_in_range = (last_close - fib_low_val) / max(fib_high_val - fib_low_val, 0.001) * 100

    # ── 下区: RSI ──
    rsi_vals = rsi_calculation(data)
    # 超买/超卖区域
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,68,68,0.08)', line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor='rgba(34,197,94,0.08)', line_width=0, row=2, col=1)

    # RSI 分段着色
    rsi_normal = np.where((rsi_vals >= 30) & (rsi_vals <= 70), rsi_vals, np.nan)
    rsi_overbought = np.where(rsi_vals > 70, rsi_vals, np.nan)
    rsi_oversold = np.where(rsi_vals < 30, rsi_vals, np.nan)

    fig.add_trace(go.Scatter(x=dates, y=rsi_normal, mode='lines',
        line=dict(color='#8b5cf6', width=1.5), name='RSI(14)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rsi_overbought, mode='lines',
        line=dict(color='#ef4444', width=2), name='超买'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rsi_oversold, mode='lines',
        line=dict(color='#22c55e', width=2), name='超卖'), row=2, col=1)

    fig.add_hline(y=70, line_dash='dash', line_color='#ef4444', line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#22c55e', line_width=1, row=2, col=1)

    # 布局
    fig.update_layout(
        title=dict(text='斐波那契回撤 & RSI', font=dict(size=16, color='#1e3a5f'), x=0.02),
        template='plotly_white',
        height=560,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    fig.update_yaxes(title_text='价格', row=1, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0, 100])

    # 信息
    rsi_last = rsi_vals[-1] if len(rsi_vals) > 0 else 50
    if rsi_last > 70:
        rsi_status = f"RSI={rsi_last:.1f} ⚠️超买"
    elif rsi_last < 30:
        rsi_status = f"RSI={rsi_last:.1f} 💡超卖"
    else:
        rsi_status = f"RSI={rsi_last:.1f}"
    info = (f"斐波那契: 高 {fib_high_val:.2f} 低 {fib_low_val:.2f} "
            f"| 50%: {retrace_levels[0.5]:.2f} "
            f"| 当前: {last_close:.2f} ({pct_in_range:.0f}%) | "
            f"{rsi_status}")
    return fig, info


def build_channel_figure(data, channels, dates):
    """构建通道分析图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.05,
                        subplot_titles=('通道 + 布林带', 'ATR (波动率)'))

    # K线
    fig.add_trace(go.Candlestick(
        x=dates, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ), row=1, col=1)

    # 布林带
    bb = bollinger_bands(data)
    fig.add_trace(go.Scatter(x=dates, y=bb['upper'], mode='lines',
                              line=dict(color='#94a3b8', width=0.8), name='布林上轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb['middle'], mode='lines',
                              line=dict(color='#64748b', width=1, dash='dash'), name='布林中轨'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=bb['lower'], mode='lines',
                              line=dict(color='#94a3b8', width=0.8), name='布林下轨'), row=1, col=1)

    # 布林挤压标注
    if bb['squeeze_idx']:
        sq_dates = [dates[i] for i in bb['squeeze_idx'] if i < len(dates)]
        sq_prices = [data['close'].iloc[i] for i in bb['squeeze_idx'] if i < len(dates)]
        fig.add_trace(go.Scatter(x=sq_dates, y=sq_prices, mode='markers',
                                  marker=dict(symbol='circle', size=6, color='#f59e0b',
                                              line=dict(color='black', width=1)),
                                  name='布林挤压'), row=1, col=1)

    # 最近通道
    for ci, ch in enumerate(channels[-2:]):
        color = '#22c55e' if ch['slope'] > 0 else '#ef4444'
        x_vals = [dates[int(idx)] for idx in ch['x'] if int(idx) < len(dates)]
        # 上轨
        fig.add_trace(go.Scatter(x=x_vals, y=ch['upper_line'][:len(x_vals)],
                                  mode='lines', line=dict(color=color, width=1.5, dash='dot'),
                                  name=f"{ch['direction']}通道上轨" if ci == len(channels[-2:]) - 1 else None,
                                  showlegend=(ci == len(channels[-2:]) - 1)), row=1, col=1)
        # 中轨
        fig.add_trace(go.Scatter(x=x_vals, y=ch['mid_line'][:len(x_vals)],
                                  mode='lines', line=dict(color=color, width=2),
                                  name=f"{ch['direction']}通道中轨" if ci == len(channels[-2:]) - 1 else None,
                                  showlegend=(ci == len(channels[-2:]) - 1)), row=1, col=1)
        # 下轨
        fig.add_trace(go.Scatter(x=x_vals, y=ch['lower_line'][:len(x_vals)],
                                  mode='lines', line=dict(color=color, width=1.5, dash='dot'),
                                  name=f"{ch['direction']}通道下轨" if ci == len(channels[-2:]) - 1 else None,
                                  showlegend=(ci == len(channels[-2:]) - 1)), row=1, col=1)

    # ATR
    atr_vals = atr_indicator(data)
    fig.add_trace(go.Scatter(x=dates, y=atr_vals, mode='lines',
                              line=dict(color='#f97316', width=1.5),
                              name='ATR(14)'), row=2, col=1)

    fig.update_layout(title='通道分析 + 布林带', template='plotly_white',
                       height=550, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='ATR', row=2, col=1)

    last_channel = channels[-1] if channels else None
    info = f"检测到 {len(channels)} 个通道"
    if last_channel:
        info += f" | 最新通道: {last_channel['direction']}, 宽度 {last_channel['width']:.2f}"
    if atr_vals[-1] > 0 and not np.isnan(atr_vals[-1]):
        info += f" | ATR: {atr_vals[-1]:.2f}"
    return fig, info


def build_support_resistance_figure(data, supports, resistances, breakouts_up, breakouts_down, dates):
    """构建支撑阻力 + 突破分析图"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35],
                        vertical_spacing=0.05,
                        subplot_titles=('支撑/阻力 + 突破', '成交量分布'))

    # K线
    fig.add_trace(go.Candlestick(
        x=dates, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ), row=1, col=1)

    # 支撑线
    for si, s in enumerate(supports[:5]):
        fig.add_hline(y=s['level'], line_dash="dash", line_color="#22c55e",
                       opacity=0.6, row=1, col=1,
                       annotation_text=f"支撑{s['level']:.2f}({s['touches']}次)",
                       annotation_position="right")

    # 阻力线
    for ri, r in enumerate(resistances[:5]):
        fig.add_hline(y=r['level'], line_dash="dash", line_color="#ef4444",
                       opacity=0.6, row=1, col=1,
                       annotation_text=f"阻力{r['level']:.2f}({r['touches']}次)",
                       annotation_position="right")

    # 突破信号
    if breakouts_up:
        bu_dates = [dates[b['idx']] for b in breakouts_up if b['idx'] < len(dates)]
        bu_prices = [b['price'] for b in breakouts_up if b['idx'] < len(dates)]
        fig.add_trace(go.Scatter(x=bu_dates, y=bu_prices, mode='markers',
                                  marker=dict(symbol='triangle-up', size=12, color='#22c55e',
                                              line=dict(color='white', width=1)),
                                  name='向上突破'), row=1, col=1)
    if breakouts_down:
        bd_dates = [dates[b['idx']] for b in breakouts_down if b['idx'] < len(dates)]
        bd_prices = [b['price'] for b in breakouts_down if b['idx'] < len(dates)]
        fig.add_trace(go.Scatter(x=bd_dates, y=bd_prices, mode='markers',
                                  marker=dict(symbol='triangle-down', size=12, color='#ef4444',
                                              line=dict(color='white', width=1)),
                                  name='向下突破'), row=1, col=1)

    # 成交量分布
    bin_centers, vol_profile, vp_info = volume_profile_analysis(data)
    if bin_centers is not None:
        fig.add_trace(go.Bar(x=vol_profile, y=bin_centers, orientation='h',
                              marker_color='#8b5cf6', opacity=0.6,
                              name='成交分布'), row=2, col=1)
        if vp_info:
            fig.add_hline(y=vp_info['poc'], line_dash="dash", line_color="#f59e0b",
                           annotation_text=f"POC {vp_info['poc']:.2f}", row=2, col=1)
            fig.add_hline(y=vp_info['vwap'], line_dash="dot", line_color="#3b82f6",
                           annotation_text=f"VWAP {vp_info['vwap']:.2f}", row=2, col=1)

    fig.update_layout(title='支撑阻力 + 成交量分布', template='plotly_white',
                       height=550, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='价格', row=2, col=1)

    info = f"支撑 {len(supports)} 个, 阻力 {len(resistances)} 个 | "
    info += f"突破向上 {len(breakouts_up)} 次, 向下 {len(breakouts_down)} 次"
    if vp_info:
        info += f" | POC: {vp_info['poc']:.2f}, VWAP: {vp_info['vwap']:.2f}"
    return fig, info


def build_pattern_figure(data, patterns, macd_data, dates):
    """构建形态识别 + MACD图"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.03,
                        subplot_titles=('K线形态识别', 'MACD 背离', '均线带'))

    # K线
    fig.add_trace(go.Candlestick(
        x=dates, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ), row=1, col=1)

    # 形态标注
    pattern_colors = {
        '十字星': '#6b7280', '锤子线(看涨)': '#22c55e', '吊颈线(看跌)': '#ef4444',
        '倒锤子(看涨)': '#22c55e', '射击之星(看跌)': '#ef4444',
        '大阳线': '#22c55e', '大阴线': '#ef4444',
        '看涨吞没': '#22c55e', '看跌吞没': '#ef4444'
    }
    for p in patterns[-10:]:
        if p['idx'] < len(dates):
            fig.add_trace(go.Scatter(
                x=[dates[p['idx']]], y=[p['close']],
                mode='markers+text',
                text=[p['name'][:4]],
                textposition='top center',
                marker=dict(symbol='circle', size=10,
                            color=pattern_colors.get(p['name'], '#6b7280')),
                name=p['name'], showlegend=False
            ), row=1, col=1)

    # MACD
    fig.add_trace(go.Bar(x=dates, y=macd_data['histogram'],
                          marker_color=['#ef4444' if h >= 0 else '#22c55e'
                                         for h in macd_data['histogram']],
                          name='MACD柱'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=macd_data['macd'], mode='lines',
                              line=dict(color='#3b82f6', width=1.5), name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=macd_data['signal'], mode='lines',
                              line=dict(color='#f97316', width=1), name='信号线'), row=2, col=1)

    # 背离标注
    for d in macd_data['divergences']:
        if d['idx'] < len(dates):
            fig.add_trace(go.Scatter(x=[dates[d['idx']]], y=[d['price']],
                                      mode='markers', marker=dict(symbol='x', size=12),
                                      name=d['type'], showlegend=False), row=1, col=1)

    # 均线带
    ribbons = moving_average_ribbon(data, windows=[5, 10, 20, 60])
    ribbon_colors = ['#ef4444', '#f97316', '#eab308', '#22c55e']
    for ci, (w, rib) in enumerate(ribbons.items()):
        fig.add_trace(go.Scatter(x=dates, y=rib, mode='lines',
                                  line=dict(color=ribbon_colors[ci % len(ribbon_colors)],
                                            width=1.2),
                                  name=f'MA{w}'), row=3, col=1)

    # 均线排列判断
    last_ribbon = {w: rib[-1] if not np.isnan(rib[-1]) else 0 for w, rib in ribbons.items()}
    sorted_ma = sorted(last_ribbon.items(), key=lambda x: x[1], reverse=True)
    is_bullish_alignment = all(
        sorted_ma[i][0] < sorted_ma[i + 1][0] for i in range(len(sorted_ma) - 1))
    alignment = "多头排列" if is_bullish_alignment else "空头排列"

    fig.update_layout(title='K线形态 + MACD + 均线带', template='plotly_white',
                       height=600, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)
    fig.update_yaxes(title_text='价格', row=3, col=1)

    info = f"形态识别: {len(patterns)} 个信号"
    if patterns:
        info += f" | 最近: {patterns[-1]['name']}"
    if macd_data['divergences']:
        info += f" | {macd_data['divergences'][-1]['type']}"
    info += f" | 均线带: {alignment}"
    return fig, info


# 收敛形态参数
_APEX_MAX_AHEAD = 60     # 顶点最多向前看60根，超出视为不相关
_MIN_PATTERN_SPAN = 15   # 形态至少跨越15根K线
_FLAT_GAP_RATIO = 0.3    # 边线到顶点的移动 < 剩余间距的30% 视为水平边


def analyze_trendline_convergence(support, resistance, n_view):
    """检测支撑/阻力趋势线的收敛形态（三角形 / 楔形）。

    两线延长线在未来 _APEX_MAX_AHEAD 根内相交、当前间距仍未收拢为 0 时视为收敛。
    按两侧斜率分类：对称三角形 / 上升楔形 / 下降楔形 / 上升三角形 / 下降三角形。

    返回 None（不收敛）或 {'pattern','tendency','apex_idx','apex_price','bars_to_apex'}，
    apex_idx/apex_price 为视图坐标（与 auto_trendlines 返回的线索引同基准）。
    """
    if not support or not resistance or not n_view:
        return None
    s_slope, s_int = support['slope'], support['intercept']
    r_slope, r_int = resistance['slope'], resistance['intercept']
    if s_slope <= r_slope:
        return None  # 间距没有收窄（发散或平行）
    gap_last = (r_slope * (n_view - 1) + r_int) - (s_slope * (n_view - 1) + s_int)
    if gap_last <= 0:
        return None  # 已经交叉
    apex_idx = (r_int - s_int) / (s_slope - r_slope)
    bars_to_apex = apex_idx - (n_view - 1)
    if bars_to_apex <= 0 or bars_to_apex > _APEX_MAX_AHEAD:
        return None
    span = n_view - 1 - max(support['start_idx'], resistance['start_idx'])
    if span < _MIN_PATTERN_SPAN:
        return None
    apex_price = s_slope * apex_idx + s_int
    if apex_price <= 0:
        return None
    flat_s = abs(s_slope) * bars_to_apex <= _FLAT_GAP_RATIO * gap_last
    flat_r = abs(r_slope) * bars_to_apex <= _FLAT_GAP_RATIO * gap_last
    if flat_s and flat_r:
        return None
    if flat_r:
        pattern, tendency = '上升三角形', '通常偏向向上突破'
    elif flat_s:
        pattern, tendency = '下降三角形', '通常偏向向下突破'
    elif s_slope > 0 and r_slope < 0:
        pattern, tendency = '对称三角形', '方向待突破确认'
    elif s_slope > 0:
        pattern, tendency = '上升楔形', '通常偏向向下突破'
    else:
        pattern, tendency = '下降楔形', '通常偏向向上突破'
    return {'pattern': pattern, 'tendency': tendency,
            'apex_idx': float(apex_idx), 'apex_price': float(apex_price),
            'bars_to_apex': float(bars_to_apex)}


def _future_date_strs(dates, k, freq='daily'):
    """推算未来k根K线的日期字符串：日线按工作日推进，其余按历史中位K线间隔"""
    if k <= 0:
        return []
    try:
        if freq == 'daily':
            fut = pd.date_range(dates[-1], periods=k + 1, freq='B')[1:]
            return [d.strftime('%Y-%m-%d') for d in fut]
        ds = pd.to_datetime(pd.Series(dates[-30:]))
        diffs = ds.diff().dropna().dt.total_seconds()
        step = float(np.median(diffs)) if len(diffs) else 86400.0
        if step <= 0:
            step = 86400.0
        last = ds.iloc[-1]
        return [(last + pd.Timedelta(seconds=step * (i + 1))).strftime('%Y-%m-%d')
                for i in range(k)]
    except (ValueError, TypeError):
        return []


def build_trendline_figure(data, tl_results, levels_result=None, freq='daily',
                           show_lines=True, show_levels=True, extend_bars=10):
    """构建趋势线 & 阻力线分析图: K线 + 各回溯期支撑/阻力趋势线 + 水平支撑阻力位 + 成交量

    tl_results: [(标签, auto_trendlines 结果)]，levels_result: 水平位列表 [{'price','touches','type'}]
    收敛形态（三角形/楔形）时两线延伸至顶点并填充收敛区域。
    """
    dates = data['trade_date'].astype(str).tolist()
    n = len(data)
    allow_future = freq not in ('60min', '30min', '15min', '5min')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28],
                        vertical_spacing=0.04,
                        subplot_titles=('趋势线（支撑/阻力连线）+ 水平支撑/阻力位', '成交量'))

    fig.add_trace(go.Candlestick(
        x=dates, open=data['open'], high=data['high'], low=data['low'], close=data['close'],
        name='K线', increasing_line_color='#ef4444', decreasing_line_color='#22c55e'
    ), row=1, col=1)

    # 各回溯期趋势线样式: 标签 → (支撑色, 阻力色, 线型)
    style_map = {'短期': ('#10B981', '#EF4444', 'dash'),
                 '中期': ('#047857', '#B91C1C', 'dashdot')}

    if show_lines:
        for tag, result in tl_results:
            color_sup, color_res, dash = style_map.get(tag, ('#10B981', '#EF4444', 'dash'))
            start = result.get('start', 0)
            n_view = result.get('n_view', 0)
            if n_view <= 0:
                continue
            g_end = min(start + n_view - 1, n - 1)
            conv = analyze_trendline_convergence(result.get('support'),
                                                 result.get('resistance'), n_view)
            # 收敛时两线延伸到顶点（最多再延伸 _APEX_MAX_AHEAD 根），否则按默认延伸
            n_future = extend_bars if allow_future else 0
            if conv and allow_future:
                n_future = min(int(round(conv['apex_idx'])) - (n_view - 1), _APEX_MAX_AHEAD)
                n_future = max(n_future, 0)
            fut = _future_date_strs(dates, n_future, freq)

            def _x(t):
                """视图坐标t → 日期字符串（含未来延伸段）"""
                g = start + t
                return dates[g] if g < n else fut[g - n]

            for key, is_sup in (('support', True), ('resistance', False)):
                line = result.get(key)
                if not line:
                    continue
                t0 = line['start_idx']
                g_start = start + t0
                if g_start >= n or g_start >= g_end:
                    continue
                color = color_sup if is_sup else color_res
                label = f"{tag}{'支撑' if is_sup else '阻力'}趋势线"
                # 直线只需两端点：起点锚 → 末根(+延伸)
                end_t = n_view - 1 + n_future if allow_future else n_view - 1
                x_pts = [_x(t0), _x(end_t)]
                y_pts = [line['slope'] * t0 + line['intercept'],
                         line['slope'] * end_t + line['intercept']]
                fig.add_trace(go.Scatter(x=x_pts, y=y_pts, mode='lines', name=label,
                                         line=dict(color=color, width=1.8, dash=dash),
                                         hovertemplate=f'{label}: %{{y:.2f}}<extra></extra>'),
                              row=1, col=1)
                # 当前值与状态标注放在最后一根历史K线处（收敛顶点处两线标注重叠）
                cur_y = line['slope'] * (n_view - 1) + line['intercept']
                fig.add_annotation(x=dates[g_end], y=cur_y, row=1, col=1,
                                   text=f"{tag}{'支撑' if is_sup else '阻力'} {cur_y:.2f} · {line.get('status', '')}",
                                   showarrow=False, xanchor='right',
                                   yanchor='top' if is_sup else 'bottom',
                                   font=dict(size=10, color=color))

            # 收敛形态：填充收敛三角区域并标注顶点
            if conv and allow_future:
                s_line, r_line = result.get('support'), result.get('resistance')
                if s_line and r_line:
                    t0c = max(s_line['start_idx'], r_line['start_idx'])
                    t_apex = min(int(round(conv['apex_idx'])), n_view - 1 + n_future)
                    fig.add_trace(go.Scatter(
                        x=[_x(t0c), _x(t_apex), _x(t0c)],
                        y=[s_line['slope'] * t0c + s_line['intercept'],
                           conv['apex_price'],
                           r_line['slope'] * t0c + r_line['intercept']],
                        mode='lines', line=dict(width=0.5, color='rgba(245,158,11,0.5)'),
                        fill='toself', fillcolor='rgba(245,158,11,0.10)',
                        name=f"{tag}·{conv['pattern']}收敛", hoverinfo='skip',
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=[_x(t_apex)], y=[conv['apex_price']],
                        mode='markers+text', text=[f"顶点 {_x(t_apex)[:10]}"],
                        textposition='top center',
                        textfont=dict(size=10, color='#b45309'),
                        marker=dict(symbol='cross-thin', size=10, color='#f59e0b',
                                    line=dict(width=1.5)),
                        showlegend=False,
                        hovertemplate='收敛顶点: %{y:.2f}<extra></extra>',
                    ), row=1, col=1)

    if show_levels and levels_result:
        for lv in levels_result:
            is_sup = lv['type'] == 'support'
            color = '#10B981' if is_sup else '#EF4444'
            fig.add_hline(y=lv['price'], line_dash='dot', line_color=color, line_width=1,
                          annotation_text=f"{'水平支撑' if is_sup else '水平阻力'} {lv['price']:.2f} · {lv['touches']}次触及",
                          annotation_position='right',
                          annotation_font=dict(size=10, color=color), row=1, col=1)

    # 成交量
    vol_colors = ['#ef4444' if c >= o else '#22c55e'
                  for c, o in zip(data['close'].values, data['open'].values)]
    fig.add_trace(go.Bar(x=dates, y=data['vol'], marker_color=vol_colors,
                         name='成交量', showlegend=False), row=2, col=1)

    fig.update_layout(
        title=dict(text='自动趋势线 & 阻力线', font=dict(size=16, color='#1e3a5f'), x=0.02),
        template='plotly_white',
        height=520,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )
    fig.update_yaxes(title_text='价格', row=1, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text='成交量', row=2, col=1)

    parts = []
    for tag, result in tl_results:
        sup, res = result.get('support'), result.get('resistance')
        if sup:
            parts.append(f"{tag}支撑 {sup['current_y']:.2f}({sup['status']})")
        if res:
            parts.append(f"{tag}阻力 {res['current_y']:.2f}({res['status']})")
    info = '趋势线: ' + (' ｜ '.join(parts) if parts else '当前未形成有效趋势线')
    for tag, result in tl_results:
        c = analyze_trendline_convergence(result.get('support'),
                                          result.get('resistance'),
                                          result.get('n_view', 0))
        if c:
            apex_g = result.get('start', 0) + int(round(c['apex_idx']))
            apex_date = '?'
            if apex_g < n:
                apex_date = str(dates[apex_g])[:10]
            else:
                fut_i = _future_date_strs(dates, max(int(round(c['bars_to_apex'])), 1), freq)
                if fut_i:
                    apex_date = fut_i[-1]
            info += f" ｜ {tag}·{c['pattern']}收敛 顶点约{apex_date}({int(round(c['bars_to_apex']))}根后)"
    if levels_result:
        info += f" ｜ 水平位 {len(levels_result)} 个"
    if tl_results and tl_results[-1][1].get('note'):
        info += f" ｜ {tl_results[-1][1]['note']}"
    return fig, info


# ===================== 页面布局 =====================

def trend_analysis_layout():
    """趋势分析页面布局 - 类似分析页面的两栏布局"""
    return html.Div([
        html.Main([
            # ===== 左侧边栏 =====
            html.Aside([
                html.Div([
                    html.Div([
                        html.H2([
                            html.I(className='fas fa-chart-line mr-2'),
                            "趋势分析工具"
                        ], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),

                    # 股票选择
                    html.Label("选择股票:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Dropdown(
                        id='ta-stock-dropdown',
                        options=[],
                        value=None,
                        placeholder="搜索或选择股票...",
                        className="mb-3"
                    ),
                    html.Label("或输入代码:", className="block text-sm font-medium text-gray-700 mb-1"),
                    html.Div([
                        dcc.Input(
                            id='ta-stock-input', type='text',
                            placeholder='例如: 600519.SH',
                            className="w-full py-2 px-3 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm",
                            style={'flex': '1'}
                        ),
                        html.Button("查询", id='ta-load-btn',
                                    className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-r-md text-sm font-medium transition-all"),
                    ], className="flex mb-3"),

                    # 数据源选择
                    html.Label("数据源:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Dropdown(
                        id='ta-data-source',
                        options=[{'label': s.capitalize(), 'value': s} for s in get_available_sources()],
                        value='tushare',
                        clearable=False,
                        className="mb-3"
                    ),

                    # K线周期选择
                    html.Label("K线周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Dropdown(
                        id='ta-kline-freq',
                        options=[
                            {'label': '日线', 'value': 'daily'},
                            {'label': '周线', 'value': 'weekly'},
                            {'label': '月线', 'value': 'monthly'},
                            {'label': '60分钟', 'value': '60min'},
                            {'label': '30分钟', 'value': '30min'},
                            {'label': '15分钟', 'value': '15min'},
                        ],
                        value='daily',
                        clearable=False,
                        className="mb-3"
                    ),

                    # 日期选择
                    html.Label("开始日期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.DatePickerSingle(
                        id='ta-start-date',
                        date=pd.Timestamp.now() - pd.Timedelta(days=730),
                        className="w-full mb-2"
                    ),
                    html.Label("结束日期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.DatePickerSingle(
                        id='ta-end-date',
                        date=pd.Timestamp.now(),
                        className="w-full mb-3"
                    ),

                    # 分析模块选择
                    html.Div([
                        html.H3([
                            html.I(className='fas fa-sliders-h mr-2'),
                            "分析模块"
                        ], className="text-sm font-semibold text-gray-600 mb-2 pt-2 border-t border-gray-200"),

                        dcc.Checklist(
                            id='ta-modules-checklist',
                            options=[
                                {'label': '  五浪结构分析', 'value': 'elliott'},
                                {'label': '  斐波那契回撤 & RSI', 'value': 'fibonacci'},
                                {'label': '  通道 & 布林带 & ATR', 'value': 'channel'},
                                {'label': '  支撑阻力 & 突破 & 成交量分布', 'value': 'sr'},
                                {'label': '  趋势线 & 阻力线（自动连线）', 'value': 'trendlines'},
                                {'label': '  K线形态 & MACD & 均线带', 'value': 'pattern'},
                            ],
                            value=['elliott', 'fibonacci', 'channel', 'sr', 'trendlines', 'pattern'],
                            className="space-y-2"
                        ),
                    ], className="mb-3"),

                    # 趋势线显示选项
                    html.Div([
                        html.Label("趋势线显示:", className="block text-sm font-semibold text-gray-600 mb-1"),
                        dcc.Checklist(
                            id='ta-tl-options',
                            options=[
                                {'label': '  趋势连线（支撑/阻力）', 'value': 'lines'},
                                {'label': '  水平支撑/阻力位', 'value': 'levels'},
                            ],
                            value=['lines', 'levels'],
                            className="space-y-1"
                        ),
                        html.Div("自动连接摆动高低点生成趋势线并标注支撑/突破状态，检测三角形/楔形收敛形态",
                                 className="text-[10px] text-gray-400 mt-1"),
                    ], className="mb-3"),

                    # 参数设置
                    html.Div([
                        html.H3([
                            html.I(className='fas fa-cog mr-2'),
                            "参数设置"
                        ], className="text-sm font-semibold text-gray-600 mb-2 pt-2 border-t border-gray-200"),

                        html.Label("摆动窗口:", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-swing-window', type='number', value=5, min=2, max=30,
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),

                        html.Label("斐波那契周期(d):", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-fib-period', type='number', value=90, min=20, max=365,
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),

                        html.Label("通道回溯期:", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-channel-lookback', type='number', value=50, min=10, max=200,
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),

                        html.Label("支撑阻力容差(%):", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-sr-tolerance', type='number', value=2.0, min=0.5, max=10.0, step=0.5,
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),

                        html.Label("趋势线回溯期:", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-tl-lookback', type='number', value=120, min=40, max=500,
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),

                        html.Label("均线窗口(逗号分隔):", className="text-xs text-gray-500"),
                        dcc.Input(id='ta-ma-windows', type='text', value='5,10,20,60',
                                  className="w-full mb-2 py-1 px-2 border border-gray-300 rounded text-sm"),
                    ], className="mb-3"),

                    # 分析按钮
                    html.Button([
                        html.I(className='fas fa-play mr-2'),
                        "执行趋势分析"
                    ], id='ta-run-btn', className="w-full bg-green-600 hover:bg-green-700 text-white py-2.5 rounded-lg font-medium transition-all"),

                    # 状态提示
                    html.Div(id='ta-status', className="mt-3 text-sm text-gray-600"),

                ], className="space-y-1")
            ], className="w-full md:w-1/4 lg:w-1/5 bg-white rounded-xl shadow-lg border border-gray-100 p-4 overflow-y-auto",
                style={'maxHeight': 'calc(100vh - 100px)'}),

            # ===== 右侧图表区 =====
            html.Div([
                # 加载状态和错误
                html.Div(id='ta-error', className="text-red-500 text-sm mb-2"),
                html.Div(id='ta-summary', className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3 text-sm text-blue-800"),

                # 图表网格
                html.Div(id='ta-charts-container', className="space-y-4"),

                # 图表存储（供全屏弹窗使用）
                dcc.Store(id='ta-modal-figure-key', data=''),
                dcc.Store(id='ta-modal-title-store', data=''),
                # 斐波那契交互数据存储（供分割线拖动回调使用）
                dcc.Store(id='ta-fib-data-store', data=None),

            ], className="w-full md:w-3/4 lg:w-4/5 pl-4")
        ], className="flex flex-col md:flex-row"),

        # ===== 全屏弹窗 =====
        html.Div([
            html.Div([
                html.Div([
                    html.Span(id='ta-modal-title', className="text-lg font-bold text-gray-800"),
                    html.Button("✕ 关闭", id='ta-modal-close-btn',
                                className="bg-gray-200 hover:bg-gray-300 text-gray-700 px-4 py-1.5 rounded-lg text-sm font-medium transition-all ml-auto"),
                ], className="flex items-center justify-between pb-3 border-b border-gray-200"),
                dcc.Graph(id='ta-modal-graph', config={'displayModeBar': True, 'scrollZoom': True},
                          style={'height': '85vh'}),
            ], className="bg-white rounded-xl shadow-2xl p-5 w-full max-w-[95vw] max-h-[95vh] overflow-auto"),
        ], id='ta-fullscreen-modal', className="fixed inset-0 z-50 flex items-center justify-center bg-black/60",
            style={'display': 'none'}),
    ], className="min-h-screen bg-gray-50 p-4")


# ===================== 回调注册 =====================

def register_trend_callbacks(app):
    """注册趋势分析页面回调到 Dash app"""

    # 股票下拉菜单选项加载 (页面初始化)
    @app.callback(
        Output('ta-stock-dropdown', 'options'),
        Input('ta-stock-dropdown', 'search_value')
    )
    def load_ta_stock_options(search_value):
        from .database_manager import get_all_stock_codes
        try:
            codes = get_all_stock_codes()
            options = [{'label': f"{c[0]} {c[1] if len(c) > 1 and c[1] else ''}".strip(),
                        'value': c[0]} for c in codes]
            return options
        except Exception as e:
            logging.warning(f"加载股票选项失败: {e}")
            return []

    # 主分析回调
    @app.callback(
        [Output('ta-error', 'children'),
         Output('ta-status', 'children'),
         Output('ta-summary', 'children'),
         Output('ta-charts-container', 'children'),
         Output('ta-fib-data-store', 'data')],
        [Input('ta-run-btn', 'n_clicks')],
        [State('ta-stock-dropdown', 'value'),
         State('ta-stock-input', 'value'),
         State('ta-data-source', 'value'),
         State('ta-kline-freq', 'value'),
         State('ta-start-date', 'date'),
         State('ta-end-date', 'date'),
         State('ta-modules-checklist', 'value'),
         State('ta-swing-window', 'value'),
         State('ta-fib-period', 'value'),
         State('ta-channel-lookback', 'value'),
         State('ta-sr-tolerance', 'value'),
         State('ta-ma-windows', 'value'),
         State('ta-tl-lookback', 'value'),
         State('ta-tl-options', 'value')]
    )
    def run_trend_analysis(n_clicks, dropdown_code, input_code, data_source, kline_freq,
                            start_date, end_date,
                            modules, swing_window, fib_period, channel_lookback,
                            sr_tolerance, ma_windows_str, tl_lookback, tl_options):
        if not n_clicks:
            return '', '', '', [], dash.no_update

        code = dropdown_code or (input_code.strip() if input_code else '')
        if not code:
            return '请选择或输入股票代码', '', '', [], dash.no_update

        # 代码标准化: 纯数字 → 9位 .SH/.SZ 格式
        # 上交所: 5xxxxx(ETF/Lof), 6xxxxx(主板), 9xxxxx(债券)
        # 深交所: 0xxxxx(主板), 2xxxxx(中小), 3xxxxx(创业板)
        if code.isdigit():
            if code.startswith(('5', '6', '9')):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"
        elif '.' not in code and len(code) == 6:
            if code.startswith(('5', '6', '9')):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"

        # 切换数据源
        if data_source:
            set_data_source(data_source)

        # 参数解析
        try:
            swing_window = int(swing_window) if swing_window else 5
            fib_period = int(fib_period) if fib_period else 90
            channel_lookback = int(channel_lookback) if channel_lookback else 50
            sr_tolerance = float(sr_tolerance) / 100.0 if sr_tolerance else 0.02
            ma_windows = [int(w.strip()) for w in ma_windows_str.split(',')] if ma_windows_str else [5, 10, 20, 60]
        except (ValueError, TypeError) as e:
            return f"参数错误: {str(e)}", '', '', [], dash.no_update

        if not modules:
            return '请至少选择一个分析模块', '', '', [], dash.no_update

        # 日期格式转换: Dash DatePickerSingle → YYYYMMDD
        try:
            start_date = pd.Timestamp(start_date).strftime('%Y%m%d') if start_date else '20240101'
            end_date = pd.Timestamp(end_date).strftime('%Y%m%d') if end_date else '20260607'
        except Exception:
            start_date = '20240101'
            end_date = '20260607'

        # 获取数据 (kline_freq: daily/weekly/monthly/60min/30min/15min)
        try:
            data = get_and_process_data(code, start_date, end_date, freq=kline_freq or 'daily')
        except Exception as e:
            return f"数据获取失败: {str(e)}", '', '', [], dash.no_update

        if data is None or len(data) < 20:
            tips = {
                'tushare': '请检查: 1) 代码是否正确(如600519.SH) 2) Tushare积分是否充足 3) 接口频率是否超限',
                'baostock': '请检查: 1) 代码是否正确(如sh.600519) 2) Baostock是否正常登录',
            }
            tip = tips.get(data_source, '两个数据源均返回空数据，请检查股票代码是否存在或网络是否正常')
            freq_note = ' (分钟线历史数据仅从2020年起)' if (kline_freq or 'daily').endswith('min') else ''
            return f"数据不足 (需≥20根K线): {code} — {tip}{freq_note}", '', '', [], dash.no_update

        data = data.sort_values('trade_date').reset_index(drop=True)
        dates = data['trade_date'].astype(str).tolist()
        n_days = len(data)

        # 计算共用数据
        swings_needed = 'elliott' in modules or 'pattern' in modules
        fib_needed = 'fibonacci' in modules
        channel_needed = 'channel' in modules
        sr_needed = 'sr' in modules
        pattern_needed = 'pattern' in modules

        swing_highs, swing_lows, waves = None, None, None
        wave_segments = []
        fib_levels = None
        fib_start = fib_end = None
        channels = None
        bb = None
        supports, resistances = None, None
        breakouts_up, breakouts_down = None, None
        patterns = None
        macd_data = None
        volume_info = None

        # 预计算共用分析
        if swings_needed:
            waves, swing_highs, swing_lows = elliott_wave_analysis(data, swing_window)

        # 斐波那契级别（供信号生成复用）
        if 'fibonacci' in modules:
            fib_levels, fib_start, fib_end = _compute_fib_levels(data)

        if channel_needed:
            channels = channel_analysis(data, channel_lookback)
            bb = bollinger_bands(data)  # 供信号生成复用

        if sr_needed:
            # 数据不足时自动降低最少触及次数 (ETF 通常只有 180 天)
            sr_min_touches = 1 if n_days < 250 else 2
            supports, resistances = support_resistance_analysis(data, min_touches=sr_min_touches, tolerance=sr_tolerance)
            breakouts_up, breakouts_down = trendline_breakout(data, 20)

        if pattern_needed:
            patterns = candlestick_patterns(data)
            macd_data = macd_divergence(data)

        # 趋势线 & 阻力线（复用自动趋势线模块：短/中期回溯各生成一组连线）
        tl_results = []
        tl_levels = []
        if 'trendlines' in modules:
            try:
                from .trendlines import auto_trendlines
                tl_lookback_val = max(40, int(tl_lookback)) if tl_lookback else 120
                if len(data) >= 40:
                    short_lb = min(60, len(data))
                    long_lb = min(tl_lookback_val, len(data))
                    tl_results.append(('短期', auto_trendlines(data, pivot_window=swing_window, lookback=short_lb)))
                    if long_lb > short_lb:
                        tl_results.append(('中期', auto_trendlines(data, pivot_window=swing_window, lookback=long_lb)))
                    # 水平位取回溯期最长的一组（摆动点更丰富）
                    tl_levels = tl_results[-1][1].get('levels', [])
            except Exception as tl_err:
                logging.warning(f"趋势线生成失败 {code}: {tl_err}")

        # 筛选可见波浪段（供信号生成复用）
        if waves:
            slice_start = max(0, len(data) - 180)
            for w in waves:
                if 0 <= w['wave_1_idx'] - slice_start < len(data) and 0 <= w['wave_5_idx'] - slice_start < len(data):
                    wave_segments.append(w)

        # 生成统一交易信号
        all_signals = generate_trading_signals(data, supports or [], resistances or [],
                                                fib_levels or {}, channels or [],
                                                bb or {}, patterns or [], macd_data or {},
                                                wave_segments)

        # 合并趋势线模块信号（趋势线状态 + 水平位）
        if tl_results:
            all_signals = _dedup_signals(all_signals + generate_trendline_signals(data, tl_results, tl_levels))

        # 生成图表
        charts = []
        _freq_labels = {'daily': '日线', 'weekly': '周线', 'monthly': '月线',
                        '60min': '60分钟', '30min': '30分钟', '15min': '15分钟', '5min': '5分钟'}
        freq_label = _freq_labels.get(kline_freq or 'daily', kline_freq)
        summary_parts = [f"股票: {code}, 周期: {freq_label}, K线数: {n_days}"]

        def _chart_card(fig, module_key, title_hint, info, module_filter=None):
            """生成带放大按钮和交易建议的图表卡片"""
            _figure_cache[module_key] = fig
            return html.Div([
                html.Div([
                    html.Span(title_hint, className="text-xs font-semibold text-gray-500 uppercase tracking-wide"),
                    html.Button("🔍 放大", id={'type': 'ta-zoom-btn', 'index': module_key},
                                className="text-xs text-blue-600 hover:text-blue-800 hover:underline bg-transparent border-0 cursor-pointer ml-auto"),
                ], className="flex items-center justify-between mb-1"),
                dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False},
                          style={'height': '520px'}),
                html.Div([
                    html.Div("💡 交易建议", className="text-xs font-bold text-gray-500 mb-1.5 border-b border-gray-100 pb-1"),
                    build_signal_panel(all_signals, module_filter=module_filter),
                ], className="mt-2 pt-1"),
            ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-3")

        for module in modules:
            try:
                if module == 'elliott':
                    fig, info = build_elliott_wave_figure(data, waves, swing_highs, swing_lows, dates)
                    charts.append(_chart_card(fig, f"{code}_elliott", "五浪结构分析", info, module_filter='elliott'))
                    summary_parts.append(f"【五浪】{info}")

                elif module == 'fibonacci':
                    # _compute_fib_levels 基于 np.min/np.max，值是 np.float64；
                    # orjson 拒绝 np.float64 作为 dict 键（RangeSlider marks），统一转 float
                    fib_high_auto = float(fib_levels[1.0]) if fib_levels else float(data['high'].max())
                    fib_low_auto = float(fib_levels[0.0]) if fib_levels else float(data['low'].min())
                    price_min = float(data['low'].min())
                    price_max = float(data['high'].max())
                    price_range = max(price_max - price_min, 0.01)

                    fig, info = build_fibonacci_figure(data, dates, fib_levels=fib_levels, fib_start=fib_start, fib_end=fib_end)
                    _figure_cache[f"{code}_fibonacci"] = fig

                    # 斐波那契交互卡片（支持拖动分割线 + 0.618 黄金分割）
                    charts.append(html.Div([
                        html.Div([
                            html.Span("斐波那契回撤 & RSI", className="text-xs font-semibold text-gray-500 uppercase tracking-wide"),
                            html.Button("🔍 放大", id={'type': 'ta-zoom-btn', 'index': f"{code}_fibonacci"},
                                        className="text-xs text-blue-600 hover:text-blue-800 hover:underline bg-transparent border-0 cursor-pointer ml-auto"),
                        ], className="flex items-center justify-between mb-1"),
                        dcc.Graph(id={'type': 'ta-fib-graph', 'index': code}, figure=fig,
                                  config={'displayModeBar': True, 'displaylogo': False},
                                  style={'height': '520px'}),
                        html.Div([
                            html.Div([
                                html.Span("🎯 拖动两端移动分割线（低点/高点）", className="text-xs font-semibold text-amber-700"),
                                dcc.RangeSlider(
                                    id={'type': 'ta-fib-range', 'index': code},
                                    min=round(price_min - price_range * 0.05, 2),
                                    max=round(price_max + price_range * 0.05, 2),
                                    value=[round(fib_low_auto, 2), round(fib_high_auto, 2)],
                                    step=round(price_range / 500.0, 3),
                                    marks={
                                        round(fib_low_auto, 2): {'label': f'低{round(fib_low_auto, 2)}', 'style': {'color': '#16a34a'}},
                                        round(fib_high_auto, 2): {'label': f'高{round(fib_high_auto, 2)}', 'style': {'color': '#dc2626'}},
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ], className="px-2"),
                            html.Div([
                                html.Span("低点(0%) → 高点(100%)，61.8% 为黄金分割", className="text-xs text-gray-500"),
                                html.Button("↺ 重置", id={'type': 'ta-fib-reset', 'index': code},
                                            className="text-xs text-blue-600 hover:text-blue-800 hover:underline bg-transparent border-0 cursor-pointer ml-auto"),
                            ], className="flex items-center justify-between mt-1 px-2"),
                        ], className="mt-2 bg-amber-50 border border-amber-200 rounded-lg p-2"),
                        html.Div([
                            html.Div("💡 交易建议", className="text-xs font-bold text-gray-500 mb-1.5 border-b border-gray-100 pb-1"),
                            build_signal_panel(all_signals, module_filter='fibonacci'),
                        ], className="mt-2 pt-1"),
                    ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-3"))
                    summary_parts.append(f"【斐波那契】{info}")

                elif module == 'channel':
                    fig, info = build_channel_figure(data, channels, dates)
                    charts.append(_chart_card(fig, f"{code}_channel", "通道 & 布林带 & ATR", info, module_filter='channel'))
                    summary_parts.append(f"【通道】{info}")

                elif module == 'sr':
                    fig, info = build_support_resistance_figure(
                        data, supports, resistances, breakouts_up, breakouts_down, dates)
                    charts.append(_chart_card(fig, f"{code}_sr", "支撑阻力 & 成交量分布", info, module_filter='sr'))
                    summary_parts.append(f"【支撑阻力】{info}")

                elif module == 'trendlines':
                    _tl_opts = set(tl_options or [])
                    fig, info = build_trendline_figure(
                        data, tl_results, levels_result=tl_levels,
                        freq=kline_freq or 'daily',
                        show_lines='lines' in _tl_opts,
                        show_levels='levels' in _tl_opts)
                    charts.append(_chart_card(fig, f"{code}_trendlines", "趋势线 & 阻力线", info, module_filter='trendlines'))
                    summary_parts.append(f"【趋势线】{info}")

                elif module == 'pattern':
                    fig, info = build_pattern_figure(data, patterns, macd_data, dates)
                    charts.append(_chart_card(fig, f"{code}_pattern", "K线形态 & MACD & 均线带", info, module_filter='pattern'))
                    summary_parts.append(f"【形态】{info}")

            except Exception as e:
                logging.error(f"模块 {module} 出错: {e}", exc_info=True)
                charts.append(html.Div(f"❌ {module} 分析失败: {str(e)}",
                                        className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-600"))

        status = f"✓ 分析完成，共 {len(charts)} 个图表"
        summary = ' | '.join(summary_parts)

        # 构建斐波那契交互数据（供分割线拖动回调使用）
        fib_data = None
        if 'fibonacci' in modules and fib_levels:
            fib_data = {
                'dates': dates,
                'open': data['open'].tolist(),
                'high': data['high'].tolist(),
                'low': data['low'].tolist(),
                'close': data['close'].tolist(),
                'auto_high': float(fib_levels[1.0]),
                'auto_low': float(fib_levels[0.0]),
                'price_min': float(data['low'].min()),
                'price_max': float(data['high'].max()),
            }
        return '', status, summary, charts if charts else [html.Div("无图表", className="text-gray-400 text-center py-8")], fib_data

    # ===== 斐波那契分割线拖动回调 =====
    @app.callback(
        [Output({'type': 'ta-fib-graph', 'index': dash.MATCH}, 'figure'),
         Output({'type': 'ta-fib-range', 'index': dash.MATCH}, 'value')],
        [Input({'type': 'ta-fib-range', 'index': dash.MATCH}, 'value'),
         Input({'type': 'ta-fib-reset', 'index': dash.MATCH}, 'n_clicks')],
        [State('ta-fib-data-store', 'data')],
        prevent_initial_call=True
    )
    def update_fib_figure(range_val, reset_clicks, fib_data):
        """用户拖动分割线（高低点）时实时重算斐波那契回撤位"""
        if not fib_data:
            return dash.no_update, dash.no_update

        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

        # 重建数据
        data = pd.DataFrame({
            'open': fib_data['open'],
            'high': fib_data['high'],
            'low': fib_data['low'],
            'close': fib_data['close'],
        })
        dates = fib_data['dates']

        if 'ta-fib-reset' in triggered_id:
            fib_high = fib_data['auto_high']
            fib_low = fib_data['auto_low']
            new_range = [fib_low, fib_high]
        else:
            if not range_val or len(range_val) != 2:
                return dash.no_update, dash.no_update
            fib_low, fib_high = range_val
            new_range = dash.no_update

        fib_levels = fibonacci_retracement(fib_high, fib_low)
        fig, _ = build_fibonacci_figure(data, dates, fib_levels=fib_levels, fib_start=0, fib_end=0)
        return fig, new_range

    # ===== 全屏放大回调 =====
    @app.callback(
        [Output('ta-modal-figure-key', 'data', allow_duplicate=True),
         Output('ta-modal-title-store', 'data', allow_duplicate=True)],
        [Input({'type': 'ta-zoom-btn', 'index': dash.ALL}, 'n_clicks')],
        [State({'type': 'ta-zoom-btn', 'index': dash.ALL}, 'id')],
        prevent_initial_call=True
    )
    def handle_zoom_click(n_clicks_list, btn_ids):
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        # 找到被点击的按钮
        triggered_id = ctx.triggered[0]['prop_id']
        for i, (nc, bid) in enumerate(zip(n_clicks_list, btn_ids)):
            if nc and f'"index":"{bid["index"]}"' in triggered_id:
                title_map = {
                    'elliott': '五浪结构分析', 'fibonacci': '斐波那契回撤 & RSI',
                    'channel': '通道 & 布林带 & ATR', 'sr': '支撑阻力 & 成交量分布',
                    'trendlines': '趋势线 & 阻力线',
                    'pattern': 'K线形态 & MACD & 均线带'
                }
                key = bid['index']
                module_part = key.split('_')[-1] if '_' in key else key
                title = title_map.get(module_part, key)
                return key, title
        return dash.no_update, dash.no_update

    @app.callback(
        [Output('ta-fullscreen-modal', 'style'),
         Output('ta-modal-graph', 'figure')],
        [Input('ta-modal-figure-key', 'data'),
         Input('ta-modal-close-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def toggle_modal(figure_key, close_clicks):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

        # 关闭按钮
        if triggered_id == 'ta-modal-close-btn':
            return {'display': 'none'}, dash.no_update

        # 显示弹窗
        if figure_key and figure_key in _figure_cache:
            fig = _figure_cache[figure_key]
            fig.update_layout(height=750)
            return {'display': 'flex'}, fig

        return {'display': 'none'}, dash.no_update

    @app.callback(
        Output('ta-modal-title', 'children'),
        [Input('ta-modal-title-store', 'data')],
        prevent_initial_call=False
    )
    def update_modal_title(title):
        if title:
            return f"📊 {title}"
        return dash.no_update
