"""
多信号共振：整合项目内各功能模块的量化方法，对个股给出综合评分

维度与权重（各维度 -2 ~ +2 分，缺失维度自动按剩余权重重新归一）：
1. 因子评分   0.30  因子训练页模型的横截面分位（factor_page）
2. 趋势状态   0.25  价格/趋势均线位置与斜率 + ADX 趋势强度（signal_enhancer 参数）
3. 技术动量   0.20  MACD 柱方向与斜率 + RSI 区位（趋势分析页方法论）
4. CLV 资金流 0.15  收盘位置资金流（板块资金强度页 CLV 方法论，用于个股）
5. 信号质量   0.10  三重过滤后均线策略的当前持仓状态（signal_enhancer）

综合分 ∈ [-2, +2]：≥1.0 强共振看多 / ≥0.4 偏多 / >-0.4 中性 / >-1.0 偏空 / 其余 共振看空
"""
import logging

import numpy as np
import pandas as pd

from dash import dcc, html

from .signal_enhancer import calculate_adx, enhance_signals
from .trend_analysis import macd_divergence, rsi_calculation

logger = logging.getLogger(__name__)

WEIGHTS = {
    'factor': 0.30,
    'trend': 0.25,
    'momentum': 0.20,
    'clv_flow': 0.15,
    'signal_quality': 0.10,
}

_VERDICTS = [
    (1.0, '强共振看多', 'bg-green-50 text-green-700 border-green-300'),
    (0.4, '偏多', 'bg-emerald-50 text-emerald-700 border-emerald-200'),
    (-0.4, '中性', 'bg-gray-50 text-gray-600 border-gray-200'),
    (-1.0, '偏空', 'bg-amber-50 text-amber-700 border-amber-200'),
]
_VERDICT_FALLBACK = ('共振看空', 'bg-red-50 text-red-700 border-red-300')

_SCORE_STYLES = {
    2: 'bg-green-100 text-green-800', 1: 'bg-emerald-50 text-emerald-700',
    0: 'bg-gray-100 text-gray-600', -1: 'bg-amber-50 text-amber-700',
    -2: 'bg-red-100 text-red-700',
}


def _score_label(score):
    if score >= 1.5:
        return '强多'
    if score > 0.3:
        return '偏多'
    if score > -0.3:
        return '中性'
    if score > -1.5:
        return '偏空'
    return '看空'


def _dim(key, label, score, evidence, weight):
    return {'key': key, 'label': label, 'score': None if score is None else round(float(score), 2),
            'evidence': evidence, 'weight': weight}


def _score_factor(factor_info):
    """因子模型横截面分位 → -2..+2"""
    if factor_info is None or not factor_info.get('covered'):
        return None, '未纳入当前因子模型覆盖池'
    pct = factor_info['pct']
    if pct <= 10:
        s, band = 2.0, f'前 {pct:.0f}%（头部）'
    elif pct <= 25:
        s, band = 1.0, f'前 {pct:.0f}%'
    elif pct <= 50:
        s, band = 0.5, f'前 {pct:.0f}%'
    elif pct >= 90:
        s, band = -2.0, f'后 {100 - pct:.0f}%（尾部）'
    elif pct >= 75:
        s, band = -1.0, f'后 {100 - pct:.0f}%'
    else:
        s, band = -0.5, f'后 {100 - pct:.0f}%'
    model = factor_info.get('model_name') or ''
    suffix = f'（{model}）' if model else ''
    return s, f"评分 {factor_info['score']:.3f}，排名 {factor_info['rank']}/{factor_info['total']}，{band}{suffix}"


def _score_trend(data, trend_ma=60, adx_threshold=20):
    """价格/趋势均线位置与斜率 + ADX → -2..+2"""
    close = data['close'].astype(float)
    ma = close.rolling(int(trend_ma), min_periods=max(20, int(trend_ma) // 2)).mean()
    if ma.dropna().empty:
        return None, f'历史数据不足，无法计算 {trend_ma} 日趋势均线'
    last_c, last_ma = float(close.iloc[-1]), float(ma.iloc[-1])
    s = 1.0 if last_c > last_ma else -1.0
    if len(ma.dropna()) >= 21:
        ma_prev = float(ma.dropna().iloc[-21])
        s += 0.5 if last_ma > ma_prev else -0.5
        slope_txt = 'MA 上行' if last_ma > ma_prev else 'MA 下行'
    else:
        slope_txt = 'MA 斜率数据不足'
    evidence = f'收盘 {last_c:.2f} {"提升" if last_c > last_ma else "跌破"}MA{int(trend_ma)}{last_ma:.2f}，{slope_txt}'

    adx = calculate_adx(data)
    if adx is not None and pd.notna(adx.iloc[-1]):
        v = float(adx.iloc[-1])
        if v >= max(25.0, float(adx_threshold)):
            # ADX 高代表趋势强：顺着已判定的方向增强信心
            s += 0.5 if s > 0 else -0.5
            evidence += f'，ADX {v:.1f}（趋势增强）'
        elif v < 15.0:
            # 震荡市：价格与均线的相对位置不可靠，信心减半
            s *= 0.5
            evidence += f'，ADX {v:.1f}（震荡市，信号降权）'
        else:
            evidence += f'，ADX {v:.1f}'
    return float(np.clip(s, -2, 2)), evidence


def _score_momentum(data):
    """MACD 柱方向与斜率 + RSI 区位 → -2..+2"""
    n = len(data)
    if n < 40:
        return None, '历史数据不足（需 40 根以上 K 线）'
    macd = macd_divergence(data)
    hist = pd.Series(macd['histogram'])
    h_last, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    s = 1.0 if h_last > 0 else -1.0
    rising = h_last > h_prev
    s += 0.25 if rising else -0.25
    trend_txt = '上行' if rising else '下行'
    evidence = f'MACD 柱 {h_last:+.3f}（{trend_txt}）'

    rsi = rsi_calculation(data, window=14)
    r_last = float(rsi[-1]) if pd.notna(rsi[-1]) else 50.0
    evidence += f'，RSI {r_last:.1f}'
    if r_last >= 70:
        s -= 0.5
        evidence += '（超买区）'
    elif r_last <= 30:
        s += 0.5
        evidence += '（超卖区）'
    div = macd.get('divergences') or []
    if div and div[-1].get('idx') == n - 2:  # 最近一根K线的背离信号
        if '底背离' in div[-1]['type']:
            s += 0.5
            evidence += '，底背离'
        else:
            s -= 0.5
            evidence += '，顶背离'
    return float(np.clip(s, -2, 2)), evidence


def _score_clv_flow(data, window=20):
    """CLV 资金流（板块资金强度页方法论，用于个股）：近 window 日 CLV×成交额 净流入占比 → -2..+2"""
    need = ('high', 'low', 'close', 'amount')
    if any(c not in data.columns for c in need):
        return None, '数据缺少 high/low/amount，无法计算 CLV 资金流'
    sub = data.tail(int(window))
    rng = sub['high'] - sub['low']
    clv = pd.Series(0.0, index=sub.index)
    normal = rng > 0
    clv[normal] = 2.0 * (sub['close'][normal] - sub['low'][normal]) / rng[normal] - 1.0
    amount = sub['amount'].astype(float)
    total_amount = float(amount.sum())
    if total_amount <= 0:
        return None, '成交额数据为 0'
    ratio = float((clv * amount).sum() / total_amount)
    if ratio >= 0.08:
        s = 2.0
    elif ratio >= 0.03:
        s = 1.0
    elif ratio > -0.03:
        s = 0.0
    elif ratio > -0.08:
        s = -1.0
    else:
        s = -2.0
    direction = '净流入' if ratio > 0 else ('净流出' if ratio < 0 else '均衡')
    return s, f'近{int(window)}日 CLV 资金{direction}占比 {ratio * 100:+.2f}%'


def _score_signal_quality(data, filter_opts, confirm_days=2, trend_ma=60, adx_threshold=20):
    """三重过滤后双均线策略的当前状态 → -2..+2"""
    try:
        from .strategy_manager import double_moving_average_strategy
        raw = double_moving_average_strategy(data, 5, 20)
        if raw is None or raw.empty:
            return None, '均线信号计算失败'
        opts = set(filter_opts or [])
        if not opts:
            state = 'in' if raw['signal'].iloc[-1] > 0 else 'out'
            return (1.0 if state == 'in' else 0.0), '未启用过滤：' + ('双均线持仓中' if state == 'in' else '双均线空仓')
        filtered, stats = enhance_signals(
            data, raw, 'dma',
            use_trend='trend' in opts, trend_ma=trend_ma,
            use_confirm='confirm' in opts, confirm_days=confirm_days,
            use_adx='adx' in opts, adx_threshold=adx_threshold)
        raw_in = raw['signal'].iloc[-1] > 0
        filt_in = filtered['signal'].iloc[-1] > 0
        if filt_in:
            return 1.0, '三重过滤后仍持仓（信号通过全部质量检查）'
        if raw_in and not filt_in:
            return -1.0, '原始买入信号被过滤（当前环境不适合入场）'
        return 0.0, f'双均线空仓（原始交易 {stats["raw_entries"]} 笔 → 过滤后 {stats["filtered_entries"]} 笔）'
    except Exception as e:
        logger.warning(f'信号质量评分失败: {e}')
        return None, '信号质量评分计算失败'


def compute_confluence(code, data, trend_ma=60, adx_threshold=20, filter_opts=None,
                       confirm_days=2, factor_info=None):
    """对单只股票计算五维度共振评分，返回可渲染的结果 dict"""
    name = code
    if data is None or data.empty or len(data) < 30:
        return {'code': code, 'name': name, 'ok': False, 'reason': 'K线数据不足（需 30 根以上）',
                'dimensions': [], 'total': None, 'verdict': '数据不足', 'verdict_style': _SCORE_STYLES[0]}

    dims = []
    f_score, f_ev = _score_factor(factor_info)
    dims.append(_dim('factor', '因子评分', f_score, f_ev, WEIGHTS['factor']))
    t_score, t_ev = _score_trend(data, trend_ma, adx_threshold)
    dims.append(_dim('trend', '趋势状态', t_score, t_ev, WEIGHTS['trend']))
    m_score, m_ev = _score_momentum(data)
    dims.append(_dim('momentum', '技术动量', m_score, m_ev, WEIGHTS['momentum']))
    c_score, c_ev = _score_clv_flow(data)
    dims.append(_dim('clv_flow', 'CLV 资金流', c_score, c_ev, WEIGHTS['clv_flow']))
    q_score, q_ev = _score_signal_quality(data, filter_opts, confirm_days, trend_ma, adx_threshold)
    dims.append(_dim('signal_quality', '信号质量', q_score, q_ev, WEIGHTS['signal_quality']))

    w_sum = sum(d['weight'] for d in dims if d['score'] is not None)
    if w_sum <= 0:
        return {'code': code, 'name': name, 'ok': False, 'reason': '所有维度均无法计算',
                'dimensions': dims, 'total': None, 'verdict': '数据不足', 'verdict_style': _SCORE_STYLES[0]}
    total = sum(d['score'] * d['weight'] for d in dims if d['score'] is not None) / w_sum

    verdict, style = _VERDICT_FALLBACK
    for threshold, label, css in _VERDICTS:
        if total >= threshold:
            verdict, style = label, css
            break
    return {'code': code, 'name': name, 'ok': True, 'reason': '',
            'dimensions': dims, 'total': round(float(total), 2), 'verdict': verdict, 'verdict_style': style}


def build_confluence_panel(results, title_map=None):
    """分析页「多信号共振」面板：results 为 compute_confluence 结果列表"""
    if not results:
        return html.Div()
    title_map = title_map or {}

    blocks = []
    for r in results[:3]:
        code = r['code']
        title = f"{title_map.get(code, '')} ({code})".strip() if title_map.get(code) else code
        if not r['ok']:
            blocks.append(html.Div([
                html.Div(title, className="text-sm font-semibold text-gray-700 mb-1"),
                html.Div(r['reason'], className="text-sm text-gray-400 py-2"),
            ], className="bg-gray-50 rounded-lg p-3 mb-3"))
            continue

        rows = []
        for d in r['dimensions']:
            if d['score'] is None:
                score_badge = html.Span('N/A', className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-400")
            else:
                bucket = min((2, 1, 0, -1, -2), key=lambda b: abs(d['score'] - b))
                score_badge = html.Span(
                    f"{d['score']:+.2f} {_score_label(d['score'])}",
                    className=f"text-xs px-2 py-0.5 rounded {_SCORE_STYLES[bucket]}")
            rows.append(html.Tr([
                html.Td(d['label'], className="py-1 pr-3 text-xs font-medium text-gray-600 whitespace-nowrap align-top"),
                html.Td(d['evidence'], className="py-1 pr-3 text-xs text-gray-500"),
                html.Td(score_badge, className="py-1 text-xs whitespace-nowrap align-top text-right"),
            ]))
        blocks.append(html.Div([
            html.Div([
                html.Span(title, className="text-sm font-semibold text-gray-700 mr-3"),
                html.Span(f"综合 {r['total']:+.2f}", className="text-sm font-bold text-gray-800 mr-2"),
                html.Span(r['verdict'], className=f"text-xs px-2 py-1 rounded-full border font-medium {r['verdict_style']}"),
            ], className="flex items-center flex-wrap mb-2"),
            html.Table(rows, className="w-full"),
        ], className="bg-gray-50 rounded-lg p-3 mb-3"))

    header = html.Div([
        html.Div([
            html.I(className="fas fa-layer-group text-teal-600 text-xl mr-2"),
            html.H3("多信号共振", className="text-lg font-bold text-gray-800"),
        ], className="flex items-center"),
        html.Span("因子评分 × 趋势状态 × 技术动量 × CLV 资金流 × 信号质量 加权共振",
                  className="text-xs text-gray-400"),
    ], className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b")

    return html.Div([header, html.Div(blocks)], className="bg-white rounded-xl shadow-md p-4 mb-4")
