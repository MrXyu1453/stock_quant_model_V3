"""
量化走势预测引擎（概率化预测：区间 + 事件概率 + 状态感知 + 实测校准）

模型：GBM 蒙特卡洛 / Holt 双参数指数平滑 / OLS 线性趋势，按回测 MAPE 加权集成。

v2 完善（相对点预测版的改进）：
1. 概率化输出：N 日后上涨概率（GBM 解析概率 × 历史同状态经验频率 混合），
   触及上方/下方障碍位的概率（蒙特卡洛路径模拟，可直接用于止盈止损参考）
2. 状态感知：识别 趋势向上/趋势向下/震荡 三种状态；震荡市明确提示方向预测不可信
3. 情景表：乐观(q75)/中性(中位)/悲观(q25) 三情景的 N 日收益
4. 回测增加概率校准检验：预测 P(涨) 的均值 vs 实际上涨频率的偏差

准确性声明：短期股价接近随机游走，任何模型都无法保证准确。方向命中率 ≤50%
等价于抛硬币；P(涨) 的价值取决于校准差（预测概率与实际频率的偏差）。所有指标
均为该股历史滚动回测的实测值，历史表现不保证未来。
"""
import logging
import math

import numpy as np
import pandas as pd

from dash import dcc, html
import plotly.graph_objects as go

from .signal_enhancer import calculate_adx

logger = logging.getLogger(__name__)

Z80 = 1.2816
_MIN_HISTORY = 150
_BACKTEST_WINDOW = 120
_BACKTEST_FOLDS = 8
_N_SIMS = 3000


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _z(p):
    table = {0.10: -Z80, 0.25: -0.6745, 0.75: 0.6745, 0.90: Z80}
    return table.get(p, 0.0)


# ==================== 状态识别 ====================

def detect_regime_series(data, trend_ma=60, adx_threshold=20):
    """逐日状态序列：'up' / 'down' / 'chop'（收盘在趋势均线上/下方 且 ADX 达标，否则震荡）"""
    close = data['close'].astype(float)
    ma = close.rolling(int(trend_ma), min_periods=max(20, int(trend_ma) // 2)).mean()
    reg = np.where(close > ma, 'up', 'down')
    adx = calculate_adx(data)
    if adx is not None:
        reg = np.where(adx >= float(adx_threshold), reg, 'chop')
    return pd.Series(reg, index=data.index)


def summarize_regime(reg):
    if reg is None or len(reg) == 0:
        return {'state': 'unknown', 'label': '状态未知', 'style': 'bg-gray-100 text-gray-600',
                'direction_ok': False,
                'hint': '状态无法识别，方向预测可靠性未知'}
    cur = reg.iloc[-1]
    if cur == 'up':
        return {'state': 'up', 'label': '趋势向上', 'style': 'bg-green-100 text-green-700',
                'direction_ok': True, 'hint': '趋势行情，方向预测参考价值较高'}
    if cur == 'down':
        return {'state': 'down', 'label': '趋势向下', 'style': 'bg-red-100 text-red-700',
                'direction_ok': True, 'hint': '趋势行情，方向预测参考价值较高'}
    return {'state': 'chop', 'label': '震荡市', 'style': 'bg-amber-100 text-amber-700',
            'direction_ok': False,
            'hint': '当前为震荡市：方向预测参考价值低，建议只参考区间与事件概率'}


# ==================== 各模型预测 ====================

def forecast_gbm(close, horizon):
    """GBM 解析外推（漂移收缩一半）。返回中位路径与 10/25/75/90 分位带"""
    close = pd.Series(close).dropna().astype(float).values
    rets = np.diff(np.log(close))
    window = rets[-120:]
    mu = float(np.mean(window)) * 0.5
    sigma = float(np.std(window, ddof=1))
    s0 = float(close[-1])
    steps = np.arange(1, horizon + 1)
    drift = (mu - 0.5 * sigma ** 2) * steps
    vol = sigma * np.sqrt(steps)
    return {
        'median': s0 * np.exp(drift),
        'q10': s0 * np.exp(drift + vol * _z(0.10)),
        'q25': s0 * np.exp(drift + vol * _z(0.25)),
        'q75': s0 * np.exp(drift + vol * _z(0.75)),
        'q90': s0 * np.exp(drift + vol * _z(0.90)),
    }, mu, sigma, s0


def simulate_paths(close, horizon, n_sims=_N_SIMS, seed=42):
    """蒙特卡洛路径矩阵 shape (n_sims, horizon)，用于障碍触及概率"""
    _, mu, sigma, s0 = forecast_gbm(close, horizon)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_sims, horizon))
    steps = np.arange(1, horizon + 1)
    log_ret = (mu - 0.5 * sigma ** 2) * steps + sigma * np.sqrt(steps) * shocks
    return s0 * np.exp(np.cumsum(log_ret, axis=1))


def barrier_probabilities(sim_prices, last, up_pct, dn_pct):
    """路径首次触及障碍的概率（任一时点穿过即算）"""
    up_barrier = last * (1 + up_pct)
    dn_barrier = last * (1 - dn_pct)
    p_up = float(np.mean(np.any(sim_prices >= up_barrier, axis=1)))
    p_dn = float(np.mean(np.any(sim_prices <= dn_barrier, axis=1)))
    return p_up, p_dn


def probability_up_gbm(mu, sigma, horizon):
    """GBM 下 S_h > S_0 的解析概率：P(Σ对数收益 > 0) = Φ(mu·√h/σ)，mu 为对数漂移"""
    if sigma <= 0:
        return 0.5
    return float(_norm_cdf(mu * math.sqrt(horizon) / sigma))


def empirical_up_prob(close_values, regimes, current_regime, horizon, lookback=250, min_samples=10):
    """历史同状态下 h 日后上涨的频率（只用当前时点之前的数据）"""
    n = len(close_values)
    ups = []
    for i in range(max(n - lookback, 0), n - horizon):
        if regimes[i] == current_regime:
            ups.append(close_values[i + horizon] > close_values[i])
    if len(ups) < min_samples:
        return None
    return float(np.mean(ups))


def forecast_holt(close, horizon):
    v = pd.Series(close).dropna().astype(float).values
    best = None
    for alpha in (0.2, 0.4, 0.6, 0.8, 0.95):
        for beta in (0.05, 0.1, 0.2, 0.4, 0.6):
            level, trend = v[0], v[1] - v[0]
            resid = []
            for t in range(1, len(v)):
                pred = level + trend
                resid.append(v[t] - pred)
                level_new = alpha * v[t] + (1 - alpha) * (level + trend)
                trend = beta * (level_new - level) + (1 - beta) * trend
                level = level_new
            sse = float(np.sum(np.square(resid)))
            if best is None or sse < best[0]:
                best = (sse, alpha, beta, level, trend, float(np.std(resid, ddof=1)) if len(resid) > 2 else 0.0)
    _, _, _, level, trend, sigma = best
    steps = np.arange(1, horizon + 1)
    path = level + trend * steps
    band = sigma * np.sqrt(steps) * Z80
    return {
        'median': path,
        'q10': path - band, 'q25': path - band / 2.0,
        'q75': path + band / 2.0, 'q90': path + band,
    }


def forecast_ols(close, horizon, window=60):
    v = pd.Series(close).dropna().astype(float).values[-int(window):]
    x = np.arange(len(v))
    coef = np.polyfit(x, v, 1)
    fitted = np.polyval(coef, x)
    sigma = float(np.std(v - fitted, ddof=1))
    steps = np.arange(1, horizon + 1)
    path = np.polyval(coef, len(v) - 1 + steps)
    band = max(sigma, 1e-6) * np.sqrt(steps) * Z80
    return {
        'median': path,
        'q10': path - band, 'q25': path - band / 2.0,
        'q75': path + band / 2.0, 'q90': path + band,
    }


def forecast_ensemble(paths, mapes):
    inv = np.array([1.0 / max(m, 1e-4) for m in mapes])
    w = inv / inv.sum()
    out = {}
    stack = np.vstack([p['median'] for p in paths])
    out['median'] = w @ stack
    for k in ('q10', 'q25', 'q75', 'q90'):
        out[k] = np.median(np.vstack([p[k] for p in paths]), axis=0)
    return out


# ==================== 滚动回测 ====================

def walk_forward_backtest(close_values, horizon, data=None, trend_ma=60, adx_threshold=20,
                          folds=_BACKTEST_FOLDS):
    """滚动回测：方向命中率 / 80%区间覆盖率 / 期末MAPE / P(涨)校准差"""
    close = pd.Series(close_values).dropna().astype(float).values
    n = len(close)
    result = {}
    if n < _MIN_HISTORY + horizon:
        return result
    regimes = None
    if data is not None and len(data) == n:
        regimes = detect_regime_series(data, trend_ma, adx_threshold)

    fold_step = max(horizon, (_BACKTEST_WINDOW - horizon) // max(folds, 1))
    starts = []
    e = n - horizon
    while e >= n - _BACKTEST_WINDOW and len(starts) < folds * 3:
        starts.append(e)
        e -= fold_step
    starts = sorted(starts)

    metrics = {name: {'dir_hits': 0, 'n': 0, 'covered': 0, 'pts': 0, 'ape': [],
                      'p_preds': [], 'p_actuals': []}
               for name in ('GBM', 'Holt', 'OLS', '集成')}
    for e in starts:
        if e + horizon > n or e < _MIN_HISTORY // 2:
            continue
        history, actual = close[:e], close[e:e + horizon]
        last = history[-1]
        gbm_path, mu, sigma, _s0 = forecast_gbm(history, horizon)
        paths = {'GBM': gbm_path,
                 'Holt': forecast_holt(history, horizon),
                 'OLS': forecast_ols(history, horizon)}
        ens = forecast_ensemble(list(paths.values()), [1.0] * len(paths))
        all_paths = dict(paths)
        all_paths['集成'] = ens
        # P(涨)：GBM 解析概率 × 历史同状态经验频率
        p_up_gbm = probability_up_gbm(mu, sigma, horizon)
        p_emp = None
        if regimes is not None:
            p_emp = empirical_up_prob(history, regimes.values[:e], regimes.values[e], horizon)
        p_up_blend = 0.5 * p_up_gbm + 0.5 * p_emp if p_emp is not None else p_up_gbm
        p_ups = {'GBM': p_up_gbm, '集成': p_up_blend}
        for name, p in all_paths.items():
            m = metrics[name]
            terminal_fc = float(p['median'][-1])
            terminal_ac = float(actual[-1])
            m['dir_hits'] += int((terminal_fc > last) == (terminal_ac > last))
            m['n'] += 1
            if terminal_fc > 0:
                m['ape'].append(abs(terminal_fc - terminal_ac) / terminal_ac)
            lo, hi = p['q10'], p['q90']
            within = [(actual[i] >= lo[i]) and (actual[i] <= hi[i]) for i in range(len(actual))]
            m['covered'] += int(np.sum(within))
            m['pts'] += max(len(actual), 1)
            if name in p_ups and p_ups[name] is not None:
                m['p_preds'].append(p_ups[name])
                m['p_actuals'].append(int(terminal_ac > last))

    for name, m in metrics.items():
        if m['n'] == 0:
            continue
        entry = {
            'dir_rate': m['dir_hits'] / m['n'],
            'dir_n': m['n'],
            'coverage': m['covered'] / m['pts'] if m['pts'] else 0.0,
            'mape': float(np.mean(m['ape'])) if m['ape'] else 0.0,
        }
        if m['p_preds']:
            entry['p_up_mean'] = float(np.mean(m['p_preds']))
            entry['cal_gap'] = abs(float(np.mean(m['p_preds'])) - float(np.mean(m['p_actuals'])))
        result[name] = entry
    return result


def forecast_gbm_paths_only(history, horizon):
    p, _, _, _ = forecast_gbm(history, horizon)
    return p


# ==================== 汇总 ====================

def run_forecast(data, horizon, code='', name='', trend_ma=60, adx_threshold=20, filter_opts=None):
    """对单只股票执行概率化预测 + 回测，返回渲染所需结果"""
    close = data['close'].dropna().astype(float)
    if len(close) < _MIN_HISTORY + max(horizon, 1):
        return {'ok': False, 'code': code, 'name': name,
                'reason': f'历史数据不足（需 {_MIN_HISTORY} 根以上 K 线，当前 {len(close)} 根）'}
    values = close.values
    regimes = detect_regime_series(data, trend_ma, adx_threshold)
    regime = summarize_regime(regimes)

    backtest = walk_forward_backtest(values, horizon, data=data, trend_ma=trend_ma,
                                     adx_threshold=adx_threshold)
    mapes = [backtest.get(k, {}).get('mape', 0.05) for k in ('GBM', 'Holt', 'OLS')]
    paths = {'GBM': forecast_gbm_paths_only(values, horizon),
             'Holt': forecast_holt(values, horizon),
             'OLS': forecast_ols(values, horizon)}
    ens = forecast_ensemble(list(paths.values()), mapes)
    last = float(close.iloc[-1])

    # 当前时点的上涨概率与障碍概率
    gbm_path, mu, sigma, s0 = forecast_gbm(values, horizon)
    p_up_gbm = probability_up_gbm(mu, sigma, horizon)
    p_emp = empirical_up_prob(values, regimes.values, regimes.values[-1], horizon)
    p_up = p_up_gbm if p_emp is None else 0.5 * p_up_gbm + 0.5 * p_emp
    sim = simulate_paths(values, horizon)
    # 障碍位自适应：±max(2%, 1.0σ√h)
    barrier_pct = max(0.02, float(sigma) * math.sqrt(horizon))
    p_touch_up, p_touch_dn = barrier_probabilities(sim, last, barrier_pct, barrier_pct)

    best_name = max(backtest, key=lambda k: backtest[k]['dir_rate']) if backtest else None
    return {
        'ok': True, 'code': code, 'name': name,
        'last': last, 'horizon': horizon,
        'paths': paths, 'ensemble': ens, 'backtest': backtest, 'best': best_name,
        'history': close.iloc[-90:],
        'regime': regime,
        'p_up': p_up, 'p_up_gbm': p_up_gbm, 'p_emp': p_emp,
        'barrier_pct': barrier_pct, 'p_touch_up': p_touch_up, 'p_touch_dn': p_touch_dn,
    }


# ==================== 面板与图表 ====================

def build_forecast_figure(res):
    hist = res['history']
    ens = res['ensemble']
    last_date = hist.index[-1]
    last_val = float(hist.iloc[-1])
    future_idx = pd.date_range(last_date, periods=res['horizon'] + 1, freq='B')[1:]
    x_fc = [last_date] + list(future_idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='历史收盘',
                             line=dict(color='#64748B', width=1.5)))
    for lo, hi, nm, color in ((ens['q10'], ens['q90'], '80%预测区间', 'rgba(59,130,246,0.12)'),
                              (ens['q25'], ens['q75'], '50%预测区间', 'rgba(59,130,246,0.25)')):
        fig.add_trace(go.Scatter(
            x=[last_date] + list(future_idx) + list(future_idx[::-1]) + [last_date],
            y=[last_val] + list(hi) + list(lo[::-1]) + [last_val],
            fill='toself', fillcolor=color, mode='lines', line=dict(width=0),
            name=nm, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=x_fc, y=[last_val] + list(ens['median']), mode='lines+markers',
                             name='集成中位预测', line=dict(color='#3B82F6', width=2, dash='dash'),
                             marker=dict(size=4)))
    # 乐观/悲观情景端点
    fig.add_trace(go.Scatter(x=[future_idx[-1]], y=[float(ens['q75'][-1])], mode='markers+text',
                             name='乐观情景', marker=dict(color='#10B981', size=9, symbol='triangle-up'),
                             text=[f"乐观 {float(ens['q75'][-1]):.2f}"], textposition='top center',
                             textfont=dict(size=10)))
    fig.add_trace(go.Scatter(x=[future_idx[-1]], y=[float(ens['q25'][-1])], mode='markers+text',
                             name='悲观情景', marker=dict(color='#EF4444', size=9, symbol='triangle-down'),
                             text=[f"悲观 {float(ens['q25'][-1]):.2f}"], textposition='bottom center',
                             textfont=dict(size=10)))
    fig.add_hline(y=res['last'], line_dash='dot', line_color='gray',
                  annotation_text=f'当前 {res["last"]:.2f}', annotation_position='right')
    fig.update_layout(title=f'未来 {res["horizon"]} 步走势预测（区间为统计置信带，非保证）',
                      template='plotly_white', height=380,
                      margin=dict(l=40, r=20, t=45, b=35), hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0))
    return fig


def _prob_card(label, prob, sub, style):
    pct = f"{prob * 100:.0f}%" if prob is not None else '—'
    return html.Div([
        html.Div(label, className="text-xs text-gray-500 mb-1"),
        html.Div(pct, className=f"text-2xl font-bold {style}"),
        html.Div(sub, className="text-xs text-gray-400 mt-1"),
    ], className="flex-1 bg-gray-50 rounded-lg p-3 text-center min-w-0")


def build_forecast_panel(res):
    if not res.get('ok'):
        return html.Div(res['reason'], className="text-sm text-gray-400 py-4 text-center")

    ens, bt, best = res['ensemble'], res['backtest'], res['best']
    reg = res['regime']
    last = res['last']
    terminal = float(ens['median'][-1])
    h_ret = terminal / last - 1

    # 头部：预测对象 + 状态
    title = f"{res['name']} ({res['code']})" if res['name'] else res['code']
    head = html.Div([
        html.Span("预测对象: ", className="text-xs text-gray-500"),
        html.Span(title, className="text-sm font-semibold text-gray-700 mr-3"),
        html.Span(reg['label'], className=f"text-xs px-2 py-0.5 rounded-full font-medium {reg['style']}"),
        html.Span(reg['hint'], className="text-xs text-gray-400"),
    ], className="flex flex-wrap items-center gap-x-2 mb-3")

    # 三个关键概率
    probs = html.Div([
        _prob_card(f'{res["horizon"]}步后上涨概率', res['p_up'],
                   f"GBM {res['p_up_gbm'] * 100:.0f}%" + (f" × 历史 {res['p_emp'] * 100:.0f}%" if res['p_emp'] is not None else ''),
                   'text-blue-700'),
        _prob_card(f'触及 +{res["barrier_pct"] * 100:.0f}%（止盈参考）', res['p_touch_up'],
                   f'当前 {last:.2f} → {last * (1 + res["barrier_pct"]):.2f}', 'text-green-700'),
        _prob_card(f'触及 -{res["barrier_pct"] * 100:.0f}%（止损参考）', res['p_touch_dn'],
                   f'当前 {last:.2f} → {last * (1 - res["barrier_pct"]):.2f}', 'text-red-700'),
    ], className="flex gap-3 mb-3")

    # 情景表
    scen_rows = []
    for label, key, style in (('乐观 (q75)', 'q75', 'text-green-600'),
                              ('中性 (中位)', 'median', 'text-gray-700'),
                              ('悲观 (q25)', 'q25', 'text-red-600')):
        px = float(ens[key][-1])
        scen_rows.append(html.Tr([
            html.Td(label, className="text-xs px-2 py-1 font-medium"),
            html.Td(f"{px:.2f}", className="text-xs px-2 py-1 text-right"),
            html.Td(f"{(px / last - 1) * 100:+.1f}%", className=f"text-xs px-2 py-1 text-right {style}"),
        ]))
    scenario = html.Table(
        [html.Tr([html.Th('情景', className='text-xs text-left px-2 py-1 bg-gray-100'),
                  html.Th(f'{res["horizon"]}步价格', className='text-xs text-right px-2 py-1 bg-gray-100'),
                  html.Th('收益率', className='text-xs text-right px-2 py-1 bg-gray-100')])] + scen_rows,
        className="w-full mb-3")

    # 回测对比表（含 P(涨)校准差）
    header = html.Tr([html.Th('模型', className="text-xs text-left px-2 py-1"),
                      html.Th('回测方向命中', className="text-xs text-right px-2 py-1"),
                      html.Th('80%区间覆盖', className="text-xs text-right px-2 py-1"),
                      html.Th('期末MAPE', className="text-xs text-right px-2 py-1"),
                      html.Th('P(涨)校准差', className="text-xs text-right px-2 py-1"),
                      html.Th(f'{res["horizon"]}步预测', className="text-xs text-right px-2 py-1")],
                     className="bg-gray-100")
    rows = [header]
    for nm in ('GBM', 'Holt', 'OLS', '集成'):
        m = bt.get(nm)
        fc = float(ens['median'][-1]) if nm == '集成' else float(res['paths'][nm]['median'][-1])
        if m:
            dir_txt = f"{m['dir_rate'] * 100:.0f}% ({m['dir_n']}折)"
            if m['dir_rate'] <= 0.5:
                dir_txt += ' ⚠≈抛硬币'
            cov = f"{m['coverage'] * 100:.0f}%"
            mape = f"{m['mape'] * 100:.1f}%"
            cal = f"{m['cal_gap'] * 100:.0f}%" if 'cal_gap' in m else '—'
            row_style = " bg-blue-50" if nm == best else ""
        else:
            dir_txt, cov, mape, cal, row_style = '—', '—', '—', '—', ''
        label = nm + (' ★回测最优' if nm == best else '')
        rows.append(html.Tr([
            html.Td(label, className=f"text-xs px-2 py-1 font-medium{row_style}"),
            html.Td(dir_txt, className=f"text-xs px-2 py-1 text-right{row_style}"),
            html.Td(cov, className=f"text-xs px-2 py-1 text-right{row_style}"),
            html.Td(mape, className=f"text-xs px-2 py-1 text-right{row_style}"),
            html.Td(cal, className=f"text-xs px-2 py-1 text-right{row_style}"),
            html.Td(f"{fc:.2f}", className=f"text-xs px-2 py-1 text-right font-medium{row_style}"),
        ]))

    note = html.Div(
        '准确性说明：方向命中≤50%的模型等价于抛硬币；区间覆盖接近80%、P(涨)校准差小说明模型在该股上可信。'
        '以上为该股历史滚动回测实测值，历史表现不保证未来，不构成投资建议。',
        className="text-xs text-gray-400 mt-2")

    return html.Div([
        head,
        probs,
        scenario,
        html.Table(rows, className="w-full mb-3"),
        dcc.Graph(figure=build_forecast_figure(res), config={'displayModeBar': False}, className="h-96"),
        note,
    ])
