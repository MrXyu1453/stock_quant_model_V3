"""
股票交易小游戏（模拟炒股）：隐藏未来行情，逐根揭示历史K线，玩家在真实数据上模拟买卖。

规则:
1. 每局随机抽取历史时间段，行情从该区间逐根揭示，玩家只能以"当根收盘价"市价成交
2. A股交易规则: 买入须100股整手、T+1(当日买入下一根才可卖)、
   佣金万2.5(最低5元, 双边)、印花税万5(仅卖出)、过户费万0.1(双边)
3. 非整手卖出只允许一次性清仓（零股规则）
4. 随时可结算: 对比"首根收盘全仓买入持有"基准，按超额收益评级 S/A/B/C/D
5. 各类参数数据(均线/MACD/RSI/KDJ/BOLL/量比)随行情逐根更新，供决策参考
"""
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_manager import get_and_process_data, get_available_sources, set_data_source

logger = logging.getLogger(__name__)

# 每个交易时段的分钟K线根数（A股 4 小时连续竞价）
_MINUTE_BARS_PER_DAY = {'60min': 4, '30min': 8, '15min': 16, '5min': 48}
_FREQ_LABELS = {'daily': '日线', 'weekly': '周线', '60min': '60分钟',
                '30min': '30分钟', '15min': '15分钟', '5min': '5分钟'}

# ===================== 交易规则参数 =====================
COMMISSION_RATE = 0.00025   # 佣金 万2.5
COMMISSION_MIN = 5.0        # 最低佣金 5 元
STAMP_TAX = 0.0005          # 印花税 万5，仅卖出
TRANSFER_FEE = 0.00001      # 过户费 万0.1，双边

# 服务端游戏数据缓存（game_id → 行情与指标），浏览器只存交易状态，避免剧透
_GAMES = OrderedDict()
_GAMES_MAX = 30

# 多拉几倍历史，让游戏窗口能在其中随机抽取时间段
_RANDOM_SPAN_FACTOR = 3
# 游戏窗口前保留的指标预热根数 / 窗口后隐藏的根数（防止从数据结尾推断未来走势）
_WARMUP_BARS = 70
_HIDDEN_TAIL_BARS = 70


# ===================== 交易逻辑（纯函数，可单测） =====================

def buy_fee(amount):
    """买入费用: 佣金(最低5元) + 过户费"""
    return max(amount * COMMISSION_RATE, COMMISSION_MIN) + amount * TRANSFER_FEE


def sell_fee(amount):
    """卖出费用: 佣金(最低5元) + 印花税 + 过户费"""
    return max(amount * COMMISSION_RATE, COMMISSION_MIN) + amount * (STAMP_TAX + TRANSFER_FEE)


def available_shares(lots, bar):
    """T+1 可卖数量: 当根买入的股票下一根才能卖"""
    return sum(l['shares'] for l in lots if l['bar'] < bar)


def avg_cost(lots):
    total_shares = sum(l['shares'] for l in lots)
    if total_shares <= 0:
        return 0.0
    return sum(l['shares'] * l['price'] for l in lots) / total_shares


def _copy_state(state):
    st = dict(state)
    st['lots'] = [dict(l) for l in state.get('lots', [])]
    st['trades'] = list(state.get('trades', []))
    return st


def apply_buy(state, shares, price, bar, date):
    """买入。返回 (新状态, 错误信息)；成功时错误信息为空"""
    shares = int(shares or 0)
    price = float(price)
    if price <= 0:
        return None, '价格异常，无法成交'
    if shares <= 0 or shares % 100 != 0:
        return None, '买入数量必须为100股整数倍（A股整手规则）'
    amount = shares * price
    fee = buy_fee(amount)
    if amount + fee > state['cash'] + 1e-6:
        return None, (f'资金不足: 需要 {amount + fee:,.2f} 元（含费用），'
                      f'可用资金 {state["cash"]:,.2f} 元')
    st = _copy_state(state)
    st['cash'] = round(st['cash'] - amount - fee, 2)
    st['lots'] = st['lots'] + [{'bar': bar, 'shares': shares, 'price': float(price)}]
    st['trades'] = st['trades'] + [{'bar': int(bar), 'date': date, 'action': 'buy',
                                    'shares': shares, 'price': float(price),
                                    'fee': round(fee, 2), 'cash_after': st['cash']}]
    return st, ''


def apply_sell(state, shares, price, bar, date):
    """卖出（FIFO 减仓，T+1 限制）。返回 (新状态, 错误信息)"""
    shares = int(shares or 0)
    price = float(price)
    if price <= 0:
        return None, '价格异常，无法成交'
    if shares <= 0:
        return None, '卖出数量必须大于0'
    avail = available_shares(state['lots'], bar)
    if shares > avail:
        return None, f'可卖数量不足（T+1，当根买入下一根才可卖）: 可卖 {avail} 股'
    if shares % 100 != 0 and shares != avail:
        return None, '非整手数量只允许一次性清仓（零股规则）'
    st = _copy_state(state)
    remaining = shares
    cost_basis = 0.0
    new_lots = []
    for lot in st['lots']:
        if remaining <= 0 or lot['bar'] >= bar:
            new_lots.append(lot)
            continue
        take = min(lot['shares'], remaining)
        cost_basis += take * lot['price']
        remaining -= take
        if take < lot['shares']:
            new_lots.append({'bar': lot['bar'], 'shares': lot['shares'] - take,
                             'price': lot['price']})
    st['lots'] = new_lots
    proceeds = shares * price
    fee = sell_fee(proceeds)
    st['cash'] = round(st['cash'] + proceeds - fee, 2)
    pnl = proceeds - fee - cost_basis
    st['trades'] = st['trades'] + [{'bar': int(bar), 'date': date, 'action': 'sell',
                                    'shares': shares, 'price': float(price),
                                    'fee': round(fee, 2), 'pnl': round(pnl, 2),
                                    'cash_after': st['cash']}]
    return st, ''


def portfolio_value(state, closes, idx):
    """第 idx 根收盘时的总资产（现金 + 持仓市值，含 T+1 未解锁部分）"""
    shares = sum(l['shares'] for l in state['lots'])
    return state['cash'] + shares * float(closes[idx])


def equity_curve(state, closes, dates):
    """逐根资产曲线 [(日期, 玩家资产, 基准资产)]；基准=首根收盘全仓买入持有(不计费用)"""
    events = {}
    for t in state['trades']:
        events.setdefault(t['bar'], []).append(t)
    cash, shares = state['initial_cash'], 0
    base = float(closes[0])
    curve = []
    for i in range(int(state['idx']) + 1):
        for t in events.get(i, []):
            shares += t['shares'] if t['action'] == 'buy' else -t['shares']
            cash = t['cash_after']
        v = cash + shares * float(closes[i])
        curve.append((dates[i], v, state['initial_cash'] / base * float(closes[i])))
    return curve


def max_drawdown(values):
    """最大回撤（0~1）"""
    peak, dd = float('-inf'), 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            dd = max(dd, (peak - v) / peak)
    return dd


def trade_stats(trades):
    """交易统计: 买入次数、卖出(平仓)次数、平仓胜率、累计实现盈亏"""
    n_buy = sum(1 for t in trades if t['action'] == 'buy')
    pnls = [t['pnl'] for t in trades if t['action'] == 'sell']
    wins = sum(1 for p in pnls if p > 0)
    return {'n_buy': n_buy, 'n_sell': len(pnls),
            'win_rate': (wins / len(pnls)) if pnls else None,
            'realized_pnl': round(sum(pnls), 2)}


def rank_of(excess, n_trades):
    """按超额收益(百分点)评级 → (段位, 评语, 样式类)"""
    if n_trades == 0:
        return '—', '一股未买，全程围观。有时候不动也是种操作，但这次不算分', \
            'bg-gray-100 text-gray-500'
    if excess >= 15:
        return 'S', '股神附体！大幅跑赢买入持有', 'bg-amber-100 text-amber-700'
    if excess >= 8:
        return 'A', '操作漂亮，明显跑赢基准', 'bg-green-100 text-green-700'
    if excess >= 2:
        return 'B', '稳健盈利，小胜基准', 'bg-blue-100 text-blue-700'
    if excess >= -3:
        return 'C', '与基准不相上下，再接再厉', 'bg-yellow-100 text-yellow-700'
    return 'D', '跑输买入持有——少操作有时比多操作强', 'bg-red-100 text-red-700'


# ===================== 指标与行情 =====================

def _indicator_df(df):
    """在行情 DataFrame 上追加常用技术指标列（MA/MACD/RSI/KDJ/BOLL/量比）"""
    out = df.copy()
    close = out['close'].astype(float)
    for w in (5, 10, 20, 60):
        out[f'ma{w}'] = close.rolling(w, min_periods=1).mean()
    macd = _macd_lines(close)
    out['macd_dif'], out['macd_dea'], out['macd_hist'] = macd
    out['rsi'] = _rsi(close, 14)
    out['kdj_k'], out['kdj_d'], out['kdj_j'] = _kdj(out, 9)
    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std().fillna(0)
    out['boll_up'], out['boll_mid'], out['boll_low'] = mid + 2 * std, mid, mid - 2 * std
    out['vol_ma5'] = out['vol'].astype(float).rolling(5, min_periods=1).mean()
    return out


def _macd_lines(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    dif = ema_f - ema_s
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea, (dif - dea) * 2


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _kdj(df, n=9):
    low_n = df['low'].astype(float).rolling(n, min_periods=1).min()
    high_n = df['high'].astype(float).rolling(n, min_periods=1).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = ((df['close'].astype(float) - low_n) / rng * 100).fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def _new_game_id():
    import time
    return f"g{int(time.time() * 1000)}"


def _create_game(code, freq, play_bars, source):
    """拉取数据并随机抽取历史时间段初始化服务端行情缓存。返回 (game_id, 错误信息)"""
    freq = freq or 'daily'
    bars_needed = int(play_bars) + 30 + _WARMUP_BARS  # 游戏窗口 + 可见上下文 + 指标预热
    span = _RANDOM_SPAN_FACTOR
    if freq == 'daily':
        days = bars_needed * 1.7 * span + 40
    elif freq == 'weekly':
        days = bars_needed * 7 * 1.7 * span + 40
    else:  # 分钟线: 按每交易日的K线根数折算
        per_day = _MINUTE_BARS_PER_DAY.get(freq, 4)
        days = max(bars_needed / per_day * 1.8 * span + 15, 40)
        days = min(days, 2000)  # 分钟线历史仅从2020年起
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime('%Y%m%d')
    end = pd.Timestamp.now().strftime('%Y%m%d')
    df = get_and_process_data(code, start, end, freq=freq)
    if df is None or len(df) < int(play_bars) + 15:
        return None, (f'数据不足（需至少 {int(play_bars) + 15} 根K线，实际 '
                      f'{0 if df is None else len(df)} 根），请换个股票或缩短游戏长度')
    df = df.sort_values('trade_date').reset_index(drop=True)
    play_len = min(int(play_bars), len(df) - 10)
    ctx_len = min(30, len(df) - play_len)
    tail = min(_HIDDEN_TAIL_BARS, max(0, len(df) - play_len - ctx_len))
    max_start = len(df) - play_len - tail  # 窗口最晚起点，保证尾部隐藏区
    pre = _WARMUP_BARS + ctx_len           # 游戏窗口前的预热 + 可见上下文
    min_start = min(pre, max_start)        # 历史不够预热时退化为贴尾部
    if max_start > min_start:
        play_start = int(np.random.default_rng().integers(min_start, max_start + 1))
    else:
        play_start = max_start
    ind = _indicator_df(df)  # 指标在完整历史上计算，再裁剪到游戏窗口附近
    keep_from = max(0, play_start - pre)
    view = ind.iloc[keep_from: play_start + play_len + tail].reset_index(drop=True)
    try:
        from .data_manager import get_stock_name
        name = get_stock_name(code, source) or ''
    except Exception:
        name = ''
    game_id = _new_game_id()
    g = {'df': view, 'view_start': play_start - keep_from - ctx_len,
         'ctx_len': ctx_len, 'play_len': play_len,
         'code': code, 'name': name, 'freq': freq}
    _precompute_render_data(g)  # 整段K线与指标提前加载为numpy数组+缓存布局
    _GAMES[game_id] = g
    while len(_GAMES) > _GAMES_MAX:
        _GAMES.popitem(last=False)
    return game_id, ''


def _save_record(state, g):
    """结算后把战绩写入数据库（当前用户）。成功返回 True"""
    try:
        from .database_manager import add_game_record, get_current_username
        closes, dates = _game_bars(g)
        gi = g['play_len'] - 1
        value = portfolio_value(state, closes, gi)
        ret = value / state['initial_cash'] - 1
        curve = equity_curve(state, closes, dates)
        bench_ret = curve[-1][2] / state['initial_cash'] - 1 if curve else 0.0
        stats = trade_stats(state['trades'])
        excess = (ret - bench_ret) * 100
        grade, comment, _ = rank_of(excess, stats['n_buy'] + stats['n_sell'])
        rec_id = add_game_record({
            'username': get_current_username() or 'guest',
            'stock_code': g['code'], 'stock_name': g['name'],
            'freq': g['freq'], 'play_bars': g['play_len'],
            'initial_cash': state['initial_cash'], 'final_value': round(value, 2),
            'return_pct': round(ret * 100, 2), 'bench_return_pct': round(bench_ret * 100, 2),
            'excess_pp': round(excess, 2), 'max_dd': round(max_drawdown([c[1] for c in curve]), 4),
            'n_buy': stats['n_buy'], 'n_sell': stats['n_sell'],
            'win_rate': round(stats['win_rate'], 4) if stats['win_rate'] is not None else 0,
            'realized_pnl': stats['realized_pnl'],
            'fees': round(sum(t['fee'] for t in state['trades']), 2),
            'grade': grade, 'comment': comment,
        })
        return rec_id is not None
    except Exception as e:
        logger.warning(f'保存战绩失败(不影响结算): {e}')
        return False


def _game_bars(g):
    """游戏窗口的收盘价与日期序列（首次调用后缓存，避免每次点击重复切表）"""
    cached = g.get('_bars')
    if cached is None:
        vs, cl, pl = g['view_start'], g['ctx_len'], g['play_len']
        seg = g['df'].iloc[vs + cl: vs + cl + pl]
        cached = (seg['close'].astype(float).tolist(),
                  seg['trade_date'].astype(str).tolist())
        g['_bars'] = cached
    return cached


def _precompute_chart(df):
    """把整段K线/均线/成交量/MACD序列一次性转为numpy数组（开局提前加载）。
    之后每次点击只做O(1)切片视图，plotly 还会按二进制紧凑编码传输，避免逐根卡顿"""
    def col(name):
        return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)

    o, c = col('open'), col('close')
    return {
        'dates': df['trade_date'].astype(str).tolist(),
        'open': o, 'high': col('high'), 'low': col('low'), 'close': c,
        'ma5': col('ma5'), 'ma10': col('ma10'), 'ma20': col('ma20'), 'ma60': col('ma60'),
        'vol': col('vol'),
        'vol_colors': ['#ef4444' if cv >= ov else '#22c55e' for cv, ov in zip(c, o)],
        'macd_hist': col('macd_hist'), 'macd_dif': col('macd_dif'), 'macd_dea': col('macd_dea'),
        'hist_colors': ['#ef4444' if h >= 0 else '#22c55e' for h in col('macd_hist')],
    }


def _param_rows(df):
    """每根K线的技术参数快照（纯标量，None 表示缺失），供 build_params_panel 查表"""
    def val(rec, key):
        v = rec.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(v) else v

    rows = []
    for rec in df.to_dict('records'):
        vol_ma5 = val(rec, 'vol_ma5')
        vol = val(rec, 'vol')
        rows.append({
            'close': val(rec, 'close'),
            'ma5': val(rec, 'ma5'), 'ma10': val(rec, 'ma10'),
            'ma20': val(rec, 'ma20'), 'ma60': val(rec, 'ma60'),
            'macd_hist': val(rec, 'macd_hist'),
            'macd_dif': val(rec, 'macd_dif'), 'macd_dea': val(rec, 'macd_dea'),
            'rsi': val(rec, 'rsi'),
            'kdj_k': val(rec, 'kdj_k'), 'kdj_d': val(rec, 'kdj_d'), 'kdj_j': val(rec, 'kdj_j'),
            'boll_up': val(rec, 'boll_up'), 'boll_mid': val(rec, 'boll_mid'),
            'boll_low': val(rec, 'boll_low'),
            'vol_ratio': (vol / vol_ma5) if (vol_ma5 and vol is not None) else None,
            'amount_yi': (val(rec, 'amount') or 0.0) / 1e8 if val(rec, 'amount') else None,
        })
    return rows


def _precompute_render_data(g):
    """提前加载：开局时一次性预计算整段行情的全部渲染数据。
    之后每次点击"下一根"只做数组切片与trace拼装，不再触发 pandas 计算，避免逐根卡顿。"""
    g['chart'] = _precompute_chart(g['df'])
    g['param_rows'] = _param_rows(g['df'])
    g['fig_layout'] = _chart_layout(g)
    _game_bars(g)  # 预热游戏窗口收盘价/日期缓存


def _chart_layout(g):
    """静态图布局（子图网格/样式/游戏起点分隔线），每局只构建一次"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.58, 0.17, 0.25],
                        vertical_spacing=0.04,
                        subplot_titles=('行情（未来不可见）', '成交量', 'MACD'))
    fig.update_layout(
        template='plotly_white', height=560,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        uirevision=f"{g['code']}-{g['freq']}",
    )
    fig.update_yaxes(title_text='价格', row=1, col=1, gridcolor='#f1f5f9')
    fig.update_yaxes(title_text='成交量', row=2, col=1)
    fig.update_yaxes(title_text='MACD', row=3, col=1)
    # 游戏起点分隔线（游戏窗口第一根 = view_start + ctx_len）。
    # 注：无 trace 的子图上 add_vline 会被 plotly 静默忽略，须用 add_shape/add_annotation 显式添加
    if g['chart']['dates']:
        gx = g['chart']['dates'][g['view_start'] + g['ctx_len']]
        fig.add_shape(type='line', xref='x', yref='y domain', x0=gx, x1=gx, y0=0, y1=1,
                      line=dict(color='#94a3b8', dash='dot', width=1))
        fig.add_annotation(xref='x', yref='y domain', x=gx, y=1, text='游戏开始',
                           showarrow=False, xanchor='right', yanchor='top',
                           font=dict(size=10, color='#64748b'))
    return fig.layout.to_plotly_json()


# ===================== 展示构建 =====================

def _pct_color(v):
    return '#ef4444' if v > 0 else ('#22c55e' if v < 0 else '#64748b')


def _fmt_pct(v):
    return f'{v * 100:+.2f}%'


def _chip(label, value, color='#1e293b', big=False):
    return html.Div([
        html.Div(label, className='text-[10px] text-gray-400 leading-tight'),
        html.Div(value, className=f'{"text-lg" if big else "text-sm"} font-bold leading-tight',
                 style={'color': color}),
    ], className='bg-white border border-gray-100 rounded-lg px-3 py-1.5 shadow-sm min-w-[86px]')


def build_status_cards(state, g, gi):
    """顶部账户状态卡片"""
    closes, dates = _game_bars(g)
    price = closes[gi]
    prev = closes[gi - 1] if gi > 0 else price
    chg = (price - prev) / prev if prev else 0.0
    value = portfolio_value(state, closes, gi)
    shares = sum(l['shares'] for l in state['lots'])
    avail = available_shares(state['lots'], gi)
    cost = avg_cost(state['lots'])
    float_pnl = (price - cost) * shares if shares else 0.0
    ret = value / state['initial_cash'] - 1
    bench = state['initial_cash'] / closes[0] * price / state['initial_cash'] - 1
    pct_played = f'{gi + 1}/{g["play_len"]}'
    return html.Div([
        _chip('日期 / 进度', f'{dates[gi][:16]} · 第{gi + 1}根'),
        _chip('现价', f'{price:.2f} ({_fmt_pct(chg)})', _pct_color(chg)),
        _chip('总资产', f'{value:,.0f}', '#1d4ed8', big=True),
        _chip('可用资金', f'{state["cash"]:,.0f}'),
        _chip('持仓 / 可卖', f'{shares} / {avail}'),
        _chip('持仓成本', f'{cost:.2f}' if shares else '—'),
        _chip('浮动盈亏', f'{float_pnl:+,.0f}', _pct_color(float_pnl) if shares else '#64748b'),
        _chip('总收益率', _fmt_pct(ret), _pct_color(ret)),
        _chip('基准收益率', _fmt_pct(bench), _pct_color(bench)),
        html.Div([
            html.Div('领先基准' if ret >= bench else '落后基准',
                     className='text-[10px] text-gray-400 leading-tight'),
            html.Div(f'{(ret - bench) * 100:+.2f}pp',
                     className='text-sm font-bold', style={'color': '#7c3aed'}),
        ], className='bg-white border border-gray-100 rounded-lg px-3 py-1.5 shadow-sm min-w-[86px]'),
    ], className='flex flex-wrap gap-2')


def build_params_panel(g, gi):
    """当前bar的技术参数面板（各类参数数据；数值开局时已提前计算，直接查表）"""
    rows = g.get('param_rows')
    if rows is None:
        rows = _param_rows(g['df'])
        g['param_rows'] = rows
    p = rows[g['view_start'] + g['ctx_len'] + gi]
    price = p['close'] if p['close'] is not None else 0.0

    def _ma_chip(w):
        v = p[f'ma{w}']
        if v is None:
            return _chip(f'MA{w}', '—')
        pos = '上方' if price >= v else '下方'
        color = '#ef4444' if price >= v else '#22c55e'
        return _chip(f'MA{w}({pos})', f'{v:.2f}', color)

    hist = p['macd_hist'] if p['macd_hist'] is not None else 0.0
    rsi = p['rsi'] if p['rsi'] is not None else 50.0
    kdj = '/'.join(f'{p[k]:.0f}' if p[k] is not None else '—'
                   for k in ('kdj_k', 'kdj_d', 'kdj_j'))
    boll = ('/'.join(f'{p[k]:.2f}' for k in ('boll_up', 'boll_mid', 'boll_low'))
            if all(p[k] is not None for k in ('boll_up', 'boll_mid', 'boll_low')) else '—')
    dif_dea = (f"{p['macd_dif']:.3f} / {p['macd_dea']:.3f}"
               if p['macd_dif'] is not None and p['macd_dea'] is not None else '—')
    chips = [
        _ma_chip(5), _ma_chip(10), _ma_chip(20), _ma_chip(60),
        _chip('MACD 柱', f'{hist:+.3f}', _pct_color(hist)),
        _chip('DIF / DEA', dif_dea),
        _chip('RSI(14)', f'{rsi:.1f}',
              '#ef4444' if rsi > 70 else ('#22c55e' if rsi < 30 else '#1e293b')),
        _chip('KDJ', kdj),
        _chip('BOLL 上/中/下', boll),
        _chip('量比(vs 5日均量)', f"{p['vol_ratio']:.2f}" if p['vol_ratio'] is not None else '—'),
        _chip('成交额', f"{p['amount_yi']:.2f}亿" if p['amount_yi'] is not None else '—'),
    ]
    return html.Div(chips, className='flex flex-wrap gap-2')


def build_game_figure(g, gi):
    """K线 + MA + 成交量 + MACD，只显示到当前bar（未来不可见）。
    行情与指标序列开局时已提前加载为numpy数组，布局每局缓存，这里只做切片拼装。"""
    chart = g.get('chart')
    if chart is None:
        chart = _precompute_chart(g['df'])
        g['chart'] = chart
    layout = g.get('fig_layout')
    if layout is None:
        layout = _chart_layout(g)
        g['fig_layout'] = layout
    vs, cl = g['view_start'], g['ctx_len']
    end = vs + cl + gi
    n = end + 1
    dates = chart['dates'][:n]
    traces = [
        go.Candlestick(x=dates, open=chart['open'][:n], high=chart['high'][:n],
                       low=chart['low'][:n], close=chart['close'][:n], name='K线',
                       increasing_line_color='#ef4444', decreasing_line_color='#22c55e',
                       xaxis='x', yaxis='y'),
    ]
    for w, color in ((5, '#f97316'), (10, '#3b82f6'), (20, '#8b5cf6'), (60, '#64748b')):
        traces.append(go.Scatter(x=dates, y=chart[f'ma{w}'][:n], mode='lines',
                                 name=f'MA{w}', line=dict(color=color, width=1.1),
                                 hovertemplate=f'MA{w}: %{{y:.2f}}<extra></extra>',
                                 xaxis='x', yaxis='y'))
    # 现价标记
    traces.append(go.Scatter(
        x=[dates[-1]], y=[float(chart['close'][end])], mode='markers',
        marker=dict(symbol='diamond', size=11, color='#1e293b',
                    line=dict(color='white', width=2)),
        name='现价', showlegend=False, xaxis='x', yaxis='y'))
    # 成交量
    traces.append(go.Bar(x=dates, y=chart['vol'][:n], marker_color=chart['vol_colors'][:n],
                         name='成交量', showlegend=False, xaxis='x2', yaxis='y2'))
    # MACD
    traces.append(go.Bar(x=dates, y=chart['macd_hist'][:n],
                         marker_color=chart['hist_colors'][:n],
                         name='MACD柱', showlegend=False, xaxis='x3', yaxis='y3'))
    traces.append(go.Scatter(x=dates, y=chart['macd_dif'][:n], mode='lines',
                             line=dict(color='#3b82f6', width=1.2), name='DIF',
                             xaxis='x3', yaxis='y3'))
    traces.append(go.Scatter(x=dates, y=chart['macd_dea'][:n], mode='lines',
                             line=dict(color='#f97316', width=1.2), name='DEA',
                             xaxis='x3', yaxis='y3'))
    return go.Figure(data=traces, layout=layout)


def build_equity_figure(state, g, gi):
    """资产曲线 vs 买入持有基准"""
    closes, dates = _game_bars(g)
    curve = equity_curve(state, closes, dates)
    if not curve:
        return go.Figure()
    xs = [c[0][:10] for c in curve]
    player = [c[1] for c in curve]
    bench = [c[2] for c in curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=player, mode='lines', name='我的资产',
                             line=dict(color='#2563eb', width=2),
                             hovertemplate='我的资产: %{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=xs, y=bench, mode='lines', name='买入持有基准',
                             line=dict(color='#94a3b8', width=1.6, dash='dash'),
                             hovertemplate='基准: %{y:,.0f}<extra></extra>'))
    fig.update_layout(template='plotly_white', height=220,
                      margin=dict(l=50, r=20, t=30, b=30),
                      hovermode='x unified',
                      title=dict(text='资产曲线 vs 买入持有基准', font=dict(size=13), x=0.01))
    return fig


def build_trades_table(state):
    """最近交易记录表"""
    trades = state.get('trades', [])[-12:]
    if not trades:
        return html.Div('暂无交易。用右侧按钮或输入股数开始第一笔操作。',
                        className='text-xs text-gray-400 italic')
    rows = [html.Tr([html.Th(h, className='px-2 py-1 text-left text-[10px] text-gray-400 font-medium')
                     for h in ('时间', '方向', '价格', '数量', '费用', '平仓盈亏', '剩余现金')],
                    className='border-b border-gray-100')]
    for t in reversed(trades):
        is_buy = t['action'] == 'buy'
        rows.append(html.Tr([
            html.Td(t.get('date', '')[:10], className='px-2 py-1 text-xs font-mono text-gray-500'),
            html.Td('🟢 买入' if is_buy else '🔴 卖出',
                    className='px-2 py-1 text-xs font-bold'),
            html.Td(f"{t['price']:.2f}", className='px-2 py-1 text-xs'),
            html.Td(str(t['shares']), className='px-2 py-1 text-xs'),
            html.Td(f"{t['fee']:.2f}", className='px-2 py-1 text-xs text-gray-500'),
            html.Td('—' if is_buy else f"{t.get('pnl', 0):+,.2f}",
                    className='px-2 py-1 text-xs',
                    style={'color': '#64748b' if is_buy else _pct_color(t.get('pnl', 0))}),
            html.Td(f"{t['cash_after']:,.0f}", className='px-2 py-1 text-xs text-gray-500'),
        ]))
    return html.Table(rows, className='w-full')


def build_scoreboard(state, g, gi):
    """结算面板"""
    closes, dates = _game_bars(g)
    value = portfolio_value(state, closes, gi)
    ret = value / state['initial_cash'] - 1
    curve = equity_curve(state, closes, dates)
    bench_ret = curve[-1][2] / state['initial_cash'] - 1 if curve else 0.0
    excess = (ret - bench_ret) * 100
    stats = trade_stats(state['trades'])
    badge, comment, style = rank_of(excess, stats['n_buy'] + stats['n_sell'])
    dd_player = max_drawdown([c[1] for c in curve])
    dd_bench = max_drawdown([c[2] for c in curve])
    total_fees = round(sum(t['fee'] for t in state['trades']), 2)
    rows = [
        ('期末总资产', f'{value:,.2f} 元', '#1d4ed8'),
        ('我的收益率', _fmt_pct(ret), _pct_color(ret)),
        ('基准(买入持有)', _fmt_pct(bench_ret), _pct_color(bench_ret)),
        ('超额收益', f'{excess:+.2f}pp', _pct_color(excess)),
        ('我的最大回撤', f'{dd_player * 100:.1f}%', '#b91c1c'),
        ('基准最大回撤', f'{dd_bench * 100:.1f}%', '#64748b'),
        ('买入 / 平仓次数', f"{stats['n_buy']} / {stats['n_sell']}", '#1e293b'),
        ('平仓胜率', f"{stats['win_rate'] * 100:.0f}%" if stats['win_rate'] is not None else '—',
         '#1e293b'),
        ('实现盈亏', f"{stats['realized_pnl']:+,.2f} 元", _pct_color(stats['realized_pnl'])),
        ('手续费合计', f'{total_fees:,.2f} 元', '#64748b'),
    ]
    return html.Div([
        html.Div([
            html.Div(badge, className=f'{style} text-5xl font-black rounded-2xl '
                                      'w-24 h-24 flex items-center justify-center shadow-inner'),
            html.Div([
                html.Div('最终段位', className='text-xs text-gray-400'),
                html.Div(comment, className='text-sm font-semibold text-gray-700 mt-1'),
                html.Div('✓ 战绩已记入下方排行榜', className='text-[10px] text-green-600 mt-1'),
            ], className='ml-4'),
        ], className='flex items-center mb-4'),
        html.Div([
            html.Div([
                html.Div(k, className='text-[11px] text-gray-400'),
                html.Div(v, className='text-sm font-bold', style={'color': c}),
            ], className='bg-gray-50 rounded-lg px-3 py-2 min-w-[130px]')
            for k, v, c in rows
        ], className='grid grid-cols-2 md:grid-cols-5 gap-2'),
    ], className='bg-white rounded-xl border border-amber-200 shadow-sm p-4')


# ===================== 排行榜 / 战绩 =====================

_GRADE_STYLES = {'S': 'bg-amber-100 text-amber-700', 'A': 'bg-green-100 text-green-700',
                 'B': 'bg-blue-100 text-blue-700', 'C': 'bg-yellow-100 text-yellow-700',
                 'D': 'bg-red-100 text-red-700', '—': 'bg-gray-100 text-gray-500'}


def _grade_badge(grade):
    return html.Span(grade,
                     className=f"{_GRADE_STYLES.get(grade, _GRADE_STYLES['—'])} "
                               'inline-flex items-center justify-center w-7 h-7 rounded-lg font-black text-sm')


def _board_th(label):
    return html.Th(label, className='px-2 py-1 text-left text-[10px] text-gray-400 font-medium whitespace-nowrap')


def build_leaderboard_panel():
    """全用户排行榜（按超额收益）"""
    from .database_manager import get_top_game_records
    rows_data = get_top_game_records(limit=10)
    if not rows_data:
        return html.Div('还没有人上榜。完成一局游戏，抢占第一名！',
                        className='text-xs text-gray-400 italic')
    header = html.Tr([_board_th(h) for h in
                      ('排名', '玩家', '股票', '周期', '长度', '收益率', '超额', '段位', '回撤', '时间')],
                     className='border-b border-gray-100')
    rows = [header]
    for i, r in enumerate(rows_data):
        rows.append(html.Tr([
            html.Td(f'#{i + 1}', className='px-2 py-1 text-xs font-bold text-gray-500'),
            html.Td(r['username'], className='px-2 py-1 text-xs font-semibold'),
            html.Td(f"{r['stock_code']} {r['stock_name']}".strip(),
                    className='px-2 py-1 text-xs whitespace-nowrap'),
            html.Td(_FREQ_LABELS.get(r['freq'], r['freq']), className='px-2 py-1 text-xs text-gray-500'),
            html.Td(f"{r['play_bars']}根", className='px-2 py-1 text-xs text-gray-500'),
            html.Td(_fmt_pct(r['return_pct'] / 100), className='px-2 py-1 text-xs font-bold',
                    style={'color': _pct_color(r['return_pct'])}),
            html.Td(f"{r['excess_pp']:+.2f}pp", className='px-2 py-1 text-xs font-bold',
                    style={'color': _pct_color(r['excess_pp'])}),
            html.Td(_grade_badge(r['grade']), className='px-2 py-1'),
            html.Td(f"{r['max_dd'] * 100:.1f}%", className='px-2 py-1 text-xs text-gray-500'),
            html.Td(str(r['created_at'])[:16], className='px-2 py-1 text-[10px] text-gray-400'),
        ]))
    return html.Table(rows, className='w-full')


def build_my_records_panel():
    """当前用户的战绩与汇总"""
    from .database_manager import get_current_user_id, get_game_user_summary, get_my_game_records
    uid = get_current_user_id()
    records = get_my_game_records(uid, limit=20)
    if not records:
        return html.Div('还没有战绩。完成一局游戏后在这里回顾。',
                        className='text-xs text-gray-400 italic')
    s = get_game_user_summary(uid)
    chips = html.Div([
        _chip('总局数', f"{s['games']}"),
        _chip('平均超额', f"{s['avg_excess']:+.2f}pp", _pct_color(s['avg_excess'])),
        _chip('跑赢基准占比', f"{s['positive_ratio'] * 100:.0f}%",
              '#22c55e' if s['positive_ratio'] >= 0.5 else '#64748b'),
        _chip('最佳一局', f"{s['best_excess']:+.2f}pp", '#f59e0b'),
    ], className='flex flex-wrap gap-2 mb-2')
    header = html.Tr([_board_th(h) for h in
                      ('时间', '股票', '周期', '长度', '收益率', '基准', '超额', '段位', '平仓胜率', '评语')],
                     className='border-b border-gray-100')
    rows = [header]
    for r in records:
        wr = f"{r['win_rate'] * 100:.0f}%" if r['n_sell'] else '—'
        rows.append(html.Tr([
            html.Td(str(r['created_at'])[:16], className='px-2 py-1 text-[10px] text-gray-400 whitespace-nowrap'),
            html.Td(f"{r['stock_code']} {r['stock_name']}".strip(),
                    className='px-2 py-1 text-xs whitespace-nowrap'),
            html.Td(_FREQ_LABELS.get(r['freq'], r['freq']), className='px-2 py-1 text-xs text-gray-500'),
            html.Td(f"{r['play_bars']}根", className='px-2 py-1 text-xs text-gray-500'),
            html.Td(_fmt_pct(r['return_pct'] / 100), className='px-2 py-1 text-xs font-bold',
                    style={'color': _pct_color(r['return_pct'])}),
            html.Td(_fmt_pct(r['bench_return_pct'] / 100), className='px-2 py-1 text-xs text-gray-500'),
            html.Td(f"{r['excess_pp']:+.2f}pp", className='px-2 py-1 text-xs font-bold',
                    style={'color': _pct_color(r['excess_pp'])}),
            html.Td(_grade_badge(r['grade']), className='px-2 py-1'),
            html.Td(wr, className='px-2 py-1 text-xs text-gray-500'),
            html.Td(r['comment'], className='px-2 py-1 text-[10px] text-gray-400'),
        ]))
    return html.Div([chips, html.Table(rows, className='w-full')])


def build_board_container(active='top'):
    """排行榜 / 我的战绩 双Tab容器内容"""
    content = build_leaderboard_panel() if active == 'top' else build_my_records_panel()
    return html.Div(content, className='overflow-x-auto')


# ===================== 页面布局 =====================

def _empty_figure(text):
    fig = go.Figure()
    fig.update_layout(template='plotly_white', height=560,
                      annotations=[dict(text=text, xref='paper', yref='paper',
                                        x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=16, color='#94a3b8'))])
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def stock_game_layout():
    """模拟炒股小游戏页面布局"""
    return html.Div([
        html.Main([
            # ===== 左侧边栏：游戏设置 =====
            html.Aside([
                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-gamepad mr-2'),
                                 '模拟炒股小游戏'], className='text-lg font-semibold text-gray-700 flex items-center'),
                    ], className='mb-4 pb-2 border-b border-gray-200'),

                    html.Label('选择股票:', className='block text-sm font-medium text-gray-700 mb-1'),
                    dcc.Dropdown(id='sg-stock-dropdown', options=[], value=None,
                                 placeholder='搜索或选择股票...', className='mb-3'),
                    html.Label('或输入代码:', className='block text-sm font-medium text-gray-700 mb-1'),
                    html.Div([
                        dcc.Input(id='sg-stock-input', type='text', placeholder='例如: 600519',
                                  className='w-full py-2 px-3 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm',
                                  style={'flex': '1'}),
                    ], className='flex mb-3'),

                    html.Label('数据源:', className='block text-sm font-medium text-gray-700 mb-1'),
                    dcc.Dropdown(id='sg-data-source',
                                 options=[{'label': s.capitalize(), 'value': s}
                                          for s in get_available_sources()],
                                 value='tushare', clearable=False, className='mb-3'),

                    html.Label('K线周期:', className='block text-sm font-medium text-gray-700 mb-1'),
                    dcc.Dropdown(id='sg-freq',
                                 options=[{'label': '日线', 'value': 'daily'},
                                          {'label': '周线', 'value': 'weekly'},
                                          {'label': '60分钟 · 快节奏', 'value': '60min'},
                                          {'label': '30分钟 · 快节奏', 'value': '30min'},
                                          {'label': '15分钟 · 快节奏', 'value': '15min'},
                                          {'label': '5分钟 · 极速', 'value': '5min'}],
                                 value='daily', clearable=False, className='mb-3'),

                    html.Label('游戏长度(K线根数):', className='block text-sm font-medium text-gray-700 mb-1'),
                    dcc.Dropdown(id='sg-play-bars',
                                 options=[{'label': '60 根（速战）', 'value': 60},
                                          {'label': '120 根（标准）', 'value': 120},
                                          {'label': '250 根（一年/马拉松）', 'value': 250}],
                                 value=120, clearable=False, className='mb-3'),

                    html.Label('初始资金(元):', className='block text-sm font-medium text-gray-700 mb-1'),
                    dcc.Dropdown(id='sg-initial-cash',
                                 options=[{'label': '10 万', 'value': 100000},
                                          {'label': '50 万', 'value': 500000},
                                          {'label': '100 万', 'value': 1000000}],
                                 value=100000, clearable=False, className='mb-3'),

                    html.Button([html.I(className='fas fa-play mr-2'), '开始新游戏'],
                                id='sg-start-btn',
                                className='w-full bg-green-600 hover:bg-green-700 text-white py-2.5 rounded-lg font-medium transition-all'),

                    html.Div([
                        html.H3('游戏规则', className='text-sm font-semibold text-gray-600 mb-1 pt-3 border-t border-gray-200 mt-3'),
                        html.Ul([
                            html.Li('每局随机抽取历史时间段，行情逐根揭示，只能以当根收盘价成交', className='text-xs text-gray-500 mb-1'),
                            html.Li('买入须100股整手；T+1，当日买入下一根可卖', className='text-xs text-gray-500 mb-1'),
                            html.Li('佣金万2.5(最低5元)、卖出印花税万5、过户费万0.1', className='text-xs text-gray-500 mb-1'),
                            html.Li('分钟线为快节奏模式，可开自动播放（每根自动揭示）', className='text-xs text-gray-500 mb-1'),
                            html.Li('分钟线历史数据仅从2020年起', className='text-xs text-gray-500 mb-1'),
                            html.Li('结算对比"首根收盘全仓买入持有"，超额收益定段位并记入排行榜', className='text-xs text-gray-500'),
                        ], className='list-disc pl-4'),
                    ]),
                ], className='space-y-1')
            ], className='w-full md:w-1/4 lg:w-1/5 bg-white rounded-xl shadow-lg border border-gray-100 p-4 overflow-y-auto',
                style={'maxHeight': 'calc(100vh - 100px)'}),

            # ===== 右侧游戏区 =====
            html.Div([
                html.Div(id='sg-error', className='text-red-500 text-sm mb-2'),
                html.Div(id='sg-status-cards',
                         children=html.Div('设置好参数后，点左侧"开始新游戏"',
                                           className='text-sm text-gray-400'),
                         className='flex flex-wrap gap-2 mb-3'),
                dcc.Graph(id='sg-chart', figure=_empty_figure('等待开始游戏…'),
                          config={'displayModeBar': True, 'displaylogo': False},
                          style={'height': '560px'}),

                # 参数面板
                html.Div([
                    html.Div('📊 当前技术参数', className='text-xs font-bold text-gray-500 mb-1.5'),
                    html.Div(id='sg-params-panel', children=html.Div('—', className='text-xs text-gray-400')),
                ], className='bg-blue-50 border border-blue-100 rounded-xl p-3 mt-3'),

                # 交易操作面板
                html.Div([
                    html.Div([
                        html.Div('💰 交易操作', className='text-xs font-bold text-gray-500 mb-2'),
                        html.Div('以当根收盘价成交 · 100股整手 · T+1',
                                 className='text-[10px] text-gray-400'),
                    ], className='flex items-baseline gap-2'),
                    html.Div([
                        dcc.Input(id='sg-buy-shares', type='number', value=100, min=100, step=100,
                                  className='w-28 py-1.5 px-2 border border-gray-300 rounded text-sm'),
                        html.Button('自定义买入', id='sg-buy-btn',
                                    className='bg-red-600 hover:bg-red-700 text-white px-3 py-1.5 rounded text-sm font-medium'),
                        html.Button('买¼仓', id='sg-buy-q1',
                                    className='bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-2.5 py-1.5 rounded text-xs'),
                        html.Button('买半仓', id='sg-buy-q2',
                                    className='bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-2.5 py-1.5 rounded text-xs'),
                        html.Button('买全仓', id='sg-buy-q3',
                                    className='bg-red-50 hover:bg-red-100 text-red-600 border border-red-200 px-2.5 py-1.5 rounded text-xs'),
                    ], className='flex flex-wrap items-center gap-2 mt-2'),
                    html.Div([
                        dcc.Input(id='sg-sell-shares', type='number', value=100, min=1, step=100,
                                  className='w-28 py-1.5 px-2 border border-gray-300 rounded text-sm'),
                        html.Button('自定义卖出', id='sg-sell-btn',
                                    className='bg-green-600 hover:bg-green-700 text-white px-3 py-1.5 rounded text-sm font-medium'),
                        html.Button('卖一半', id='sg-sell-half',
                                    className='bg-green-50 hover:bg-green-100 text-green-700 border border-green-200 px-2.5 py-1.5 rounded text-xs'),
                        html.Button('全部清仓', id='sg-sell-all',
                                    className='bg-green-50 hover:bg-green-100 text-green-700 border border-green-200 px-2.5 py-1.5 rounded text-xs'),
                    ], className='flex flex-wrap items-center gap-2 mt-2'),
                    html.Div([
                        html.Button([html.I(className='fas fa-forward mr-1'), '下一根'],
                                    id='sg-next-btn',
                                    className='bg-blue-600 hover:bg-blue-700 text-white px-4 py-1.5 rounded text-sm font-medium'),
                        html.Button('快进5根', id='sg-next5-btn',
                                    className='bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200 px-3 py-1.5 rounded text-xs'),
                        html.Button([html.I(className='fas fa-play mr-1'), '自动播放'],
                                    id='sg-autoplay-btn',
                                    className='bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 px-3 py-1.5 rounded text-xs'),
                        dcc.Dropdown(id='sg-autoplay-speed',
                                     options=[{'label': '每秒1根', 'value': 1000},
                                              {'label': '每2秒1根', 'value': 2000},
                                              {'label': '每4秒1根', 'value': 4000}],
                                     value=2000, clearable=False,
                                     className='w-28 text-xs'),
                        html.Button([html.I(className='fas fa-flag-checkered mr-1'), '结算'],
                                    id='sg-finish-btn',
                                    className='bg-amber-500 hover:bg-amber-600 text-white px-4 py-1.5 rounded text-sm font-medium ml-auto'),
                    ], className='flex items-center gap-2 mt-3 pt-2 border-t border-gray-100'),
                    dcc.Interval(id='sg-autoplay-timer', interval=2000, n_intervals=0, disabled=True),
                ], className='bg-white rounded-xl border border-gray-100 shadow-sm p-3 mt-3'),

                # 结算面板
                html.Div(id='sg-scoreboard', className='mt-3'),

                # 资产曲线 + 交易记录
                html.Div([
                    html.Div([
                        dcc.Graph(id='sg-equity-graph', figure=go.Figure(),
                                  config={'displayModeBar': False},
                                  style={'height': '220px'}),
                    ], className='lg:w-1/2'),
                    html.Div([
                        html.Div('📜 交易记录（最近12笔）',
                                 className='text-xs font-bold text-gray-500 mb-1'),
                        html.Div(id='sg-trades-table',
                                 children=html.Div('—', className='text-xs text-gray-400'),
                                 className='overflow-y-auto', style={'maxHeight': '200px'}),
                    ], className='lg:w-1/2 lg:pl-4'),
                ], className='flex flex-col lg:flex-row mt-3 bg-white rounded-xl border border-gray-100 shadow-sm p-3'),

                # 排行榜 / 我的战绩
                html.Div([
                    html.Div([
                        html.Div('🏆 排行榜 / 战绩', className='text-xs font-bold text-gray-500'),
                        dcc.Tabs(id='sg-board-tabs', value='top',
                                 children=[dcc.Tab(label='🏆 排行榜', value='top'),
                                           dcc.Tab(label='📜 我的战绩', value='my')],
                                 className='text-xs'),
                        html.Button([html.I(className='fas fa-sync-alt mr-1'), '刷新'],
                                    id='sg-board-refresh-btn',
                                    className='text-xs text-blue-600 hover:text-blue-800 bg-transparent border-0 cursor-pointer ml-auto'),
                    ], className='flex items-center gap-3 mb-2'),
                    html.Div(id='sg-board-content',
                             children=build_board_container('top'),
                             className='overflow-y-auto', style={'maxHeight': '280px'}),
                ], className='bg-white rounded-xl border border-gray-100 shadow-sm p-3 mt-3'),

                dcc.Store(id='sg-game-store', data=None),
            ], className='w-full md:w-3/4 lg:w-4/5 pl-4'),
        ], className='flex flex-col md:flex-row'),
    ], className='min-h-screen bg-gray-50 p-4')


# ===================== 回调注册 =====================

def register_stock_game_callbacks(app):
    """注册模拟炒股页面回调"""

    @app.callback(
        Output('sg-stock-dropdown', 'options'),
        Input('sg-stock-dropdown', 'search_value')
    )
    def load_sg_stock_options(search_value):
        from .database_manager import get_all_stock_codes
        try:
            codes = get_all_stock_codes()
            return [{'label': f"{c[0]} {c[1] if len(c) > 1 and c[1] else ''}".strip(),
                     'value': c[0]} for c in codes]
        except Exception as e:
            logging.warning(f'加载股票选项失败: {e}')
            return []

    @app.callback(
        [Output('sg-game-store', 'data'),
         Output('sg-chart', 'figure'),
         Output('sg-equity-graph', 'figure'),
         Output('sg-status-cards', 'children'),
         Output('sg-params-panel', 'children'),
         Output('sg-trades-table', 'children'),
         Output('sg-error', 'children'),
         Output('sg-scoreboard', 'children'),
         Output('sg-autoplay-timer', 'disabled'),
         Output('sg-autoplay-btn', 'children')],
        [Input('sg-start-btn', 'n_clicks'),
         Input('sg-buy-btn', 'n_clicks'),
         Input('sg-buy-q1', 'n_clicks'),
         Input('sg-buy-q2', 'n_clicks'),
         Input('sg-buy-q3', 'n_clicks'),
         Input('sg-sell-btn', 'n_clicks'),
         Input('sg-sell-half', 'n_clicks'),
         Input('sg-sell-all', 'n_clicks'),
         Input('sg-next-btn', 'n_clicks'),
         Input('sg-next5-btn', 'n_clicks'),
         Input('sg-finish-btn', 'n_clicks'),
         Input('sg-autoplay-timer', 'n_intervals'),
         Input('sg-autoplay-btn', 'n_clicks')],
        [State('sg-game-store', 'data'),
         State('sg-stock-dropdown', 'value'),
         State('sg-stock-input', 'value'),
         State('sg-data-source', 'value'),
         State('sg-freq', 'value'),
         State('sg-play-bars', 'value'),
         State('sg-initial-cash', 'value'),
         State('sg-buy-shares', 'value'),
         State('sg-sell-shares', 'value')],
        prevent_initial_call=True
    )
    def sg_game_action(_start, _buy, _bq1, _bq2, _bq3, _sell, _shalf, _sall,
                       _next, _next5, _finish, _tick, _auto,
                       state, dropdown_code, input_code, data_source, freq,
                       play_bars, initial_cash, buy_shares, sell_shares):
        ctx = dash.callback_context
        trig = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
        empty_fig = _empty_figure('等待开始游戏…')
        blank = html.Div('—', className='text-xs text-gray-400')

        # ---- 开始新游戏 ----
        if trig == 'sg-start-btn':
            code = dropdown_code or (input_code.strip() if input_code else '')
            if not code:
                return (None, empty_fig, go.Figure(), blank, blank, blank,
                        '请先选择或输入股票代码', '', True, _autoplay_label(False))
            # 代码标准化（与趋势分析页一致）
            if code.isdigit():
                code = f'{code}.SH' if code.startswith(("5", "6", "9")) else f'{code}.SZ'
            elif '.' not in code and len(code) == 6:
                code = f'{code}.SH' if code.startswith(("5", "6", "9")) else f'{code}.SZ'
            if data_source:
                set_data_source(data_source)
            play_bars = int(play_bars) if play_bars else 120
            initial_cash = float(initial_cash) if initial_cash else 100000.0
            try:
                game_id, err = _create_game(code, freq or 'daily', play_bars, data_source)
            except Exception as e:
                logger.error(f'创建游戏失败 {code}: {e}', exc_info=True)
                return (None, empty_fig, go.Figure(), blank, blank, blank,
                        f'游戏创建失败: {e}', '', True, _autoplay_label(False))
            if err:
                return (None, empty_fig, go.Figure(), blank, blank, blank, err, '',
                        True, _autoplay_label(False))
            g = _GAMES[game_id]
            state = {'game_id': game_id, 'started': True, 'finished': False,
                     'autoplay': False, 'idx': 0, 'cash': initial_cash, 'lots': [],
                     'trades': [], 'initial_cash': initial_cash,
                     'config': {'code': code, 'freq': freq or 'daily',
                                'play_bars': play_bars, 'name': g['name']}}
            return _finish_render(state, g, gi=0, error='')

        # ---- 其余操作都需要已开始的游戏 ----
        if not state or not state.get('started'):
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update, '请先在左侧开始新游戏', dash.no_update,
                    dash.no_update, dash.no_update)
        g = _GAMES.get(state['game_id'])
        if g is None:
            return (None, empty_fig, go.Figure(), blank, blank, blank,
                    '游戏行情缓存已失效（服务已重启），请重新开始游戏', '',
                    True, _autoplay_label(False))
        closes, dates = _game_bars(g)
        gi = int(state['idx'])

        if state.get('finished'):
            return (dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                    dash.no_update, dash.no_update,
                    '本局已结算。点左侧"开始新游戏"再来一局', dash.no_update,
                    True, dash.no_update)

        error = ''
        price = closes[gi]
        cur_date = dates[gi][:16]

        if trig in ('sg-buy-btn', 'sg-buy-q1', 'sg-buy-q2', 'sg-buy-q3'):
            if trig == 'sg-buy-btn':
                shares = int(buy_shares or 0)
            else:
                ratio = {'sg-buy-q1': 0.25, 'sg-buy-q2': 0.5, 'sg-buy-q3': 1.0}[trig]
                shares = int(state['cash'] * ratio // (price * 100)) * 100
            if shares <= 0:
                error = '资金不足一手（100股），换个更小的仓位试试'
            else:
                new_state, err = apply_buy(state, shares, price, gi, cur_date)
                if new_state is not None:
                    state = new_state
                error = err
        elif trig in ('sg-sell-btn', 'sg-sell-half', 'sg-sell-all'):
            avail = available_shares(state['lots'], gi)
            if trig == 'sg-sell-btn':
                shares = int(sell_shares or 0)
            elif trig == 'sg-sell-half':
                shares = (avail // 2 // 100) * 100
            else:
                shares = avail
            if shares <= 0:
                error = '没有可卖持仓（T+1: 当根买入下一根才可卖）'
            else:
                new_state, err = apply_sell(state, shares, price, gi, cur_date)
                if new_state is not None:
                    state = new_state
                error = err
        elif trig in ('sg-next-btn', 'sg-next5-btn', 'sg-autoplay-timer'):
            if trig == 'sg-autoplay-timer' and not state.get('autoplay'):
                return _finish_render(state, g, gi=gi, error='')  # 播放已关，忽略残余tick
            step = 5 if trig == 'sg-next5-btn' else 1
            state = _copy_state(state)
            if gi >= g['play_len'] - 1:
                state['finished'] = True  # 已在最后一根，再推进即结算
            else:
                state['idx'] = min(gi + step, g['play_len'] - 1)
        elif trig == 'sg-finish-btn':
            state = _copy_state(state)
            state['finished'] = True
        elif trig == 'sg-autoplay-btn':
            state = _copy_state(state)
            state['autoplay'] = not state.get('autoplay')

        if state.get('finished') and not state.get('saved'):
            if _save_record(state, g):
                state['saved'] = True

        return _finish_render(state, g, gi=int(state['idx']), error=error)

    def _autoplay_label(on):
        """自动播放按钮文案"""
        return ([html.I(className='fas fa-play mr-1'), '自动播放'] if not on
                else [html.I(className='fas fa-pause mr-1'), '暂停'])

    def _finish_render(state, g, gi, error):
        """统一收尾: 渲染组件 + 自动播放器状态"""
        result = _render(state, g, gi, error)
        autoplay_on = bool(state.get('autoplay')) and not state.get('finished')
        return result + (not autoplay_on, _autoplay_label(bool(state.get('autoplay'))))

    def _render(state, g, gi, error):
        """根据状态重建展示组件；K线图仅在推进到新bar时重建，
        买卖/结算等不改变K线进度的操作不重发整图，减少卡顿"""
        if state.get('finished'):
            scoreboard = build_scoreboard(state, g, g['play_len'] - 1)
        else:
            scoreboard = ''
        if state.get('finished'):
            # 结算按最后一根收盘价
            final_state = _copy_state(state)
            final_state['idx'] = g['play_len'] - 1
            cards = build_status_cards(final_state, g, g['play_len'] - 1)
            params = build_params_panel(g, g['play_len'] - 1)
            eq = build_equity_figure(final_state, g, g['play_len'] - 1)
        else:
            cards = build_status_cards(state, g, gi)
            params = build_params_panel(g, gi)
            eq = build_equity_figure(state, g, gi)
        if gi == state.get('chart_gi'):
            fig = dash.no_update  # K线进度未变，图保持原样
        else:
            fig = build_game_figure(g, min(gi, g['play_len'] - 1))
            state['chart_gi'] = gi
        return (state, fig, eq, cards, params,
                build_trades_table(state), error, scoreboard)

    @app.callback(
        Output('sg-autoplay-timer', 'interval'),
        Input('sg-autoplay-speed', 'value'),
        prevent_initial_call=True
    )
    def update_autoplay_speed(speed):
        try:
            return int(speed) if speed else 2000
        except (TypeError, ValueError):
            return 2000

    @app.callback(
        Output('sg-board-content', 'children'),
        [Input('sg-board-tabs', 'value'),
         Input('sg-board-refresh-btn', 'n_clicks'),
         Input('sg-game-store', 'data')],
        prevent_initial_call=True
    )
    def refresh_board(tab, _refresh, _state):
        """切换Tab / 手动刷新 / 结算(store变化)时重绘战绩区。
        游戏进行中每次点击也会更新 store，但战绩并未变化，
        直接跳过，避免逐根点击都重查数据库造成卡顿。"""
        try:
            trig = dash.ctx.triggered_id  # Dash 2.4+；兼容各版本
        except Exception:
            trig = None
        if trig == 'sg-game-store' and not (_state and _state.get('finished')):
            return dash.no_update
        return build_board_container(tab or 'top')
