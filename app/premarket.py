"""
每日盘前提醒：开盘前的市场速览，聚合以下真实数据（无编造，数据不可得时明确降级）。

数据源策略：baostock 为主（免积分、无限频），tushare 仅补充 baostock 没有的数据。

1. 交易日历    - baostock trade_dates（主）；tushare trade_cal（备）；周末推断（兜底）
2. 隔夜海外    - 腾讯行情（主）→ 东方财富行情（备）→ tushare index_daily（兜底）；
                 全球指数 baostock 不覆盖
3. A股昨日复盘 - baostock 日线：主要指数涨跌幅、两市成交额及增减
4. 今日新股    - tushare new_share（baostock 无此数据，限频时降级缺席）
5. 自选股异动  - 数据库自选分组 + baostock 日线计算：大涨大跌/放量/20日新高新低
6. 持仓关注    - 交易日志持仓股 + baostock 日线：昨日涨跌与持仓浮盈

缓存：Redis 键 premarket:{date}（1小时），无 Redis 时退化为进程内缓存。
"""
import logging
import time
from datetime import datetime, timedelta

import dash
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd

from .baostock_fetcher import fetch_kline_data, fetch_trade_dates
from .data_manager import pro, _rate_limit, redis_client
from .database_manager import (get_stock_watchlist_map, get_trade_entries,
                               get_current_user_id)
from .trading_journal import compute_portfolio

logger = logging.getLogger(__name__)

# A股主要指数（与每日五面页一致）
_A_INDEX = [
    ('shanghai', '000001.SH', '上证指数'),
    ('shenzhen', '399001.SZ', '深证成指'),
    ('chinext', '399006.SZ', '创业板指'),
    ('hs300', '000300.SH', '沪深300'),
]

# 全球指数（tushare global index 代码 → 展示名）
_GLOBAL_INDEX = [
    ('DJI', '道琼斯'), ('SPX', '标普500'), ('NDX', '纳斯达克100'),
    ('HSI', '恒生指数'), ('N225', '日经225'),
]

# 各行情源的全球指数代码映射（标普500 腾讯代码为 usINX；日经225 腾讯无有效代码，
# 由东财/tushare 源补充）
_TENCENT_CODES = {'DJI': 'usDJI', 'SPX': 'usINX', 'NDX': 'usNDX',
                  'HSI': 'hkHSI', 'N225': 'jpN225'}

_CACHE_TTL = 3600  # 盘前数据缓存 1 小时
_memory_cache = {}  # Redis 不可用时的进程内兜底缓存 {key: (ts, payload)}

_WATCHLIST_MAX = 20   # 自选股异动最多检查的股票数
_HOLDINGS_MAX = 10    # 持仓关注最多显示的股票数
_NOTABLE_CHG = 3.0    # 自选股异动: 涨跌幅阈值 %
_NOTABLE_VOL = 2.0    # 自选股异动: 量比阈值


# ===================== 缓存 =====================

def _cache_get(key):
    if redis_client is not None:
        try:
            raw = redis_client.get(key)
            if raw:
                import json
                return json.loads(raw.decode('utf-8'))
        except Exception as e:
            logger.warning(f'盘前缓存读取失败: {e}')
    else:
        hit = _memory_cache.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            return hit[1]
    return None


def _cache_set(key, payload):
    if redis_client is not None:
        try:
            import json
            redis_client.setex(key, _CACHE_TTL, json.dumps(payload, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning(f'盘前缓存写入失败: {e}')
    _memory_cache[key] = (time.time(), payload)


# ===================== 各数据板块（全部真实数据源） =====================

def trading_day_info(today=None):
    """交易日历: {'is_open': True/False/None, 'next': 'YYYYMMDD', 'source': str}

    降级链: baostock trade_dates（免积分）→ tushare trade_cal → 周末推断
    （周末必休市；工作日无法排除节假日故不妄断，返回 is_open=None）。
    """
    today = today or datetime.now().strftime('%Y%m%d')
    horizon = (datetime.strptime(today, '%Y%m%d') + timedelta(days=20)).strftime('%Y%m%d')

    # 1) baostock（主）
    try:
        cal = fetch_trade_dates(today, horizon)
        if cal is not None and not cal.empty:
            opened = sorted(d.replace('-', '')
                            for d, flag in zip(cal['calendar_date'], cal['is_trading_day'])
                            if str(flag) == '1' and d.replace('-', '') >= today)
            is_open = bool(opened and opened[0] == today)
            next_open = next((d for d in opened if d > today), None)
            return {'is_open': is_open, 'next': next_open, 'today': today,
                    'source': 'baostock'}
    except Exception as e:
        logger.warning(f'baostock 交易日历失败: {e}')

    # 2) tushare（备）
    try:
        _rate_limit()
        df = pro.trade_cal(exchange='SSE', start_date=today, end_date=horizon)
        if df is not None and not df.empty:
            df = df.sort_values('cal_date')
            today_row = df[df['cal_date'] == today]
            is_open = bool(today_row.iloc[0]['is_open'] == 1) if len(today_row) else None
            opened = df[(df['cal_date'] > today) & (df['is_open'] == 1)]
            next_open = str(opened.iloc[0]['cal_date']) if len(opened) else None
            return {'is_open': is_open, 'next': next_open, 'today': today,
                    'source': 'tushare'}
    except Exception as e:
        logger.warning(f'tushare 交易日历失败(退化为周末推断): {e}')

    # 3) 周末推断（兜底）
    weekday = datetime.strptime(today, '%Y%m%d').weekday()
    if weekday >= 5:  # 周六/周日必休市
        return {'is_open': False, 'next': None, 'today': today, 'source': 'weekend'}
    return None


def _global_via_tencent():
    """腾讯行情接口(qt.gtimg.cn): 单次请求拿全部全球指数，GBK 编码"""
    import requests
    q = ','.join(_TENCENT_CODES[code] for code, _ in _GLOBAL_INDEX)
    resp = requests.get(f'https://qt.gtimg.cn/q={q}', timeout=8,
                        headers={'User-Agent': 'Mozilla/5.0'})
    text = resp.content.decode('gbk', errors='replace')
    parsed = {}
    for line in text.strip().split(';'):
        line = line.strip()
        if not line or '=' not in line:
            continue
        key, _, val = line.partition('=')
        key = key.strip().lstrip('v_')
        parsed[key] = val.strip('"').split('~')
    out = []
    for code, name in _GLOBAL_INDEX:
        p = parsed.get(_TENCENT_CODES[code])
        if not p or len(p) < 5:
            continue
        try:
            close, prev = float(p[3]), float(p[4])
            if close <= 0 or prev <= 0:
                continue
            out.append({'name': name, 'close': round(close, 2),
                        'chg': round((close / prev - 1) * 100, 2)})
        except (ValueError, IndexError):
            continue
    if not out:
        raise RuntimeError('腾讯行情未返回有效数据')
    return out


def _global_via_eastmoney():
    """东方财富行情接口: 一次请求拿全部全球指数（最新价/涨跌幅），带一次重试"""
    import requests
    secids = ','.join(f'100.{code}' for code, _ in _GLOBAL_INDEX)
    url = 'https://push2.eastmoney.com/api/qt/ulist.np/get'
    headers = {'User-Agent': 'Mozilla/5.0'}
    last_err = None
    for _ in range(2):  # 偶发连接重置重试一次
        try:
            resp = requests.get(url, params={'secids': secids, 'fields': 'f2,f3,f14',
                                             'fltt': 2},
                                headers=headers, timeout=8)
            data = resp.json()
            diff = (data.get('data') or {}).get('diff') or []
            out = []
            for item in diff:
                price, chg = item.get('f2'), item.get('f3')
                if price in ('-', None) or chg in ('-', None):
                    continue
                out.append({'name': item.get('f14', item.get('f12', '')),
                            'close': round(float(price), 2), 'chg': round(float(chg), 2)})
            if out:
                return out
        except Exception as e:
            last_err = e
    raise RuntimeError(f'东方财富全球指数获取失败: {last_err}')


def global_overnight():
    """隔夜海外市场: 全球主要指数最新收盘与涨跌幅。

    数据源链: 腾讯行情 → 东方财富行情 → tushare index_daily（限频严格，仅兜底）。
    """
    for provider in (_global_via_tencent, _global_via_eastmoney):
        try:
            out = provider()
            if out:
                return out
        except Exception as e:
            logger.warning(f'{provider.__name__} 失败，尝试下一数据源: {e}')
    out = []
    for code, name in _GLOBAL_INDEX:
        try:
            _rate_limit()
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now() - timedelta(days=15)).strftime('%Y%m%d')
            df = pro.index_daily(ts_code=code, start_date=start, end_date=end)
            if df is None or len(df) < 2:
                continue
            df = df.sort_values('trade_date')
            close, prev = float(df['close'].iloc[-1]), float(df['close'].iloc[-2])
            chg = (close / prev - 1) * 100
            out.append({'name': name, 'close': round(close, 2), 'chg': round(chg, 2),
                        'date': str(df['trade_date'].iloc[-1])})
        except Exception as e:
            logger.warning(f'全球指数 {name} 获取失败: {e}')
    return out


def _amount_to_yi(v):
    """成交额统一换算为亿元。

    数据源单位不一致: tushare daily 的 amount 为千元，baostock 为元。
    A股两市单日成交额恒在千亿~万亿量级，按数值量级自动判别单位。
    """
    if v is None or v <= 0 or pd.isna(v):
        return None
    if v >= 1e10:    # 单位: 元（A股单市场日成交额 ≥ 千亿元级 = 1e11 元；阈值取 1e10 留裕量）
        return v / 1e8
    if v >= 1e5:     # 单位: 千元（tushare daily，1万亿 = 1e9 千元）
        return v / 1e5
    return None      # 量级异常，宁缺毋假


def a_share_recap():
    """A股昨日复盘: 主要指数涨跌幅 + 两市成交额变化（直连 baostock，免积分）

    baostock 的 amount 单位为元；指数代码 tushare daily 接口查不到，
    直连 baostock 同时避免了先打一次注定失败的 tushare 请求。
    """
    today = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
    indices = []
    sh_now = sh_prev = sz_now = sz_prev = None
    for key, code, name in _A_INDEX:
        try:
            df = fetch_kline_data(code, start, today, adjustflag='3', freq='daily')
            if df is None or len(df) < 2:
                continue
            close = float(df['close'].iloc[-1])
            prev = float(df['close'].iloc[-2])
            indices.append({'name': name, 'close': round(close, 2),
                            'chg': round((close / prev - 1) * 100, 2),
                            'date': str(df['trade_date'].iloc[-1])[:10]})
            amount_now = _amount_to_yi(float(df['amount'].iloc[-1]))
            amount_prev = _amount_to_yi(float(df['amount'].iloc[-2]))
            if key == 'shanghai':
                sh_now, sh_prev = amount_now, amount_prev
            if key == 'shenzhen':
                sz_now, sz_prev = amount_now, amount_prev
        except Exception as e:
            logger.warning(f'A股指数 {name} 获取失败: {e}')

    recap = {'indices': indices, 'last_date': indices[0]['date'] if indices else None}
    if sh_now is not None and sz_now is not None:
        total = sh_now + sz_now
        recap['amount_total'] = round(total, 0)  # 亿元
        prev_total = (sh_prev + sz_prev) if (sh_prev is not None and sz_prev is not None) else None
        recap['amount_chg'] = round((total / prev_total - 1) * 100, 1) if prev_total else None
    return recap


def new_shares_today():
    """今日新股申购（tushare new_share），无申购返回空列表"""
    today = datetime.now().strftime('%Y%m%d')
    try:
        _rate_limit()
        df = pro.new_share(start_date=today, end_date=today)
        if df is None or df.empty:
            return []
        return [{'name': str(r['name']), 'code': str(r['subtype_code'])}
                for _, r in df.iterrows()]
    except Exception as e:
        logger.warning(f'新股申购获取失败: {e}')
        return None  # None=获取失败, []=确认无


def _stock_moves(code, start, today):
    """单只股票的异动信号: 涨跌幅/量比/20日新高新低（直连 baostock，前复权）"""
    df = fetch_kline_data(code, start, today, adjustflag='2', freq='daily')
    if df is None or len(df) < 21:
        return None
    df = df.sort_values('trade_date').reset_index(drop=True)
    close = float(df['close'].iloc[-1])
    prev = float(df['close'].iloc[-2])
    chg = (close / prev - 1) * 100
    vol = float(df['vol'].iloc[-1])
    vol_ma5 = float(df['vol'].iloc[-6:-1].mean()) if len(df) >= 6 else 0.0
    vol_ratio = vol / vol_ma5 if vol_ma5 > 0 else None
    # 创20日新高/新低: 收盘价突破之前20根的收盘价区间（不含当日自身）
    prev_closes = df['close'].iloc[-21:-1]
    high20 = float(prev_closes.max())
    low20 = float(prev_closes.min())
    signals = []
    if abs(chg) >= _NOTABLE_CHG:
        signals.append('大涨' if chg > 0 else '大跌')
    if vol_ratio is not None and vol_ratio >= _NOTABLE_VOL:
        signals.append(f'放量{vol_ratio:.1f}倍')
    if close > high20:
        signals.append('创20日新高')
    if close < low20:
        signals.append('创20日新低')
    return {'chg': round(chg, 2), 'close': round(close, 2),
            'vol_ratio': round(vol_ratio, 2) if vol_ratio else None,
            'signals': signals, 'date': str(df['trade_date'].iloc[-1])[:10]}


def watchlist_moves():
    """自选股异动（仅当前登录用户的自选分组）"""
    try:
        wl_map = get_stock_watchlist_map()
    except Exception as e:
        logger.warning(f'自选分组获取失败: {e}')
        return None
    if not wl_map:
        return []
    today = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=130)).strftime('%Y%m%d')
    notable = []
    for code in list(wl_map.keys())[:_WATCHLIST_MAX]:
        try:
            m = _stock_moves(code, start, today)
            if m and m['signals']:
                m['code'] = code
                notable.append(m)
        except Exception as e:
            logger.warning(f'自选股 {code} 行情获取失败: {e}')
    return notable


def holdings_watch():
    """持仓关注（交易日志中持仓中的股票: 昨日涨跌 + 持仓浮盈）"""
    try:
        entries = get_trade_entries()
        if not entries:
            return []
        holdings = compute_portfolio(entries).get('holdings', [])
    except Exception as e:
        logger.warning(f'持仓获取失败: {e}')
        return None
    if not holdings:
        return []
    today = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    out = []
    for pos in holdings[:_HOLDINGS_MAX]:
        code = pos['stock_code']
        try:
            df = fetch_kline_data(code, start, today, adjustflag='2', freq='daily')
            if df is None or len(df) < 2:
                continue
            close = float(df['close'].iloc[-1])
            prev = float(df['close'].iloc[-2])
            chg = (close / prev - 1) * 100
            pnl = (close - pos['avg_cost']) * pos['quantity']
            pnl_pct = (close / pos['avg_cost'] - 1) * 100 if pos['avg_cost'] else 0.0
            out.append({'code': code, 'name': pos['stock_name'],
                        'chg': round(chg, 2), 'close': round(close, 2),
                        'qty': pos['quantity'], 'pnl': round(pnl, 0),
                        'pnl_pct': round(pnl_pct, 2)})
        except Exception as e:
            logger.warning(f'持仓 {code} 行情获取失败: {e}')
    return out


# ===================== 汇总 =====================

def _personal_sections():
    """个性化板块（自选股异动 + 持仓关注），各板块独立降级"""
    personal = {}
    try:
        personal['watchlist'] = watchlist_moves()
    except Exception as e:
        logger.warning(f'自选股异动板块失败: {e}')
    try:
        personal['holdings'] = holdings_watch()
    except Exception as e:
        logger.warning(f'持仓关注板块失败: {e}')
    return personal


def fetch_premarket_briefing(data_source=None, use_cache=True):
    """聚合盘前提醒数据。返回 dict（各板块独立降级，失败的板块缺失）

    数据隔离: 公共板块（交易日历/海外/复盘/新股）全局缓存；
    个性化板块（自选股异动/持仓关注）按 user_id 独立缓存，
    未登录时不包含个性化数据。
    data_source 参数已废弃（改用 baostock 为主源），保留以兼容旧调用。
    """
    today = datetime.now().strftime('%Y%m%d')

    # ── 公共板块（全局共享缓存）──
    public_key = f'premarket:{today}'
    brief = _cache_get(public_key) if use_cache else None
    if brief is None:
        brief = {'date': today, 'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                 'from_cache': False}

        cal = trading_day_info()
        if cal:
            brief['is_open'] = cal['is_open']
            brief['next_open'] = cal.get('next')
            brief['cal_source'] = cal.get('source', 'baostock')

        try:
            brief['global'] = global_overnight()
        except Exception as e:
            logger.warning(f'海外市场板块失败: {e}')

        try:
            brief['recap'] = a_share_recap()
        except Exception as e:
            logger.warning(f'A股复盘板块失败: {e}')

        ns = new_shares_today()
        if ns is not None:
            brief['new_shares'] = ns

        _cache_set(public_key, brief)
    else:
        brief = {**brief, 'from_cache': True}  # 不改写缓存对象（避免别名污染）

    # ── 个性化板块（按 user_id 隔离，未登录不含）──
    uid = get_current_user_id()
    if uid is not None:
        personal_key = f'premarket:{today}:u:{uid}'
        personal = _cache_get(personal_key) if use_cache else None
        if personal is None:
            personal = _personal_sections()
            _cache_set(personal_key, personal)
        brief.update(personal)

    return brief


# ===================== 展示 =====================

def _pct_color(v):
    return '#ef4444' if v > 0 else ('#22c55e' if v < 0 else '#64748b')


def _fmt_pct(v):
    return f'{v:+.2f}%'


def _index_chip(name, close, chg, date=None):
    sub = f' · {date}' if date else ''
    return html.Div([
        html.Div(name, className='text-[10px] text-gray-400'),
        html.Div(f'{close:,.2f}', className='text-sm font-bold'),
        html.Div(_fmt_pct(chg), className='text-xs font-bold',
                 style={'color': _pct_color(chg)}),
        html.Div(sub, className='text-[9px] text-gray-300'),
    ], className='bg-gray-50 rounded-lg px-3 py-1.5 min-w-[92px] text-center')


def build_premarket_card(brief):
    """根据盘前数据 dict 构建展示卡片"""
    if not brief:
        return html.Div([
            html.Span('🌅 盘前提醒', className='text-base font-bold text-gray-700'),
            html.Div('数据加载失败，请点右侧"刷新"重试', className='text-sm text-gray-400 mt-1'),
        ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-4')

    blocks = []

    # ── 交易日历 ──
    if brief.get('is_open') is False:
        src = brief.get('cal_source')
        label = '🏖️ 今日休市（周末）' if src == 'weekend' else '🏖️ 今日休市'
        blocks.append(html.Div([
            html.Span(label, className='text-sm font-bold text-gray-600'),
            html.Span(f" 下一交易日: {brief.get('next_open', '待定')}",
                      className='text-xs text-gray-400 ml-2'),
        ], className='bg-blue-50 border border-blue-100 rounded-lg px-3 py-2 mb-2'))
    elif brief.get('is_open') is True:
        blocks.append(html.Div([
            html.Span('🔔 今日为交易日', className='text-sm font-bold text-green-700'),
        ], className='bg-green-50 border border-green-100 rounded-lg px-3 py-2 mb-2'))

    # ── 隔夜海外 ──
    glob = brief.get('global') or []
    if glob:
        blocks.append(html.Div([
            html.Div('🌍 隔夜海外市场', className='text-xs font-bold text-gray-500 mb-1'),
            html.Div([_index_chip(g['name'], g['close'], g['chg'], g.get('date')) for g in glob],
                     className='flex flex-wrap gap-2'),
        ], className='mb-3'))

    # ── A股昨日复盘 ──
    recap = brief.get('recap') or {}
    if recap.get('indices'):
        rows = [_index_chip(i['name'], i['close'], i['chg']) for i in recap['indices']]
        amount_line = None
        if recap.get('amount_total'):
            chg_txt = ''
            if recap.get('amount_chg') is not None:
                chg_txt = f"（较上一日{_fmt_pct(recap['amount_chg'])}）"
            amount_line = html.Div(
                f"两市成交额 ≈ {recap['amount_total']:,.0f} 亿元{chg_txt}",
                className='text-xs text-gray-500 mt-1')
        blocks.append(html.Div([
            html.Div(f"🇨🇳 A股复盘（{recap.get('last_date', '')}）",
                     className='text-xs font-bold text-gray-500 mb-1'),
            html.Div(rows, className='flex flex-wrap gap-2'),
            amount_line,
        ], className='mb-3'))

    # ── 自选股异动 ──
    wl = brief.get('watchlist')
    if wl is not None:
        items = []
        if wl:
            for w in wl:
                sig = ' · '.join(w['signals'])
                items.append(html.Div([
                    html.Span(w['code'], className='text-xs font-mono font-bold mr-2'),
                    html.Span(_fmt_pct(w['chg']), className='text-xs font-bold mr-2',
                              style={'color': _pct_color(w['chg'])}),
                    html.Span(sig, className='text-xs text-gray-600'),
                    html.Span(f"收盘 {w['close']:.2f}", className='text-[10px] text-gray-400 ml-auto'),
                ], className='flex items-center bg-gray-50 rounded px-2 py-1 mb-1'))
            body = items
        else:
            body = [html.Div('自选股昨日无异动（或未设置自选股）',
                             className='text-xs text-gray-400 italic')]
        blocks.append(html.Div(
            [html.Div('⭐ 自选股异动', className='text-xs font-bold text-gray-500 mb-1')] + body,
            className='mb-3'))

    # ── 持仓关注 ──
    hd = brief.get('holdings')
    if hd:
        rows = []
        for h in hd:
            rows.append(html.Div([
                html.Span(f"{h['code']} {h['name']}".strip(), className='text-xs font-bold mr-2'),
                html.Span(f"昨收 {h['close']:.2f}（{_fmt_pct(h['chg'])}）", className='text-xs mr-2',
                          style={'color': _pct_color(h['chg'])}),
                html.Span(f"持仓{h['qty']}股", className='text-xs text-gray-500 mr-2'),
                html.Span(f"浮盈 {h['pnl']:+,.0f}元（{_fmt_pct(h['pnl_pct'])}）",
                          className='text-xs font-bold', style={'color': _pct_color(h['pnl'])}),
            ], className='flex flex-wrap items-center bg-gray-50 rounded px-2 py-1 mb-1'))
        blocks.append(html.Div(
            [html.Div('💼 持仓关注', className='text-xs font-bold text-gray-500 mb-1')] + rows,
            className='mb-3'))

    # ── 今日新股 ──
    ns = brief.get('new_shares')
    if ns is not None:
        if ns:
            txt = '、'.join(f"{n['name']}({n['code']})" for n in ns)
            icon, color = '🆕', '#b45309'
        else:
            txt, icon, color = '今日无新股申购', '🆕', '#64748b'
        blocks.append(html.Div([
            html.Span(f'{icon} 今日新股申购: ', className='text-xs font-bold text-gray-500'),
            html.Span(txt, className='text-xs font-semibold', style={'color': color}),
        ], className='mb-1'))

    header = html.Div([
        html.Span('🌅 盘前提醒', className='text-base font-bold text-gray-800'),
        html.Span(f" {brief.get('generated_at', '')} 生成", className='text-[10px] text-gray-400 ml-2'),
        html.Span('（缓存）' if brief.get('from_cache') else '', className='text-[10px] text-gray-300'),
    ], className='flex items-baseline mb-2')

    return html.Div([header] + blocks,
                    className='bg-white rounded-xl shadow-sm border border-blue-100 p-4')


def register_premarket_callbacks(app):
    """盘前提醒回调：页面加载即生成 + 手动刷新"""

    @app.callback(
        Output('mo-premarket-card', 'children'),
        Input('mo-premarket-refresh-btn', 'n_clicks'),
        prevent_initial_call=False
    )
    def refresh_premarket(n_clicks):
        use_cache = not n_clicks  # 手动刷新绕过缓存
        try:
            brief = fetch_premarket_briefing(use_cache=use_cache)
        except Exception as e:
            logger.error(f'盘前提醒生成失败: {e}', exc_info=True)
            brief = None
        return build_premarket_card(brief)
