"""
板块资金强度页面 — 数据源: baostock(行业分类) + tushare(全市场日线)
资金强度 = 主力资金(估算) / 成交额 × 100
主力资金估算: CLV 量价资金流模型
  个股CLV = 2×(收盘-最低)/(最高-最低)-1 (一字板按涨跌方向取±1)
  板块主力净流入(估) = Σ(CLV×成交额) × 0.15 (按主力成交占比折算量级)
  baostock/tushare 均无逐笔大单数据, 故用量价模型估算
分级规则: >=3 抢筹 | 1~3 建仓 | -1~1 洗盘 | <=-1 出货
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import time
import json
import os
import logging
import threading
from datetime import datetime, timedelta
from collections import Counter

from .data_manager import pro, _rate_limit
from .paths import cache_path

# ===================== 常量 =====================
STATE_ALL = '全部'
STATE_QC, STATE_JC, STATE_XP, STATE_CH = '抢筹', '建仓', '洗盘', '出货'
STATES = [STATE_QC, STATE_JC, STATE_XP, STATE_CH]

# 状态样式（A股习惯: 红涨绿跌）
STATE_STYLE = {
    STATE_QC: ('bg-red-100 text-red-700 border-red-200', '#dc2626', 'fa-fire'),
    STATE_JC: ('bg-orange-100 text-orange-700 border-orange-200', '#ea580c', 'fa-cart-plus'),
    STATE_XP: ('bg-slate-100 text-slate-600 border-slate-200', '#64748b', 'fa-arrows-rotate'),
    STATE_CH: ('bg-green-100 text-green-700 border-green-200', '#16a34a', 'fa-truck-fast'),
}

SORT_OPTIONS = [
    {'label': '资金强度 降序', 'value': 'strength_desc'},
    {'label': '资金强度 升序', 'value': 'strength_asc'},
    {'label': '主力净流入 降序', 'value': 'main_desc'},
    {'label': '成交额 降序', 'value': 'amount_desc'},
    {'label': '涨跌幅 降序', 'value': 'pct_desc'},
]

MAX_TABLE_ROWS = 150    # 表格最多渲染行数
TREND_DAYS = 10         # 趋势图回溯交易日数
# 主力资金折算系数: CLV 为全市场收盘位置资金流(量级±40), 按主力成交占比约15%折算,
# 使资金强度量级与主力净占比可比, 令 ±1/±3 分级阈值语义成立
MAIN_FLOW_SCALE = 0.15
_DAILY_TTL_TODAY = 300  # 当日数据缓存(秒), 历史日期永久缓存
_IND_OK_TTL = 7 * 86400   # 行业分类成功缓存
_IND_FAIL_TTL = 1800      # 行业分类失败重试间隔

# ===================== 缓存 ====================
_industry_map = None     # {ts_code: {'name','industry'}}
_industry_ts = 0.0
_industry_ok = False     # True=在线获取(7天TTL); False=离线缓存兜底(30分钟后重试在线)
_industry_source = ''
_industry_fetch_lock = threading.Lock()  # 防止并发重复抓取行业分类
_daily_cache = {}        # {yyyymmdd: (fetch_ts, DataFrame)}
_strength_cache = {}     # {yyyymmdd: (fetch_ts, rows)}


def _is_today(date_str):
    return date_str == datetime.now().strftime('%Y%m%d')


def _cache_get(cache, key):
    """历史日期永久缓存, 当日按 TTL"""
    hit = cache.get(key)
    if not hit:
        return None
    ts, val = hit
    if not _is_today(key) or (time.time() - ts) < _DAILY_TTL_TODAY:
        return val
    return None


def _cache_set(cache, key, val):
    cache[key] = (time.time(), val)


# ===================== 分级 =====================

def classify_strength(v):
    """资金强度分级: >=3 抢筹 | [1,3) 建仓 | (-1,1) 洗盘 | <=-1 出货"""
    if v >= 3:
        return STATE_QC
    if v >= 1:
        return STATE_JC
    if v > -1:
        return STATE_XP
    return STATE_CH


# ===================== 行业分类 (baostock 优先) =====================

INDUSTRY_CACHE_FILE = cache_path('sector_industry.json')


def _fetch_industry_baostock():
    """baostock 行业分类 → {ts_code: {'name','industry'}}; 失败返回 None
    复用 baostock_fetcher 的全局登录会话(不 logout, 避免破坏其他页面的会话),
    查询持有全局查询锁(baostock socket 非线程安全), 失败时重新登录后重试"""
    try:
        import baostock as bs
        from .baostock_fetcher import _ensure_login, _query_lock
    except ImportError:
        return None

    for attempt in range(2):
        try:
            if not _ensure_login():
                return None
            with _query_lock:
                rs = bs.query_stock_industry()
                if rs is None or rs.error_code != '0':
                    return None
                data = []
                while (rs.error_code == '0') and rs.next():
                    data.append(rs.get_row_data())
            result = {}
            for row in data:  # updateDate, code, code_name, industry, industryClassification
                bs_code, code_name, industry = row[1], row[2], row[3]
                if not industry or '.' not in bs_code:
                    continue
                # sh.600000 -> 600000.SH
                mkt, num = bs_code.split('.')
                result[f"{num}.{mkt.upper()}"] = {'name': code_name, 'industry': industry}
            if len(result) > 100:
                return result
            return None
        except Exception as e:
            logging.warning(f"[板块资金] baostock 行业分类第{attempt + 1}次获取失败: {e}")
            # 会话可能已损坏(socket 串扰/服务端断开), 重新登录后重试
            try:
                bs.login()
            except Exception:
                pass
    return None


def _fetch_industry_tushare():
    """tushare 行业分类兜底"""
    try:
        _rate_limit()
        df = pro.stock_basic(fields='ts_code,name,industry', list_status='L')
        if df is None or df.empty:
            return None
        result = {}
        for _, r in df.iterrows():
            if r.get('industry'):
                result[r['ts_code']] = {'name': r['name'], 'industry': r['industry']}
        return result if len(result) > 100 else None
    except Exception as e:
        logging.warning(f"[板块资金] tushare 行业分类获取失败: {e}")
        return None


def _load_industry_disk():
    """读取磁盘缓存的行业分类, 返回 (map, 保存时间戳, 来源) 或 (None, 0, '')"""
    try:
        if os.path.exists(INDUSTRY_CACHE_FILE):
            with open(INDUSTRY_CACHE_FILE, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            m = obj.get('map') or {}
            if len(m) > 100:
                return m, float(obj.get('ts') or 0), str(obj.get('src') or '')
    except Exception as e:
        logging.warning(f"[板块资金] 读取行业分类磁盘缓存失败: {e}")
    return None, 0.0, ''


def _save_industry_disk(m, src):
    """行业分类落盘 (跨重启容错, baostock 服务不稳定)"""
    try:
        os.makedirs(os.path.dirname(INDUSTRY_CACHE_FILE), exist_ok=True)
        with open(INDUSTRY_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'ts': time.time(), 'src': src, 'map': m}, f, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"[板块资金] 保存行业分类磁盘缓存失败: {e}")


def _industry_cache_valid():
    """内存缓存是否有效"""
    if _industry_map is None:
        return False
    ttl = _IND_OK_TTL if _industry_ok else _IND_FAIL_TTL
    return time.time() - _industry_ts < ttl


def get_industry_map(force=False):
    """获取 {ts_code: {'name','industry'}}
    优先级: 内存缓存 → 磁盘缓存(7天内) → baostock → tushare → 过期磁盘缓存(30天内)"""
    global _industry_map, _industry_ts, _industry_source, _industry_ok
    if not force and _industry_cache_valid():
        return _industry_map or None

    # 抓取锁 + 双检: 并发调用只触发一次在线抓取, 其余等待后直接读缓存
    with _industry_fetch_lock:
        if not force and _industry_cache_valid():
            return _industry_map or None

        # 1. 磁盘缓存 (7 天内有效)
        if not force:
            m, ts, src = _load_industry_disk()
            if m and time.time() - ts < _IND_OK_TTL:
                _industry_map, _industry_ts, _industry_ok = m, ts, True
                _industry_source = f'{src}(缓存)'
                return m

        # 2. 在线获取: baostock 优先, tushare 兜底
        m = _fetch_industry_baostock()
        src = 'baostock行业'
        if not m:
            m = _fetch_industry_tushare()
            src = 'tushare行业'
        if m:
            _industry_map, _industry_ts, _industry_ok, _industry_source = m, time.time(), True, src
            _save_industry_disk(m, src)
            logging.info(f"[板块资金] 行业分类加载成功: {len(m)} 只 ({src})")
            return m

        # 3. 在线均失败: 使用过期磁盘缓存兜底 (30 天内, 30 分钟后重试在线)
        m, ts, src = _load_industry_disk()
        if m and time.time() - ts < 30 * 86400:
            _industry_map, _industry_ts, _industry_ok = m, time.time(), False
            _industry_source = f'{src}(离线缓存)'
            logging.warning(f"[板块资金] 在线获取失败, 使用过期磁盘缓存 ({_industry_source})")
            return m

        # 4. 全部失败, 30 分钟后重试
        _industry_map, _industry_ok, _industry_source = {}, False, ''
        return None


# ===================== 全市场日线 (兼作交易日探测) =====================

_daily_empty = set()     # 非交易日 (历史, 永久)
_empty_probe = {}        # 当日探测为空的负缓存 {date: ts}, 10分钟


def get_daily_df(trade_date, force=False):
    """获取全市场日线 DataFrame (amount 已转为元); 非交易日/无数据返回 None"""
    if not force:
        cached = _cache_get(_daily_cache, trade_date)
        if cached is not None:
            return cached
        if trade_date in _daily_empty:
            return None
        ts = _empty_probe.get(trade_date)
        if ts is not None and time.time() - ts < 600:
            return None
    try:
        _rate_limit()
        df = pro.daily(trade_date=trade_date,
                       fields='ts_code,open,high,low,close,pre_close,pct_chg,amount')
        if df is None or df.empty:
            # 成功返回但为空 => 非交易日或当日数据未就绪
            if trade_date < datetime.now().strftime('%Y%m%d'):
                _daily_empty.add(trade_date)
            else:
                _empty_probe[trade_date] = time.time()
            return None
        df = df.dropna(subset=['close', 'high', 'low', 'amount'])
        df = df[df['amount'] > 0]
        if df.empty:
            return None
        df['pct_chg'] = df['pct_chg'].fillna(0)
        df['amount'] = df['amount'] * 1000.0  # 千元 -> 元
        _cache_set(_daily_cache, trade_date, df)
        return df
    except Exception as e:
        logging.warning(f"[板块资金] 获取 {trade_date} 全市场日线失败: {e}")
        return None


# ===================== 交易日探测 =====================

def get_trade_dates(n=60, end_date=None):
    """从 end_date (默认今天) 向前回溯最近 n 个交易日 (YYYYMMDD, 升序)
    通过 daily 接口探测 (trade_cal/index_daily 该账号有每小时频次限制)"""
    today = datetime.now().strftime('%Y%m%d')
    end = (end_date or today).replace('-', '')
    try:
        d = datetime.strptime(end, '%Y%m%d')
    except ValueError:
        return []
    dates = []
    tries = 0
    max_tries = n * 3 + 12
    while len(dates) < n and tries < max_tries:
        ds = d.strftime('%Y%m%d')
        if get_daily_df(ds) is not None:
            dates.append(ds)
        d -= timedelta(days=1)
        tries += 1
    return list(reversed(dates))


def resolve_trade_date(picked):
    """将所选日期解析为 <=picked 的最近交易日"""
    picked = str(picked).replace('-', '')
    d = datetime.strptime(picked, '%Y%m%d')
    for _ in range(12):
        ds = d.strftime('%Y%m%d')
        if get_daily_df(ds) is not None:
            return ds
        d -= timedelta(days=1)
    return picked


# ===================== 板块资金强度计算 =====================

def compute_sector_strength(trade_date, force=False):
    """按行业聚合计算资金强度, 返回 rows list 或 None"""
    if not force:
        cached = _cache_get(_strength_cache, trade_date)
        if cached is not None:
            return cached

    ind = get_industry_map(force=force)
    if not ind:
        return None
    df = get_daily_df(trade_date, force=force)
    if df is None or df.empty:
        return None

    # CLV 资金流乘数: 2×(收盘-最低)/(最高-最低)-1, 一字板按涨跌方向 ±1
    rng = df['high'] - df['low']
    clv = pd.Series(0.0, index=df.index)
    normal = rng > 0
    clv[normal] = 2.0 * (df['close'][normal] - df['low'][normal]) / rng[normal] - 1.0
    flat_up = (~normal) & (df['pct_chg'] > 0)
    flat_dn = (~normal) & (df['pct_chg'] < 0)
    clv[flat_up] = 1.0
    clv[flat_dn] = -1.0

    df = df.assign(_flow=clv * df['amount'],
                   _industry=df['ts_code'].map(lambda c: ind.get(c, {}).get('industry', '')))
    df = df[df['_industry'] != '']
    if df.empty:
        return None

    rows = []
    for industry, sub in df.groupby('_industry'):
        # 主力净流入估算(元) = Σ(CLV × 成交额) × 折算系数
        main_net = float(sub['_flow'].sum()) * MAIN_FLOW_SCALE
        amount = float(sub['amount'].sum())        # 成交额(元)
        strength = main_net / amount * 100 if amount else 0.0
        # 成交额加权板块涨跌幅
        pct = float((sub['pct_chg'] * sub['amount']).sum() / amount) if amount else 0.0
        # 领涨股
        top = sub.loc[sub['pct_chg'].idxmax()]
        top_code = top['ts_code']
        rows.append({
            'name': industry,
            'count': int(len(sub)),
            'pct_change': round(pct, 2),
            'main_net': round(main_net / 1e8, 2),   # 亿
            'amount': round(amount / 1e8, 2),       # 亿
            'strength': round(strength, 2),
            'state': classify_strength(strength),
            'top_stock': ind.get(top_code, {}).get('name', top_code),
            'top_stock_code': top_code,
            'top_stock_pct': round(float(top['pct_chg']), 2),
        })
    rows.sort(key=lambda r: r['strength'], reverse=True)
    _cache_set(_strength_cache, trade_date, rows)
    logging.info(f"[板块资金] {trade_date}: {len(rows)} 个行业计算完成")
    return rows


def compute_sector_strength_resolved(trade_date, force=False):
    """计算资金强度; 若当日盘后数据未就绪则回退到上一交易日, 返回 (实际交易日, rows)"""
    rows = compute_sector_strength(trade_date, force=force)
    if rows:
        return trade_date, rows
    dates = get_trade_dates(n=6, end_date=trade_date)
    for d in reversed(dates[:-1]):  # 跳过自身, 逐日回退(最多5日)
        rows = compute_sector_strength(d, force=force)
        if rows:
            return d, rows
    return trade_date, None


# ===================== 过滤排序 =====================

def filter_rows(rows, search, state, sort):
    result = rows or []
    if search:
        kw = str(search).strip().lower()
        if kw:
            result = [r for r in result if kw in r['name'].lower()
                      or kw in (r['top_stock'] or '').lower()]
    if state and state != STATE_ALL:
        result = [r for r in result if r['state'] == state]
    keys = {
        'strength_desc': lambda r: -r['strength'],
        'strength_asc': lambda r: r['strength'],
        'main_desc': lambda r: -r['main_net'],
        'amount_desc': lambda r: -r['amount'],
        'pct_desc': lambda r: -(r['pct_change'] or 0),
    }
    return sorted(result, key=keys.get(sort) or keys['strength_desc'])


# ===================== 统计卡片 =====================

def _build_stat_cards(rows):
    counts = Counter(r['state'] for r in rows)
    total_main = sum(r['main_net'] for r in rows)
    avg_strength = sum(r['strength'] for r in rows) / len(rows) if rows else 0.0

    children = []
    for label in STATES:
        style_cls, color, icon = STATE_STYLE[label]
        children.append(html.Div([
            html.Div([
                html.I(className=f"fas {icon} {color} text-lg mr-2"),
                html.Div([
                    html.Div(label, className="text-xs text-gray-500"),
                    html.Div(f"{counts.get(label, 0)} 个", className=f"text-lg font-bold {color}"),
                ]),
            ], className="flex items-center"),
        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-3"))

    main_color = 'text-red-600' if total_main > 0 else ('text-green-600' if total_main < 0 else 'text-gray-800')
    for label, value, icon, color in [
        ('平均资金强度', f'{avg_strength:+.2f}', 'fa-gauge-high', 'text-indigo-600'),
        ('主力净流入合计(估)', f'{total_main:+,.1f} 亿', 'fa-money-bill-transfer', main_color),
    ]:
        children.append(html.Div([
            html.Div([
                html.I(className=f"fas {icon} {color} text-lg mr-2"),
                html.Div([
                    html.Div(label, className="text-xs text-gray-500"),
                    html.Div(value, className=f"text-lg font-bold {color}"),
                ]),
            ], className="flex items-center"),
        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-3"))

    return html.Div(children, className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-2")


# ===================== 表格 =====================

def _build_table(rows):
    if not rows:
        return html.Div([
            html.I(className="fas fa-magnifying-glass-chart text-gray-300 text-5xl mb-3"),
            html.P("没有符合条件的板块", className="text-gray-400 text-sm"),
        ], className="text-center py-16")

    header = html.Thead(html.Tr([
        html.Th(h, className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap")
        for h in ['排名', '板块', '家数', '涨跌幅', '主力净流入(亿,估)', '成交额(亿)', '资金强度', '状态', '领涨股']
    ]), className="bg-gray-50")

    body_rows = []
    for i, r in enumerate(rows[:MAX_TABLE_ROWS]):
        style_cls, color, _ = STATE_STYLE[r['state']]
        pct = r['pct_change'] or 0
        pct_color = 'text-red-600' if pct > 0 else ('text-green-600' if pct < 0 else 'text-gray-500')
        main_color = 'text-red-600' if r['main_net'] > 0 else ('text-green-600' if r['main_net'] < 0 else 'text-gray-500')

        body_rows.append(html.Tr([
            html.Td(str(i + 1), className="px-3 py-2 text-xs text-gray-400"),
            html.Td(r['name'], className="px-3 py-2 text-sm font-medium text-gray-800 whitespace-nowrap"),
            html.Td(str(r['count']), className="px-3 py-2 text-sm text-gray-500"),
            html.Td(f"{pct:+.2f}%", className=f"px-3 py-2 text-sm font-medium {pct_color}"),
            html.Td(f"{r['main_net']:+,.2f}", className=f"px-3 py-2 text-sm font-medium {main_color}"),
            html.Td(f"{r['amount']:,.1f}", className="px-3 py-2 text-sm text-gray-700"),
            html.Td(f"{r['strength']:+.2f}", className=f"px-3 py-2 text-sm font-bold {main_color}"),
            html.Td(html.Span(r['state'], className=f"px-2 py-0.5 text-xs rounded-full border {style_cls}"),
                    className="px-3 py-2 whitespace-nowrap"),
            html.Td(f"{r['top_stock']} ({r['top_stock_pct']:+.2f}%)",
                    className="px-3 py-2 text-xs text-gray-600 whitespace-nowrap"),
        ], className="border-t border-gray-100 hover:bg-gray-50 transition-colors"))

    return html.Table([header, html.Tbody(body_rows)],
                      className="min-w-full divide-y divide-gray-200")


# ===================== 图表 =====================

def _build_top_chart(rows):
    """TOP15 资金强度条形图（按状态着色）"""
    fig = go.Figure()
    top = [r for r in rows if r['strength'] > 0][:15] or rows[:15]
    if top:
        top = list(reversed(top))
        fig.add_trace(go.Bar(
            x=[r['strength'] for r in top], y=[r['name'] for r in top],
            orientation='h',
            marker_color=[STATE_STYLE[r['state']][1] for r in top], marker_opacity=0.9,
            text=[f"{r['strength']:+.2f}" for r in top], textposition='outside',
            textfont=dict(size=10),
            hovertemplate='%{y}<br>资金强度: %{x:+.2f}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='资金强度 TOP15', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=360,
        xaxis=dict(title='资金强度(%)', showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#cbd5e1'),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=10, r=40, t=40, b=20),
    )
    return fig


def _build_dist_chart(rows):
    """状态分布环形图"""
    fig = go.Figure()
    counts = Counter(r['state'] for r in rows)
    labels = [s for s in STATES if counts.get(s)]
    if labels:
        values = [counts[s] for s in labels]
        fig.add_trace(go.Pie(
            labels=[f'{l} {v}个' for l, v in zip(labels, values)], values=values, hole=0.55,
            marker=dict(colors=[STATE_STYLE[l][1] for l in labels]),
            textinfo='percent', textfont=dict(size=11),
        ))
    fig.update_layout(
        title=dict(text='状态分布', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=360,
        showlegend=True, legend=dict(font=dict(size=10), orientation='h', y=-0.05),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _build_scatter_chart(rows):
    """资金强度 × 涨跌幅 散点图（气泡=成交额）"""
    fig = go.Figure()
    for state in STATES:
        sub = [r for r in rows if r['state'] == state]
        if not sub:
            continue
        fig.add_trace(go.Scatter(
            x=[r['strength'] for r in sub], y=[r['pct_change'] for r in sub],
            mode='markers', name=state,
            marker=dict(size=[max(6, min(30, (r['amount'] or 0) ** 0.5 / 3)) for r in sub],
                        color=STATE_STYLE[state][1], opacity=0.7,
                        sizemode='diameter', line=dict(width=0)),
            text=[f"{r['name']}<br>强度 {r['strength']:+.2f} | 涨跌 {r['pct_change']:+.2f}%<br>主力 {r['main_net']:+,.1f}亿 | 成交 {r['amount']:,.0f}亿" for r in sub],
            hovertemplate='%{text}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text='资金强度 × 涨跌幅（气泡=成交额）', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=360,
        xaxis=dict(title='资金强度(%)', showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0'),
        yaxis=dict(title='涨跌幅(%)', showgrid=True, gridcolor='#f1f5f9', zeroline=True, zerolinecolor='#e2e8f0'),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(font=dict(size=10), orientation='h', y=-0.18),
    )
    for xv in (-1, 1, 3):
        fig.add_vline(x=xv, line_width=1, line_dash='dot', line_color='#94a3b8', opacity=0.6)
    return fig


def build_trend_figure(trade_date, rows):
    """近 N 个交易日 TOP10 板块（按|强度|）资金强度趋势"""
    fig = go.Figure()
    if not rows:
        return fig
    dates = get_trade_dates(n=TREND_DAYS, end_date=trade_date)
    if len(dates) < 2:
        return fig
    top10 = sorted(rows, key=lambda r: -abs(r['strength']))[:10]
    names = [r['name'] for r in top10]

    series = {name: [] for name in names}
    valid_dates = []
    for d in dates:
        d_rows = compute_sector_strength(d)
        if not d_rows:
            continue
        strength_map = {r['name']: r['strength'] for r in d_rows}
        valid_dates.append(datetime.strptime(d, '%Y%m%d').strftime('%m-%d'))
        for name in names:
            series[name].append(strength_map.get(name))

    palette = ['#dc2626', '#ea580c', '#6366f1', '#16a34a', '#0891b2',
               '#7c3aed', '#db2777', '#65a30d', '#d97706', '#475569']
    for i, name in enumerate(names):
        ys = series[name]
        if all(v is None for v in ys):
            continue
        fig.add_trace(go.Scatter(
            x=valid_dates, y=ys, mode='lines+markers', name=name,
            line=dict(color=palette[i % len(palette)], width=2),
            marker=dict(size=5),
            hovertemplate=f'{name}<br>%{{x}} 强度: %{{y:+.2f}}<extra></extra>',
        ))
    fig.update_layout(
        title=dict(text=f'近{len(valid_dates)}个交易日 资金强度趋势（|强度|TOP10）',
                   font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=360,
        xaxis=dict(showgrid=False),
        yaxis=dict(title='资金强度(%)', showgrid=True, gridcolor='#f1f5f9',
                   zeroline=True, zerolinecolor='#cbd5e1'),
        margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(font=dict(size=9), orientation='h', y=-0.22),
    )
    return fig


# ===================== 页面布局 =====================

def _warmup():
    """后台预热行业分类与交易日 (baostock 行业查询约需 35 秒, 避免阻塞首屏)"""
    try:
        get_trade_dates(n=TREND_DAYS)
        get_industry_map()
    except Exception as e:
        logging.warning(f"[板块资金] 预热失败: {e}")


def sector_fund_strength_layout():
    # 服务启动时后台预热 (磁盘缓存 7 天内有效时预热瞬时完成)
    threading.Thread(target=_warmup, daemon=True).start()
    default_display = datetime.now().strftime('%Y-%m-%d')

    return html.Div([
        html.Div([
            # 标题 + 公式说明
            html.Div([
                html.H2([html.I(className='fas fa-gauge-high mr-2'), '板块资金强度'],
                        className='text-xl font-bold text-gray-800'),
                html.Div([
                    html.Span('资金强度 = 主力资金(估) ÷ 成交额 × 100', className='text-xs text-gray-600 bg-blue-50 px-2 py-0.5 rounded'),
                    html.Span('主力资金为CLV量价估算（收盘位置×成交额×0.15折算），非逐笔大单统计',
                              className='text-xs text-gray-400'),
                    html.Span('≥3 抢筹', className='text-xs px-2 py-0.5 rounded-full border bg-red-100 text-red-700 border-red-200'),
                    html.Span('1~3 建仓', className='text-xs px-2 py-0.5 rounded-full border bg-orange-100 text-orange-700 border-orange-200'),
                    html.Span('-1~1 洗盘', className='text-xs px-2 py-0.5 rounded-full border bg-slate-100 text-slate-600 border-slate-200'),
                    html.Span('≤-1 出货', className='text-xs px-2 py-0.5 rounded-full border bg-green-100 text-green-700 border-green-200'),
                ], className='flex flex-wrap items-center gap-2 mt-2'),
            ], className='mb-4'),

            # 日期 + 操作
            html.Div([
                html.Div([
                    html.Span('数据日期', className='text-xs text-gray-500 mr-2'),
                    dcc.DatePickerSingle(id='sfs-date', date=default_display,
                                         max_date_allowed=datetime.now().strftime('%Y-%m-%d'),
                                         className='text-sm'),
                ], className='flex items-center bg-white border border-gray-300 rounded-lg px-3 py-1.5'),
                html.Span(id='sfs-status', className='text-xs text-gray-400 ml-4'),
                html.Div([
                    html.Button([html.I(className='fas fa-sync-alt mr-1'), '刷新'],
                                id='sfs-refresh-btn',
                                className='bg-blue-50 hover:bg-blue-100 text-blue-600 px-3 py-1.5 rounded-lg text-sm font-medium transition-all'),
                    html.Button([html.I(className='fas fa-file-csv mr-1'), '导出'],
                                id='sfs-export-btn',
                                className='bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-all ml-2'),
                ], className='ml-auto flex'),
            ], className='flex flex-wrap items-center mb-3'),

            # 统计卡片
            html.Div(id='sfs-stat-cards', className='mb-3'),

            # 工具栏: 搜索 + 状态筛选 + 排序
            html.Div([
                html.Div([
                    html.I(className='fas fa-search text-gray-400 mr-2'),
                    dcc.Input(id='sfs-search', type='text', placeholder='搜索板块/领涨股...',
                              className='flex-1 border-0 outline-none text-sm bg-transparent', debounce=True),
                ], className='flex items-center bg-white border border-gray-300 rounded-lg px-3 py-2 w-64'),
                html.Div([
                    html.Button('全部', id={'type': 'sfs-state-btn', 'index': STATE_ALL},
                                className='px-3 py-1.5 rounded-lg text-xs font-medium transition-all bg-indigo-600 text-white'),
                    *[html.Button(s, id={'type': 'sfs-state-btn', 'index': s},
                                  className=f'px-3 py-1.5 rounded-lg text-xs font-medium bg-white border {STATE_STYLE[s][0]} hover:opacity-80 transition-all')
                      for s in STATES],
                ], className='flex gap-1 ml-2'),
                dcc.Dropdown(id='sfs-sort', options=SORT_OPTIONS, value='strength_desc',
                             clearable=False, className='w-44 text-sm ml-auto'),
            ], className='flex flex-wrap items-center mb-3'),

            # 表格
            html.Div([html.Span(id='sfs-count', className='text-xs text-gray-400 ml-auto')],
                     className='flex mb-1'),
            html.Div(id='sfs-table-container',
                     className='bg-white rounded-xl shadow-sm border border-gray-100 overflow-x-auto max-h-[52vh] overflow-y-auto'),

            # 图表 2×2
            html.Div([
                html.Div([dcc.Graph(id='sfs-top-chart', figure=go.Figure(), config={'displayModeBar': False})],
                         className='w-full lg:w-1/2 pr-0 lg:pr-2'),
                html.Div([dcc.Graph(id='sfs-dist-chart', figure=go.Figure(), config={'displayModeBar': False})],
                         className='w-full lg:w-1/2 pl-0 lg:pl-2 mt-4 lg:mt-0'),
            ], className='flex flex-col lg:flex-row mt-4'),
            html.Div([
                html.Div([dcc.Graph(id='sfs-scatter-chart', figure=go.Figure(), config={'displayModeBar': False})],
                         className='w-full lg:w-1/2 pr-0 lg:pr-2'),
                html.Div([dcc.Graph(id='sfs-trend-chart', figure=go.Figure(), config={'displayModeBar': False})],
                         className='w-full lg:w-1/2 pl-0 lg:pl-2 mt-4 lg:mt-0'),
            ], className='flex flex-col lg:flex-row mt-4'),
        ], className='container mx-auto px-4 py-6'),

        # Stores & 组件
        dcc.Store(id='sfs-store', storage_type='memory'),
        dcc.Store(id='sfs-state-store', data=STATE_ALL, storage_type='memory'),
        dcc.Interval(id='sfs-interval', interval=300000, n_intervals=0),  # 5分钟自动刷新
        dcc.Download(id='sfs-download'),
    ])


# ===================== 回调注册 =====================

def register_sector_fund_callbacks(app):
    """注册板块资金强度页面回调"""

    @app.callback(
        [Output({'type': 'sfs-state-btn', 'index': dash.ALL}, 'className'),
         Output('sfs-stat-cards', 'children'),
         Output('sfs-table-container', 'children'),
         Output('sfs-count', 'children'),
         Output('sfs-status', 'children'),
         Output('sfs-top-chart', 'figure'),
         Output('sfs-dist-chart', 'figure'),
         Output('sfs-scatter-chart', 'figure'),
         Output('sfs-store', 'data'),
         Output('sfs-state-store', 'data')],
        [Input('sfs-date', 'date'),
         Input('sfs-refresh-btn', 'n_clicks'),
         Input('sfs-interval', 'n_intervals'),
         Input({'type': 'sfs-state-btn', 'index': dash.ALL}, 'n_clicks'),
         Input('sfs-search', 'value'),
         Input('sfs-sort', 'value')],
        [State('sfs-store', 'data'),
         State('sfs-state-store', 'data')],
        prevent_initial_call=False
    )
    def render(picked_date, refresh_clicks, intervals, state_clicks,
               search, sort, store, state_store):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

        # ── 数据日期：日期选择触发时解析交易日，否则沿用 store ──
        if 'sfs-date' in triggered and picked_date:
            trade_date = resolve_trade_date(picked_date)
        else:
            trade_date = (store or {}).get('trade_date') or (
                resolve_trade_date(picked_date or datetime.now().strftime('%Y-%m-%d')))

        # ── 状态筛选：按钮点击优先，否则沿用 store ──
        if 'sfs-state-btn' in triggered:
            try:
                current_state = json.loads(triggered.split('.')[0]).get('index', STATE_ALL)
            except Exception:
                current_state = state_store or STATE_ALL
        else:
            current_state = state_store or STATE_ALL

        # ── 获取数据（刷新按钮强制绕过当日缓存; 当日盘后未就绪自动回退上一交易日）──
        force = 'sfs-refresh-btn' in triggered
        trade_date, rows = compute_sector_strength_resolved(trade_date, force=force)
        filtered = filter_rows(rows, search, current_state, sort)

        # 状态按钮样式（顺序与布局一致: 全部/抢筹/建仓/洗盘/出货）
        btn_classes = []
        for st in [STATE_ALL] + STATES:
            active = (st == current_state)
            if st == STATE_ALL:
                btn_classes.append('px-3 py-1.5 rounded-lg text-xs font-medium transition-all '
                                   + ('bg-indigo-600 text-white' if active
                                      else 'bg-white border border-gray-300 text-gray-600 hover:border-indigo-400'))
            else:
                base = f'px-3 py-1.5 rounded-lg text-xs font-medium border transition-all {STATE_STYLE[st][0]}'
                btn_classes.append(base + (' ring-2 ring-indigo-400' if active else ' opacity-70 hover:opacity-100'))

        src = _industry_source or '行业分类获取失败'
        if rows:
            status = f'数据日期 {trade_date} · {len(rows)} 个行业 · {src}'
        else:
            status = f'数据日期 {trade_date} · 数据获取失败，请点击「刷新」重试'

        count = f'筛选后 {len(filtered)} 个行业'
        if len(filtered) > MAX_TABLE_ROWS:
            count += f'，表格仅显示前 {MAX_TABLE_ROWS} 个（导出含全部）'

        store_data = {'trade_date': trade_date, 'rows': rows or []}

        return (
            btn_classes,
            _build_stat_cards(filtered),
            _build_table(filtered),
            count,
            status,
            _build_top_chart(filtered),
            _build_dist_chart(filtered),
            _build_scatter_chart(filtered),
            store_data,
            current_state,
        )

    # ─── 趋势图（依赖主渲染结果，历史数据缓存后秒出）───
    @app.callback(
        Output('sfs-trend-chart', 'figure'),
        [Input('sfs-store', 'data')],
        prevent_initial_call=True
    )
    def update_trend(store):
        if not store or not store.get('rows'):
            return go.Figure()
        return build_trend_figure(store.get('trade_date'), store['rows'])

    # ─── 导出 CSV ───
    @app.callback(
        Output('sfs-download', 'data'),
        [Input('sfs-export-btn', 'n_clicks')],
        [State('sfs-store', 'data'),
         State('sfs-state-store', 'data'),
         State('sfs-search', 'value'),
         State('sfs-sort', 'value')],
        prevent_initial_call=True
    )
    def export_csv(n_clicks, store, state, search, sort):
        if not n_clicks or not store or not store.get('rows'):
            return dash.no_update
        filtered = filter_rows(store['rows'], search, state, sort)
        if not filtered:
            return dash.no_update

        df = pd.DataFrame(filtered)
        col_map = [
            ('name', '板块名称'), ('count', '家数'), ('state', '状态'), ('strength', '资金强度'),
            ('main_net', '主力净流入(亿,估)'), ('amount', '成交额(亿)'), ('pct_change', '涨跌幅%'),
            ('top_stock', '领涨股'), ('top_stock_code', '领涨股代码'), ('top_stock_pct', '领涨股涨跌幅%'),
        ]
        cols = [c for c, _ in col_map if c in df.columns]
        df = df[cols]
        df.columns = [dict(col_map)[c] for c in cols]
        fname = f"sector_fund_{store.get('trade_date', 'unknown')}.csv"
        return dcc.send_data_frame(df.to_csv, fname, index=False, encoding='utf-8-sig')
