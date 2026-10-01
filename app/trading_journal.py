"""
交易日志页面 — 持仓管理、加减仓、自动获取行情计算盈亏、总资金管理
功能: 开仓/加仓/减仓/清仓、实时行情盈亏、出入金、资金概览、统计图表、CSV 导出
数据存储: SQLite (trading_journal / account / capital_flows 表)
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import io
import json
import logging

from .database_manager import (get_trade_entries, get_trade_entry,
                               add_trade_entry, update_trade_entry, delete_trade_entry,
                               clear_all_trade_data,
                               get_all_stock_codes, get_stock_name_map,
                               get_account, set_initial_capital,
                               get_capital_flows, add_capital_flow, delete_capital_flow,
                               normalize_stock_code)

# ===================== 常量 =====================
DIRECTIONS = ['买入', '卖出']
TRADE_TYPES = ['短线', '波段', '中长线', '打板', '日内', '其他']
FLOW_TYPES = ['转入', '转出']
ALL_OPTION = '全部'


# ===================== 工具函数 =====================

def _to_float(v):
    if v is None or v == '':
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_int(v):
    if v is None or v == '':
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _normalize_code(code):
    """将代码统一为标准格式 (如 600519.SH / 920025.BJ)，规则与数据库层一致"""
    return normalize_stock_code((code or '').strip())


def _lookup_name(code):
    """根据代码从股票库查找名称，兼容多种代码格式"""
    code = (code or '').strip()
    if not code:
        return ''
    name_map = get_stock_name_map()
    for key in (code, _normalize_code(code), code.split('.')[0]):
        if key and name_map.get(key):
            return name_map[key]
    return ''


def _fmt_money(v, decimals=2):
    v = v or 0
    try:
        return f"{v:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _fmt_pnl(v, decimals=2):
    v = v or 0
    try:
        return f"{v:+,.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _current_position(entries, code):
    """从交易记录计算某股票当前持仓数量"""
    qty = 0
    for e in sorted(entries, key=lambda x: ((x.get('trade_date') or ''), x.get('id') or 0)):
        if e.get('stock_code') != code:
            continue
        if e.get('direction') == '买入':
            qty += (e.get('quantity') or 0)
        else:
            qty -= (e.get('quantity') or 0)
    return max(0, qty)


def _find_position(holdings, code):
    """在持仓列表中查找指定代码的持仓（双方规范化后匹配）"""
    norm = _normalize_code(code)
    for h in holdings or []:
        if _normalize_code(h.get('stock_code', '')) == norm:
            return h
    return None


def _valid_click_trigger():
    """校验 pattern-matching 回调是否由真实点击触发
    表格重渲染(组件增删)也会触发 dash.ALL 回调, 此时 triggered[0].value 为 None;
    必须检查本次触发组件自身的 n_clicks, 否则会用错误的按钮覆盖表单甚至误删数据。
    注意: 按钮 index 必须纯 ASCII —— Dash 4.1 的 renderer 发送的 changedPropIds 为
    中文原文, 而服务端 stringify_id 按 ensure_ascii 转义, 中文 index 会导致键不匹配、
    triggered value 恒为 None, 真实点击也会被误判为重渲染"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    trig = ctx.triggered[0]
    if not (trig.get('value') or 0):
        return None  # 非真实点击(组件增删引起的重渲染)
    return trig


def _parse_pattern_id(prop_id):
    """解析 pattern-matching 回调的 prop_id 为组件 id 字典
    prop_id 形如 '{"index":"600519.SH|买入","type":"journal-pos-btn"}.n_clicks'
    注意 index 可能包含 '.' (股票代码/小数), 必须用 rsplit 从右侧剥离属性名,
    不能 split('.')[0] (会截断含 '.' 的 index 导致按钮点击无效)"""
    if not prop_id:
        return {}
    try:
        return json.loads(prop_id.rsplit('.', 1)[0])
    except Exception:
        return {}


def compute_portfolio(entries):
    """从交易记录按时间顺序计算持仓、资金变化、已实现盈亏"""
    positions = {}
    cash_delta = 0.0
    total_realized = 0.0
    realized_series = []

    for e in sorted(entries, key=lambda x: ((x.get('trade_date') or ''), x.get('id') or 0)):
        code = e.get('stock_code', '')
        name = e.get('stock_name', '')
        qty = e.get('quantity') or 0
        price = e.get('price') or 0
        fee = e.get('fee') or 0
        direction = e.get('direction', '买入')

        if code not in positions:
            positions[code] = {'stock_code': code, 'stock_name': name,
                               'quantity': 0, 'avg_cost': 0.0, 'realized_pnl': 0.0}
        pos = positions[code]
        if name:
            pos['stock_name'] = name

        if direction == '买入':
            new_qty = pos['quantity'] + qty
            if new_qty > 0:
                pos['avg_cost'] = (pos['quantity'] * pos['avg_cost'] + qty * price + fee) / new_qty
            pos['quantity'] = new_qty
            cash_delta -= (qty * price + fee)
        else:
            sell_qty = min(qty, pos['quantity'])
            realized = (price - pos['avg_cost']) * sell_qty - fee
            pos['realized_pnl'] += realized
            total_realized += realized
            pos['quantity'] -= sell_qty
            if pos['quantity'] <= 0:
                pos['quantity'] = 0
                pos['avg_cost'] = 0.0
            cash_delta += (sell_qty * price - fee)
            realized_series.append({
                'trade_date': e.get('trade_date', ''),
                'realized': realized,
                'stock_code': code,
                'stock_name': name,
            })

    holdings = [p for p in positions.values() if p['quantity'] > 0]
    holdings.sort(key=lambda x: x['stock_code'])

    cum = 0.0
    for r in realized_series:
        cum += r['realized']
        r['cumulative'] = cum

    return {
        'holdings': holdings,
        'cash_delta': cash_delta,
        'total_realized': total_realized,
        'realized_series': realized_series,
    }


def _filter_entries(entries, search, direction, trade_type):
    result = entries or []
    if search:
        kw = str(search).strip().lower()
        if kw:
            def _match(e):
                blob = ' '.join(str(e.get(k, '')) for k in
                                ('stock_code', 'stock_name', 'reason', 'memo', 'tags', 'action'))
                return kw in blob.lower()
            result = [e for e in result if _match(e)]
    if direction and direction != ALL_OPTION:
        result = [e for e in result if e.get('direction') == direction]
    if trade_type and trade_type != ALL_OPTION:
        result = [e for e in result if e.get('trade_type') == trade_type]
    return result


_BS_CACHE_TTL = 600  # baostock 行情缓存(秒)


def _fetch_baostock_daily(code, start_yyyymmdd, end_yyyymmdd):
    """仅从 baostock 获取不复权日线（交易日志页专用，Redis 缓存 10 分钟）
    返回 DataFrame(trade_date, open, high, low, close, vol, amount) 或 None
    不复权是为了与录入的实际成交价对齐"""
    from .baostock_fetcher import fetch_daily_data
    from . import data_manager as dm
    code = _normalize_code(code)
    if not code:
        return None

    cache_key = f"journal-bs-unadj:{code}:{start_yyyymmdd}:{end_yyyymmdd}"
    if dm.redis_client:
        try:
            cached = dm.redis_client.get(cache_key)
            if cached:
                # pandas>=2 的 read_json 需要用 StringIO 包裹 JSON 字符串
                df = pd.read_json(io.StringIO(cached.decode('utf-8')), orient='records')
                if df is not None and not df.empty and 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    return df
        except Exception:
            try:
                dm.redis_client.delete(cache_key)
            except Exception:
                pass

    df = None
    try:
        df = fetch_daily_data(code, start_yyyymmdd, end_yyyymmdd, adjustflag='3')
    except Exception as e:
        logging.warning(f"[交易日志] baostock 获取 {code} 日线失败: {e}")
    if df is not None and not df.empty:
        if dm.redis_client:
            try:
                dm.redis_client.setex(cache_key, _BS_CACHE_TTL,
                                      df.to_json(orient='records', date_format='iso'))
            except Exception:
                pass
        return df
    return None


def _fetch_current_price(code):
    """获取股票最新收盘价（仅 baostock，不复权）"""
    end = datetime.now()
    start = end - timedelta(days=60)
    try:
        df = _fetch_baostock_daily(code, start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))
        if df is not None and not df.empty and 'close' in df.columns:
            return float(df.iloc[-1]['close'])
    except Exception as e:
        logging.warning(f"获取 {code} 现价失败: {e}")
    return None


# ===================== 账户概览渲染 =====================

def _build_account_cards(initial, net_flows, available_cash, total_mv,
                         total_assets, total_pnl, total_pnl_pct, flows):
    pnl_color = 'text-red-600' if total_pnl > 0 else ('text-green-600' if total_pnl < 0 else 'text-gray-800')
    pnl_icon = 'fa-arrow-trend-up' if total_pnl >= 0 else 'fa-arrow-trend-down'

    cards = [
        ('总资产', _fmt_money(total_assets), 'fa-wallet', 'text-blue-600', 'text-gray-800'),
        ('可用资金', _fmt_money(available_cash), 'fa-coins', 'text-emerald-600', 'text-gray-800'),
        ('持仓市值', _fmt_money(total_mv), 'fa-chart-pie', 'text-indigo-600', 'text-gray-800'),
        ('总盈亏', f"{_fmt_pnl(total_pnl)} ({total_pnl_pct:+.2f}%)", pnl_icon, pnl_color, pnl_color),
        ('初始资金', _fmt_money(initial), 'fa-piggy-bank', 'text-amber-600', 'text-gray-800'),
        ('累计净转入', _fmt_pnl(net_flows), 'fa-arrow-right-arrow-left', 'text-cyan-600', 'text-gray-800'),
    ]

    card_children = []
    for label, value, icon, color, value_color in cards:
        card_children.append(html.Div([
            html.Div([
                html.Div([
                    html.I(className=f"fas {icon} {color} text-xl mr-3"),
                    html.Div([
                        html.Div(label, className="text-xs text-gray-500"),
                        html.Div(value, className=f"text-xl font-bold {value_color}"),
                    ]),
                ], className="flex items-center"),
            ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 transition-all duration-300 hover:shadow-md"),
        ], className="w-full sm:w-1/2 lg:w-1/3 xl:w-1/6 p-2"))

    # 出入金流水列表
    flow_items = []
    if flows:
        for f in flows:
            is_in = f.get('flow_type') == '转入'
            flow_items.append(html.Div([
                html.Span([
                    html.I(className=f"fas {'fa-circle-arrow-down text-emerald-500' if is_in else 'fa-circle-arrow-up text-orange-500'} mr-1"),
                    html.Span(f.get('flow_type', ''), className='font-medium'),
                ], className='text-xs'),
                html.Span(f"{'+' if is_in else '-'}{_fmt_money(f.get('amount'))}",
                          className=f"text-sm font-medium ml-2 {'text-emerald-600' if is_in else 'text-orange-600'}"),
                html.Span((f.get('flow_date') or '')[:10], className='text-xs text-gray-400 ml-auto'),
                html.Span(f.get('note', ''), className='text-xs text-gray-500 ml-2 truncate'),
                html.Button([html.I(className='fas fa-times text-red-400 hover:text-red-600')],
                            id={'type': 'journal-flow-del', 'index': f['id']},
                            className='bg-transparent border-none cursor-pointer p-1 ml-1', title='删除'),
            ], className='flex items-center py-1.5 px-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors'))
    else:
        flow_items = [html.Div('暂无出入金记录', className='text-xs text-gray-400 text-center py-2')]

    flow_list = html.Div([
        html.Div([
            html.Span([html.I(className='fas fa-arrow-right-arrow-left mr-1'), '出入金流水'],
                      className='text-xs font-semibold text-gray-600'),
        ], className='mb-1 pb-1 border-b border-gray-100'),
        html.Div(flow_items, className='max-h-28 overflow-y-auto'),
    ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-3 mt-3')

    return html.Div([
        html.Div(card_children, className='flex flex-wrap -mx-2'),
        flow_list,
    ])


# ===================== 持仓表格 =====================

def _build_holdings_table(holdings):
    if not holdings:
        return html.Div([
            html.I(className="fas fa-box-open text-gray-300 text-5xl mb-3"),
            html.P("当前无持仓，录入第一笔买入记录后即可查看", className="text-gray-400 text-sm"),
        ], className="text-center py-12")

    header = html.Thead(html.Tr([
        html.Th(h, className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap")
        for h in ['股票', '持仓数量', '持仓均价', '现价', '市值', '浮动盈亏', '盈亏比例', '已实现盈亏', '操作']
    ]), className="bg-gray-50")

    rows = []
    for h in holdings:
        cur = h.get('current_price')
        mv = h.get('market_value')
        float_pnl = h.get('float_pnl')
        float_pct = h.get('float_pnl_pct')
        code = h.get('stock_code', '')

        pnl_color = 'text-red-600' if (float_pnl or 0) > 0 else ('text-green-600' if (float_pnl or 0) < 0 else 'text-gray-500')
        realized_color = 'text-red-600' if (h.get('realized_pnl') or 0) > 0 else ('text-green-600' if (h.get('realized_pnl') or 0) < 0 else 'text-gray-500')

        rows.append(html.Tr([
            html.Td(html.Div([
                html.Div(code, className="text-sm font-medium text-gray-800"),
                html.Div(h.get('stock_name', ''), className="text-xs text-gray-400"),
            ]), className="px-3 py-2"),
            html.Td(str(h.get('quantity') or 0), className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_money(h.get('avg_cost')), className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_money(cur) if cur is not None else '--', className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_money(mv) if mv is not None else '--', className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_pnl(float_pnl) if float_pnl is not None else '--', className=f"px-3 py-2 text-sm font-medium {pnl_color} whitespace-nowrap"),
            html.Td(f"{float_pct:+.2f}%" if float_pct is not None else '--', className=f"px-3 py-2 text-sm {pnl_color} whitespace-nowrap"),
            html.Td(_fmt_pnl(h.get('realized_pnl')), className=f"px-3 py-2 text-sm font-medium {realized_color} whitespace-nowrap"),
            html.Td(html.Div([
                html.Button(
                    [html.I(className="fas fa-plus text-red-500 hover:text-red-700")],
                    id={'type': 'journal-pos-btn', 'index': f"{code}|buy"},
                    className="bg-red-50 hover:bg-red-100 border border-red-100 rounded px-1.5 py-1 text-xs", title="加仓（预填表单）"),
                html.Button(
                    [html.I(className="fas fa-minus text-green-600 hover:text-green-800")],
                    id={'type': 'journal-pos-btn', 'index': f"{code}|sell"},
                    className="bg-green-50 hover:bg-green-100 border border-green-100 rounded px-1.5 py-1 text-xs ml-1", title="减仓/清仓（预填表单）"),
                html.Button(
                    [html.I(className="fas fa-chart-simple text-indigo-400 hover:text-indigo-600")],
                    id={'type': 'journal-kline-btn', 'index': code},
                    className="bg-transparent border-none cursor-pointer p-1 ml-1", title="查看K线交易标记"),
            ], className="flex items-center"), className="px-3 py-2"),
        ], className="border-t border-gray-100 hover:bg-gray-50 transition-colors"))

    return html.Table([header, html.Tbody(rows)], className="min-w-full divide-y divide-gray-200")


# ===================== 交易记录表格 =====================

_ACTION_STYLE = {
    '开仓': 'bg-blue-100 text-blue-700',
    '加仓': 'bg-indigo-100 text-indigo-700',
    '减仓': 'bg-amber-100 text-amber-700',
    '清仓': 'bg-gray-200 text-gray-600',
}


def _build_table(entries):
    if not entries:
        return html.Div([
            html.I(className="fas fa-book-open text-gray-300 text-5xl mb-3"),
            html.P("暂无交易记录", className="text-gray-400 text-sm"),
        ], className="text-center py-12")

    header = html.Thead(html.Tr([
        html.Th(h, className="px-3 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider whitespace-nowrap")
        for h in ['日期', '股票', '操作', '方向', '类型', '价格', '数量', '金额', '手续费', '标签', '操作']
    ]), className="bg-gray-50")

    rows = []
    for e in entries:
        direction = e.get('direction', '')
        action = e.get('action', '') or ('开仓/加仓' if direction == '买入' else '减仓/清仓')
        action_style = _ACTION_STYLE.get(action, 'bg-gray-100 text-gray-600')
        direction_color = 'text-red-600' if direction == '买入' else 'text-green-600'

        tags = (e.get('tags') or '').strip()
        tag_badges = []
        if tags:
            for t in tags.split(','):
                t = t.strip()
                if t:
                    tag_badges.append(html.Span(t, className="px-1.5 py-0.5 bg-indigo-50 text-indigo-600 text-xs rounded mr-1"))

        rows.append(html.Tr([
            html.Td((e.get('trade_date') or '')[:10], className="px-3 py-2 text-sm text-gray-600 whitespace-nowrap"),
            html.Td(html.Div([
                html.Div(e.get('stock_code', ''), className="text-sm font-medium text-gray-800"),
                html.Div(e.get('stock_name', ''), className="text-xs text-gray-400"),
            ]), className="px-3 py-2"),
            html.Td(html.Span(action, className=f"px-2 py-0.5 text-xs rounded-full {action_style}"), className="px-3 py-2 whitespace-nowrap"),
            html.Td(direction, className=f"px-3 py-2 text-sm font-medium {direction_color} whitespace-nowrap"),
            html.Td(e.get('trade_type', ''), className="px-3 py-2 text-sm text-gray-600 whitespace-nowrap"),
            html.Td(_fmt_money(e.get('price')), className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(str(e.get('quantity') or 0), className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_money(e.get('amount')), className="px-3 py-2 text-sm text-gray-700 whitespace-nowrap"),
            html.Td(_fmt_money(e.get('fee')), className="px-3 py-2 text-sm text-gray-500 whitespace-nowrap"),
            html.Td(html.Div(tag_badges, className="flex flex-wrap gap-1"), className="px-3 py-2"),
            html.Td(html.Div([
                html.Button([html.I(className="fas fa-pen text-blue-500 hover:text-blue-700")],
                            id={'type': 'journal-edit-btn', 'index': e['id']},
                            className="bg-transparent border-none cursor-pointer p-1", title="编辑"),
                html.Button([html.I(className="fas fa-trash text-red-400 hover:text-red-600")],
                            id={'type': 'journal-delete-btn', 'index': e['id']},
                            className="bg-transparent border-none cursor-pointer p-1", title="删除"),
            ], className="flex items-center gap-1"), className="px-3 py-2 whitespace-nowrap"),
        ], className="border-t border-gray-100 hover:bg-gray-50 transition-colors"))

    return html.Table([header, html.Tbody(rows)], className="min-w-full divide-y divide-gray-200")


# ===================== 图表 =====================

def _build_holdings_chart(holdings):
    fig = go.Figure()
    with_mv = [h for h in holdings if h.get('market_value') is not None]
    if with_mv:
        labels = [f"{h.get('stock_code', '')}" for h in with_mv]
        values = [h.get('market_value', 0) for h in with_mv]
        fig.add_trace(go.Pie(
            labels=labels, values=values, hole=0.45,
            textinfo='label+percent',
            marker=dict(colors=['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444']),
        ))
    fig.update_layout(
        title=dict(text='持仓市值占比', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig


# ===================== 每日盈亏 =====================

def _daily_pnl_stats(realized_series):
    """按自然周期统计已实现盈亏: 今日/本周/本月/今年"""
    result = {'今日': 0.0, '本周': 0.0, '本月': 0.0, '今年': 0.0}
    now = datetime.now()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    for r in realized_series or []:
        d = str(r.get('trade_date') or '')[:10]
        try:
            dt = datetime.strptime(d, '%Y-%m-%d')
        except ValueError:
            continue
        if dt.year != now.year:
            continue
        pnl = r.get('realized') or 0
        result['今年'] += pnl
        if dt.month == now.month:
            result['本月'] += pnl
        if dt >= week_start:
            result['本周'] += pnl
        if dt.date() == now.date():
            result['今日'] += pnl
    return result


def _build_daily_pnl_badges(realized_series):
    """每日盈亏统计徽章行"""
    stats = _daily_pnl_stats(realized_series)
    badges = []
    for label in ['今日', '本周', '本月', '今年']:
        v = stats[label]
        cls = ('text-red-600 bg-red-50 border-red-100' if v > 0
               else ('text-green-600 bg-green-50 border-green-100' if v < 0
                     else 'text-gray-500 bg-gray-50 border-gray-100'))
        badges.append(html.Span([
            html.Span(f'{label}盈亏 ', className='text-xs'),
            html.Span(_fmt_pnl(v), className='font-bold'),
        ], className=f'px-3 py-1.5 rounded-lg border text-sm {cls}'))
    badges.append(html.Span('已实现口径（卖出成交后计算）', className='text-xs text-gray-400 self-center ml-1'))
    return html.Div(badges, className='flex flex-wrap gap-2 items-center')


def _build_daily_pnl_chart(realized_series):
    """每日已实现盈亏柱状图 + 累计盈亏曲线"""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    if realized_series:
        daily = {}
        for r in sorted(realized_series, key=lambda x: (x.get('trade_date') or '')):
            d = (r.get('trade_date') or '')[:10] or '未知'
            daily[d] = daily.get(d, 0) + (r.get('realized') or 0)
        dates = list(daily.keys())
        values = [round(v, 2) for v in daily.values()]
        colors = ['#ef4444' if v > 0 else ('#10b981' if v < 0 else '#9ca3af') for v in values]
        fig.add_trace(go.Bar(
            x=dates, y=values, marker_color=colors, name='每日盈亏',
            text=[_fmt_pnl(v) for v in values], textposition='outside', textfont=dict(size=9),
            hovertemplate='%{x}<br>盈亏: %{y:+,.2f}<extra></extra>',
        ), secondary_y=False)
        # 累计曲线 (按日汇总后累加)
        cum, s = [], 0.0
        for v in values:
            s += v
            cum.append(round(s, 2))
        fig.add_trace(go.Scatter(
            x=dates, y=cum, mode='lines+markers', name='累计盈亏',
            line=dict(color='#6366f1', width=2), marker=dict(size=4),
            hovertemplate='%{x}<br>累计: %{y:+,.2f}<extra></extra>',
        ), secondary_y=True)
        fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', zeroline=True,
                         zerolinecolor='#cbd5e1', secondary_y=False)
        fig.update_yaxes(showgrid=False, secondary_y=True)
    fig.update_layout(
        title=dict(text='每日盈亏与累计', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True, legend=dict(font=dict(size=9), orientation='h', y=1.14, x=0),
    )
    return fig


# ===================== 主图K线 · 交易画线标记 =====================

def _snap_date(d, dates_sorted):
    """将交易日期对齐到最近的行情日期"""
    if d in dates_sorted:
        return d
    import bisect
    i = bisect.bisect_left(dates_sorted, d)
    candidates = ([dates_sorted[i - 1]] if i > 0 else []) + ([dates_sorted[i]] if i < len(dates_sorted) else [])
    if not candidates:
        return None
    try:
        dd = datetime.strptime(d, '%Y-%m-%d')
        return min(candidates, key=lambda c: abs((datetime.strptime(c, '%Y-%m-%d') - dd).days))
    except ValueError:
        return None


def _empty_kline_figure(msg):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text='主图 · K线交易标记', font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=440,
        annotations=[dict(text=msg, showarrow=False, x=0.5, y=0.5, xref='paper', yref='paper',
                          font=dict(size=13, color='#9ca3af'))],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _build_kline_figure(code, entries):
    """主图K线 + 交易画线标记: 买卖点(▲/▼)、交易轨迹虚线、持仓成本线"""
    if not code:
        return _empty_kline_figure('选择股票或在持仓表点击 📈 查看K线交易标记')
    entries = entries or []
    trades = sorted([e for e in entries if e.get('stock_code') == code],
                    key=lambda e: (e.get('trade_date') or ''))
    if not trades:
        return _empty_kline_figure(f'{code} 暂无交易记录')

    # 行情范围: 首笔交易前 60 个自然日 ~ 今天
    first_date = (trades[0].get('trade_date') or '')[:10]
    try:
        start = (datetime.strptime(first_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y%m%d')
    except ValueError:
        start = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    end = datetime.now().strftime('%Y%m%d')

    df = None
    try:
        df = _fetch_baostock_daily(code, start, end)
    except Exception as e:
        logging.warning(f"[交易日志] 获取 {code} K线失败: {e}")
    if df is None or df.empty:
        return _empty_kline_figure(f'{code} K线数据获取失败（数据源: baostock），请稍后重试')

    dates = [d.strftime('%Y-%m-%d') for d in df['trade_date']]
    dates_sorted = sorted(set(dates))
    vol_colors = ['#ef4444' if c >= o else '#10b981' for o, c in zip(df['open'], df['close'])]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=dates, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K线',
        increasing_line_color='#ef4444', decreasing_line_color='#10b981',
        increasing_fillcolor='#ef4444', decreasing_fillcolor='#10b981',
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=dates, y=df['vol'], marker_color=vol_colors, marker_opacity=0.7,
                         name='成交量', hoverinfo='skip'), row=2, col=1)

    # 交易标记: ▲买入 ▼卖出 + 轨迹连线
    buys, sells, traj_x, traj_y = [], [], [], []
    for e in trades:
        snapped = _snap_date((e.get('trade_date') or '')[:10], dates_sorted)
        if not snapped:
            continue
        price = e.get('price') or 0
        qty = e.get('quantity') or 0
        action = e.get('action') or ('买入' if e.get('direction') == '买入' else '卖出')
        label = f"{action} {qty}股 @ {price:.2f}"
        if e.get('direction') == '买入':
            buys.append((snapped, price, label))
        else:
            sells.append((snapped, price, label))
        traj_x.append(snapped)
        traj_y.append(price)

    if len(traj_x) >= 2:
        fig.add_trace(go.Scatter(
            x=traj_x, y=traj_y, mode='lines', name='交易轨迹',
            line=dict(color='#6366f1', width=1.5, dash='dot'), opacity=0.75, hoverinfo='skip',
        ), row=1, col=1)
    if buys:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in buys], y=[p[1] for p in buys], mode='markers', name='买入',
            marker=dict(symbol='triangle-up', size=13, color='#dc2626', line=dict(width=1, color='#7f1d1d')),
            text=[p[2] for p in buys], hovertemplate='%{text}<extra></extra>',
        ), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in sells], y=[p[1] for p in sells], mode='markers', name='卖出',
            marker=dict(symbol='triangle-down', size=13, color='#16a34a', line=dict(width=1, color='#14532d')),
            text=[p[2] for p in sells], hovertemplate='%{text}<extra></extra>',
        ), row=1, col=1)

    # 持仓成本线
    portfolio = compute_portfolio(entries)
    for h in portfolio['holdings']:
        if h['stock_code'] == code and h.get('quantity', 0) > 0 and h.get('avg_cost'):
            fig.add_hline(y=h['avg_cost'], line_dash='dash', line_color='#f59e0b', line_width=1.2,
                          annotation_text=f"成本 {h['avg_cost']:.2f}",
                          annotation_position='top left', row=1, col=1)

    name = trades[0].get('stock_name') or ''
    fig.update_layout(
        title=dict(text=f"主图 · {code} {name} · 交易标记",
                   font=dict(size=13, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=440,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(font=dict(size=9), orientation='h', y=1.06, x=0),
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', row=1, col=1)
    return fig


def _build_distribution_chart(entries):
    entries = entries or []
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'domain'}, {'type': 'xy'}]],
        subplot_titles=('交易方向分布', '交易类型分布'),
    )
    d_counts = Counter(e.get('direction', '') for e in entries)
    if d_counts:
        fig.add_trace(go.Pie(
            labels=list(d_counts.keys()), values=list(d_counts.values()), hole=0.45,
            marker=dict(colors=['#ef4444', '#10b981']),
            textinfo='label+percent',
        ), 1, 1)
    t_counts = Counter(e.get('trade_type', '') for e in entries)
    if t_counts:
        fig.add_trace(go.Bar(
            x=list(t_counts.keys()), y=list(t_counts.values()),
            marker_color='#6366f1', marker_opacity=0.85,
            text=list(t_counts.values()), textposition='outside', textfont=dict(size=10),
        ), 1, 2)
    fig.update_layout(
        template='plotly_white', height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    return fig


# ===================== 页面布局 =====================

def trading_journal_layout():
    stocks = get_all_stock_codes()
    code_options = [html.Option(value=code) for code, _ in stocks]
    today = datetime.now().strftime('%Y-%m-%d')

    return html.Div([
        html.Div([
            # 标题
            html.Div([
                html.H2([html.I(className='fas fa-book-journal-whills mr-2'), '交易日志'],
                        className='text-xl font-bold text-gray-800'),
                html.P('持仓管理 · 加减仓 · 实时行情盈亏 · 总资金', className='text-xs text-gray-500 mt-1'),
            ], className='mb-4'),

            # 账户资金概览
            html.Div(id='journal-account-cards'),

            html.Div([
                # 左侧：资金管理 + 录入表单
                html.Div([
                    # 资金管理
                    html.Div([
                        html.H3([html.I(className='fas fa-coins mr-2'), '资金管理'],
                                className='text-sm font-bold text-gray-700'),
                        html.Div([
                            html.Div([
                                html.Label('初始资金', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Input(id='journal-initial-capital', type='number', placeholder='如 100000',
                                          className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                            ], className='flex-1 pr-1'),
                            html.Button([html.I(className='fas fa-check mr-1'), '设置'],
                                        id='journal-set-capital-btn',
                                        className='self-end bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-md text-sm font-medium transition-all'),
                        ], className='flex items-end mb-3'),

                        html.Div([
                            html.Div([
                                html.Label('出入金类型', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Dropdown(id='journal-flow-type',
                                             options=[{'label': t, 'value': t} for t in FLOW_TYPES],
                                             value='转入', clearable=False, className='text-sm'),
                            ], className='w-1/3 pr-1'),
                            html.Div([
                                html.Label('金额', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Input(id='journal-flow-amount', type='number', placeholder='0',
                                          className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                            ], className='w-1/3 px-1'),
                            html.Div([
                                html.Label('日期', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.DatePickerSingle(id='journal-flow-date', date=today, className='w-full'),
                            ], className='w-1/3 pl-1'),
                        ], className='flex mb-2'),
                        html.Div([
                            dcc.Input(id='journal-flow-note', type='text', placeholder='备注（可选）',
                                      className='flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 mr-2'),
                            html.Button([html.I(className='fas fa-plus mr-1'), '入账'],
                                        id='journal-add-flow-btn',
                                        className='bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-2 rounded-md text-sm font-medium transition-all'),
                        ], className='flex'),
                        html.Div(id='journal-account-status', className='text-xs mt-2'),

                        # 危险操作区：一键清除数据
                        html.Div([
                            html.Button([html.I(className='fas fa-trash-can mr-1'), '一键清除数据'],
                                        id='journal-clear-btn',
                                        className='mt-3 w-full bg-red-50 hover:bg-red-100 text-red-600 px-3 py-2 rounded-md text-sm font-medium transition-all border border-red-200'),
                            html.Div(id='journal-clear-status', className='text-xs mt-1'),
                        ], className='mt-2'),
                    ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-4'),

                    # 录入表单
                    html.Div([
                        html.H3([html.I(className='fas fa-pen-to-square mr-2'), '录入交易'],
                                className='text-sm font-bold text-gray-700'),
                        html.Div(id='journal-form-status', className='text-xs mt-1'),
                        html.Div([
                            # 股票代码 + 名称
                            html.Div([
                                html.Label('股票代码 *', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Input(id='journal-stock-code', type='text', list='journal-code-list',
                                          placeholder='如 600519.SH 或 600519',
                                          className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                                html.Datalist(id='journal-code-list', children=code_options),
                            ], className='mb-3'),
                            html.Div([
                                html.Label('股票名称', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Input(id='journal-stock-name', type='text', placeholder='自动识别或手动填写',
                                          className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                            ], className='mb-1'),

                            # 持仓上下文（实时显示当前持仓/加仓后均价/快捷减仓数量）
                            html.Div(id='journal-position-context', className='mb-3 min-h-[28px]'),

                            html.Div([
                                html.Div([
                                    html.Label('交易方向', className='block text-xs font-medium text-gray-600 mb-1'),
                                    dcc.Dropdown(id='journal-direction',
                                                 options=[{'label': d, 'value': d} for d in DIRECTIONS],
                                                 value='买入', clearable=False, className='text-sm'),
                                ], className='w-1/2 pr-1'),
                                html.Div([
                                    html.Label('交易日期', className='block text-xs font-medium text-gray-600 mb-1'),
                                    dcc.DatePickerSingle(id='journal-trade-date', date=today, className='w-full'),
                                ], className='w-1/2 pl-1'),
                            ], className='flex mb-3'),

                            html.Div([
                                html.Div([
                                    html.Label('成交价格', className='block text-xs font-medium text-gray-600 mb-1'),
                                    html.Div([
                                        dcc.Input(id='journal-price', type='number', placeholder='0.00',
                                                  className='flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                                        html.Button([html.I(className='fas fa-tag'), ' 现价'],
                                                    id='journal-fill-price-btn',
                                                    className='ml-1 px-2 bg-gray-100 hover:bg-blue-100 text-blue-600 rounded-md text-xs whitespace-nowrap transition-all',
                                                    title='填入 baostock 最新收盘价'),
                                    ], className='flex'),
                                ], className='w-1/2 pr-1'),
                                html.Div([
                                    html.Label('数量(股)', className='block text-xs font-medium text-gray-600 mb-1'),
                                    dcc.Input(id='journal-quantity', type='number', placeholder='0',
                                              className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                                ], className='w-1/2 pl-1'),
                            ], className='flex mb-3'),

                            html.Div([
                                html.Div([
                                    html.Label('手续费', className='block text-xs font-medium text-gray-600 mb-1'),
                                    dcc.Input(id='journal-fee', type='number', placeholder='0',
                                              className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                                ], className='w-1/2 pr-1'),
                                html.Div([
                                    html.Label('交易类型', className='block text-xs font-medium text-gray-600 mb-1'),
                                    dcc.Dropdown(id='journal-trade-type',
                                                 options=[{'label': t, 'value': t} for t in TRADE_TYPES],
                                                 value='短线', clearable=False, className='text-sm'),
                                ], className='w-1/2 pl-1'),
                            ], className='flex mb-3'),

                            html.Div([
                                html.Label('标签（逗号分隔）', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Input(id='journal-tags', type='text', placeholder='如 题材, 龙头, 突破',
                                          className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'),
                            ], className='mb-3'),

                            html.Div([
                                html.Label('交易理由', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Textarea(id='journal-reason', placeholder='买入/卖出的逻辑依据...',
                                             className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                                             style={'height': '56px'}),
                            ], className='mb-3'),
                            html.Div([
                                html.Label('备注', className='block text-xs font-medium text-gray-600 mb-1'),
                                dcc.Textarea(id='journal-memo', placeholder='其他补充说明...',
                                             className='w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500',
                                             style={'height': '56px'}),
                            ], className='mb-3'),

                            html.Div([
                                html.Button([html.I(className='fas fa-save mr-1'), '保存记录'],
                                            id='journal-save-btn',
                                            className='flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-all'),
                                html.Button([html.I(className='fas fa-times mr-1'), '取消'],
                                            id='journal-cancel-btn',
                                            className='ml-2 px-4 py-2 rounded-md text-sm font-medium bg-gray-100 text-gray-600 hover:bg-gray-200 transition-all'),
                            ], className='flex'),
                        ]),
                    ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-4'),
                ], className='w-full lg:w-1/3'),

                # 右侧：持仓 + 记录 + 图表
                html.Div([
                    # 持仓面板
                    html.Div([
                        html.Div([
                            html.H3([html.I(className='fas fa-boxes-stacked mr-2'), '当前持仓'],
                                    className='text-sm font-bold text-gray-700'),
                            html.Div([
                                html.Span(id='journal-price-status', className='text-xs text-gray-400 mr-2'),
                                html.Button([html.I(className='fas fa-rotate mr-1'), '刷新行情'],
                                            id='journal-refresh-btn',
                                            className='bg-blue-50 hover:bg-blue-100 text-blue-600 px-3 py-1.5 rounded-md text-xs font-medium transition-all'),
                            ], className='flex items-center'),
                        ], className='flex items-center justify-between mb-2 pb-2 border-b border-gray-100'),
                        html.Div(id='journal-holdings-container', className='overflow-x-auto'),
                    ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-4'),

                    # 主图K线 · 交易画线标记
                    html.Div([
                        html.Div([
                            html.H3([html.I(className='fas fa-chart-simple mr-2'), '主图K线 · 交易标记'],
                                    className='text-sm font-bold text-gray-700'),
                            dcc.Dropdown(id='journal-kline-stock',
                                         placeholder='选择股票查看买卖标记 / 轨迹线 / 成本线',
                                         className='w-72 text-sm', clearable=True),
                        ], className='flex items-center justify-between mb-1 pb-2 border-b border-gray-100 flex-wrap gap-2'),
                        html.P('▲买入 ▼卖出 · 蓝色虚线为交易轨迹 · 橙色虚线为持仓成本（数据源: baostock 不复权，与录入成交价对齐）',
                               className='text-xs text-gray-400 mb-1'),
                        dcc.Graph(id='journal-kline-chart', figure=go.Figure()),
                    ], className='bg-white rounded-xl shadow-sm border border-gray-100 p-4 mb-4'),

                    # 筛选栏
                    html.Div([
                        html.Div([
                            html.I(className='fas fa-search text-gray-400 mr-2'),
                            dcc.Input(id='journal-filter-search', type='text', placeholder='搜索代码/名称/理由/标签...',
                                      className='flex-1 border-0 outline-none text-sm bg-transparent', debounce=True),
                        ], className='flex items-center bg-white border border-gray-300 rounded-lg px-3 py-2 flex-1'),
                        dcc.Dropdown(id='journal-filter-direction',
                                     options=[{'label': d, 'value': d} for d in [ALL_OPTION] + DIRECTIONS],
                                     value=ALL_OPTION, clearable=False, className='w-28 text-sm ml-2'),
                        dcc.Dropdown(id='journal-filter-type',
                                     options=[{'label': t, 'value': t} for t in [ALL_OPTION] + TRADE_TYPES],
                                     value=ALL_OPTION, clearable=False, className='w-28 text-sm ml-2'),
                        html.Button([html.I(className='fas fa-file-csv mr-1'), '导出'],
                                    id='journal-export-btn',
                                    className='ml-2 bg-emerald-600 hover:bg-emerald-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap'),
                    ], className='flex items-center mb-3'),

                    html.Div([
                        html.Span('交易记录', className='text-sm font-semibold text-gray-700'),
                        html.Span(id='journal-entry-count', className='text-xs text-gray-400 ml-2'),
                    ], className='flex items-center justify-between mb-2'),

                    html.Div(id='journal-table-container', className='overflow-x-auto max-h-[40vh] overflow-y-auto'),

                    # 每日盈亏统计
                    html.Div(id='journal-daily-summary', className='mt-4 mb-2'),

                    # 图表
                    html.Div([
                        html.Div([
                            dcc.Graph(id='journal-holdings-chart', figure=go.Figure(), config={'displayModeBar': False}),
                        ], className='w-full lg:w-1/3 pr-0 lg:pr-2'),
                        html.Div([
                            dcc.Graph(id='journal-pnl-chart', figure=go.Figure(), config={'displayModeBar': False}),
                        ], className='w-full lg:w-1/3 px-0 lg:px-2 mt-4 lg:mt-0'),
                        html.Div([
                            dcc.Graph(id='journal-distribution-chart', figure=go.Figure(), config={'displayModeBar': False}),
                        ], className='w-full lg:w-1/3 pl-0 lg:pl-2 mt-4 lg:mt-0'),
                    ], className='flex flex-col lg:flex-row mt-4'),
                ], className='w-full lg:w-2/3 mt-4 lg:mt-0 lg:ml-4'),
            ], className='flex flex-col lg:flex-row mt-4'),
        ], className='container mx-auto px-4 py-6'),

        # Stores & 组件
        dcc.Store(id='journal-data-store', data=get_trade_entries(), storage_type='memory'),
        dcc.Store(id='journal-prices-store', data={}, storage_type='memory'),
        dcc.Store(id='journal-account-version', data=0, storage_type='memory'),
        dcc.Store(id='journal-edit-id', data=None, storage_type='memory'),
        dcc.Interval(id='journal-price-interval', interval=300000, n_intervals=0),
        dcc.Download(id='journal-download'),
        # 一键清除数据二次确认对话框
        dcc.ConfirmDialog(
            id='journal-clear-confirm',
            message='确定要清除当前账户的全部交易日志数据吗？\n将删除所有交易记录、出入金流水，并重置初始资金。此操作不可恢复，请谨慎确认！',
        ),
    ])


# ===================== 回调注册 =====================

def register_trading_journal_callbacks(app):
    """注册交易日志页面回调"""

    # ─── 登录状态变化时重新加载交易日志（数据隔离 + 防切换用户泄露）───
    @app.callback(
        Output('journal-data-store', 'data', allow_duplicate=True),
        Input('auth-state', 'data'),
        prevent_initial_call='initial_duplicate'
    )
    def load_journal_on_auth_change(auth_state):
        """用户身份变化（登录/退出/切换账户）时，重新按当前用户拉取交易日志。
        既保证登录后正确加载数据，也避免同一浏览器切换账户时残留上一用户数据。"""
        return get_trade_entries()

    # ─── 一键清除数据：打开二次确认对话框 ───
    @app.callback(
        Output('journal-clear-confirm', 'displayed'),
        Input('journal-clear-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def open_clear_confirm(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return True

    # ─── 一键清除数据：确认执行 / 取消 ───
    @app.callback(
        [Output('journal-clear-confirm', 'displayed', allow_duplicate=True),
         Output('journal-clear-status', 'children'),
         Output('journal-data-store', 'data', allow_duplicate=True),
         Output('journal-account-version', 'data', allow_duplicate=True)],
        [Input('journal-clear-confirm', 'submit_n_clicks'),
         Input('journal-clear-confirm', 'cancel_n_clicks')],
        [State('journal-account-version', 'data')],
        prevent_initial_call=True
    )
    def handle_clear_data(submit_clicks, cancel_clicks, version):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        prop_id = ctx.triggered[0]['prop_id']
        if prop_id.endswith('cancel_n_clicks'):
            return False, dash.no_update, dash.no_update, dash.no_update
        if prop_id.endswith('submit_n_clicks'):
            ok = clear_all_trade_data()
            if ok:
                msg = html.Span([html.I(className='fas fa-circle-check text-green-500 mr-1'), '数据已全部清除'],
                                className='text-green-600')
                return False, msg, [], (version or 0) + 1
            msg = html.Span([html.I(className='fas fa-circle-exclamation text-red-500 mr-1'), '清除失败，请稍后重试'],
                            className='text-red-500')
            return False, msg, dash.no_update, dash.no_update
        return False, dash.no_update, dash.no_update, dash.no_update

    # ─── 行情刷新 ───
    @app.callback(
        [Output('journal-prices-store', 'data'),
         Output('journal-price-status', 'children')],
        [Input('journal-refresh-btn', 'n_clicks'),
         Input('journal-price-interval', 'n_intervals'),
         Input('journal-data-store', 'data')],
    )
    def refresh_prices(n_clicks, n_intervals, entries):
        entries = entries or []
        portfolio = compute_portfolio(entries)
        codes = [h['stock_code'] for h in portfolio['holdings']]
        prices = {}
        for code in codes:
            prices[code] = _fetch_current_price(code)

        now = datetime.now().strftime('%H:%M:%S')
        status = f'行情更新于 {now} (baostock)'
        if codes:
            fetched = sum(1 for v in prices.values() if v is not None)
            status = f'行情更新于 {now} (baostock) · {fetched}/{len(codes)} 只'
        return prices, status

    # ─── 主渲染：账户 + 持仓 + 记录 + 图表 ───
    @app.callback(
        [Output('journal-account-cards', 'children'),
         Output('journal-holdings-container', 'children'),
         Output('journal-table-container', 'children'),
         Output('journal-entry-count', 'children'),
         Output('journal-code-list', 'children'),
         Output('journal-holdings-chart', 'figure'),
         Output('journal-pnl-chart', 'figure'),
         Output('journal-distribution-chart', 'figure'),
         Output('journal-daily-summary', 'children')],
        [Input('journal-data-store', 'data'),
         Input('journal-prices-store', 'data'),
         Input('journal-account-version', 'data'),
         Input('journal-filter-search', 'value'),
         Input('journal-filter-direction', 'value'),
         Input('journal-filter-type', 'value')],
    )
    def render_journal(entries, prices, account_version, search, direction, trade_type):
        entries = entries or []
        prices = prices or {}

        account = get_account()
        flows = get_capital_flows()
        initial = account.get('initial_capital') or 0
        net_flows = sum((f.get('amount') or 0) if f.get('flow_type') == '转入'
                        else -(f.get('amount') or 0) for f in flows)

        portfolio = compute_portfolio(entries)
        holdings = portfolio['holdings']

        # 附加上现价、市值、浮动盈亏
        total_mv = 0.0
        total_float_pnl = 0.0
        for h in holdings:
            cur = prices.get(h['stock_code'])
            h['current_price'] = cur
            if cur is not None:
                h['market_value'] = cur * h['quantity']
                h['float_pnl'] = (cur - h['avg_cost']) * h['quantity']
                h['float_pnl_pct'] = (cur - h['avg_cost']) / h['avg_cost'] * 100 if h['avg_cost'] else 0.0
                total_mv += h['market_value']
                total_float_pnl += h['float_pnl']
            else:
                h['market_value'] = None
                h['float_pnl'] = None
                h['float_pnl_pct'] = None

        available_cash = initial + net_flows + portfolio['cash_delta']
        total_assets = available_cash + total_mv
        principal = initial + net_flows
        total_pnl = total_assets - principal
        total_pnl_pct = (total_pnl / principal * 100) if principal else 0.0

        account_cards = _build_account_cards(initial, net_flows, available_cash, total_mv,
                                             total_assets, total_pnl, total_pnl_pct, flows)
        holdings_table = _build_holdings_table(holdings)

        filtered = _filter_entries(entries, search, direction, trade_type)
        trade_table = _build_table(filtered)
        count = f'共 {len(filtered)} 条记录'

        stocks = get_all_stock_codes()
        code_options = [html.Option(value=code) for code, _ in stocks]

        holdings_chart = _build_holdings_chart(holdings)
        daily_pnl_chart = _build_daily_pnl_chart(portfolio['realized_series'])
        daily_summary = _build_daily_pnl_badges(portfolio['realized_series'])
        dist_chart = _build_distribution_chart(filtered)

        return (account_cards, holdings_table, trade_table, count, code_options,
                holdings_chart, daily_pnl_chart, dist_chart, daily_summary)

    # ─── 主图K线: 持仓表按钮点击 → 选中股票 ───
    @app.callback(
        Output('journal-kline-stock', 'value'),
        [Input({'type': 'journal-kline-btn', 'index': dash.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def select_kline_stock(clicks):
        trig = _valid_click_trigger()
        if not trig:
            return dash.no_update
        code = _parse_pattern_id(trig['prop_id']).get('index')
        return code if code else dash.no_update

    # ─── 主图K线: 选中股票 → K线 + 买卖标记/轨迹/成本线 ───
    @app.callback(
        [Output('journal-kline-chart', 'figure'),
         Output('journal-kline-stock', 'options')],
        [Input('journal-kline-stock', 'value'),
         Input('journal-data-store', 'data')],
    )
    def render_kline(code, entries):
        entries = entries or []
        # 下拉选项: 持仓股票优先, 其余有交易记录的股票
        portfolio = compute_portfolio(entries)
        holding_codes = [h['stock_code'] for h in portfolio['holdings']]
        all_codes = []
        for e in entries:
            c = e.get('stock_code')
            if c and c not in all_codes:
                all_codes.append(c)
        name_map = {e.get('stock_code'): e.get('stock_name') for e in entries if e.get('stock_code')}
        options = []
        for c in holding_codes + [x for x in all_codes if x not in holding_codes]:
            label = f"{c} {name_map.get(c, '')}".strip()
            options.append({'label': label, 'value': c})
        return _build_kline_figure(code, entries), options

    # ─── 保存 / 取消 ───
    @app.callback(
        [Output('journal-form-status', 'children'),
         Output('journal-stock-code', 'value'),
         Output('journal-stock-name', 'value'),
         Output('journal-direction', 'value'),
         Output('journal-trade-date', 'date'),
         Output('journal-price', 'value'),
         Output('journal-quantity', 'value'),
         Output('journal-fee', 'value'),
         Output('journal-trade-type', 'value'),
         Output('journal-tags', 'value'),
         Output('journal-reason', 'value'),
         Output('journal-memo', 'value'),
         Output('journal-edit-id', 'data'),
         Output('journal-data-store', 'data')],
        [Input('journal-save-btn', 'n_clicks'),
         Input('journal-cancel-btn', 'n_clicks')],
        [State('journal-stock-code', 'value'),
         State('journal-stock-name', 'value'),
         State('journal-direction', 'value'),
         State('journal-trade-date', 'date'),
         State('journal-price', 'value'),
         State('journal-quantity', 'value'),
         State('journal-fee', 'value'),
         State('journal-trade-type', 'value'),
         State('journal-tags', 'value'),
         State('journal-reason', 'value'),
         State('journal-memo', 'value'),
         State('journal-edit-id', 'data')],
        prevent_initial_call=True
    )
    def handle_save_cancel(save_n, cancel_n, code, name, direction, trade_date, price, quantity,
                           fee, trade_type, tags, reason, memo, edit_id):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

        today = datetime.now().strftime('%Y-%m-%d')
        reset_values = ['', '', '买入', today, None, None, None, '短线', '', '', '']
        reset_msg = html.Span('', className='text-xs')

        if triggered_id == 'journal-cancel-btn':
            return [reset_msg] + reset_values + [None, dash.no_update]

        if triggered_id != 'journal-save-btn':
            return [dash.no_update] * 14

        code = _normalize_code(code)
        if not code:
            err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '请输入股票代码'],
                            className='text-xs')
            return [err] + [dash.no_update] * 13

        price = _to_float(price) or 0
        quantity = _to_int(quantity) or 0
        fee = _to_float(fee) or 0

        if quantity <= 0:
            err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '请输入有效数量'],
                            className='text-xs')
            return [err] + [dash.no_update] * 13

        # 计算当前持仓（排除正在编辑的记录）
        all_entries = get_trade_entries()
        base_entries = [e for e in all_entries if e.get('id') != edit_id]
        cur_qty = _current_position(base_entries, code)

        if direction == '买入':
            action = '开仓' if cur_qty == 0 else '加仓'
        else:
            if cur_qty <= 0:
                err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '无持仓可卖出'],
                                className='text-xs')
                return [err] + [dash.no_update] * 13
            if quantity > cur_qty:
                err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'),
                                 f'卖出数量 {quantity} 超过当前持仓 {cur_qty}'],
                                className='text-xs')
                return [err] + [dash.no_update] * 13
            action = '清仓' if quantity == cur_qty else '减仓'

        amount = round(price * quantity, 2)
        data = {
            'stock_code': code,
            'stock_name': (name or '').strip() or _lookup_name(code),
            'direction': direction,
            'action': action,
            'trade_date': trade_date or '',
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'fee': fee,
            'trade_type': trade_type or '短线',
            'tags': (tags or '').strip(),
            'reason': (reason or '').strip(),
            'memo': (memo or '').strip(),
        }

        if edit_id:
            ok = update_trade_entry(edit_id, data)
            msg = html.Span([html.I(className='fas fa-check-circle text-green-500 mr-1'), '已更新'],
                            className='text-xs')
        else:
            new_id = add_trade_entry(data)
            ok = new_id is not None
            msg = html.Span([html.I(className='fas fa-check-circle text-green-500 mr-1'),
                             f'已保存 · {action}'], className='text-xs')

        if not ok:
            msg = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '保存失败'],
                            className='text-xs')
            return [msg] + [dash.no_update] * 13

        return [msg] + reset_values + [None, get_trade_entries()]

    # ─── 编辑：回填表单 ───
    @app.callback(
        [Output('journal-form-status', 'children', allow_duplicate=True),
         Output('journal-stock-code', 'value', allow_duplicate=True),
         Output('journal-stock-name', 'value', allow_duplicate=True),
         Output('journal-direction', 'value', allow_duplicate=True),
         Output('journal-trade-date', 'date', allow_duplicate=True),
         Output('journal-price', 'value', allow_duplicate=True),
         Output('journal-quantity', 'value', allow_duplicate=True),
         Output('journal-fee', 'value', allow_duplicate=True),
         Output('journal-trade-type', 'value', allow_duplicate=True),
         Output('journal-tags', 'value', allow_duplicate=True),
         Output('journal-reason', 'value', allow_duplicate=True),
         Output('journal-memo', 'value', allow_duplicate=True),
         Output('journal-edit-id', 'data', allow_duplicate=True)],
        [Input({'type': 'journal-edit-btn', 'index': dash.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def load_entry_for_edit(clicks):
        trig = _valid_click_trigger()
        if not trig:
            return [dash.no_update] * 13

        entry_id = _parse_pattern_id(trig['prop_id']).get('index')
        if entry_id is None:
            return [dash.no_update] * 13

        entry = get_trade_entry(entry_id)
        if not entry:
            return [dash.no_update] * 13

        msg = html.Span([html.I(className='fas fa-pen text-blue-500 mr-1'), f'正在编辑 #{entry_id}'],
                        className='text-xs')

        return [
            msg,
            entry.get('stock_code', ''),
            entry.get('stock_name', ''),
            entry.get('direction', '买入'),
            entry.get('trade_date', '') or datetime.now().strftime('%Y-%m-%d'),
            entry.get('price') or None,
            entry.get('quantity') or None,
            entry.get('fee') or None,
            entry.get('trade_type', '短线'),
            entry.get('tags', ''),
            entry.get('reason', ''),
            entry.get('memo', ''),
            entry_id,
        ]

    # ─── 删除交易 ───
    @app.callback(
        [Output('journal-data-store', 'data', allow_duplicate=True),
         Output('journal-edit-id', 'data', allow_duplicate=True)],
        [Input({'type': 'journal-delete-btn', 'index': dash.ALL}, 'n_clicks')],
        prevent_initial_call=True
    )
    def delete_entry(clicks):
        trig = _valid_click_trigger()
        if not trig:
            return [dash.no_update, dash.no_update]

        entry_id = _parse_pattern_id(trig['prop_id']).get('index')
        if entry_id is None:
            return [dash.no_update, dash.no_update]

        delete_trade_entry(entry_id)
        return [get_trade_entries(), None]

    # ─── 删除出入金流水 ───
    @app.callback(
        Output('journal-account-version', 'data', allow_duplicate=True),
        [Input({'type': 'journal-flow-del', 'index': dash.ALL}, 'n_clicks')],
        [State('journal-account-version', 'data')],
        prevent_initial_call=True
    )
    def delete_flow(clicks, version):
        trig = _valid_click_trigger()
        if not trig:
            return dash.no_update
        flow_id = _parse_pattern_id(trig['prop_id']).get('index')
        if flow_id is None:
            return dash.no_update
        delete_capital_flow(flow_id)
        return (version or 0) + 1

    # ─── 加减仓优化: 持仓表快捷按钮 → 预填表单 ───
    @app.callback(
        [Output('journal-stock-code', 'value', allow_duplicate=True),
         Output('journal-stock-name', 'value', allow_duplicate=True),
         Output('journal-direction', 'value', allow_duplicate=True),
         Output('journal-price', 'value', allow_duplicate=True),
         Output('journal-quantity', 'value', allow_duplicate=True),
         Output('journal-edit-id', 'data', allow_duplicate=True),
         Output('journal-form-status', 'children', allow_duplicate=True)],
        [Input({'type': 'journal-pos-btn', 'index': dash.ALL}, 'n_clicks')],
        [State('journal-data-store', 'data'),
         State('journal-prices-store', 'data')],
        prevent_initial_call=True
    )
    def quick_position_prefill(clicks, entries, prices):
        trig = _valid_click_trigger()
        if not trig:
            return [dash.no_update] * 7
        payload = _parse_pattern_id(trig['prop_id']).get('index', '')
        try:
            code, act = payload.split('|')
        except ValueError:
            return [dash.no_update] * 7
        # index 必须纯 ASCII (buy/sell): Dash 4.1 下中文 index 会导致 triggered value 恒为 None
        direction = {'buy': '买入', 'sell': '卖出'}.get(act)
        if not direction:
            return [dash.no_update] * 7

        entries = entries or []
        prices = prices or {}
        name = ''
        for e in entries:
            if _normalize_code(e.get('stock_code', '')) == _normalize_code(code):
                name = e.get('stock_name', '') or name
                break
        price = prices.get(code) or prices.get(_normalize_code(code))
        msg = html.Span([html.I(className='fas fa-bolt text-amber-500 mr-1'),
                         f'{direction} {code} · {"已填入现价" if price else "请填写价格"}'],
                        className='text-xs')
        return [code, name, direction,
                round(price, 3) if price else None,
                None, None, msg]

    # ─── 加减仓优化: 录入时实时持仓上下文 ───
    @app.callback(
        Output('journal-position-context', 'children'),
        [Input('journal-stock-code', 'value'),
         Input('journal-direction', 'value'),
         Input('journal-price', 'value'),
         Input('journal-quantity', 'value'),
         Input('journal-data-store', 'data'),
         Input('journal-prices-store', 'data')],
    )
    def update_position_context(code, direction, price, quantity, entries, prices):
        code = (code or '').strip()
        if not code:
            return html.Div()
        entries = entries or []
        portfolio = compute_portfolio(entries)
        pos = _find_position(portfolio['holdings'], code)
        norm = _normalize_code(code)
        has_trades = any(_normalize_code(e.get('stock_code', '')) == norm for e in entries)

        children = []
        if pos:
            qty, avg = pos['quantity'], pos['avg_cost'] or 0
            children.append(html.Span([
                html.I(className='fas fa-box-open text-amber-500 mr-1'),
                f'当前持仓 {qty} 股 · 均价 {avg:.2f}',
            ], className='text-amber-700 bg-amber-50 border border-amber-100 px-2 py-1 rounded'))

            price_v = _to_float(price)
            qty_v = _to_int(quantity)
            if price_v and qty_v and qty_v > 0:
                amt = price_v * qty_v
                if direction == '买入':
                    new_qty = qty + qty_v
                    new_avg = (qty * avg + amt) / new_qty if new_qty else 0
                    children.append(html.Span(
                        f'预计金额 {_fmt_money(amt)} 元 · 加仓后均价 ≈ {new_avg:.2f}',
                        className='text-gray-600 bg-gray-50 px-2 py-1 rounded'))
                    # 资金充足性提示
                    account = get_account()
                    if account.get('initial_capital'):
                        flows = get_capital_flows()
                        net_flows = sum((f.get('amount') or 0) if f.get('flow_type') == '转入'
                                        else -(f.get('amount') or 0) for f in flows)
                        available = account['initial_capital'] + net_flows + portfolio['cash_delta']
                        if amt > available:
                            children.append(html.Span([
                                html.I(className='fas fa-triangle-exclamation text-orange-500 mr-1'),
                                f'超过可用资金 {_fmt_money(available)} 元',
                            ], className='text-orange-600 bg-orange-50 border border-orange-100 px-2 py-1 rounded'))
                else:
                    sell_qty = min(qty_v, qty)
                    children.append(html.Span(
                        f'预计回笼 {_fmt_money(price_v * sell_qty)} 元' + ('（超持仓部分无效）' if qty_v > qty else ''),
                        className=('text-orange-600 bg-orange-50 border border-orange-100' if qty_v > qty
                                   else 'text-gray-600 bg-gray-50') + ' px-2 py-1 rounded'))
            if direction == '卖出':
                # 快捷减仓数量
                for label, frac in [('¼仓', 0.25), ('半仓', 0.5), ('全部', 1.0)]:
                    children.append(html.Button(
                        label, id={'type': 'journal-qty-btn', 'index': str(frac)},
                        className='px-2 py-1 bg-white border border-gray-200 rounded text-xs text-gray-600 '
                                  'hover:border-blue-400 hover:text-blue-600 transition-all'))
        elif has_trades:
            children.append(html.Span('该股已清仓 · 有历史交易记录',
                                      className='text-gray-500 bg-gray-50 px-2 py-1 rounded'))
        else:
            children.append(html.Span('无持仓 · 买入将开新仓',
                                      className='text-gray-400 bg-gray-50 px-2 py-1 rounded'))
        return html.Div(children, className='flex flex-wrap items-center gap-2')

    # ─── 加减仓优化: 快捷减仓数量按钮 ───
    @app.callback(
        Output('journal-quantity', 'value', allow_duplicate=True),
        [Input({'type': 'journal-qty-btn', 'index': dash.ALL}, 'n_clicks')],
        [State('journal-stock-code', 'value'),
         State('journal-data-store', 'data')],
        prevent_initial_call=True
    )
    def set_quick_quantity(clicks, code, entries):
        trig = _valid_click_trigger()
        if not trig:
            return dash.no_update
        try:
            frac = float(_parse_pattern_id(trig['prop_id']).get('index'))
        except (TypeError, ValueError):
            return dash.no_update
        portfolio = compute_portfolio(entries or [])
        pos = _find_position(portfolio['holdings'], (code or '').strip())
        if not pos or pos['quantity'] <= 0:
            return dash.no_update
        pos_qty = pos['quantity']
        if frac >= 1:
            return pos_qty
        # 卖出允许零股, 直接按比例取整 (至少100股, 除非持仓不足100股)
        q = int(pos_qty * frac)
        if pos_qty >= 100:
            q = max(100, q)
        return min(q, pos_qty)

    # ─── 加减仓优化: 填入现价按钮 ───
    @app.callback(
        Output('journal-price', 'value', allow_duplicate=True),
        [Input('journal-fill-price-btn', 'n_clicks')],
        [State('journal-stock-code', 'value'),
         State('journal-prices-store', 'data')],
        prevent_initial_call=True
    )
    def fill_current_price(n_clicks, code, prices):
        if not n_clicks:
            return dash.no_update
        code = (code or '').strip()
        if not code:
            return dash.no_update
        norm = _normalize_code(code)
        p = (prices or {}).get(code) or (prices or {}).get(norm)
        if p is None:
            p = _fetch_current_price(code)  # baostock (带缓存)
        return round(p, 3) if p else dash.no_update

    # ─── 自动填充股票名称 ───
    @app.callback(
        Output('journal-stock-name', 'value', allow_duplicate=True),
        [Input('journal-stock-code', 'value')],
        [State('journal-stock-name', 'value'),
         State('journal-edit-id', 'data')],
        prevent_initial_call=True
    )
    def autofill_stock_name(code, current_name, edit_id):
        if not code or edit_id is not None:
            return dash.no_update
        if current_name and str(current_name).strip():
            return dash.no_update
        name = _lookup_name(code)
        return name if name else dash.no_update

    # ─── 设置初始资金 ───
    @app.callback(
        [Output('journal-account-status', 'children'),
         Output('journal-initial-capital', 'value'),
         Output('journal-account-version', 'data')],
        [Input('journal-set-capital-btn', 'n_clicks')],
        [State('journal-initial-capital', 'value'),
         State('journal-account-version', 'data')],
        prevent_initial_call=True
    )
    def handle_set_capital(n_clicks, value, version):
        if not n_clicks:
            return [dash.no_update, dash.no_update, dash.no_update]
        value = _to_float(value)
        if value is None or value < 0:
            err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '请输入有效金额'],
                            className='text-xs')
            return [err, dash.no_update, dash.no_update]
        set_initial_capital(value)
        msg = html.Span([html.I(className='fas fa-check-circle text-green-500 mr-1'), '初始资金已更新'],
                        className='text-xs')
        return [msg, None, (version or 0) + 1]

    # ─── 新增出入金 ───
    @app.callback(
        [Output('journal-account-status', 'children', allow_duplicate=True),
         Output('journal-flow-amount', 'value', allow_duplicate=True),
         Output('journal-flow-note', 'value', allow_duplicate=True),
         Output('journal-account-version', 'data', allow_duplicate=True)],
        [Input('journal-add-flow-btn', 'n_clicks')],
        [State('journal-flow-type', 'value'),
         State('journal-flow-amount', 'value'),
         State('journal-flow-date', 'date'),
         State('journal-flow-note', 'value'),
         State('journal-account-version', 'data')],
        prevent_initial_call=True
    )
    def handle_add_flow(n_clicks, flow_type, amount, flow_date, note, version):
        if not n_clicks:
            return [dash.no_update] * 4
        amount = _to_float(amount)
        if amount is None or amount <= 0:
            err = html.Span([html.I(className='fas fa-exclamation-circle text-red-500 mr-1'), '请输入有效金额'],
                            className='text-xs')
            return [err, dash.no_update, dash.no_update, dash.no_update]
        add_capital_flow(flow_type or '转入', amount, flow_date or '', (note or '').strip())
        msg = html.Span([html.I(className='fas fa-check-circle text-green-500 mr-1'), '出入金已入账'],
                        className='text-xs')
        return [msg, None, '', (version or 0) + 1]

    # ─── 导出 CSV ───
    @app.callback(
        Output('journal-download', 'data'),
        [Input('journal-export-btn', 'n_clicks')],
        [State('journal-data-store', 'data')],
        prevent_initial_call=True
    )
    def export_csv(n_clicks, entries):
        if not n_clicks or not entries:
            return dash.no_update

        df = pd.DataFrame(entries)
        col_order = ['trade_date', 'stock_code', 'stock_name', 'direction', 'action',
                     'trade_type', 'price', 'quantity', 'amount', 'fee',
                     'tags', 'reason', 'memo', 'created_at', 'updated_at']
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
        return dcc.send_data_frame(df.to_csv, 'trading_journal.csv', index=False)