import dash
import json
import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from .data_manager import (get_and_process_data, validate_code, get_stock_name, batch_get_stock_names,
                           set_data_source, get_data_source, get_available_sources)
from .strategy_manager import double_moving_average_strategy, simple_moving_average_strategy, exponential_moving_average_strategy
from .metrics_calculator import calculate_sharpe_ratio, calculate_max_drawdown, simulate_trades
from .signal_enhancer import enhance_signals, simulate_trades_detailed
from .confluence import compute_confluence, build_confluence_panel
from .forecast_engine import run_forecast, build_forecast_panel
from .trendlines import auto_trendlines, add_trendline_overlays, trendline_note
from .signal_scanner import save_scan_settings, get_latest_scan
from .database_manager import (init_db, insert_stock_code, get_all_stock_codes, read_cached_codes,
                               save_codes, get_stock_name_map, delete_stock_code, search_stock_codes,
                               update_stock_code, is_default_admin_password,
                               create_watchlist, get_all_watchlists, delete_watchlist,
                               add_stock_to_watchlist, remove_stock_from_watchlist,
                               get_watchlist_stocks, get_stock_watchlist_map, normalize_stock_code,
                               get_marked_stocks, add_marked_stock, remove_marked_stock)
from . import auth
from .security import load_or_create_secret_key
from .factor_page import (factor_page_layout, register_factor_callbacks, build_factor_score_panel,
                          get_factor_score_snapshot, get_stock_factor_info, set_active_model,
                          list_saved_models, get_active_model_name, _normalize_stock_code)
from .trend_analysis import trend_analysis_layout, register_trend_callbacks
from .stock_game import stock_game_layout, register_stock_game_callbacks
from .market_overview import market_overview_layout, register_market_callbacks
from .sector_fund_strength import sector_fund_strength_layout, register_sector_fund_callbacks
from .trading_journal import trading_journal_layout, register_trading_journal_callbacks

# 初始化 Dash 应用
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# 会话密钥（用于登录状态签名 cookie，随机生成并持久化，避免硬编码被伪造冒用）
app.server.secret_key = load_or_create_secret_key()
app.server.config['SESSION_TYPE'] = 'filesystem'

# 引入外部资源
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>股票分析平台</title>
        {%favicon%}
        {%css%}
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"/>
    </head>
    <body class="bg-slate-50 font-sans">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# 初始化数据库
init_db()

# 获取缓存的代码
cached_codes = read_cached_codes()

# 缓存股票名称映射，避免频繁查询数据库
_cached_stock_name_map = None


def _get_cached_name_map():
    """获取缓存的股票名称映射"""
    global _cached_stock_name_map
    if _cached_stock_name_map is None:
        _cached_stock_name_map = get_stock_name_map()
    return _cached_stock_name_map


def _refresh_name_map_cache():
    """刷新股票名称映射缓存"""
    global _cached_stock_name_map
    _cached_stock_name_map = get_stock_name_map()

# 共享导航栏
def create_header(current_path='/', username=None):
    is_analysis = current_path == '/' or current_path == ''
    is_management = current_path == '/stock-management'
    is_factor = current_path == '/factor-training'
    is_trend = current_path == '/trend-analysis'
    is_game = current_path == '/stock-game'
    is_market = current_path == '/market-overview'
    is_sector = current_path == '/sector-fund-strength'
    is_journal = current_path == '/trading-journal'
    return html.Header([
        html.Div([
            html.Div([
                html.I(className="fas fa-chart-line text-yellow-400 text-xl"),
                html.Span("股票分析平台", className="text-xl md:text-2xl font-bold ml-2")
            ], className="flex items-center text-white"),
            html.Nav([
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-chart-simple {'text-white' if is_analysis else 'text-blue-200'} mr-1"),
                        html.Span("分析", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_analysis else 'hover:bg-white/10'}"),
                    href='/', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-database {'text-white' if is_management else 'text-blue-200'} mr-1"),
                        html.Span("股票管理", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_management else 'hover:bg-white/10'}"),
                    href='/stock-management', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-chart-line {'text-white' if is_factor else 'text-blue-200'} mr-1"),
                        html.Span("因子训练", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_factor else 'hover:bg-white/10'}"),
                    href='/factor-training', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-magnifying-glass-chart {'text-white' if is_trend else 'text-blue-200'} mr-1"),
                        html.Span("趋势分析", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_trend else 'hover:bg-white/10'}"),
                    href='/trend-analysis', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-gamepad {'text-white' if is_game else 'text-blue-200'} mr-1"),
                        html.Span("模拟炒股", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_game else 'hover:bg-white/10'}"),
                    href='/stock-game', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-globe {'text-white' if is_market else 'text-blue-200'} mr-1"),
                        html.Span("每日五面", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_market else 'hover:bg-white/10'}"),
                    href='/market-overview', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-gauge-high {'text-white' if is_sector else 'text-blue-200'} mr-1"),
                        html.Span("板块资金强度", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_sector else 'hover:bg-white/10'}"),
                    href='/sector-fund-strength', className="no-underline"
                ),
                dcc.Link(
                    html.Div([
                        html.I(className=f"fas fa-book-journal-whills {'text-white' if is_journal else 'text-blue-200'} mr-1"),
                        html.Span("交易日志", className="hidden sm:inline")
                    ], className=f"px-3 py-2 rounded-lg transition-all duration-300 flex items-center text-sm font-medium {'bg-white/20 shadow-lg' if is_journal else 'hover:bg-white/10'}"),
                    href='/trading-journal', className="no-underline"
                )
            ], className="flex items-center space-x-1 md:space-x-3"),
            html.Div([
                html.Div([
                    html.I(className="fas fa-user-circle text-blue-200 mr-1"),
                    html.Span(username or "未登录", className="text-white text-sm font-medium")
                ], className="flex items-center mr-1"),
                html.Button(
                    [html.I(className='fas fa-sync-alt mr-1'), "刷新"],
                    id='refresh-button',
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-all duration-300 flex items-center text-sm",
                    style={'display': 'block' if is_analysis else 'none'}
                ),
                html.Button(
                    [html.I(className="fas fa-key mr-1"), "改密"],
                    id='change-password-button',
                    className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-2 rounded-md transition-all duration-300 flex items-center text-sm"
                ),
                html.Button(
                    [html.I(className="fas fa-sign-out-alt mr-1"), "退出"],
                    id='logout-button',
                    className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-md transition-all duration-300 flex items-center text-sm"
                )
            ], className="flex items-center space-x-2")
        ], className="flex items-center justify-between space-x-4")
    ], className="bg-gradient-to-r from-blue-700 to-blue-900 px-4 md:px-6 py-3 shadow-lg sticky top-0 z-50")

# ===================== 分析页面布局 =====================
def analysis_layout():
    return html.Div([
        html.Main([
            html.Aside([
                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-search mr-2'), "选择股票"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("选择要显示图表的股票:", className="block text-sm font-medium text-gray-700 mb-2"),
                    html.Div([
                        dcc.Dropdown(
                            id='stock-codes-dropdown',
                            options=[],
                            value=[],
                            multi=True,
                            placeholder="选择股票...",
                            persistence=True, persistence_type='local',
                            className="w-full"
                        )
                    ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg")
                ]),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-database mr-2'), "数据源"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("选择数据源:", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Dropdown(
                        id='data-source-dropdown',
                        options=[
                            {'label': 'Tushare (推荐)', 'value': 'tushare'},
                            {'label': 'Baostock (免费)', 'value': 'baostock'},
                        ],
                        value='tushare',
                        clearable=False,
                        persistence=True, persistence_type='local',
                        className="w-full"
                    ),
                    html.Div(id='data-source-status', className="text-xs text-gray-500 mt-2")
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-plus-circle mr-2'), "添加代码"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("单个股票代码:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Input(
                        id='single-code-input',
                        type='text',
                        placeholder='输入单个股票代码',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Button(
                            [html.I(className='fas fa-plus mr-1'), "插入"],
                            id='insert-code-button',
                            className="bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 w-full flex items-center justify-center"
                        ),
                    html.Div(id='single-code-error', className="text-sm mt-2 text-gray-700 min-h-[1.25rem]")
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Button(
                        [html.I(className='fas fa-list mr-1'), "查询所有代码"],
                        id='query-codes-button',
                        className="bg-yellow-500 hover:bg-yellow-600 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 w-full flex items-center justify-center"
                    )
                ], className="mb-6"),
                html.Div(id='query-codes-result', className="text-gray-700 mt-1 bg-white rounded-lg p-3 shadow-sm"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-calendar-alt mr-2'), "日期选择"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("开始日期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.DatePickerSingle(
                        id='start-date-picker',
                        date=pd.Timestamp.now() - pd.Timedelta(days=365),
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Label("结束日期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.DatePickerSingle(
                        id='end-date-picker',
                        date=pd.Timestamp.now(),
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Label("K线周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Dropdown(
                        id='kline-freq-dropdown',
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
                        persistence=True, persistence_type='local',
                        className="w-full"
                    )
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-sliders-h mr-2'), "策略参数"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("短期均线周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Input(
                        id='short-window-input',
                        type='number',
                        value=5,
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Label("长期均线周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Input(
                        id='long-window-input',
                        type='number',
                        value=20,
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Label("简单移动平均周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Input(
                        id='sma-window-input',
                        type='number',
                        value=20,
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm mb-3"
                    ),
                    html.Label("指数移动平均周期:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Input(
                        id='ema-window-input',
                        type='number',
                        value=20,
                        persistence=True, persistence_type='local',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                    )
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-filter mr-2'), "信号过滤"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("减少虚假信号（只过滤买入，离场不拦截）:", className="block text-sm font-medium text-gray-700 mb-2"),
                    dcc.Checklist(id='signal-filter-options',
                        options=[
                            {'label': ' 趋势过滤（顺势才买入）', 'value': 'trend'},
                            {'label': ' 信号确认（连续N日成立）', 'value': 'confirm'},
                            {'label': ' ADX过滤（避开震荡市）', 'value': 'adx'},
                        ], value=['trend', 'confirm', 'adx'],
                        persistence=True, persistence_type='local',
                        className="text-sm text-gray-700 mb-3"),
                    html.Div([
                        html.Div([
                            html.Label("趋势均线:", className="block text-xs font-medium text-gray-600 mb-1"),
                            dcc.Input(id='trend-ma-input', type='number', value=60, min=10, max=250,
                                persistence=True, persistence_type='local',
                                className="w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm"),
                        ], className="w-1/3 pr-1"),
                        html.Div([
                            html.Label("确认天数:", className="block text-xs font-medium text-gray-600 mb-1"),
                            dcc.Input(id='confirm-days-input', type='number', value=2, min=1, max=10,
                                persistence=True, persistence_type='local',
                                className="w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm"),
                        ], className="w-1/3 px-1"),
                        html.Div([
                            html.Label("ADX阈值:", className="block text-xs font-medium text-gray-600 mb-1"),
                            dcc.Input(id='adx-threshold-input', type='number', value=20, min=0, max=50,
                                persistence=True, persistence_type='local',
                                className="w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm"),
                        ], className="w-1/3 pl-1"),
                    ], className="flex"),
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-drafting-compass mr-2'), "自动趋势线"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    dcc.Checklist(id='trendline-options',
                        options=[
                            {'label': ' 趋势线（支撑/阻力连线）', 'value': 'trendlines'},
                            {'label': ' 水平支撑/阻力位', 'value': 'levels'},
                        ], value=['trendlines', 'levels'],
                        persistence=True, persistence_type='local',
                        className="text-sm text-gray-700"),
                    html.Div("自动连接摆动高低点生成趋势线并标注突破状态，供人工判断",
                             className="text-xs text-gray-400 mt-2"),
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-tag mr-2'), "股票标记"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    html.Label("标记已购买股票:", className="block text-sm font-medium text-gray-700 mb-1"),
                    dcc.Dropdown(
                        id='marked-stock-dropdown',
                        placeholder='选择已购买的股票代码',
                        clearable=True,
                        searchable=True,
                        className="mb-3"
                    ),
                    html.Button(
                        [html.I(className='fas fa-flag mr-1'), "标记"],
                        id='mark-stock-button',
                        className="bg-purple-500 hover:bg-purple-600 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 w-full flex items-center justify-center"
                    )
                ], className="bg-white rounded-xl shadow-md p-4 mb-6 transition-all duration-300 hover:shadow-lg"),
                html.Div(id='marked-stock-result', className="text-gray-700 mt-1 bg-white rounded-lg p-3 shadow-sm mb-6"),
                html.Div(id='marked-stock-list', className="mt-1"),

                html.Div([
                    html.Div([
                        html.H2([html.I(className='fas fa-sticky-note mr-2'), "记事本"], className="text-lg font-semibold text-gray-700 flex items-center"),
                    ], className="mb-4 pb-2 border-b border-gray-200"),
                    dcc.Textarea(
                        id='notepad-input',
                        placeholder='输入笔记内容',
                        className="w-full py-2 px-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm h-32",
                    ),
                    html.Div([
                        html.Button(
                            [html.I(className='fas fa-save mr-1'), "保存笔记"],
                            id='save-note-button',
                            className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 flex-1 flex items-center justify-center"
                        ),
                        html.Button(
                            [html.I(className='fas fa-trash-alt mr-1'), "删除笔记"],
                            id='delete-note-button',
                            className="bg-red-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-md transition-all duration-300 flex-1 flex items-center justify-center ml-2"
                        )
                    ], className="flex mt-3")
                ], className="bg-white rounded-xl shadow-md p-4 transition-all duration-300 hover:shadow-lg")

            ], className="w-full md:w-1/4 lg:w-1/5 p-4 space-y-4"),

            html.Div([
                html.Div([
                    html.Div(id='stock-codes-error', className="text-red-500 text-sm mt-1 animate__animated animate__fadeIn"),
                    html.Div(id='start-date-error', className="text-red-500 text-sm mt-1 animate__animated animate__fadeIn"),
                    html.Div(id='end-date-error', className="text-red-500 text-sm mt-1 animate__animated animate__fadeIn"),
                    html.Div(id='short-window-error', className="text-red-500 text-sm mt-1 animate__animated animate__fadeIn"),
                    html.Div(id='long-window-error', className="text-red-500 text-sm mt-1 animate__animated animate__fadeIn"),
                    html.Div(id='notepad-result', className="text-gray-700 mt-1 animate__animated animate__fadeIn")
                ], className="mb-4 bg-red-50 rounded-lg p-3 hidden"),

                html.Div(id='refresh-status', className="mb-4"),
                html.Div(id='fs-model-picker-wrap', style={'display': 'none'}, className="mb-4", children=[
                    html.Div([
                        html.Label("使用模型:", className="text-sm font-medium text-gray-600 mr-3 whitespace-nowrap"),
                        dcc.Dropdown(id='fs-model-select', options=[], placeholder='选择已保存的因子模型...',
                                     className="flex-1 text-sm", clearable=False),
                    ], className="bg-white rounded-xl shadow-md p-3 flex items-center"),
                ]),
                html.Div(id='factor-score-panel'),
                html.Div(id='confluence-panel'),
                html.Div(className="bg-white rounded-xl shadow-md p-4 mb-4", children=[
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-arrow-trend-up text-violet-600 text-xl mr-2"),
                            html.H3("量化走势预测", className="text-lg font-bold text-gray-800"),
                        ], className="flex items-center"),
                        html.Div([
                            html.Span("预测周期:", className="text-xs text-gray-500 mr-2"),
                            dcc.RadioItems(id='forecast-horizon',
                                options=[{'label': '5日 ', 'value': 5},
                                         {'label': '10日 ', 'value': 10},
                                         {'label': '20日', 'value': 20}],
                                value=10, inline=True,
                                persistence=True, persistence_type='local',
                                className="text-sm text-gray-700"),
                        ], className="flex items-center"),
                    ], className="flex flex-wrap items-center justify-between gap-2 mb-2 pb-2 border-b"),
                    html.Div(id='forecast-panel',
                             children=html.Div("选择股票后展示预测", className="text-sm text-gray-400 py-4 text-center")),
                ]),
                html.Div(id='stock-graphs-container', className="space-y-8 grid grid-cols-1 gap-6")
            ], className="w-full md:w-3/4 lg:w-4/5 p-4"),

        ], className="container mx-auto flex flex-col md:flex-row gap-4"),

        dcc.Interval(
            id='interval-component',
            interval=300 * 1000,
            n_intervals=0
        ),

        html.Footer([
            html.Div([
                html.P("© 2023 股票分析平台 | 数据仅供参考，不构成投资建议", className="text-gray-600")
            ], className="container mx-auto px-4 py-3 text-center")
        ], className="bg-gray-800 text-white py-4 mt-8")
    ])

# ===================== 编辑模态框 =====================
def create_edit_modal():
    return html.Div([
        html.Div([
            html.Div([
                html.H3([html.I(className="fas fa-edit mr-2 text-blue-500"), "编辑股票"], className="text-xl font-bold text-gray-800 mb-4"),
                html.Div([
                    html.Label("股票代码", className="block text-sm font-medium text-gray-600 mb-1"),
                    dcc.Input(
                        id='edit-code-input',
                        type='text',
                        placeholder='例如 600519.SH',
                        className="w-full py-2 px-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                    ),
                ], className="mb-4"),
                html.Div([
                    html.Label("股票名称", className="block text-sm font-medium text-gray-600 mb-1"),
                    dcc.Input(
                        id='edit-name-input',
                        type='text',
                        placeholder='例如 贵州茅台',
                        className="w-full py-2 px-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                    ),
                ], className="mb-6"),
                html.Div([
                    html.Button(
                        [html.I(className="fas fa-save mr-1"), "保存"],
                        id='edit-save-btn',
                        className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-all duration-300 text-sm"
                    ),
                    html.Button(
                        [html.I(className="fas fa-times mr-1"), "取消"],
                        id='edit-cancel-btn',
                        className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-6 rounded-lg transition-all duration-300 text-sm ml-2"
                    )
                ], className="flex justify-end space-x-2")
            ], className="p-6")
        ], className="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4")
    ], id='edit-modal-content', className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden")

# ===================== 股票管理页面布局 =====================
def stock_management_layout():
    return html.Div([
        html.Main([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-database text-blue-600 text-3xl"),
                        html.H2("股票管理", className="text-2xl font-bold text-gray-800 ml-3")
                    ], className="flex items-center"),
                    html.P("管理您的自选股票列表，支持添加、删除、搜索等操作", className="text-sm text-gray-500 mt-1 ml-1")
                ], className="mb-6"),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-chart-pie text-blue-500 text-2xl"),
                                ], className="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center"),
                                html.Div([
                                    html.P("股票总数", className="text-sm text-gray-500"),
                                    html.P(id='mgmt-total-count', className="text-2xl font-bold text-gray-800"),
                                ], className="ml-3")
                            ], className="flex items-center"),
                            html.Div([
                                html.I(className="fas fa-arrow-up text-green-500", id='mgmt-total-icon', style={'opacity': '0'})
                            ], className="ml-auto")
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 hover:shadow-md transition-shadow duration-300"),
                        html.Div([
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-check-circle text-green-500 text-2xl"),
                                ], className="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center"),
                                html.Div([
                                    html.P("已获取名称", className="text-sm text-gray-500"),
                                    html.P(id='mgmt-named-count', className="text-2xl font-bold text-gray-800"),
                                ], className="ml-3")
                            ], className="flex items-center"),
                            html.Div([
                                html.I(className="fas fa-arrow-up text-green-500")
                            ], className="ml-auto")
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-5 hover:shadow-md transition-shadow duration-300"),
                    ], className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6"),
                ]),

                html.Div([
                    # 自选股分组侧边栏
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3([html.I(className="fas fa-layer-group mr-2 text-purple-500"), "自选股分组"], className="text-lg font-semibold text-gray-700 flex items-center"),
                            ], className="mb-3"),
                            # 新建分组
                            html.Div([
                                dcc.Input(id='wl-new-name', type='text', placeholder='新建分组...',
                                    className="flex-1 py-2 px-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 text-sm"),
                                html.Button([html.I(className="fas fa-plus")], id='wl-create-btn',
                                    className="bg-purple-500 hover:bg-purple-600 text-white px-3 py-2 rounded-r-lg transition-colors text-sm"),
                            ], className="flex mb-3"),
                            # 分组列表
                            html.Div(id='wl-list-container', className="space-y-1"),
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-4"),
                        # 添加到分组
                        html.Div([
                            html.P("快速添加股票到分组:", className="text-xs text-gray-500 mb-2"),
                            dcc.Dropdown(id='wl-assign-dropdown', placeholder='选择分组...',
                                className="mb-2"),
                            html.Div([
                                dcc.Input(id='wl-assign-code', type='text', placeholder='股票代码',
                                    className="flex-1 py-2 px-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"),
                                html.Button([html.I(className="fas fa-arrow-right")], id='wl-assign-btn',
                                    className="bg-purple-500 hover:bg-purple-600 text-white px-3 py-2 rounded-r-lg transition-colors text-sm"),
                            ], className="flex"),
                            html.Div(id='wl-assign-msg', className="text-xs text-gray-500 mt-1"),
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 mt-3"),
                        dcc.Store(id='wl-selected-store', data={'watchlist_id': None}),
                    ], className="lg:w-1/5"),

                    # 中间：添加股票
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3([html.I(className="fas fa-plus-circle mr-2 text-blue-500"), "添加股票"], className="text-lg font-semibold text-gray-700 flex items-center"),
                            ], className="mb-4"),
                            html.Div([
                                html.Label("股票代码", className="block text-sm font-medium text-gray-600 mb-1"),
                                html.Div([
                                    dcc.Input(
                                        id='mgmt-code-input',
                                        type='text',
                                        placeholder='例如 600519.SH，支持逗号/空格批量',
                                        className="flex-1 py-2 px-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                                    ),
                                    html.Button(
                                        [html.I(className="fas fa-search mr-1"), "获取名称"],
                                        id='mgmt-fetch-name-btn',
                                        className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 text-sm border-t border-b border-r border-gray-300 transition-colors"
                                    )
                                ], className="flex"),
                                html.Div(id='mgmt-fetched-name', className="text-xs text-gray-500 mt-1 min-h-[1rem]"),
                                html.Div([
                                    html.Button(
                                        [html.I(className="fas fa-plus mr-1"), "添加"],
                                        id='mgmt-add-btn',
                                        className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-5 rounded-lg transition-all duration-300 text-sm flex-1"
                                    ),
                                    html.Button(
                                        [html.I(className="fas fa-sync-alt mr-1"), "刷新列表"],
                                        id='mgmt-refresh-btn',
                                        className="bg-gray-500 hover:bg-gray-600 text-white font-medium py-2 px-5 rounded-lg transition-all duration-300 text-sm"
                                    )
                                ], className="flex space-x-2 mt-3"),
                                html.Div(id='mgmt-add-msg', className="text-xs text-gray-500 mt-2 min-h-[1rem]")
                            ])
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-5")
                    ], className="lg:w-1/4"),

                    # 右侧：股票列表
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H3([html.I(className="fas fa-list mr-2 text-green-500"), "自选股票列表"], className="text-lg font-semibold text-gray-700 flex items-center"),
                                # 分组过滤标签
                                html.Div(id='wl-filter-badge', className="ml-2"),
                            ], className="mb-4 flex items-center"),
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-search text-gray-400"),
                                    dcc.Input(
                                        id='mgmt-search-input',
                                        type='text',
                                        placeholder='搜索代码或名称...',
                                        className="flex-1 py-2 px-3 border-none focus:outline-none text-sm bg-transparent"
                                    ),
                                    html.Button(
                                        [html.I(className="fas fa-times")],
                                        id='mgmt-clear-search-btn',
                                        className="text-gray-400 hover:text-gray-600 px-2",
                                        style={'display': 'none'}
                                    )
                                ], className="flex items-center border border-gray-300 rounded-lg px-3 focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all mb-4 bg-white")
                            ]),
                            html.Div(id='mgmt-table-container', className="overflow-x-auto"),
                            html.Div([
                                html.Button(
                                    [html.I(className="fas fa-chevron-left mr-1"), "上一页"],
                                    id='mgmt-prev-btn',
                                    disabled=True,
                                    className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                ),
                                html.Span(id='mgmt-page-info', className="text-sm text-gray-600 mx-3"),
                                html.Button(
                                    ["下一页", html.I(className="fas fa-chevron-right ml-1")],
                                    id='mgmt-next-btn',
                                    disabled=True,
                                    className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 text-gray-600 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                                ),
                            ], className="flex items-center justify-center mt-4")
                        ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-5")
                    ], className="lg:w-2/4 lg:flex-1")
                ], className="flex flex-col lg:flex-row gap-4")
            ], className="container mx-auto px-4 py-6")
        ]),

        html.Footer([
            html.Div([
                html.P("© 2023 股票分析平台 | 数据仅供参考，不构成投资建议", className="text-gray-600")
            ], className="container mx-auto px-4 py-3 text-center")
        ], className="bg-gray-800 text-white py-4 mt-8"),

        html.Div(id='edit-modal-container', children=create_edit_modal()),
        dcc.Store(id='edit-stock-data', data={'code': '', 'name': ''}),
        dcc.Store(id='mgmt-pending-delete', data={'code': '', 'name': ''}),
        dcc.Store(id='mgmt-pending-wl-delete', data={'id': None, 'name': ''}),
        dcc.Store(id='mgmt-page-store', data=1),
        dcc.ConfirmDialog(id='mgmt-delete-confirm', message=''),
        dcc.ConfirmDialog(id='mgmt-wl-delete-confirm', message=''),
    ])

# ===================== 登录 / 注册页面 =====================

def login_layout():
    """登录页整体布局（未登录时展示）"""
    return html.Div([
        html.Div([
            html.Div([
                html.I(className="fas fa-chart-line text-5xl text-blue-600"),
                html.H1("股票分析平台", className="text-2xl font-bold text-gray-800 mt-4"),
                html.P("请登录后使用，各账户数据相互隔离", className="text-gray-500 text-sm mt-1"),
            ], className="text-center mb-6"),
            dcc.Tabs(id='auth-tabs', value='login', className="mb-5", children=[
                dcc.Tab(label='登录', value='login', className="font-medium"),
                dcc.Tab(label='注册', value='register', className="font-medium"),
            ]),
            html.Div(id='auth-form-content'),
        ], className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md"),
    ], className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-blue-100 px-4")


def _login_form():
    hint = []
    if is_default_admin_password():
        hint = [html.Div(
            "首次使用默认账号：admin / admin123（登录后请及时修改密码）",
            className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-3"
        )]
    return html.Div([
        html.Div(hint),
        html.Label("用户名", className="block text-sm font-medium text-gray-700 mb-1"),
        dcc.Input(id='login-username', type='text', placeholder="请输入用户名",
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"),
        html.Label("密码", className="block text-sm font-medium text-gray-700 mb-1"),
        dcc.Input(id='login-password', type='password', placeholder="请输入密码",
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"),
        html.Button("登 录", id='login-button', n_clicks=0,
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition-all duration-300"),
        html.Div(id='login-message', className="text-sm mt-3 min-h-[1.25rem]"),
    ])


def _register_form():
    return html.Div([
        html.Label("用户名", className="block text-sm font-medium text-gray-700 mb-1"),
        dcc.Input(id='register-username', type='text', placeholder="至少 2 个字符",
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"),
        html.Label("密码", className="block text-sm font-medium text-gray-700 mb-1"),
        dcc.Input(id='register-password', type='password', placeholder="至少 6 位",
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"),
        html.Label("确认密码", className="block text-sm font-medium text-gray-700 mb-1"),
        dcc.Input(id='register-confirm', type='password', placeholder="再次输入密码",
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"),
        html.Button("注 册", id='register-button', n_clicks=0,
                    className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg transition-all duration-300"),
        html.Div(id='register-message', className="text-sm mt-3 min-h-[1.25rem]"),
    ])


def _change_password_modal():
    return html.Div([
        html.Div([
            html.H3("修改密码", className="text-lg font-bold text-gray-800 mb-4"),
            html.Label("原密码", className="block text-sm font-medium text-gray-700 mb-1"),
            dcc.Input(id='cp-old-password', type='password', placeholder="请输入原密码",
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"),
            html.Label("新密码", className="block text-sm font-medium text-gray-700 mb-1"),
            dcc.Input(id='cp-new-password', type='password', placeholder="至少 6 位",
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500"),
            html.Label("确认新密码", className="block text-sm font-medium text-gray-700 mb-1"),
            dcc.Input(id='cp-confirm-password', type='password', placeholder="再次输入新密码",
                      className="w-full border border-gray-300 rounded-lg px-3 py-2 mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"),
            html.Div([
                html.Button("取消", id='cp-cancel-button', n_clicks=0,
                            className="bg-gray-300 hover:bg-gray-400 text-gray-800 px-4 py-2 rounded-lg mr-3"),
                html.Button("保存", id='cp-save-button', n_clicks=0,
                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"),
            ], className="flex justify-end"),
            html.Div(id='cp-message', className="text-sm mt-3 min-h-[1.25rem]"),
        ], className="bg-white rounded-xl shadow-2xl p-6 w-full max-w-sm"),
    ], id='change-password-modal', className="fixed inset-0 z-50 flex items-center justify-center bg-black/50", style={'display': 'none'})


# ===================== 主布局 =====================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='auth-state', storage_type='session'),
    html.Div(id='login-page', children=login_layout(), style={'display': 'none'}),
        html.Div(id='app-content', children=[
            html.Div(id='page-header', children=create_header('/')),
            html.Div(id='analysis-page', children=analysis_layout(), style={'display': 'block'}),
            html.Div(id='management-page', children=stock_management_layout(), style={'display': 'none'}),
            html.Div(id='factor-page', children=factor_page_layout(), style={'display': 'none'}),
            html.Div(id='trend-page', children=trend_analysis_layout(), style={'display': 'none'}),
            html.Div(id='game-page', children=stock_game_layout(), style={'display': 'none'}),
            html.Div(id='market-page', children=market_overview_layout(), style={'display': 'none'}),
            html.Div(id='sector-page', children=sector_fund_strength_layout(), style={'display': 'none'}),
            html.Div(id='journal-page', children=trading_journal_layout(), style={'display': 'none'}),
            _change_password_modal(),
        ], style={'display': 'none'}),

        # 自选股双均线买入信号提示（后台扫描器，全页面可见）
        html.Button([html.I(className='fas fa-bell text-xl'),
                     html.Span(id='signal-alert-count',
                               className='absolute -top-1 -right-1 bg-red-500 text-white text-xs font-bold rounded-full px-1.5 min-w-[1.25rem] text-center')],
                    id='signal-alert-btn',
                    className='fixed bottom-6 right-6 z-40 bg-indigo-600 hover:bg-indigo-700 text-white rounded-full w-14 h-14 shadow-lg flex items-center justify-center'),
        html.Div(id='signal-alert-panel', style={'display': 'none'},
                 className='fixed bottom-24 right-6 z-40 bg-white rounded-xl shadow-2xl border border-gray-200 p-4 w-[26rem] max-h-[28rem] overflow-y-auto'),
        dcc.Interval(id='signal-alert-interval', interval=15000, n_intervals=0),
    ])


# ===================== 认证回调 =====================

@app.callback(
    Output('auth-form-content', 'children'),
    Input('auth-tabs', 'value')
)
def render_auth_form(tab):
    return _register_form() if tab == 'register' else _login_form()


@app.callback(
    [Output('login-message', 'children'),
     Output('auth-state', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('login-button', 'n_clicks'),
    [State('login-username', 'value'), State('login-password', 'value')],
    prevent_initial_call=True
)
def do_login(n_clicks, username, password):
    if not n_clicks:
        raise PreventUpdate
    ok, msg, uid = auth.authenticate(username, password)
    if not ok:
        return html.Span(msg, className="text-red-500"), dash.no_update, dash.no_update
    username = (username or '').strip()
    auth.login_session(uid, username)
    return "", username, "/"


@app.callback(
    [Output('register-message', 'children'),
     Output('auth-state', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('register-button', 'n_clicks'),
    [State('register-username', 'value'), State('register-password', 'value'), State('register-confirm', 'value')],
    prevent_initial_call=True
)
def do_register(n_clicks, username, password, confirm):
    if not n_clicks:
        raise PreventUpdate
    if password != confirm:
        return html.Span("两次输入的密码不一致", className="text-red-500"), dash.no_update, dash.no_update
    ok, msg, uid = auth.register_user(username, password)
    if not ok:
        return html.Span(msg, className="text-red-500"), dash.no_update, dash.no_update
    username = (username or '').strip()
    auth.login_session(uid, username)
    return "", username, "/"


@app.callback(
    [Output('auth-state', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def do_logout(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    auth.logout_session()
    return None, "/login"


@app.callback(
    Output('change-password-modal', 'style'),
    [Input('change-password-button', 'n_clicks'),
     Input('cp-cancel-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_change_password_modal(open_clicks, cancel_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    # 仅当"改密"按钮被真实点击（n_clicks 为正整数）才打开弹窗；
    # 路由切换时 header 重建会把 n_clicks 重置为 None，此时应保持关闭而非误弹窗
    if trig == 'change-password-button' and open_clicks:
        return {'display': 'flex'}
    return {'display': 'none'}


@app.callback(
    [Output('cp-message', 'children'),
     Output('change-password-modal', 'style', allow_duplicate=True),
     Output('cp-old-password', 'value'),
     Output('cp-new-password', 'value'),
     Output('cp-confirm-password', 'value')],
    Input('cp-save-button', 'n_clicks'),
    [State('cp-old-password', 'value'), State('cp-new-password', 'value'), State('cp-confirm-password', 'value')],
    prevent_initial_call=True
)
def do_change_password(n_clicks, old_pwd, new_pwd, confirm_pwd):
    if not n_clicks:
        raise PreventUpdate
    no_update = [dash.no_update, dash.no_update, dash.no_update]
    user = auth.get_current_user()
    if not user:
        return [html.Span("未登录", className="text-red-500")] + [dash.no_update] + no_update
    if new_pwd != confirm_pwd:
        return [html.Span("两次输入的新密码不一致", className="text-red-500")] + [dash.no_update] + no_update
    ok, msg = auth.change_password(user['id'], old_pwd, new_pwd)
    color = "text-green-600" if ok else "text-red-500"
    if ok:
        # 修改成功：关闭弹窗并清空输入框
        return [html.Span(msg, className=color), {'display': 'none'}, None, None, None]
    return [html.Span(msg, className=color)] + [dash.no_update] + no_update


# ===================== 路由回调 =====================
@app.callback(
    [Output('login-page', 'style'),
     Output('app-content', 'style'),
     Output('page-header', 'children'),
     Output('analysis-page', 'style'),
     Output('management-page', 'style'),
     Output('factor-page', 'style'),
     Output('trend-page', 'style'),
     Output('game-page', 'style'),
     Output('market-page', 'style'),
     Output('sector-page', 'style'),
     Output('journal-page', 'style')],
    [Input('url', 'pathname'),
     Input('auth-state', 'data')]
)
def router(pathname, auth_state):
    user = auth.get_current_user()
    if not user:
        # 未登录: 显示登录页；page-header 的输出是 children，必须给组件
        # （给 style 字典会被 React 当子节点渲染而报错）
        return (
            {'display': 'block'},
            {'display': 'none'},
            create_header(pathname),
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
        )
    is_analysis = pathname == '/' or pathname == ''
    is_management = pathname == '/stock-management'
    is_factor = pathname == '/factor-training'
    is_trend = pathname == '/trend-analysis'
    is_game = pathname == '/stock-game'
    is_market = pathname == '/market-overview'
    is_sector = pathname == '/sector-fund-strength'
    is_journal = pathname == '/trading-journal'
    return (
        {'display': 'none'},
        {'display': 'block'},
        create_header(pathname, user['username']),
        {'display': 'block'} if is_analysis else {'display': 'none'},
        {'display': 'block'} if is_management else {'display': 'none'},
        {'display': 'block'} if is_factor else {'display': 'none'},
        {'display': 'block'} if is_trend else {'display': 'none'},
        {'display': 'block'} if is_game else {'display': 'none'},
        {'display': 'block'} if is_market else {'display': 'none'},
        {'display': 'block'} if is_sector else {'display': 'none'},
        {'display': 'block'} if is_journal else {'display': 'none'},
    )

# ===================== 分析页面回调（保持原有） =====================
@app.callback(
    [Output('stock-codes-dropdown', 'options'),
     Output('stock-codes-dropdown', 'value')],
    [Input('interval-component', 'n_intervals'),
     Input('insert-code-button', 'n_clicks'),
     Input('query-codes-button', 'n_clicks'),
     Input('refresh-button', 'n_clicks')],
    [State('stock-codes-dropdown', 'value')]
)
def update_stock_dropdown(n, insert_clicks, query_clicks, refresh_clicks, current_value):
    stock_list = get_all_stock_codes()
    if not stock_list:
        return [], []
    
    options = []
    for code, name in stock_list:
        display_name = f"{code} - {name}" if name else code
        options.append({'label': display_name, 'value': code})
    
    return options, current_value


@app.callback(
    [Output('data-source-status', 'children'),
     Output('stock-graphs-container', 'children', allow_duplicate=True)],
    [Input('data-source-dropdown', 'value')],
    prevent_initial_call=True
)
def switch_data_source(source_name):
    """切换数据源并清空图表"""
    success = set_data_source(source_name)
    if success:
        status = html.Span([
            html.I(className="fas fa-check-circle text-green-500 mr-1"),
            f"当前数据源: {source_name}"
        ])
    else:
        status = html.Span([
            html.I(className="fas fa-exclamation-circle text-red-500 mr-1"),
            f"数据源切换失败: {source_name}"
        ])
    return status, []


@app.callback(
    [Output('stock-codes-error', 'children'),
     Output('start-date-error', 'children'),
     Output('end-date-error', 'children'),
     Output('short-window-error', 'children'),
     Output('long-window-error', 'children'),
     Output('refresh-status', 'children'),
     Output('stock-graphs-container', 'children', allow_duplicate=True)],
    [Input('interval-component', 'n_intervals'),
     Input('stock-codes-dropdown', 'value'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('short-window-input', 'value'),
     Input('long-window-input', 'value'),
     Input('sma-window-input', 'value'),
     Input('ema-window-input', 'value'),
     Input('refresh-button', 'n_clicks'),
     Input('data-source-dropdown', 'value'),
     Input('kline-freq-dropdown', 'value'),
     Input('signal-filter-options', 'value'),
     Input('trend-ma-input', 'value'),
     Input('confirm-days-input', 'value'),
     Input('adx-threshold-input', 'value'),
     Input('trendline-options', 'value')],
    prevent_initial_call='initial_duplicate'
)
def combined_callback(n, stock_codes, start_date, end_date, short_window, long_window, sma_window, ema_window, refresh_clicks, data_source, kline_freq,
                      filter_options, trend_ma, confirm_days, adx_threshold, trendline_options):
    import dash
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    stock_codes_error = ""
    start_date_error = ""
    end_date_error = ""
    short_window_error = ""
    long_window_error = ""
    refresh_status = ""
    graph_children = []

    # 判断触发来源
    is_dropdown = triggered_id == 'stock-codes-dropdown'
    is_date_change = triggered_id in ('start-date-picker', 'end-date-picker')
    is_window_change = triggered_id in ('short-window-input', 'long-window-input', 'sma-window-input', 'ema-window-input')
    is_refresh_btn = triggered_id == 'refresh-button'
    is_data_source = triggered_id == 'data-source-dropdown'
    is_timer = triggered_id == 'interval-component'

    if is_timer and n == 0:
        return stock_codes_error, start_date_error, end_date_error, short_window_error, long_window_error, refresh_status, []

    code_list = stock_codes if isinstance(stock_codes, list) else []

    if not code_list:
        return stock_codes_error, start_date_error, end_date_error, short_window_error, long_window_error, refresh_status, []

    # 定时刷新时才显示刷新状态
    if is_timer and n > 0:
        refresh_status = html.Div([
            html.Div([
                html.I(className="fas fa-sync-alt fa-spin text-blue-500 mr-2"),
                html.Span("正在刷新数据...", className="text-blue-600 font-medium"),
                html.Span(f"（共 {len(code_list)} 只股票，数据源: {data_source}）", className="text-gray-400 text-sm ml-2")
            ], className="flex items-center bg-blue-50 border border-blue-200 rounded-lg px-4 py-2")
        ], className="animate__animated animate__fadeIn")

    # 仅用户主动触发时才生成图表
    if is_timer and n > 0 and not code_list:
        return stock_codes_error, start_date_error, end_date_error, short_window_error, long_window_error, refresh_status, []

    for i, code in enumerate(code_list):
        if code.isdigit():
            if int(code) >= 500000:
                code_list[i] = code + '.SH'
            else:
                code_list[i] = code + '.SZ'

    if len(code_list) > 100:
        stock_codes_error = "最多只能选择 100 个股票。"
    for code in code_list:
        if not validate_code(code):
            stock_codes_error = "请选择有效的股票或 ETF 代码。"
            break

    if start_date and end_date and pd.Timestamp(start_date) > pd.Timestamp(end_date):
        start_date_error = "开始日期不能晚于结束日期。"
        end_date_error = "结束日期不能早于开始日期。"

    try:
        short_window = int(short_window) if short_window else 5
        long_window = int(long_window) if long_window else 20
        sma_window = int(sma_window) if sma_window else 20
        ema_window = int(ema_window) if ema_window else 20
    except (ValueError, TypeError):
        short_window_error = "均线周期必须是正整数。"
        long_window_error = "均线周期必须是正整数。"

    if short_window <= 0:
        short_window_error = "短期均线周期必须大于0。"
    if long_window <= 0:
        long_window_error = "长期均线周期必须大于0。"

    if any([stock_codes_error, start_date_error, end_date_error, short_window_error, long_window_error]):
        error_msg = stock_codes_error or start_date_error or end_date_error or short_window_error or long_window_error
        error_fig = go.Figure()
        error_fig.update_layout(title='参数错误: ' + error_msg, xaxis_title='', yaxis_title='')
        graph_children = [html.Div(dcc.Graph(figure=error_fig, config={'displayModeBar': True}), className="bg-white rounded-xl shadow-md p-4")]
    else:
        # 记录当前双均线参数与过滤设置，供后台自选股信号扫描使用
        try:
            save_scan_settings({'short_window': short_window, 'long_window': long_window,
                                'filter_options': list(filter_options or []),
                                'confirm_days': int(confirm_days) if confirm_days else 2,
                                'adx_threshold': float(adx_threshold) if adx_threshold is not None else 20,
                                'trend_ma': int(trend_ma) if trend_ma else 60})
        except Exception as e:
            logging.warning(f"保存扫描设置失败: {e}")
        start_date_str = pd.Timestamp(start_date).strftime('%Y%m%d')
        end_date_str = pd.Timestamp(end_date).strftime('%Y%m%d')

        # 已训练的因子模型评分快照（叠加到个股图注释中）
        _factor_snap = None
        try:
            _factor_snap = get_factor_score_snapshot()
        except Exception as e:
            logging.warning(f"获取因子评分快照失败: {e}")

        def _build_stock_graph(code, data, idx):
            if data is None:
                error_fig = go.Figure()
                error_fig.update_layout(title=code + ' 数据获取失败', xaxis_title='', yaxis_title='')
                return dcc.Graph(figure=error_fig, config={'displayModeBar': True})
            dma_signals = double_moving_average_strategy(data, short_window, long_window)
            sma_signals = simple_moving_average_strategy(data, sma_window)
            ema_signals = exponential_moving_average_strategy(data, ema_window)

            sharpe_ratio = calculate_sharpe_ratio(data)
            max_drawdown = calculate_max_drawdown(data)

            data = data.dropna(subset=['close'])

            if dma_signals is None or sma_signals is None or ema_signals is None:
                error_fig = go.Figure()
                error_fig.update_layout(title=code + f' 数据不足（{len(data)}根K线），需要至少 {max(short_window, long_window, sma_window, ema_window)} 根K线',
                                        xaxis_title='', yaxis_title='')
                return dcc.Graph(figure=error_fig, config={'displayModeBar': True})

            dma_signals = dma_signals.dropna()
            sma_signals = sma_signals.dropna()
            ema_signals = ema_signals.dropna()

            common_index = data.index.copy()
            for sig in [dma_signals, sma_signals, ema_signals]:
                common_index = common_index.intersection(sig.index)

            data = data.loc[common_index]
            dma_signals = dma_signals.loc[common_index]
            sma_signals = sma_signals.loc[common_index]
            ema_signals = ema_signals.loc[common_index]

            if data.empty:
                error_fig = go.Figure()
                error_fig.update_layout(title=code + ' 数据为空', xaxis_title='', yaxis_title='')
                return dcc.Graph(figure=error_fig, config={'displayModeBar': True})

            # ===== 信号过滤（减少虚假信号）：趋势/确认/ADX 三重入场过滤，离场不拦截 =====
            _opts = set(filter_options or [])
            filters_on = len(_opts) > 0
            _confirm_days = int(confirm_days) if confirm_days else 2

            def _prepare_strategy(sig, kind, label):
                """返回 (标记用信号, 被过滤的原始买入, 注释行)"""
                if sig is None or sig.empty:
                    return sig, None, f"{label}: 信号不可用"
                raw_detail = simulate_trades_detailed(data, sig['signal'])
                if not filters_on:
                    line = (f"{label}: 交易 {raw_detail['n_trades']} 笔 ｜ "
                            f"胜率 {raw_detail['win_rate']*100:.0f}% ｜ "
                            f"累计收益 {raw_detail['cumulative_return']*100:.1f}%")
                    return sig, None, line
                filtered, stats = enhance_signals(
                    data, sig, kind,
                    use_trend='trend' in _opts, trend_ma=int(trend_ma) if trend_ma else 60,
                    use_confirm='confirm' in _opts, confirm_days=_confirm_days,
                    use_adx='adx' in _opts, adx_threshold=float(adx_threshold) if adx_threshold is not None else 20)
                det = simulate_trades_detailed(data, filtered['signal'])

                # 被过滤掉的原始买入点（因确认而延迟入场的点不算被过滤）
                raw_entries = list(sig.index[sig['signal'].diff() == 1])
                filt_entries = list(filtered.index[filtered['signal'].diff() == 1])
                tol = _confirm_days if 'confirm' in _opts else 0
                removed_idx = []
                for ri in raw_entries:
                    ri_pos = sig.index.get_loc(ri)
                    matched = any(abs(sig.index.get_loc(fi) - ri_pos) <= tol for fi in filt_entries)
                    if not matched:
                        removed_idx.append(ri)
                removed_sig = None
                if removed_idx:
                    removed_sig = sig.loc[removed_idx].copy()
                    removed_sig['positions'] = 1.0

                line = (f"{label}: 交易 {raw_detail['n_trades']}→{det['n_trades']} 笔 ｜ "
                        f"胜率 {raw_detail['win_rate']*100:.0f}%→{det['win_rate']*100:.0f}% ｜ "
                        f"累计收益 {raw_detail['cumulative_return']*100:.1f}%→{det['cumulative_return']*100:.1f}%")
                return filtered, removed_sig, line

            dma_signals, dma_removed, dma_note = _prepare_strategy(dma_signals, 'dma', '双均线')
            sma_signals, sma_removed, sma_note = _prepare_strategy(sma_signals, 'sma', '简单均线')
            ema_signals, ema_removed, ema_note = _prepare_strategy(ema_signals, 'ema', '指数均线')

            # ===== 自动趋势线与水平支撑/阻力位 =====
            _tl_opts = set(trendline_options or [])
            trendline_result = None
            if _tl_opts:
                try:
                    trendline_result = auto_trendlines(data, lookback=min(120, len(data)))
                except Exception as tl_err:
                    logging.warning(f"趋势线生成失败 {code}: {tl_err}")
            allow_future = (kline_freq or 'daily') not in ('60min', '30min', '15min', '5min')
            tl_note = ('<br>' + trendline_note(trendline_result)) if ('trendlines' in _tl_opts and trendline_result) else ''

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['trade_date'], y=data['close'],
                mode='lines', name='收盘价',
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate='日期: %{x}<br>收盘价: %{y:.2f}'
            ))

            fig.add_trace(go.Scatter(
                x=data['trade_date'], y=dma_signals['short_mavg'], mode='lines',
                name='双均线短期 (' + str(short_window) + ')',
                line=dict(color='#10B981', width=1.5, dash='dot'),
                hovertemplate='日期: %{x}<br>短期均线 (' + str(short_window) + '): %{y:.2f}'
            ))
            fig.add_trace(go.Scatter(
                x=data['trade_date'], y=dma_signals['long_mavg'], mode='lines',
                name='双均线长期 (' + str(long_window) + ')',
                line=dict(color='#F59E0B', width=1.5, dash='dash'),
                hovertemplate='日期: %{x}<br>长期均线 (' + str(long_window) + '): %{y:.2f}'
            ))
            def _add_strategy_markers(sig, removed_sig, y_col, label, buy_color, symbol):
                buy = sig[sig['positions'] == 1]
                sell = sig[sig['positions'] == -1]
                fig.add_trace(go.Scatter(
                    x=data.loc[buy.index, 'trade_date'], y=buy[y_col],
                    mode='markers', name=label + '买入',
                    marker=dict(color=buy_color, size=12, symbol=symbol, line=dict(color='black', width=1)),
                    hovertemplate='日期: %{x}<br>买入价格: %{y:.2f}'
                ))
                fig.add_trace(go.Scatter(
                    x=data.loc[sell.index, 'trade_date'], y=sell[y_col],
                    mode='markers', name=label + '卖出',
                    marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='black', width=1)),
                    hovertemplate='日期: %{x}<br>卖出价格: %{y:.2f}'
                ))
                if removed_sig is not None and len(removed_sig) > 0:
                    fig.add_trace(go.Scatter(
                        x=data.loc[removed_sig.index, 'trade_date'], y=removed_sig[y_col],
                        mode='markers', name=label + '原始买入(已过滤)',
                        marker=dict(color='gray', size=7, symbol=symbol + '-open', opacity=0.6),
                        hovertemplate='日期: %{x}<br>已过滤的原始买入信号'
                    ))

            _add_strategy_markers(dma_signals, dma_removed, 'short_mavg', '双均线', 'lime', 'triangle-up')

            fig.add_trace(go.Scatter(
                x=data['trade_date'], y=sma_signals['mavg'], mode='lines',
                name='简单移动平均 (' + str(sma_window) + ')',
                line=dict(color='#3B82F6', width=1.5, dash='dashdot'),
                hovertemplate='日期: %{x}<br>简单移动平均 (' + str(sma_window) + '): %{y:.2f}'
            ))
            _add_strategy_markers(sma_signals, sma_removed, 'mavg', '简单均线', 'blue', 'circle')

            fig.add_trace(go.Scatter(
                x=data['trade_date'], y=ema_signals['emavg'], mode='lines',
                name='指数移动平均 (' + str(ema_window) + ')',
                line=dict(color='#8B5CF6', width=1.5, dash='longdash'),
                hovertemplate='日期: %{x}<br>指数移动平均 (' + str(ema_window) + '): %{y:.2f}'
            ))
            _add_strategy_markers(ema_signals, ema_removed, 'emavg', '指数均线', 'cyan', 'diamond')

            if trendline_result:
                add_trendline_overlays(fig, data, trendline_result,
                                       show_lines='trendlines' in _tl_opts,
                                       show_levels='levels' in _tl_opts,
                                       allow_future=allow_future)

            freq_labels = {'daily': '日线', 'weekly': '周线', 'monthly': '月线',
                           '60min': '60分钟', '30min': '30分钟', '15min': '15分钟'}
            freq_label = freq_labels.get(kline_freq or 'daily', kline_freq)
            fig.update_layout(
                title=f'{code} 股票分析（{freq_label}）', xaxis_title='日期', yaxis_title='价格',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(type='date'), template='plotly_white', height=600,
                margin=dict(l=40, r=40, t=80, b=40)
            )
            score_line = ''
            if _factor_snap is not None:
                try:
                    finfo = get_stock_factor_info(_factor_snap, code)
                    if finfo.get('covered'):
                        score_line = (f"<br>因子评分: {finfo['score']:.3f}"
                                      f"（第 {finfo['rank']}/{finfo['total']} 名，前 {finfo['pct']:.0f}%）")
                except Exception:
                    pass
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.95,
                text=f"夏普比率: {sharpe_ratio:.4f} ｜ 最大回撤: {max_drawdown:.4f}<br>"\
                     f"{dma_note}<br>{sma_note}<br>{ema_note}{tl_note}{score_line}",
                showarrow=False, bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E5E7EB", borderwidth=1, font=dict(size=12)
            )
            return dcc.Graph(figure=fig, config={'displayModeBar': True})

        colors = px.colors.qualitative.Plotly

        for i, code in enumerate(code_list):
            try:
                data = get_and_process_data(code, start_date_str, end_date_str, freq=kline_freq or 'daily')
                graph_children.append(_build_stock_graph(code, data, i))
            except Exception as e:
                logging.error(f"处理 {code} 时发生未知错误: {e}")
                error_fig = go.Figure()
                error_fig.update_layout(title=f"{code} 处理时发生未知错误: {e}", xaxis_title='', yaxis_title='')
                graph_children.append(dcc.Graph(figure=error_fig, config={'displayModeBar': True}))

    if is_timer and n > 0 and graph_children:
        refresh_status = html.Div([
            html.Div([
                html.I(className="fas fa-check-circle text-green-500 mr-2"),
                html.Span("数据刷新完成", className="text-green-600 font-medium"),
                html.Span(f"（共 {len(graph_children)} 只股票，数据源: {data_source}）", className="text-gray-400 text-sm ml-2"),
                html.Span(f" - {pd.Timestamp.now().strftime('%H:%M:%S')}", className="text-gray-300 text-xs ml-1")
            ], className="flex items-center bg-green-50 border border-green-200 rounded-lg px-4 py-2")
        ], className="animate__animated animate__fadeIn")

    return stock_codes_error, start_date_error, end_date_error, short_window_error, long_window_error, refresh_status, graph_children


def _factor_model_picker_options():
    options = []
    for m in list_saved_models():
        method_label = 'ICIR' if m.get('method') == 'icir' else 'LGBM'
        options.append({
            'label': f"【{method_label}】{m.get('name', '')} · {m.get('n_factors', '?')}因子 · "
                     f"{m.get('n_stocks', '?')}股 · {m.get('latest_date', '')}",
            'value': m['name'],
        })
    return options


@app.callback(
    [Output('factor-score-panel', 'children'),
     Output('fs-model-picker-wrap', 'style'),
     Output('fs-model-select', 'options'),
     Output('fs-model-select', 'value')],
    [Input('stock-codes-dropdown', 'value'),
     Input('interval-component', 'n_intervals'),
     Input('fs-model-select', 'value')]
)
def update_factor_score_panel(selected_codes, n_intervals, fs_model_select):
    """分析页「训练因子评分」面板：跟随所选股票刷新；顶部下拉可切换已保存模型"""
    try:
        if fs_model_select:
            set_active_model(fs_model_select)
    except Exception as e:
        logging.warning(f"切换因子模型失败: {e}")
    try:
        panel = build_factor_score_panel(selected_codes or [], _get_cached_name_map())
    except Exception as e:
        logging.error(f"因子评分面板更新失败: {e}")
        panel = html.Div(f"因子评分面板加载失败: {e}", className="text-sm text-red-500")

    options = _factor_model_picker_options()
    if options:
        picker_style = {'display': 'block'}
        active = get_active_model_name()
        picker_value = active if active in [o['value'] for o in options] else None
    else:
        picker_style = {'display': 'none'}
        picker_value = None
    return panel, picker_style, options, picker_value


@app.callback(
    Output('confluence-panel', 'children'),
    [Input('stock-codes-dropdown', 'value'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('kline-freq-dropdown', 'value'),
     Input('signal-filter-options', 'value'),
     Input('trend-ma-input', 'value'),
     Input('confirm-days-input', 'value'),
     Input('adx-threshold-input', 'value'),
     Input('fs-model-select', 'value'),
     Input('data-source-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_confluence_panel(selected_codes, start_date, end_date, kline_freq, filter_opts,
                            trend_ma, confirm_days, adx_threshold, fs_model_select,
                            data_source, n_intervals):
    """多信号共振面板：整合因子评分/趋势状态/技术动量/CLV资金流/信号质量（最多展示3只）"""
    codes = [_normalize_stock_code(c) for c in (selected_codes or [])][:3]
    if not codes:
        return html.Div()

    if fs_model_select:
        try:
            set_active_model(fs_model_select)
        except Exception as e:
            logging.warning(f"切换因子模型失败: {e}")
    try:
        snap = get_factor_score_snapshot()
    except Exception as e:
        logging.warning(f"获取因子评分快照失败: {e}")
        snap = None

    try:
        sd = pd.Timestamp(start_date).strftime('%Y%m%d') if start_date else '20200101'
        ed = pd.Timestamp(end_date).strftime('%Y%m%d') if end_date else pd.Timestamp.now().strftime('%Y%m%d')
    except Exception:
        sd, ed = '20200101', pd.Timestamp.now().strftime('%Y%m%d')
    freq = kline_freq or 'daily'

    name_map = _get_cached_name_map()
    title_map = {c: (name_map.get(c) or name_map.get(c.split('.')[0]) or '') for c in codes}

    results = []
    for code in codes:
        data = None
        try:
            data = get_and_process_data(code, sd, ed, source=data_source, freq=freq)
        except Exception as e:
            logging.warning(f"共振面板获取 {code} 数据失败: {e}")
        factor_info = None
        if snap is not None:
            try:
                factor_info = get_stock_factor_info(snap, code)
                factor_info['model_name'] = snap.get('name')
            except Exception:
                factor_info = None
        try:
            results.append(compute_confluence(
                code, data,
                trend_ma=int(trend_ma) if trend_ma else 60,
                adx_threshold=float(adx_threshold) if adx_threshold is not None else 20,
                filter_opts=filter_opts,
                confirm_days=int(confirm_days) if confirm_days else 2,
                factor_info=factor_info))
        except Exception as e:
            logging.error(f"共振评分计算失败 {code}: {e}")
            results.append({'code': code, 'name': title_map.get(code, ''), 'ok': False,
                            'reason': f'评分计算失败: {e}', 'dimensions': [],
                            'total': None, 'verdict': '计算失败', 'verdict_style': ''})
    return build_confluence_panel(results, title_map)


@app.callback(
    Output('forecast-panel', 'children'),
    [Input('forecast-horizon', 'value'),
     Input('stock-codes-dropdown', 'value'),
     Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('kline-freq-dropdown', 'value'),
     Input('data-source-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_forecast_panel(horizon, selected_codes, start_date, end_date, kline_freq, data_source, n_intervals):
    """量化走势预测面板：对第一只选中股票做多模型预测 + 滚动回测准确率展示"""
    codes = [_normalize_stock_code(c) for c in (selected_codes or [])]
    if not codes:
        return html.Div("选择股票后展示预测", className="text-sm text-gray-400 py-4 text-center")
    horizon = int(horizon) if horizon else 10
    try:
        sd = pd.Timestamp(start_date).strftime('%Y%m%d') if start_date else '20200101'
        ed = pd.Timestamp(end_date).strftime('%Y%m%d') if end_date else pd.Timestamp.now().strftime('%Y%m%d')
    except Exception:
        sd, ed = '20200101', pd.Timestamp.now().strftime('%Y%m%d')
    try:
        data = get_and_process_data(codes[0], sd, ed, source=data_source, freq=kline_freq or 'daily')
    except Exception as e:
        logging.warning(f"预测面板获取 {codes[0]} 数据失败: {e}")
        data = None
    if data is None or data.empty:
        return html.Div("K线数据获取失败，无法预测", className="text-sm text-gray-400 py-4 text-center")
    try:
        name = _get_cached_name_map().get(codes[0]) or _get_cached_name_map().get(codes[0].split('.')[0]) or ''
        res = run_forecast(data, horizon, code=codes[0], name=name)
        return build_forecast_panel(res)
    except Exception as e:
        logging.error(f"走势预测失败 {codes[0]}: {e}")
        return html.Div(f"预测计算失败: {e}", className="text-sm text-red-500 py-2")


# ===================== 自选股双均线买入信号提示（后台扫描） =====================

def _build_signal_alert_body(scan):
    header = html.Div([
        html.Div([html.I(className='fas fa-bell mr-2 text-indigo-600'),
                  html.Span('自选股双均线买入信号', className='font-semibold text-gray-800')],
                 className='flex items-center'),
        html.Span(f"扫描于 {scan['scanned_at']}" if scan.get('scanned_at')
                  else ('扫描进行中…' if scan.get('scanning') else '尚未完成扫描'),
                  className='text-xs text-gray-400'),
    ], className='flex items-center justify-between mb-2 pb-2 border-b')

    if scan.get('scanning'):
        total = scan.get('total') or 0
        done = scan.get('n_scanned') or 0
        pct = (done / total * 100) if total else 0
        started = scan.get('started_at')
        eta_txt = ''
        if started and done >= 3 and total > done:
            elapsed = time.time() - started
            remain = (elapsed / done) * (total - done)
            eta_txt = f" ｜ 预计剩余 {remain / 60:.0f} 分钟" if remain >= 90 else f" ｜ 预计剩余 {remain:.0f} 秒"
        cur = scan.get('current_code') or ''
        prev = ''
        if scan.get('scanned_at'):
            prev = f" ｜ 上次扫描 {scan['scanned_at']} 共 {len(scan.get('results') or [])} 只信号"
        return [header,
                html.Div([
                    html.Div(f"正在扫描 {done}/{total} 只（{pct:.0f}%）{eta_txt}",
                             className='text-sm text-gray-600 mb-2'),
                    html.Div(html.Div(className='bg-indigo-600 h-2 rounded-full transition-all',
                                      style={'width': f'{max(pct, 3):.0f}%'}),
                             className='w-full bg-gray-200 rounded-full h-2 overflow-hidden'),
                    html.Div(f"当前: {cur} ｜ 数据源 baostock{prev}",
                             className='text-xs text-gray-400 mt-2'),
                ], className='py-2')]

    results = scan.get('results') or []
    if not results:
        body = scan.get('note') or '应用启动后将自动开始首轮扫描（数据源 baostock），请稍候，进度会实时显示在这里'
        if scan.get('last_error'):
            body += f"（{scan['last_error']}）"
        return [header, html.Div(body, className='text-sm text-gray-500 py-3 text-center')]

    name_map = _get_cached_name_map()
    items = []
    for h in results[:30]:
        code = h['code']
        nm = name_map.get(code) or name_map.get(code.split('.')[0]) or ''
        filt = h.get('filtered_active')
        if filt is True:
            filt_badge = html.Span('✓ 过滤确认', className='text-xs px-1.5 py-0.5 rounded bg-green-100 text-green-700')
        elif filt is False:
            filt_badge = html.Span('✗ 被过滤', className='text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-500')
        else:
            filt_badge = html.Span('未过滤', className='text-xs px-1.5 py-0.5 rounded bg-blue-50 text-blue-600')
        children = [
            html.Span(code, className='text-sm font-semibold text-gray-800 mr-1'),
            html.Span(nm, className='text-sm text-gray-600'),
        ]
        if h.get('fresh'):
            children.append(html.Span('新', className='text-xs px-1.5 py-0.5 rounded bg-red-100 text-red-600 ml-1'))
        items.append(html.Div([
            html.Div(children, className='flex items-center'),
            html.Div([
                html.Span(f"信号日 {h['signal_date']}", className='text-xs text-gray-500 mr-2'),
                html.Span(f"现价 {h['price']:.2f}", className='text-xs text-gray-500 mr-2'),
                html.Span(f"已持仓 {h['days_held']} 天", className='text-xs text-gray-500 mr-2'),
                filt_badge,
            ], className='flex items-center flex-wrap gap-y-1 mt-0.5'),
        ], className='py-2 border-b border-gray-100 last:border-0'))
    return [header, html.Div(items)]


@app.callback(
    [Output('signal-alert-count', 'children'),
     Output('signal-alert-panel', 'children')],
    [Input('signal-alert-interval', 'n_intervals'),
     Input('signal-alert-btn', 'n_clicks')]
)
def update_signal_alerts(n_intervals, n_clicks):
    try:
        scan = get_latest_scan()
    except Exception as e:
        logging.error(f"读取信号扫描结果失败: {e}")
        return '', html.Div('读取扫描结果失败', className='text-sm text-red-500 p-2')
    results = scan.get('results') or []
    badge = str(len(results)) if results else ''
    return badge, _build_signal_alert_body(scan)


@app.callback(
    Output('signal-alert-panel', 'style'),
    [Input('signal-alert-btn', 'n_clicks')],
    [State('signal-alert-panel', 'style')],
    prevent_initial_call=True
)
def toggle_signal_alert_panel(n_clicks, style):
    if not n_clicks:
        return dash.no_update
    style = dict(style or {})
    style['display'] = 'none' if style.get('display') == 'block' else 'block'
    return style


@app.callback(
    Output('single-code-error', 'children'),
    [Input('insert-code-button', 'n_clicks')],
    [State('single-code-input', 'value')]
)
def insert_single_code(n_clicks, code):
    if n_clicks:
        if validate_code(code):
            name = get_stock_name(code)
            insert_stock_code(code, name)
            return f"股票 {code} ({name}) 已插入。"
        else:
            return "请输入有效的股票或 ETF 代码（如 600519.SH 或 000001.SZ）。"
    return ""

@app.callback(
    Output('query-codes-result', 'children'),
    [Input('query-codes-button', 'n_clicks')]
)
def query_all_codes(n_clicks):
    if n_clicks:
        stock_list = get_all_stock_codes()
        if not stock_list:
            return "暂无保存的股票代码。"
        rows_html = []
        for code, name in stock_list:
            display_name = f"（{name}）" if name else ""
            rows_html.append(html.Div([
                html.Span(f"{code} {display_name}", className="text-gray-700")
            ], className="py-1 px-2 even:bg-gray-50 rounded"))
        return html.Div([
            html.Div(f"共 {len(stock_list)} 只股票", className="font-semibold text-gray-800 mb-2"),
            html.Div(rows_html, className="divide-y divide-gray-100")
        ])
    return ""

@app.callback(
    Output('marked-stock-dropdown', 'options'),
    [Input('interval-component', 'n_intervals'),
     Input('insert-code-button', 'n_clicks'),
     Input('query-codes-button', 'n_clicks')]
)
def update_marked_stock_dropdown(n_intervals, insert_clicks, query_clicks):
    _refresh_name_map_cache()
    stock_list = get_all_stock_codes()
    if not stock_list:
        return []
    # 读取已标记的股票
    marked_stocks = _read_marked_stocks()

    options = []
    for code, name in stock_list:
        display_name = f"{code} - {name}" if name else code
        if code in marked_stocks:
            display_name = f"✓ {display_name} (已标记)"
        options.append({'label': display_name, 'value': code, 'disabled': code in marked_stocks})
    return options


def _user_scoped_file(base_name):
    """为文件名追加当前用户名后缀，实现文件数据按用户隔离（如 notepad.txt -> notepad_admin.txt）。
    （已标记股票已迁移到数据库 marked_stocks 表，不再走文件）"""
    username = auth.get_current_username()
    if not username:
        return base_name
    name, ext = os.path.splitext(base_name)
    return f"{name}_{username}{ext}"


def _read_marked_stocks():
    """读取当前用户已标记的股票代码集合（数据库按 user_id 隔离）"""
    return get_marked_stocks()


@app.callback(
    [Output('marked-stock-result', 'children'),
     Output('marked-stock-dropdown', 'value'),
     Output('marked-stock-list', 'children')],
    [Input('mark-stock-button', 'n_clicks'),
     Input({'type': 'unmark-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('marked-stock-dropdown', 'value')]
)
def handle_marked_stocks(mark_clicks, unmark_clicks, selected_code):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    # 取消标记
    if 'unmark-btn' in triggered_id:
        try:
            parsed = json.loads(triggered_id.replace("'", '"'))
            code_to_unmark = parsed.get('index', '')
            if code_to_unmark:
                if code_to_unmark in _read_marked_stocks():
                    remove_marked_stock(code_to_unmark)
                    marked_stocks = _read_marked_stocks()
                    return f"已取消标记 {code_to_unmark}。", dash.no_update, build_marked_stock_list(marked_stocks)
        except Exception as e:
            return f"取消标记时出错: {e}", dash.no_update, dash.no_update

    # 标记按钮点击
    if triggered_id == 'mark-stock-button' and mark_clicks and selected_code:
        if validate_code(selected_code):
            try:
                marked_stocks = _read_marked_stocks()

                if selected_code in marked_stocks:
                    return f"股票 {selected_code} 之前已标记过。", None, build_marked_stock_list()

                add_marked_stock(selected_code)
                marked_stocks = _read_marked_stocks()
                return f"已标记股票 {selected_code} 为已购买。", None, build_marked_stock_list(marked_stocks)
            except Exception as e:
                return f"标记股票时出错: {e}", dash.no_update, dash.no_update
        else:
            return "请选择有效的股票代码。", dash.no_update, dash.no_update

    # 页面首次加载
    return dash.no_update, dash.no_update, build_marked_stock_list()


def build_marked_stock_list(marked_stocks=None):
    """构建已标记股票列表的 HTML"""
    if marked_stocks is None:
        marked_stocks = _read_marked_stocks()

    if not marked_stocks:
        return html.Div()

    # 使用缓存的名称映射，避免每次查询数据库
    name_map = _get_cached_name_map()
    items = []
    for code in sorted(marked_stocks):
        name = name_map.get(code, '')
        display = f"{code} - {name}" if name else code
        items.append(html.Div([
            html.I(className="fas fa-check-circle text-green-500 mr-2"),
            html.Span(display, className="text-sm text-gray-700"),
            html.Button(
                [html.I(className="fas fa-times text-red-400 hover:text-red-600")],
                id={'type': 'unmark-btn', 'index': code},
                className="ml-auto bg-transparent border-none cursor-pointer p-1",
                title="取消标记"
            )
        ], className="flex items-center py-1 px-2 bg-green-50 rounded mb-1"))

    return html.Div([
        html.Div([
            html.I(className="fas fa-flag text-purple-500 mr-2"),
            html.Span("已标记股票", className="text-sm font-semibold text-gray-700"),
        ], className="mb-2"),
        html.Div(items)
    ], className="bg-white rounded-lg p-3 shadow-sm")

@app.callback(
    [Output('notepad-result', 'children'),
     Output('notepad-input', 'value')],
    [Input('save-note-button', 'n_clicks'),
     Input('delete-note-button', 'n_clicks')],
    [State('notepad-input', 'value')]
)
def handle_notepad(save_clicks, delete_clicks, note):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if triggered_id == 'save-note-button' and save_clicks:
        try:
            with open(_user_scoped_file('notepad.txt'), 'w') as f:
                f.write(note or '')
            return "笔记已保存。", dash.no_update
        except Exception as e:
            return f"保存笔记时出错: {e}", dash.no_update

    if triggered_id == 'delete-note-button' and delete_clicks:
        try:
            with open(_user_scoped_file('notepad.txt'), 'w') as f:
                f.write('')
            return "笔记已删除。", ""
        except Exception as e:
            return f"删除笔记时出错: {e}", dash.no_update

    return "", dash.no_update

# ===================== 股票管理页面回调 =====================

# 更新统计卡片
def build_stock_table(stock_list, selected_wl_id=None, selected_wl_name=None, page=1, page_size=10):
    """构建股票表格，支持分组标签、按分组过滤和分页。
    返回 (表格HTML, 过滤后总数, 总页数, 实际页码)"""
    import math
    wl_map = get_stock_watchlist_map()

    if not stock_list:
        return html.Div([
            html.Div([
                html.I(className="fas fa-inbox text-gray-300 text-5xl mb-3"),
                html.P("暂无股票数据", className="text-gray-400 text-lg"),
                html.P("请在左侧添加股票代码", className="text-gray-400 text-sm")
            ], className="flex flex-col items-center justify-center py-16")
        ]), 0, 1, 1

    # 先按分组过滤得到有效列表
    effective = []
    for code, name in stock_list:
        tags = wl_map.get(code, [])
        if selected_wl_id and selected_wl_name and selected_wl_name not in tags:
            continue
        effective.append((code, name, tags))

    total = len(effective)
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    page_items = effective[start:start + page_size]

    rows = []
    for idx, (code, name, tags) in enumerate(page_items, start=start):
        display_name = name if name else '（名称未知）'
        tag_spans = [html.Span(tag, className="inline-block bg-purple-100 text-purple-700 text-xs rounded-full px-2 py-0.5 mr-1") for tag in tags]

        rows.append(html.Tr([
            html.Td([
                html.Div([
                    html.Span(str(idx + 1), className="text-gray-400 text-xs")
                ], className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center")
            ], className="py-3 px-2"),
            html.Td([
                html.Span(code, className="font-mono font-semibold text-gray-800 text-sm tracking-wide")
            ], className="py-3 px-3"),
            html.Td([
                html.Div([
                    html.Span(display_name, className="text-sm text-gray-600"),
                    html.Div(tag_spans, className="mt-1") if tag_spans else None
                ])
            ], className="py-3 px-3"),
            html.Td([
                html.Div([
                    html.Button(
                        [html.I(className="fas fa-edit text-blue-400 hover:text-blue-600 transition-colors")],
                        id={'type': 'mgmt-edit-btn', 'index': code},
                        className="bg-blue-50 hover:bg-blue-100 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 mr-1",
                        title="编辑"
                    ),
                    html.Button(
                        [html.I(className="fas fa-minus-circle text-orange-400 hover:text-orange-600 transition-colors")],
                        id={'type': 'mgmt-remove-wl-btn', 'index': code},
                        className="bg-orange-50 hover:bg-orange-100 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200 mr-1",
                        title="移出当前分组"
                    ) if selected_wl_id else None,
                    html.Button(
                        [html.I(className="fas fa-trash-alt text-red-400 hover:text-red-600 transition-colors")],
                        id={'type': 'mgmt-delete-btn', 'index': code},
                        className="bg-red-50 hover:bg-red-100 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-200",
                        title="删除"
                    )
                ], className="flex items-center justify-center")
            ], className="py-3 px-3 text-center")
        ], className="border-b border-gray-50 hover:bg-blue-50/40 transition-colors duration-150"))

    if not rows and selected_wl_id:
        rows = [html.Tr([
            html.Td([
                html.Div("该分组暂无股票", className="text-gray-400 text-sm py-6 text-center")
            ], colSpan=4)
        ])]

    table = html.Div([
        html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("#", className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider py-3 px-2 w-12"),
                        html.Th("股票代码", className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider py-3 px-3"),
                        html.Th("股票名称", className="text-left text-xs font-semibold text-gray-400 uppercase tracking-wider py-3 px-3"),
                        html.Th("操作", className="text-center text-xs font-semibold text-gray-400 uppercase tracking-wider py-3 px-3 w-20"),
                    ])
                ], className="bg-gray-50/80 rounded-t-lg"),
                html.Tbody(rows)
            ], className="w-full")
        ], className="rounded-lg overflow-hidden border border-gray-200")
    ])

    return table, total, total_pages, page

@app.callback(
    [Output('mgmt-table-container', 'children'),
     Output('mgmt-total-count', 'children'),
     Output('mgmt-named-count', 'children'),
     Output('edit-modal-content', 'style'),
     Output('edit-stock-data', 'data'),
     Output('wl-filter-badge', 'children'),
     Output('mgmt-add-msg', 'children'),
     Output('mgmt-prev-btn', 'disabled'),
     Output('mgmt-next-btn', 'disabled'),
     Output('mgmt-page-info', 'children'),
     Output('mgmt-page-store', 'data')],
    [Input('mgmt-refresh-btn', 'n_clicks'),
     Input('mgmt-add-btn', 'n_clicks'),
     Input('mgmt-search-input', 'value'),
     Input({'type': 'mgmt-edit-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
     Input('edit-save-btn', 'n_clicks'),
     Input('edit-cancel-btn', 'n_clicks'),
     Input('wl-selected-store', 'data'),
     Input('mgmt-prev-btn', 'n_clicks'),
     Input('mgmt-next-btn', 'n_clicks')],
    [State('mgmt-code-input', 'value'),
     State('mgmt-fetched-name', 'children'),
     State('edit-code-input', 'value'),
     State('edit-name-input', 'value'),
     State('edit-stock-data', 'data'),
     State('mgmt-page-store', 'data')]
)
def update_stock_list(refresh_clicks, add_clicks, search_keyword, edit_clicks, save_clicks, cancel_clicks, wl_store, prev_clicks, next_clicks, code_input, fetched_name, edit_code, edit_name, edit_stock_data, page_store):
    import re
    ctx = dash.callback_context
    # 使用 rsplit 避免股票代码中的 "." 被 split('.') 截断
    triggered_id = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0] if ctx.triggered else ''

    modal_style = {'display': 'none'}
    if edit_stock_data is None:
        edit_stock_data = {'code': '', 'name': ''}
    add_msg = ''

    # 读取分组过滤状态
    selected_wl_id = wl_store.get('watchlist_id') if wl_store else None
    selected_wl_name = wl_store.get('watchlist_name') if wl_store else None

    if triggered_id == 'mgmt-add-btn' and code_input:
        raw_codes = [c for c in re.split(r'[,，;；\s]+', code_input.strip()) if c]
        if len(raw_codes) == 1:
            # 单个添加：尝试获取名称
            code = normalize_stock_code(raw_codes[0])
            if re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code):
                name = ''
                if isinstance(fetched_name, str):
                    m = re.search(r'名称[：:]\s*(.+)', fetched_name)
                    if m:
                        name = m.group(1).strip()
                if not name:
                    name = get_stock_name(code)
                if insert_stock_code(code, name):
                    add_msg = f"已添加 {code}"
                else:
                    add_msg = f"{code} 已存在，未重复添加"
            else:
                add_msg = f"{raw_codes[0]} 格式无效"
        else:
            # 批量添加：不逐个在线查询名称，避免接口限频阻塞
            added, dup, invalid = [], [], []
            for raw in raw_codes:
                code = normalize_stock_code(raw)
                if not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', code):
                    invalid.append(raw)
                elif insert_stock_code(code, ''):
                    added.append(code)
                else:
                    dup.append(code)
            parts = []
            if added:
                parts.append(f"新增 {len(added)} 只")
            if dup:
                parts.append(f"已存在 {len(dup)} 只")
            if invalid:
                parts.append(f"无效 {len(invalid)} 只")
            add_msg = "、".join(parts) if parts else "未添加"

    if 'mgmt-edit-btn' in triggered_id:
        try:
            parsed = json.loads(triggered_id.replace("'", '"'))
            code_to_edit = parsed.get('index', '')
            if code_to_edit:
                stock_list = get_all_stock_codes()
                for code, name in stock_list:
                    if code == code_to_edit:
                        edit_stock_data = {'code': code, 'name': name}
                        modal_style = {'display': 'flex'}
                        break
        except Exception:
            pass

    if triggered_id == 'edit-save-btn' and edit_code and edit_stock_data.get('code'):
        old_code = edit_stock_data.get('code')
        new_code = normalize_stock_code(edit_code.strip())
        new_name = edit_name.strip()
        if re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', new_code):
            update_stock_code(old_code, new_code, new_name)
            modal_style = {'display': 'none'}
            edit_stock_data = {'code': '', 'name': ''}

    if triggered_id == 'edit-cancel-btn':
        modal_style = {'display': 'none'}
        edit_stock_data = {'code': '', 'name': ''}

    keyword = search_keyword if search_keyword else ''
    if keyword:
        stock_list = search_stock_codes(keyword)
    else:
        stock_list = get_all_stock_codes()

    total = len(stock_list)
    named = sum(1 for _, n in stock_list if n)

    # 分组过滤标签
    if selected_wl_id and selected_wl_name:
        filter_badge = html.Span([
            html.I(className="fas fa-layer-group text-purple-500 mr-1"),
            html.Span(f"筛选: {selected_wl_name}", className="text-xs text-purple-700 bg-purple-50 rounded-full px-2 py-0.5"),
        ])
    else:
        filter_badge = None

    # 分页：翻页按钮调整页码，其它操作（搜索/分组/添加/删除/刷新）重置回第一页
    page = page_store if isinstance(page_store, int) else 1
    if triggered_id == 'mgmt-next-btn' and next_clicks:
        page += 1
    elif triggered_id == 'mgmt-prev-btn' and prev_clicks:
        page -= 1
    else:
        page = 1

    table_html, filtered_total, total_pages, page = build_stock_table(
        stock_list, selected_wl_id, selected_wl_name, page=page, page_size=10)

    prev_disabled = page <= 1
    next_disabled = page >= total_pages
    if total_pages <= 1:
        page_info = f"共 {filtered_total} 条"
    else:
        page_info = f"第 {page} / {total_pages} 页 · 共 {filtered_total} 条"

    return table_html, str(total), str(named), modal_style, edit_stock_data, filter_badge, add_msg, prev_disabled, next_disabled, page_info, page

# 点击删除按钮 → 弹出确认框
@app.callback(
    [Output('mgmt-delete-confirm', 'displayed'),
     Output('mgmt-delete-confirm', 'message'),
     Output('mgmt-pending-delete', 'data')],
    [Input({'type': 'mgmt-delete-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def prompt_delete_stock(delete_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or not any(delete_clicks):
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0]
    code = ''
    try:
        parsed = json.loads(triggered_id.replace("'", '"'))
        code = parsed.get('index', '')
    except Exception:
        pass
    if not code:
        raise PreventUpdate
    name = ''
    for c, n in get_all_stock_codes():
        if c == code:
            name = n
            break
    display = name or code
    return True, f"确定要删除股票「{display}」吗？删除后将同时从所有自选股分组中移除。", {'code': code, 'name': name}

# 确认删除 → 执行删除并刷新列表
@app.callback(
    [Output('mgmt-pending-delete', 'data', allow_duplicate=True),
     Output('mgmt-refresh-btn', 'n_clicks', allow_duplicate=True)],
    [Input('mgmt-delete-confirm', 'submit_n_clicks'),
     Input('mgmt-delete-confirm', 'cancel_n_clicks')],
    [State('mgmt-pending-delete', 'data'),
     State('mgmt-refresh-btn', 'n_clicks')],
    prevent_initial_call=True
)
def confirm_delete_stock(submit_clicks, cancel_clicks, pending, refresh_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if 'submit_n_clicks' in ctx.triggered[0]['prop_id'] and pending and pending.get('code'):
        delete_stock_code(pending['code'])
    return {'code': '', 'name': ''}, (refresh_clicks or 0) + 1

# 从当前分组移除股票并刷新
@app.callback(
    Output('mgmt-refresh-btn', 'n_clicks', allow_duplicate=True),
    [Input({'type': 'mgmt-remove-wl-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('wl-selected-store', 'data'),
     State('mgmt-refresh-btn', 'n_clicks')],
    prevent_initial_call=True
)
def remove_stock_from_current_watchlist(remove_clicks, wl_store, refresh_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or not any(remove_clicks):
        raise PreventUpdate
    wl_id = wl_store.get('watchlist_id') if wl_store else None
    if not wl_id:
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0]
    try:
        parsed = json.loads(triggered_id.replace("'", '"'))
        code = parsed.get('index', '')
    except Exception:
        code = ''
    if code:
        remove_stock_from_watchlist(int(wl_id), code)
    return (refresh_clicks or 0) + 1

# 搜索框清除按钮显示/隐藏
@app.callback(
    Output('mgmt-clear-search-btn', 'style'),
    [Input('mgmt-search-input', 'value')]
)
def toggle_clear_search(keyword):
    if keyword:
        return {'display': 'inline-block'}
    return {'display': 'none'}

@app.callback(
    Output('mgmt-fetched-name', 'children'),
    [Input('mgmt-fetch-name-btn', 'n_clicks')],
    [State('mgmt-code-input', 'value')]
)
def fetch_name_preview(n_clicks, code):
    if n_clicks and code:
        code = code.strip()
        if validate_code(code):
            name = get_stock_name(code)
            if name:
                return html.Span([
                    html.I(className="fas fa-check-circle text-green-500 mr-1"),
                    f"名称: {name}"
                ], className="text-green-600")
            else:
                return html.Span([
                    html.I(className="fas fa-exclamation-circle text-yellow-500 mr-1"),
                    "未获取到名称，添加后将自动重试"
                ], className="text-yellow-600")
        else:
            return html.Span([
                html.I(className="fas fa-times-circle text-red-500 mr-1"),
                "无效的股票代码格式"
            ], className="text-red-500")
    return ""

@app.callback(
    Output('mgmt-code-input', 'value'),
    [Input('mgmt-add-btn', 'n_clicks')],
    [State('mgmt-code-input', 'value')]
)
def clear_after_add(n_clicks, code):
    if n_clicks and code:
        return ''
    return dash.no_update

@app.callback(
    [Output('edit-code-input', 'value'),
     Output('edit-name-input', 'value')],
    [Input('edit-stock-data', 'data')]
)
def update_edit_inputs(edit_data):
    if edit_data and edit_data.get('code'):
        return edit_data.get('code', ''), edit_data.get('name', '')
    return '', ''

@app.callback(
    Output('mgmt-search-input', 'value'),
    [Input('mgmt-clear-search-btn', 'n_clicks')]
)
def clear_search(n_clicks):
    if n_clicks:
        return ''
    return dash.no_update

# ==================== 自选股分组回调 ====================

@app.callback(
    [Output('wl-list-container', 'children'),
     Output('wl-assign-dropdown', 'options'),
     Output('wl-new-name', 'value')],
    [Input('wl-create-btn', 'n_clicks'),
      Input({'type': 'wl-select-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
      Input('mgmt-refresh-btn', 'n_clicks'),
      Input('wl-assign-btn', 'n_clicks')],
    [State('wl-new-name', 'value')]
)
def manage_watchlists(create_clicks, select_clicks, refresh_clicks, assign_clicks, new_name):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0] if ctx.triggered else ''

    # 创建分组
    if triggered_id == 'wl-create-btn' and create_clicks and new_name and new_name.strip():
        create_watchlist(new_name.strip())

    # 刷新列表
    wlists = get_all_watchlists()
    items = []
    dropdown_opts = []

    for wl_id, wl_name in wlists:
        stock_count = len(get_watchlist_stocks(wl_id))
        items.append(html.Div([
            html.Button([
                html.I(className="fas fa-chevron-right text-purple-400 mr-2"),
                html.Span(wl_name, className="text-sm font-medium text-gray-700"),
                html.Span(f" ({stock_count})", className="text-xs text-gray-400"),
            ], id={'type': 'wl-select-btn', 'index': wl_id},
               className="flex items-center w-full text-left py-2 px-3 rounded-lg hover:bg-purple-50 transition-colors"),
            html.Button([html.I(className="fas fa-times-circle text-red-300 hover:text-red-500 text-xs")],
                       id={'type': 'wl-delete-btn', 'index': wl_id},
                       className="ml-auto px-1"),
        ], className="flex items-center"))
        dropdown_opts.append({'label': f'{wl_name} ({stock_count})', 'value': wl_id})

    if not items:
        items = [html.P("暂无分组，请新建", className="text-xs text-gray-400 py-2")]

    return items, dropdown_opts, ''


# 点击删除分组按钮 → 弹出确认框
@app.callback(
    [Output('mgmt-wl-delete-confirm', 'displayed'),
     Output('mgmt-wl-delete-confirm', 'message'),
     Output('mgmt-pending-wl-delete', 'data')],
    [Input({'type': 'wl-delete-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    prevent_initial_call=True
)
def prompt_delete_watchlist(delete_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or not any(delete_clicks):
        raise PreventUpdate
    triggered_id = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0]
    try:
        parsed = json.loads(triggered_id.replace("'", '"'))
        wl_id = parsed.get('index')
    except Exception:
        wl_id = None
    if wl_id is None:
        raise PreventUpdate
    wl_name = ''
    for wid, wname in get_all_watchlists():
        if wid == wl_id:
            wl_name = wname
            break
    return True, f"确定要删除分组「{wl_name}」吗？分组内的股票不会被删除，仅解除分组关系。", {'id': wl_id, 'name': wl_name}

# 确认删除分组 → 执行删除并刷新
@app.callback(
    [Output('mgmt-pending-wl-delete', 'data', allow_duplicate=True),
     Output('mgmt-refresh-btn', 'n_clicks', allow_duplicate=True)],
    [Input('mgmt-wl-delete-confirm', 'submit_n_clicks'),
     Input('mgmt-wl-delete-confirm', 'cancel_n_clicks')],
    [State('mgmt-pending-wl-delete', 'data'),
     State('mgmt-refresh-btn', 'n_clicks')],
    prevent_initial_call=True
)
def confirm_delete_watchlist(submit_clicks, cancel_clicks, pending, refresh_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if 'submit_n_clicks' in ctx.triggered[0]['prop_id'] and pending and pending.get('id') is not None:
        delete_watchlist(int(pending['id']))
    return {'id': None, 'name': ''}, (refresh_clicks or 0) + 1


@app.callback(
    Output('wl-selected-store', 'data'),
    [Input({'type': 'wl-select-btn', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('wl-selected-store', 'data')],
    prevent_initial_call=True
)
def select_watchlist(select_clicks, current_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        parsed = json.loads(triggered_id.replace("'", '"'))
        wl_id = parsed.get('index')
    except:
        return dash.no_update

    if not any(c for c in select_clicks if c):
        return dash.no_update

    current_id = current_store.get('watchlist_id') if current_store else None
    if current_id == wl_id:
        return {'watchlist_id': None, 'watchlist_name': None}
    else:
        wlists = get_all_watchlists()
        wl_name = None
        for wid, wname in wlists:
            if wid == wl_id:
                wl_name = wname
                break
        return {'watchlist_id': wl_id, 'watchlist_name': wl_name}


@app.callback(
    [Output('wl-assign-msg', 'children'),
     Output('wl-assign-code', 'value')],
    [Input('wl-assign-btn', 'n_clicks')],
    [State('wl-assign-dropdown', 'value'),
     State('wl-assign-code', 'value')],
    prevent_initial_call=True
)
def assign_stock_to_watchlist(n_clicks, wl_id, code):
    if not n_clicks or not code:
        return dash.no_update, dash.no_update

    code = code.strip()
    if not code:
        return "请输入股票代码", dash.no_update
    if not wl_id:
        return "请选择分组", dash.no_update

    if add_stock_to_watchlist(int(wl_id), code):
        return f"已添加 {code}", ''
    else:
        return "添加失败", dash.no_update


# 注册因子训练页面回调
register_factor_callbacks(app)

# 注册趋势分析页面回调
register_trend_callbacks(app)

# 注册每日五面分析页面回调
register_market_callbacks(app)
register_sector_fund_callbacks(app)

# 注册交易日志页面回调
register_trading_journal_callbacks(app)

# 注册模拟炒股小游戏页面回调
register_stock_game_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)