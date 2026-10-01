"""
因子训练可视化页面 - 5大功能模块
"""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import glob
import io
import os
import re
import logging
import pickle
import hashlib
import time

from .factor_engine import (FactorDataProvider, preprocess_factor, calculate_rank_ic,
                            calculate_ic_statistics, layered_backtest, compute_group_returns,
                            train_lgbm_rolling, compute_icir_weighted_score, portfolio_backtest)
from .data_manager import batch_get_data, set_data_source, get_data_source, get_available_sources
from .database_manager import get_all_stock_codes

# ==================== 预设因子公式库 ====================
PRESET_FACTOR_OPTIONS = [
    {'label': '-- 选择公式 --', 'value': '', 'title': ''},
    # 反转类
    {'label': '【反转】5日动量反转 - close/MA(close,5)-1', 'value': 'close/MA(close,5)-1', 'title': '5日反转因子'},
    {'label': '【反转】20日反转 - close/MA(close,20)-1', 'value': 'close/MA(close,20)-1', 'title': '20日反转因子'},
    {'label': '【反转】60日反转 - close/MA(close,60)-1', 'value': 'close/MA(close,60)-1', 'title': '60日反转因子'},
    # 波动率类
    {'label': '【波动】5日波动率 - (high-low)/MA(close,5)', 'value': '(high-low)/MA(close,5)', 'title': '5日真实波幅/均值'},
    {'label': '【波动】20日波动率 - (high-low)/MA(close,20)', 'value': '(high-low)/MA(close,20)', 'title': '20日真实波幅/均值'},
    # 成交量类
    {'label': '【量价】20日量比 - vol/MA(vol,20)', 'value': 'vol/MA(vol,20)', 'title': '当日成交量/20日均量'},
    {'label': '【量价】5日换手率变化 - vol/MA(vol,5)-1', 'value': 'vol/MA(vol,5)-1', 'title': '5日量变动'},
    {'label': '【量价】成交额比 - amount/MA(amount,20)', 'value': 'amount/MA(amount,20)', 'title': '20日额比'},
    # 技术形态
    {'label': '【技术】5日涨跌比 - close/close.shift(5)-1', 'value': 'close/close.shift(5)-1', 'title': '5日涨跌比率'},
    {'label': '【技术】10日涨跌幅 - close/close.shift(10)-1', 'value': 'close/close.shift(10)-1', 'title': '10日动量'},
    {'label': '【技术】20日涨跌幅 - close/close.shift(20)-1', 'value': 'close/close.shift(20)-1', 'title': '20日动量'},
    {'label': '【技术】振幅比率 - (high-low)/open', 'value': '(high-low)/open', 'title': '日内振幅/开盘价'},
    # 价格形态
    {'label': '【形态】上影线占比', 'value': '(high-((open+close+abs(open-close))/2))/(high-low+0.001)', 'title': '上影线长度/振幅'},
    {'label': '【形态】收盘位置 - (close-low)/(high-low+0.001)', 'value': '(close-low)/(high-low+0.001)', 'title': '收盘价在日内的相对位置'},
    # 衍生
    {'label': '【其他】日收益率 - close/open-1', 'value': 'close/open-1', 'title': '日内收益率'},
    {'label': '【其他】5日均价偏离 - (close-MA(close,5))/MA(close,5)', 'value': '(close-MA(close,5))/MA(close,5)', 'title': '偏离5日均线幅度'},
]

# ==================== 磁盘缓存 ====================
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'factor_data')
os.makedirs(_CACHE_DIR, exist_ok=True)

def _disk_cache_path(pool, sd, ed, data_source):
    """生成磁盘缓存文件路径"""
    key_str = f"{pool}_{sd}_{ed}_{data_source}"
    fname = hashlib.md5(key_str.encode()).hexdigest() + '.pkl'
    return os.path.join(_CACHE_DIR, fname)

def _load_from_disk(cache_path):
    """从磁盘加载缓存的 FactorDataProvider"""
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        provider = FactorDataProvider(data['codes'], data['start'], data['end'], source=data['source'])
        provider.data_cache = data['data_cache']
        provider._loaded = True
        return provider, data['codes']
    except Exception as e:
        logging.warning(f"磁盘缓存加载失败: {e}")
        return None, None

def _save_to_disk(cache_path, provider, codes):
    """将 FactorDataProvider 的数据缓存到磁盘"""
    try:
        data = {
            'codes': codes,
            'start': provider.start_date,
            'end': provider.end_date,
            'source': provider.source,
            'data_cache': provider.data_cache,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.warning(f"磁盘缓存保存失败: {e}")

# ==================== 页面布局 ====================

def factor_page_layout():
    return html.Div([
        html.Main([
            # ===== 模块1: 数据源配置区 =====
            html.Div([
                html.Div([
                    html.I(className="fas fa-database text-blue-600 text-xl mr-2"),
                    html.H3("1. 数据源配置", className="text-lg font-bold text-gray-800"),
                ], className="flex items-center mb-4 pb-2 border-b"),
                html.Div([
                    html.Div([
                        html.Label("股票池:", className="text-sm font-medium text-gray-600"),
                        dcc.Dropdown(id='ft-stock-pool', options=[
                            {'label': '自选股池', 'value': 'custom'},
                            {'label': '全A股', 'value': 'all_a'},
                            {'label': '沪深300', 'value': 'hs300'},
                            {'label': '中证500', 'value': 'zz500'},
                        ], value='custom', clearable=False, className="text-sm"),
                    ], className="w-1/4 pr-2"),
                    html.Div([
                        html.Label("开始日期:", className="text-sm font-medium text-gray-600"),
                        dcc.DatePickerSingle(id='ft-start-date',
                            date=pd.Timestamp.now() - pd.Timedelta(days=365*3),
                            className="w-full"),
                    ], className="w-1/5 pr-2"),
                    html.Div([
                        html.Label("结束日期:", className="text-sm font-medium text-gray-600"),
                        dcc.DatePickerSingle(id='ft-end-date',
                            date=pd.Timestamp.now(),
                            className="w-full"),
                    ], className="w-1/5 pr-2"),
                    html.Div([
                        html.Label("预测标签:", className="text-sm font-medium text-gray-600"),
                        dcc.Dropdown(id='ft-label-type', options=[
                            {'label': '未来5日收益', 'value': '5'},
                            {'label': '未来20日收益', 'value': '20'},
                        ], value='5', clearable=False, className="text-sm"),
                    ], className="w-1/6 pr-2"),
                    html.Div([
                        html.Label("数据源:", className="text-sm font-medium text-gray-600"),
                        dcc.Dropdown(id='ft-data-source', options=[
                            {'label': 'Tushare (推荐)', 'value': 'tushare'},
                            {'label': 'Baostock (免费)', 'value': 'baostock'},
                        ], value='tushare', clearable=False, className="text-sm"),
                    ], className="w-1/6"),
                ], className="flex flex-wrap items-end mb-3"),
                html.Div([
                    html.Button([html.I(className="fas fa-download mr-1"), "加载数据"],
                        id='ft-load-data-btn', className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm"),
                    dcc.Checklist(id='ft-force-reload', options=[
                        {'label': ' 强制更新', 'value': 'force'}
                    ], value=[], className="text-sm text-gray-500 ml-3", inline=True),
                    dcc.Loading(
                        id='ft-load-loading',
                        type='default',
                        children=html.Span(id='ft-load-status', className="text-sm text-gray-500 ml-3"),
                    ),
                ], className="flex items-center"),
                dcc.Store(id='ft-data-store'),
            ], className="bg-white rounded-xl shadow-md p-4 mb-4"),

            # ===== 模块2: 单因子加工 =====
            html.Div([
                html.Div([
                    html.I(className="fas fa-cogs text-green-600 text-xl mr-2"),
                    html.H3("2. 单因子加工", className="text-lg font-bold text-gray-800"),
                ], className="flex items-center mb-4 pb-2 border-b"),
                html.Div([
                    html.Div([
                        html.Label("选择因子公式:", className="text-sm font-medium text-gray-600"),
                        dcc.Dropdown(
                            id='ft-factor-dropdown',
                            options=PRESET_FACTOR_OPTIONS,
                            value='',
                            clearable=False,
                            className="mb-2",
                        ),
                    ]),
                    html.Div([
                        html.Label("或自定义公式:", className="text-sm font-medium text-gray-600"),
                        dcc.Input(id='ft-factor-expr', type='text', placeholder='例如: close/open-1',
                            className="w-full border rounded-lg px-3 py-2 text-sm"),
                    ], className="mb-3"),
                    html.Div([
                        html.Button([html.I(className="fas fa-play mr-1"), "执行预处理"],
                            id='ft-run-preprocess-btn', className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm"),
                    ], className="flex items-center"),
                ], className="mb-3"),
                html.Div(id='ft-factor-preprocess-msg', className="text-sm text-gray-600 mb-2"),
                dcc.Graph(id='ft-preprocess-graph', config={'displayModeBar': False}, className="h-80"),
                dcc.Store(id='ft-processed-factors-store'),
            ], className="bg-white rounded-xl shadow-md p-4 mb-4"),

            # ===== 模块3: 单因子有效性校验 =====
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line text-purple-600 text-xl mr-2"),
                    html.H3("3. 单因子有效性校验", className="text-lg font-bold text-gray-800"),
                ], className="flex items-center mb-4 pb-2 border-b"),
                html.Div([
                    html.Button([html.I(className="fas fa-check-circle mr-1"), "计算IC & 分层回测"],
                        id='ft-validate-factor-btn', className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg text-sm"),
                    html.Span(id='ft-validate-status', className="text-sm text-gray-500 ml-3"),
                ], className="mb-3"),
                html.Div(id='ft-ic-stats-table', className="mb-3"),
                dcc.Graph(id='ft-ic-graph', config={'displayModeBar': False}, className="h-72 mb-3"),
                dcc.Graph(id='ft-layer-return-graph', config={'displayModeBar': False}, className="h-72 mb-3"),
                dcc.Graph(id='ft-group-nav-graph', config={'displayModeBar': False}, className="h-72"),
                dcc.Store(id='ft-validation-store'),
            ], className="bg-white rounded-xl shadow-md p-4 mb-4"),

            # ===== 模块4: 多因子建模 =====
            html.Div([
                html.Div([
                    html.I(className="fas fa-brain text-orange-600 text-xl mr-2"),
                    html.H3("4. 多因子建模", className="text-lg font-bold text-gray-800"),
                ], className="flex items-center mb-4 pb-2 border-b"),
                html.Div([
                    html.Label("选择建模因子:", className="text-sm font-medium text-gray-600 mb-1 block"),
                    dcc.Checklist(id='ft-model-factors-checklist', className="text-sm"),
                ], className="mb-3"),
                html.Div([
                    html.Label("建模方式:", className="text-sm font-medium text-gray-600"),
                    dcc.RadioItems(id='ft-model-method', options=[
                        {'label': 'ICIR 线性加权', 'value': 'icir'},
                        {'label': 'LGBM 滚动训练', 'value': 'lgbm'},
                    ], value='icir', inline=True, className="text-sm mb-2"),
                ], className="mb-2"),
                html.Div([
                    html.Div([
                        html.Label("LGBM训练月数:", className="text-sm font-medium text-gray-600"),
                        dcc.Input(id='ft-lgbm-months', type='number', value=60, min=12, max=120, className="w-24 border rounded px-2 py-1 text-sm ml-2"),
                    ], className="flex items-center"),
                ], className="mb-3"),
                html.Div([
                    html.Button([html.I(className="fas fa-rocket mr-1"), "运行模型"],
                        id='ft-run-model-btn', className="bg-orange-600 hover:bg-orange-700 text-white px-4 py-2 rounded-lg text-sm"),
                    dcc.Input(id='ft-model-name', type='text', placeholder='模型名称（留空自动生成）',
                        className="border rounded-lg px-3 py-2 text-sm ml-3 w-56"),
                    html.Span(id='ft-model-status', className="text-sm text-gray-500 ml-3"),
                ], className="mb-3"),
                dcc.Graph(id='ft-feature-importance-graph', config={'displayModeBar': False}, className="h-72"),
                html.Div([
                    html.Label("已保存模型:", className="text-sm font-medium text-gray-600 mr-3 whitespace-nowrap"),
                    dcc.Dropdown(id='ft-saved-model-dropdown', options=[],
                        placeholder='选择已保存的模型...', className="flex-1 text-sm"),
                    html.Button([html.I(className="fas fa-folder-open mr-1"), "加载"],
                        id='ft-load-model-btn', className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg text-sm ml-3 whitespace-nowrap"),
                    html.Button([html.I(className="fas fa-trash-alt mr-1"), "删除"],
                        id='ft-delete-model-btn', className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm ml-2 whitespace-nowrap"),
                    html.Span(id='ft-model-mgmt-status', className="text-sm text-gray-500 ml-3"),
                ], className="flex items-center mt-3 pt-3 border-t flex-wrap gap-y-2"),
                dcc.Store(id='ft-model-mgmt-store'),
                dcc.Store(id='ft-model-store'),
            ], className="bg-white rounded-xl shadow-md p-4 mb-4"),

            # ===== 模块5: 组合回测 & 导出 =====
            html.Div([
                html.Div([
                    html.I(className="fas fa-file-export text-red-600 text-xl mr-2"),
                    html.H3("5. 组合回测 & 导出", className="text-lg font-bold text-gray-800"),
                ], className="flex items-center mb-4 pb-2 border-b"),
                html.Div([
                    html.Div([
                        html.Label("选股比例(%):", className="text-sm font-medium text-gray-600"),
                        dcc.Input(id='ft-top-pct', type='number', value=20, min=5, max=50, className="w-20 border rounded px-2 py-1 text-sm"),
                    ], className="mr-4"),
                    html.Div([
                        html.Label("单票上限(%):", className="text-sm font-medium text-gray-600"),
                        dcc.Input(id='ft-single-cap', type='number', value=10, min=1, max=30, className="w-20 border rounded px-2 py-1 text-sm"),
                    ], className="mr-4"),
                    html.Button([html.I(className="fas fa-play mr-1"), "运行回测"],
                        id='ft-run-backtest-btn', className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg text-sm"),
                ], className="flex items-center flex-wrap mb-3"),
                html.Div(id='ft-backtest-stats', className="mb-3 text-sm"),
                dcc.Graph(id='ft-backtest-nav-graph', config={'displayModeBar': False}, className="h-72 mb-3"),
                html.Div([
                    html.Button([html.I(className="fas fa-file-excel mr-1"), "导出Excel"],
                        id='ft-export-btn', className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm"),
                    dcc.Download(id='ft-download-excel'),
                    html.Span(id='ft-export-status', className="text-sm text-gray-500 ml-3"),
                ]),
                dcc.Store(id='ft-backtest-store'),
            ], className="bg-white rounded-xl shadow-md p-4"),
        ], className="container mx-auto px-4 py-6"),
    ])


# ==================== 模块级缓存 ====================
_data_provider = None
_factor_cache = {}
_processed_factors = {}
_ic_results = {}
_layer_results = {}
_model_scores = None
_backtest_result = None
_model_method = None  # 'icir' | 'lgbm'，供分析页展示模型类型

# Excel 工作表名不允许出现 \ / ? * [ ] : 且最长 31 字符，因子公式名需要清洗
_SHEET_NAME_INVALID = re.compile(r'[\\*?:/\[\]]')


def _safe_sheet_name(name, used_names):
    """清洗为合法且不重复的 Excel sheet 名"""
    safe = _SHEET_NAME_INVALID.sub('_', str(name)).strip() or 'sheet'
    safe = safe[:31]
    base = safe
    i = 1
    while safe.lower() in used_names:
        suffix = f'~{i}'
        safe = base[:31 - len(suffix)] + suffix
        i += 1
    used_names.add(safe.lower())
    return safe


# ==================== 模型持久化 ====================
_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'factor_models')
os.makedirs(_MODELS_DIR, exist_ok=True)
_active_model_name = None
_autoload_attempted = False

_FILENAME_INVALID = re.compile(r'[\\/:*?"<>|]')


def _safe_model_filename(name):
    """模型名 → 安全文件名"""
    return _FILENAME_INVALID.sub('_', str(name)).strip().strip('.')[:80] or 'model'


def save_model_to_disk(custom_name=None):
    """把当前训练好的模型（评分+因子暴露+IC）保存到 cache/factor_models/，返回保存名。
    pkl 存数据，同名 json 存元数据（列表展示时无需读大文件）。"""
    global _active_model_name
    if _model_scores is None or _model_scores.empty:
        raise ValueError('当前没有已训练的模型可保存')

    latest_date = pd.Timestamp(_model_scores.index.max()).strftime('%Y-%m-%d')
    method_label = 'ICIR' if (_model_method or 'icir') == 'icir' else 'LGBM'
    if custom_name and str(custom_name).strip():
        name = str(custom_name).strip()
    else:
        name = f'{method_label}_{latest_date}_{time.strftime("%H%M")}'

    fname = _safe_model_filename(name)
    meta = {
        'name': name, 'file': fname + '.pkl', 'method': _model_method or 'icir',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_factors': len(_processed_factors),
        'n_stocks': int(_model_scores.loc[_model_scores.index.max()].notna().sum()),
        'latest_date': latest_date,
    }
    payload = {
        'schema': 1, 'name': name, 'method': _model_method or 'icir',
        'model_scores': _model_scores,
        'processed_factors': dict(_processed_factors),
        'ic_results': dict(_ic_results),
    }
    with open(os.path.join(_MODELS_DIR, fname + '.pkl'), 'wb') as f:
        pickle.dump(payload, f)
    with open(os.path.join(_MODELS_DIR, fname + '.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _active_model_name = name
    return name


def list_saved_models():
    """列出已保存模型（按保存时间倒序），只读 json 元数据"""
    models = []
    for path in glob.glob(os.path.join(_MODELS_DIR, '*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('file') and os.path.exists(os.path.join(_MODELS_DIR, meta['file'])):
                models.append(meta)
        except Exception as e:
            logging.warning(f"读取模型元数据失败 {path}: {e}")
    models.sort(key=lambda m: m.get('created_at', ''), reverse=True)
    return models


def load_saved_model(name):
    """按名称加载已保存模型到内存并设为当前模型，返回模型信息"""
    global _model_scores, _processed_factors, _ic_results, _model_method, _active_model_name
    fname = _safe_model_filename(name)
    path = os.path.join(_MODELS_DIR, fname + '.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'模型文件不存在: {fname}.pkl')
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    _model_scores = payload['model_scores']
    _processed_factors = payload.get('processed_factors', {})
    _ic_results = payload.get('ic_results', {})
    _model_method = payload.get('method', 'icir')
    _active_model_name = payload.get('name', str(name))
    return {
        'name': _active_model_name, 'method': _model_method,
        'n_factors': len(_processed_factors),
        'n_stocks': int(_model_scores.loc[_model_scores.index.max()].notna().sum()),
    }


def delete_saved_model(name):
    """删除已保存模型文件，返回是否删除了文件（不影响内存中的当前模型）"""
    fname = _safe_model_filename(name)
    removed = False
    for suffix in ('.pkl', '.json'):
        p = os.path.join(_MODELS_DIR, fname + suffix)
        if os.path.exists(p):
            os.remove(p)
            removed = True
    return removed


def set_active_model(name):
    """切换当前使用的模型；与当前相同且已有内存模型时为空操作"""
    if name and _active_model_name == str(name) and _model_scores is not None:
        return {'name': _active_model_name, 'method': _model_method}
    return load_saved_model(name)


def get_active_model_name():
    """当前生效的模型名（可能为 None）"""
    return _active_model_name


# ==================== 分析页接入：训练因子评分 ====================

def _normalize_stock_code(code):
    """分析页下拉可能是纯数字代码，统一补齐 .SH/.SZ 后缀（与分析页主图逻辑一致）"""
    code = str(code).strip()
    if code.isdigit() and len(code) == 6:
        return code + ('.SH' if int(code) >= 500000 else '.SZ')
    return code


def get_factor_score_snapshot():
    """返回已训练模型的评分快照（评分矩阵 + 最新日期横截面），无模型时返回 None。
    重启后内存为空时，自动加载最近保存的模型。"""
    global _autoload_attempted
    if _model_scores is None or _model_scores.empty:
        if not _autoload_attempted:
            _autoload_attempted = True
            try:
                models = list_saved_models()
                if models:
                    load_saved_model(models[0]['name'])
                    logging.info(f"自动加载最近保存的因子模型: {models[0]['name']}")
            except Exception as e:
                logging.warning(f"自动加载因子模型失败: {e}")
    if _model_scores is None or _model_scores.empty:
        return None
    scores = _model_scores.dropna(how='all').dropna(axis=1, how='all')
    if scores.empty:
        return None
    latest_date = scores.index.max()
    return {
        'latest_date': pd.Timestamp(latest_date),
        'scores': scores,
        'latest': scores.loc[latest_date].dropna().sort_values(ascending=False),
        'method': _model_method or 'icir',
        'n_factors': len(_processed_factors),
        'factors': list(_processed_factors.keys()),
        'name': _active_model_name,
    }


def get_stock_factor_info(snap, code):
    """从评分快照中取个股信息：最新评分、全池排名分位、近30个交易日走势、各因子最新暴露"""
    code = _normalize_stock_code(code)
    latest = snap['latest']
    info = {'code': code, 'covered': code in latest.index}
    if info['covered']:
        rank = int((latest > latest[code]).sum()) + 1
        info.update({
            'score': float(latest[code]),
            'rank': rank,
            'total': int(len(latest)),
            'pct': rank / max(len(latest), 1) * 100.0,
            'history': snap['scores'][code].dropna().tail(30),
        })
    exposures = {}
    for fname, df in _processed_factors.items():
        try:
            if code in df.columns and snap['latest_date'] in df.index:
                v = df.loc[snap['latest_date'], code]
                if pd.notna(v):
                    exposures[fname] = float(v)
        except Exception:
            pass
    info['exposures'] = exposures
    return info


def build_factor_score_panel(codes, name_map=None):
    """分析页「训练因子评分」面板：展示所选股票的模型评分、排名分位、评分走势与因子暴露，
    并附当日评分 Top10 列表。未训练模型时返回引导提示。"""
    name_map = name_map or {}
    snap = get_factor_score_snapshot()
    if snap is None:
        return html.Div([
            html.I(className="fas fa-flask text-gray-400 mr-2"),
            html.Span("尚未训练因子模型：", className="text-gray-500 text-sm"),
            dcc.Link("前往「因子训练」页", href='/factor-training',
                     className="text-blue-600 text-sm hover:underline"),
            html.Span("完成预处理与建模后，这里会展示所选股票的因子评分与排名。",
                      className="text-gray-500 text-sm"),
        ], className="bg-white rounded-xl shadow-md p-3 mb-4 flex items-center justify-center flex-wrap")

    method_label = 'ICIR 加权' if snap['method'] == 'icir' else 'LGBM 滚动'
    date_str = snap['latest_date'].strftime('%Y-%m-%d')

    def badge(text, color):
        return html.Span(text, className=f"text-xs px-2 py-1 rounded-full bg-{color}-50 text-{color}-700 border border-{color}-200")

    current_name = snap.get('name')
    header = html.Div([
        html.Div([
            html.I(className="fas fa-brain text-indigo-600 text-xl mr-2"),
            html.H3("训练因子评分", className="text-lg font-bold text-gray-800"),
        ], className="flex items-center"),
        html.Div([
            html.Span(f"当前: {current_name}" if current_name else f"模型: {method_label}",
                      className="text-xs px-2 py-1 rounded-full bg-purple-50 text-purple-700 border border-purple-200 max-w-xs truncate"),
            badge(f"模型: {method_label}", 'indigo'),
            badge(f"因子: {snap['n_factors']} 个", 'blue'),
            badge(f"覆盖: {len(snap['latest'])} 只", 'gray'),
            badge(f"评分日期: {date_str}", 'gray'),
        ], className="flex flex-wrap gap-2"),
    ], className="flex flex-wrap items-center justify-between gap-2 mb-3 pb-2 border-b")

    # ===== 左侧：所选股票评分卡片（最多 3 只，避免面板过重） =====
    stock_cards = []
    for raw_code in (codes or [])[:3]:
        code = _normalize_stock_code(raw_code)
        try:
            info = get_stock_factor_info(snap, code)
        except Exception as e:
            logging.warning(f"因子评分卡片构建失败 {code}: {e}")
            continue
        name = name_map.get(code) or name_map.get(str(raw_code)) or ''
        title = f"{name} ({code})" if name else code

        if not info['covered']:
            body = html.Div(f"该股不在本次训练的股票池中（覆盖 {len(snap['latest'])} 只）",
                            className="text-sm text-gray-400 py-4 text-center")
        else:
            pct = info['pct']
            if pct <= 20:
                zone_badge, zone_color = f"前 {pct:.0f}%（强势区）", 'green'
            elif pct <= 50:
                zone_badge, zone_color = f"前 {pct:.0f}%", 'amber'
            else:
                zone_badge, zone_color = f"后 {100 - pct:.0f}%（弱势区）", 'gray'

            hist = info['history']
            fig_hist = go.Figure()
            if len(hist) > 0:
                fig_hist.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines+markers',
                                              line=dict(color='#6366F1', width=2), marker=dict(size=4),
                                              hovertemplate='%{x|%m-%d}: %{y:.3f}<extra></extra>'))
            fig_hist.update_layout(title='近30个交易日评分走势', template='plotly_white', height=180,
                                   margin=dict(l=30, r=10, t=35, b=25))

            exp_items = sorted(info['exposures'].items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            fig_exp = go.Figure()
            if exp_items:
                fig_exp.add_trace(go.Bar(
                    y=[f"{k if len(k) <= 18 else k[:17] + '…'}" for k, _ in exp_items][::-1],
                    x=[v for _, v in exp_items][::-1], orientation='h',
                    marker_color=['#10B981' if v >= 0 else '#EF4444' for _, v in exp_items][::-1],
                    customdata=[k for k, _ in exp_items][::-1],
                    hovertemplate='%{customdata}: %{x:.3f}<extra></extra>'))
            fig_exp.update_layout(title='各因子最新暴露(z分)', template='plotly_white', height=180,
                                  margin=dict(l=120, r=10, t=35, b=25))

            body = html.Div([
                html.Div([
                    html.Span(f"{info['score']:.3f}", className="text-3xl font-bold text-indigo-700 mr-3"),
                    html.Span(f"第 {info['rank']}/{info['total']} 名", className="text-sm text-gray-600 mr-2"),
                    html.Span(zone_badge, className=f"text-xs px-2 py-1 rounded-full bg-{zone_color}-50 text-{zone_color}-700 border border-{zone_color}-200"),
                ], className="flex items-center mb-2"),
                html.Div([
                    html.Div(dcc.Graph(figure=fig_hist, config={'displayModeBar': False}), className="w-1/2 pr-2"),
                    html.Div(dcc.Graph(figure=fig_exp, config={'displayModeBar': False}), className="w-1/2 pl-2"),
                ], className="flex"),
            ])

        stock_cards.append(html.Div([
            html.Div(title, className="text-sm font-semibold text-gray-700 mb-1"),
            body,
        ], className="bg-gray-50 rounded-lg p-3 mb-3"))

    if not stock_cards:
        stock_cards = [html.Div("在左侧选择股票后，这里展示其因子评分详情",
                                className="text-sm text-gray-400 py-6 text-center")]

    # ===== 右侧：当日评分 Top10 =====
    latest = snap['latest']
    top_rows = []
    for i, (code, score) in enumerate(latest.head(10).items(), start=1):
        name = name_map.get(code, '')
        top_rows.append(html.Tr([
            html.Td(str(i), className="border px-2 py-1 text-sm text-gray-500 w-8"),
            html.Td(f"{code} {name}".strip(), className="border px-2 py-1 text-sm font-medium"),
            html.Td(f"{score:.3f}", className="border px-2 py-1 text-sm text-indigo-700 text-right"),
        ]))
    top_table = html.Table(
        [html.Tr([html.Th("#", className="border px-2 py-1 text-xs w-8"),
                  html.Th("股票", className="border px-2 py-1 text-xs"),
                  html.Th("评分", className="border px-2 py-1 text-xs text-right")],
                 className="bg-gray-100")] + top_rows,
        className="border-collapse border w-full")
    if not top_rows:
        top_table = html.Div("暂无评分数据", className="text-sm text-gray-400 text-center py-4")

    return html.Div([
        header,
        html.Div([
            html.Div(stock_cards, className="flex-1 min-w-0 pr-0 lg:pr-4"),
            html.Div([
                html.Div("当日评分 Top10", className="text-sm font-semibold text-gray-700 mb-2"),
                top_table,
            ], className="w-full lg:w-80 shrink-0 mt-3 lg:mt-0"),
        ], className="flex flex-col lg:flex-row"),
    ], className="bg-white rounded-xl shadow-md p-4 mb-4")


# ==================== 回调注册 ====================

def register_factor_callbacks(app):

    # ---------- 模块2：下拉选公式 → 填入输入框 ----------
    @app.callback(
        Output('ft-factor-expr', 'value'),
        [Input('ft-factor-dropdown', 'value')],
        prevent_initial_call=True
    )
    def select_factor_formula(value):
        if value:
            return value
        return dash.no_update

    # ---------- 模块1：加载数据 ----------
    @app.callback(
        [Output('ft-load-status', 'children'),
         Output('ft-data-store', 'data')],
        [Input('ft-load-data-btn', 'n_clicks')],
        [State('ft-stock-pool', 'value'),
         State('ft-start-date', 'date'),
         State('ft-end-date', 'date'),
         State('ft-data-source', 'value'),
         State('ft-force-reload', 'value')]
    )
    def load_factor_data(n_clicks, pool, start_date, end_date, data_source, force_reload):
        global _data_provider, _factor_cache
        if not n_clicks:
            return "点击加载数据", dash.no_update

        try:
            if data_source:
                set_data_source(data_source)

            codes = _get_pool_codes(pool)
            if not codes:
                return "股票池为空：自选股请先在交易日志中添加，或改用全A股/沪深300", dash.no_update

            sd = start_date.replace('-', '') if start_date else '20200101'
            ed = end_date.replace('-', '') if end_date else pd.Timestamp.now().strftime('%Y%m%d')
            if isinstance(sd, str) and len(sd) > 8:
                sd = pd.Timestamp(sd).strftime('%Y%m%d')
            if isinstance(ed, str) and len(ed) > 8:
                ed = pd.Timestamp(ed).strftime('%Y%m%d')

            is_force = 'force' in (force_reload or [])
            cache_path = _disk_cache_path(pool, sd, ed, data_source)

            # 非强制更新时，优先从磁盘缓存加载
            if not is_force and os.path.exists(cache_path):
                cached_provider, _ = _load_from_disk(cache_path)
                if cached_provider and cached_provider._loaded:
                    _data_provider = cached_provider
                    _factor_cache = {}
                    loaded = len(_data_provider.data_cache)
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(cache_path)))
                    return (f"从磁盘缓存加载: {loaded} 只股票 ({sd} ~ {ed}) 缓存于 {mtime}",
                            {'codes': codes, 'start': sd, 'end': ed, 'loaded': True, 'cached': True})

            # 从数据源获取新数据
            t0 = time.time()
            _data_provider = FactorDataProvider(codes, sd, ed, source=data_source)
            _data_provider.load_data()
            _factor_cache = {}
            elapsed = time.time() - t0

            # 保存到磁盘缓存
            _save_to_disk(cache_path, _data_provider, codes)

            loaded = len(_data_provider.data_cache)
            failed = len(codes) - loaded
            action = "已重新加载" if is_force else "加载完成（已缓存）"
            status = f"{action}: {loaded}/{len(codes)} 只有数据 (耗时 {elapsed:.0f}s)"
            if failed > 0:
                status += f", {failed} 只无数据（已过滤）"
            return (status, {'codes': codes, 'start': sd, 'end': ed, 'loaded': True, 'cached': False})
        except Exception as e:
            logging.error(f"加载数据失败: {e}")
            return f"加载失败: {str(e)}", dash.no_update

    # ---------- 模块2：因子加工 ----------
    @app.callback(
        [Output('ft-factor-preprocess-msg', 'children'),
         Output('ft-preprocess-graph', 'figure'),
         Output('ft-processed-factors-store', 'data')],
        [Input('ft-run-preprocess-btn', 'n_clicks')],
        [State('ft-factor-expr', 'value'),
         State('ft-label-type', 'value'),
         State('ft-data-store', 'data')]
    )
    def run_preprocess(n_clicks, expr, label_horizon, data_store):
        global _data_provider, _factor_cache, _processed_factors
        if not n_clicks or not expr or not _data_provider or not _data_provider._loaded:
            return "请先加载数据并输入因子公式", go.Figure(), dash.no_update

        try:
            factor_name = expr.strip()
            horizon = int(label_horizon or 5)

            raw_factor = _data_provider.eval_expression(factor_name)
            if raw_factor.dropna(how='all').empty:
                return "因子公式求值结果为空，请检查公式", go.Figure(), dash.no_update

            # 去重列名，避免 stack() 报错
            raw_factor = raw_factor.loc[:, ~raw_factor.columns.duplicated()]

            _factor_cache[factor_name] = raw_factor
            processed = preprocess_factor(raw_factor, do_mad=True, do_zscore=True)
            _processed_factors[factor_name] = processed

            before = raw_factor.stack().dropna()
            after = processed.stack().dropna()

            fig = go.Figure()
            if len(before) > 0:
                fig.add_trace(go.Histogram(x=before, name='预处理前', opacity=0.7, nbinsx=50, marker_color='#EF4444'))
            if len(after) > 0:
                fig.add_trace(go.Histogram(x=after, name='预处理后', opacity=0.7, nbinsx=50, marker_color='#10B981'))
            fig.update_layout(title=f'因子 [{factor_name}] 预处理前后分布', barmode='overlay',
                              template='plotly_white', height=320, margin=dict(l=40, r=20, t=40, b=40))

            return f"因子 [{factor_name}] 预处理完成 (样本: {len(before)})", fig, {'factor_name': factor_name, 'processed': True}
        except Exception as e:
            logging.error(f"预处理失败: {e}")
            return f"预处理失败: {str(e)}", go.Figure(), dash.no_update

    # ---------- 模块3：因子校验 ----------
    @app.callback(
        [Output('ft-validate-status', 'children'),
         Output('ft-ic-stats-table', 'children'),
         Output('ft-ic-graph', 'figure'),
         Output('ft-layer-return-graph', 'figure'),
         Output('ft-group-nav-graph', 'figure'),
         Output('ft-validation-store', 'data')],
        [Input('ft-validate-factor-btn', 'n_clicks')],
        [State('ft-processed-factors-store', 'data'),
         State('ft-label-type', 'value')]
    )
    def validate_factor(n_clicks, factor_store, label_horizon):
        global _data_provider, _processed_factors, _ic_results, _layer_results
        if not n_clicks or not factor_store or not _data_provider:
            return "请先执行因子预处理", None, go.Figure(), go.Figure(), go.Figure(), dash.no_update

        try:
            factor_name = factor_store['factor_name']
            processed = _processed_factors.get(factor_name)
            if processed is None:
                return "预处理因子数据丢失", None, go.Figure(), go.Figure(), go.Figure(), dash.no_update

            horizon = int(label_horizon or 5)
            ret = _data_provider.compute_future_returns(horizon)

            ic = calculate_rank_ic(processed, ret)
            if len(ic.dropna()) < 2:
                return "有效数据不足，无法计算IC", None, go.Figure(), go.Figure(), go.Figure(), dash.no_update
            _ic_results[factor_name] = ic.dropna()

            stats = calculate_ic_statistics(ic.dropna())
            ic_pass = abs(stats['ic_mean']) > 0.02

            # 分层回测使用单日收益，避免未来 N 日收益被按日复利造成的前视偏差
            daily_ret = _data_provider.compute_future_returns(1)
            group_nav = layered_backtest(processed, daily_ret, n_groups=10)
            _layer_results[factor_name] = group_nav
            group_stats = compute_group_returns(group_nav, 10)

            stat_rows = [
                html.Tr([html.Td(k, className="border px-3 py-1 text-sm font-medium"), html.Td(str(v), className="border px-3 py-1 text-sm")])
                for k, v in stats.items()
            ]
            stat_rows.append(html.Tr([
                html.Td("判定", className="border px-3 py-1 text-sm font-medium"),
                html.Td("✅ 合格" if ic_pass else "❌ 需剔除",
                        className=f"border px-3 py-1 text-sm {'text-green-600' if ic_pass else 'text-red-600'}")
            ]))
            ic_table = html.Table(stat_rows, className="border-collapse border w-full max-w-md")

            fig_ic = go.Figure()
            fig_ic.add_trace(go.Scatter(x=ic.index, y=ic.values, mode='lines+markers', name='Rank IC',
                                         line=dict(color='#7C3AED')))
            fig_ic.add_hline(y=0, line_dash='dash', line_color='gray')
            fig_ic.add_hline(y=0.02, line_dash='dot', line_color='green')
            fig_ic.add_hline(y=-0.02, line_dash='dot', line_color='red')
            fig_ic.update_layout(title=f'Rank IC 时序 ({factor_name})', template='plotly_white',
                                  height=280, margin=dict(l=40, r=20, t=40, b=40))

            fig_layer = go.Figure()
            colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#ffffbf', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850', '#006837']
            for i in range(10):
                nav = group_nav[i].dropna()
                fig_layer.add_trace(go.Scatter(x=nav.index, y=nav.values, mode='lines',
                    name=f'G{i+1}', line=dict(color=colors[i * len(colors) // 10])))
            fig_layer.update_layout(title='分组净值曲线', template='plotly_white',
                                     height=280, margin=dict(l=40, r=20, t=40, b=40))

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=group_stats['group'], y=group_stats['annual_return'],
                                      marker_color=[colors[i * len(colors) // 10] for i in range(10)],
                                      text=[f"{v*100:.1f}%" for v in group_stats['annual_return']],
                                      textposition='outside'))
            fig_bar.update_layout(title='分组年化收益', template='plotly_white',
                                   height=280, margin=dict(l=40, r=20, t=40, b=40),
                                   xaxis_title='分组', yaxis_title='年化收益')

            return (f"校验完成: IC均值={stats['ic_mean']}, ICIR={stats['icir']}, {'合格' if ic_pass else '不合格'}",
                    ic_table, fig_ic, fig_layer, fig_bar,
                    {'factor_name': factor_name, 'ic_pass': ic_pass})
        except Exception as e:
            logging.error(f"因子校验失败: {e}")
            return f"校验失败: {str(e)}", None, go.Figure(), go.Figure(), go.Figure(), dash.no_update

    # ---------- 因子列表更新 ----------
    @app.callback(
        Output('ft-model-factors-checklist', 'options'),
        [Input('ft-validation-store', 'data')]
    )
    def update_factor_checklist(val_store):
        global _processed_factors
        options = []
        for name in _processed_factors.keys():
            label = f"✅ {name}" if val_store and val_store.get('factor_name') == name and val_store.get('ic_pass') else name
            options.append({'label': label, 'value': name})
        return options

    @app.callback(
        Output('ft-model-factors-checklist', 'value'),
        [Input('ft-model-factors-checklist', 'options')]
    )
    def auto_select_valid_factors(options):
        return [opt['value'] for opt in options] if options else []

    # ---------- 模块4：多因子建模 ----------
    @app.callback(
        [Output('ft-model-status', 'children'),
         Output('ft-feature-importance-graph', 'figure'),
         Output('ft-model-store', 'data')],
        [Input('ft-run-model-btn', 'n_clicks')],
        [State('ft-model-factors-checklist', 'value'),
         State('ft-model-method', 'value'),
         State('ft-lgbm-months', 'value'),
         State('ft-label-type', 'value'),
         State('ft-model-name', 'value')]
    )
    def run_model(n_clicks, selected_factors, method, lgbm_months, label_horizon, model_name):
        global _data_provider, _processed_factors, _model_scores, _ic_results, _model_method
        if not n_clicks or not selected_factors or not _data_provider:
            return "请先加载数据并选择因子", go.Figure(), dash.no_update

        def _save_and_hint(base_msg):
            """训练成功后自动持久化，返回附加提示"""
            try:
                saved_name = save_model_to_disk(model_name)
                return f'{base_msg} ｜ 已保存模型 [{saved_name}]'
            except Exception as e:
                logging.warning(f"模型自动保存失败: {e}")
                return f'{base_msg} ｜ 模型保存失败: {e}'

        try:
            factor_data = {name: _processed_factors[name] for name in selected_factors if name in _processed_factors}
            if not factor_data:
                return "无有效预处理因子", go.Figure(), dash.no_update

            horizon = int(label_horizon or 5)
            ret = _data_provider.compute_future_returns(horizon)

            if method == 'icir':
                ic_stats = {}
                for name in factor_data:
                    ic = calculate_rank_ic(factor_data[name], ret)
                    ic_stats[name] = calculate_ic_statistics(ic.dropna()) if len(ic.dropna()) > 0 else {'icir': 0}
                _model_scores, weights = compute_icir_weighted_score(factor_data, ic_stats)
                _model_method = 'icir'

                # 构建 ICIR 信息
                icir_info = ', '.join([f"{name}: ICIR={ic_stats[name]['icir']:.4f}" for name in weights])

                fig = go.Figure()
                names = list(weights.keys())
                vals = list(weights.values())
                fig.add_trace(go.Bar(x=names, y=vals, text=[f"{v*100:.1f}%" for v in vals],
                                      textposition='outside', marker_color='#F97316'))
                fig.update_layout(title='ICIR 加权权重', template='plotly_white', height=280,
                                  margin=dict(l=40, r=20, t=40, b=60))

                # ICIR 全 ≤ 0 时提示回退到等权
                has_positive = any(ic_stats[name]['icir'] > 0 for name in weights)
                hint = "" if has_positive else "（ICIR 均 ≤ 0，回退到等权）"
                msg = f"ICIR 合成完成 ({len(factor_data)} 个因子) — {icir_info} {hint}"
                return _save_and_hint(msg), fig, {'method': 'icir', 'done': True}

            else:  # lgbm
                pred_scores, feat_imp = train_lgbm_rolling(factor_data, ret,
                                                            train_months=int(lgbm_months or 60))
                if pred_scores is None:
                    return "LGBM 训练失败（可能数据不足或未安装lightgbm）", go.Figure(), dash.no_update
                _model_scores = pred_scores
                _model_method = 'lgbm'

                fig = go.Figure()
                names = list(feat_imp.keys())
                vals = list(feat_imp.values())
                fig.add_trace(go.Bar(x=names, y=vals, text=[f"{v:.4f}" for v in vals],
                                      textposition='outside', marker_color='#8B5CF6'))
                fig.update_layout(title='LGBM 特征重要度', template='plotly_white', height=280,
                                  margin=dict(l=40, r=20, t=40, b=60))
                msg = f"LGBM 训练完成 ({len(factor_data)} 个因子)"
                return _save_and_hint(msg), fig, {'method': 'lgbm', 'done': True}
        except Exception as e:
            logging.error(f"建模失败: {e}")
            return f"建模失败: {str(e)}", go.Figure(), dash.no_update

    # ---------- 已保存模型：列表刷新 ----------
    @app.callback(
        Output('ft-saved-model-dropdown', 'options'),
        [Input('ft-model-store', 'data'),
         Input('ft-model-mgmt-store', 'data')]
    )
    def refresh_saved_model_options(model_store, mgmt_store):
        options = []
        for m in list_saved_models():
            method_label = 'ICIR' if m.get('method') == 'icir' else 'LGBM'
            options.append({
                'label': f"【{method_label}】{m.get('name', '')} · {m.get('n_factors', '?')}因子 · "
                         f"{m.get('n_stocks', '?')}股 · {m.get('latest_date', '')}",
                'value': m['name'],
            })
        return options

    # ---------- 已保存模型：加载 / 删除 ----------
    @app.callback(
        [Output('ft-model-mgmt-status', 'children'),
         Output('ft-model-mgmt-store', 'data'),
         Output('ft-saved-model-dropdown', 'value')],
        [Input('ft-load-model-btn', 'n_clicks'),
         Input('ft-delete-model-btn', 'n_clicks')],
        [State('ft-saved-model-dropdown', 'value')],
        prevent_initial_call=True
    )
    def manage_saved_model(load_clicks, delete_clicks, name):
        ctx = dash.callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
        if triggered not in ('ft-load-model-btn', 'ft-delete-model-btn'):
            return dash.no_update, dash.no_update, dash.no_update
        if not name:
            return "请先在下拉框选择一个模型", dash.no_update, dash.no_update
        try:
            if triggered == 'ft-load-model-btn':
                info = load_saved_model(name)
                method_label = 'ICIR' if info['method'] == 'icir' else 'LGBM'
                return (f"已加载 [{info['name']}]（{method_label}, {info['n_factors']}因子, "
                        f"覆盖 {info['n_stocks']} 只）— 分析页评分面板已切换",
                        {'ts': time.time()}, name)
            if triggered == 'ft-delete-model-btn':
                delete_saved_model(name)
                return f"已删除模型 [{name}]（不影响内存中正在使用的模型）", {'ts': time.time()}, None
        except Exception as e:
            logging.error(f"模型管理操作失败: {e}")
            return f"操作失败: {e}", dash.no_update, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update

    # ---------- 模块5：组合回测 ----------
    @app.callback(
        [Output('ft-backtest-stats', 'children'),
         Output('ft-backtest-nav-graph', 'figure'),
         Output('ft-backtest-store', 'data')],
        [Input('ft-run-backtest-btn', 'n_clicks')],
        [State('ft-top-pct', 'value'),
         State('ft-single-cap', 'value'),
         State('ft-label-type', 'value')],
        prevent_initial_call=True
    )
    def run_backtest(n_clicks, top_pct, single_cap, label_horizon):
        global _model_scores, _data_provider, _backtest_result
        if _model_scores is None or _data_provider is None:
            return "请先运行多因子模型（模块4）", go.Figure(), dash.no_update

        try:
            if isinstance(_model_scores, pd.DataFrame) and _model_scores.empty:
                return "模型得分为空，请重新运行建模", go.Figure(), dash.no_update

            horizon = int(label_horizon or 5)
            logging.info(f"开始回测: 选股比例={top_pct}%, 单票上限={single_cap}%, 预测期={horizon}日")

            # 组合回测使用单日收益，避免未来 N 日收益被按日复利造成的前视偏差
            ret = _data_provider.compute_future_returns(1)
            if ret is None or (isinstance(ret, pd.DataFrame) and ret.empty):
                return "单日收益数据为空", go.Figure(), dash.no_update

            top_fraction = (top_pct or 20) / 100.0
            cap = (single_cap or 10) / 100.0

            nav, stats = portfolio_backtest(_model_scores, ret, top_pct=top_fraction, single_stock_cap=cap)
            _backtest_result = {'nav': nav, 'stats': stats}

            stat_display = html.Div([
                html.Span(f"年化收益: {stats['annual_return']*100:.2f}%", className="mr-4 text-green-600 font-bold"),
                html.Span(f"夏普比率: {stats['sharpe']:.4f}", className="mr-4 text-blue-600"),
                html.Span(f"最大回撤: {stats['max_dd']*100:.2f}%", className="text-red-600"),
            ])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stats['dates'], y=stats['nav'], mode='lines',
                                      name='策略净值', fill='tozeroy', line=dict(color='#EF4444', width=2)))
            fig.add_hline(y=1, line_dash='dash', line_color='gray')
            fig.update_layout(title='策略净值曲线', template='plotly_white', height=280,
                              margin=dict(l=40, r=20, t=40, b=40))

            return stat_display, fig, {'done': True}
        except Exception as e:
            logging.error(f"回测失败: {e}")
            return f"回测失败: {str(e)}", go.Figure(), dash.no_update

    # ---------- Excel 导出 ----------
    @app.callback(
        [Output('ft-download-excel', 'data'),
         Output('ft-export-status', 'children')],
        [Input('ft-export-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def export_excel(n_clicks):
        global _model_scores, _processed_factors, _backtest_result, _ic_results
        if not n_clicks:
            return dash.no_update, dash.no_update

        try:
            used_names = set()
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for name, df in _processed_factors.items():
                    flat = df.stack().reset_index()
                    flat.columns = ['date', 'stock', 'value']
                    flat.to_excel(writer, sheet_name=_safe_sheet_name(f'factor_{name}', used_names), index=False)

                if _model_scores is not None and not _model_scores.empty:
                    flat = _model_scores.stack().reset_index()
                    flat.columns = ['date', 'stock', 'score']
                    flat.to_excel(writer, sheet_name=_safe_sheet_name('pred_scores', used_names), index=False)

                for name, ic in _ic_results.items():
                    pd.DataFrame({'date': ic.index, 'ic': ic.values}).to_excel(
                        writer, sheet_name=_safe_sheet_name(f'IC_{name}', used_names), index=False)

                if _backtest_result and 'stats' in _backtest_result:
                    stats_df = pd.DataFrame([{k: v for k, v in _backtest_result['stats'].items()
                                              if k not in ('nav', 'dates')}])
                    stats_df.to_excel(writer, sheet_name=_safe_sheet_name('backtest_stats', used_names), index=False)

            if not used_names:
                return dash.no_update, "没有可导出的数据，请先执行前面的预处理/建模/回测步骤"

            output.seek(0)
            return (dcc.send_bytes(output.getvalue(), filename='factor_report.xlsx'),
                    f"已导出 factor_report.xlsx（{len(used_names)} 个工作表）")
        except Exception as e:
            logging.error(f"导出失败: {e}")
            return dash.no_update, f"导出失败: {e}"


# ==================== 辅助函数 ====================

def _get_pool_codes(pool):
    """根据股票池类型获取代码列表"""
    if pool == 'custom':
        rows = get_all_stock_codes()
        codes = []
        for row in rows:
            code = row[0].strip()
            if code.endswith('.SH') or code.endswith('.SZ'):
                codes.append(code)
            elif code.isdigit() and len(code) == 6:
                suffix = '.SH' if code.startswith(('5', '6', '9')) else '.SZ'
                codes.append(code + suffix)
            else:
                codes.append(code)
        # 去重：同一代码可能以不同格式存在（如 601669 和 601669.SH）
        seen = set()
        unique = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:100]
    elif pool == 'hs300':
        try:
            from .data_manager import pro
            df = pro.index_weight(index_code='000300.SH', trade_date=pd.Timestamp.now().strftime('%Y%m%d'))
            if df is not None and not df.empty:
                return df['con_code'].tolist()[:100]
        except Exception as e:
            logging.warning(f"获取沪深300成分股失败: {e}")
        return []
    elif pool == 'zz500':
        try:
            from .data_manager import pro
            df = pro.index_weight(index_code='000905.SH', trade_date=pd.Timestamp.now().strftime('%Y%m%d'))
            if df is not None and not df.empty:
                return df['con_code'].tolist()[:100]
        except Exception as e:
            logging.warning(f"获取中证500成分股失败: {e}")
        return []
    elif pool == 'all_a':
        try:
            from .data_manager import pro
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
            if df is not None and not df.empty:
                return df['ts_code'].tolist()[:200]
        except Exception as e:
            logging.warning(f"获取全A股列表失败: {e}")
        return []
    return []
