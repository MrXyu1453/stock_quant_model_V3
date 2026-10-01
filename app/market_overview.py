"""
市场五面分析页面
- 政策面 / 消息面 / 情绪面 / 资金面 / 走势技术
- AI 驱动分析 + 图表可视化
- AI 设置 (provider / api_key / model) 持久化缓存
"""
import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from collections import defaultdict

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

from .data_manager import get_and_process_data, set_data_source, get_available_sources
from .trend_analysis import rsi_calculation, bollinger_bands, macd_divergence, support_resistance_analysis
from .paths import cache_path

# ===================== 常量 =====================
AI_SETTINGS_FILE = cache_path('ai_settings.json')
_ai_settings_cache = None

# 主要指数代码
INDEX_CODES = {
    'shanghai': {'code': '000001.SH', 'name': '上证指数'},
    'shenzhen': {'code': '399001.SZ', 'name': '深证成指'},
    'chinext': {'code': '399006.SZ', 'name': '创业板指'},
    'hs300': {'code': '000300.SH', 'name': '沪深300'},
    'zz500': {'code': '000905.SH', 'name': '中证500'},
    'sz50': {'code': '000016.SH', 'name': '上证50'},
}

# 板块/行业 ETF
SECTOR_ETFS = {
    '证券': '512880.SH', '银行': '512800.SH', '白酒': '512690.SH',
    '医药': '512010.SH', '半导体': '512480.SH', '军工': '512660.SH',
    '新能源': '516160.SH', '光伏': '515790.SH', 'AI': '159819.SZ',
    '有色金属': '512400.SH',
}

# ─── AI 模型选项（按 provider）───
MODEL_OPTIONS = {
    'deepseek': [
        {'label': 'DeepSeek-V3', 'value': 'deepseek-chat'},
        {'label': 'DeepSeek-R1', 'value': 'deepseek-reasoner'},
    ],
    'qwen': [
        {'label': 'Qwen-Plus', 'value': 'qwen-plus'},
        {'label': 'Qwen-Max', 'value': 'qwen-max'},
        {'label': 'Qwen-Turbo', 'value': 'qwen-turbo'},
        {'label': 'Qwen3-235B', 'value': 'qwen3-235b-a22b'},
        {'label': 'Qwen-Coder-Plus', 'value': 'qwen-coder-plus'},
        {'label': 'Qwen2.5-72B', 'value': 'qwen2.5-72b-instruct'},
        {'label': 'Qwen2.5-32B', 'value': 'qwen2.5-32b-instruct'},
        {'label': 'Qwen2.5-14B', 'value': 'qwen2.5-14b-instruct'},
        {'label': 'Qwen2.5-7B', 'value': 'qwen2.5-7b-instruct'},
    ],
    'openai': [
        {'label': 'GPT-4o', 'value': 'gpt-4o'},
        {'label': 'GPT-4o-mini', 'value': 'gpt-4o-mini'},
        {'label': 'GPT-4-Turbo', 'value': 'gpt-4-turbo'},
        {'label': 'GPT-4', 'value': 'gpt-4'},
        {'label': 'GPT-3.5-Turbo', 'value': 'gpt-3.5-turbo'},
        {'label': 'o1', 'value': 'o1'},
        {'label': 'o1-mini', 'value': 'o1-mini'},
        {'label': 'o3-mini', 'value': 'o3-mini'},
    ],
    'azure': [
        {'label': 'GPT-4o', 'value': 'gpt-4o'},
        {'label': 'GPT-4', 'value': 'gpt-4'},
        {'label': 'GPT-4-Turbo', 'value': 'gpt-4-turbo'},
        {'label': 'GPT-3.5-Turbo', 'value': 'gpt-3.5-turbo'},
    ],
    'custom': [
        {'label': 'Qwen2.5: 7B (Ollama)', 'value': 'qwen2.5:7b'},
        {'label': 'Qwen2.5: 14B (Ollama)', 'value': 'qwen2.5:14b'},
        {'label': 'Qwen2.5: 32B (Ollama)', 'value': 'qwen2.5:32b'},
        {'label': 'DeepSeek-R1: 8B (Ollama)', 'value': 'deepseek-r1:8b'},
        {'label': 'DeepSeek-R1: 14B (Ollama)', 'value': 'deepseek-r1:14b'},
        {'label': 'Llama3.1: 8B (Ollama)', 'value': 'llama3.1:8b'},
        {'label': 'Mistral: 7B (Ollama)', 'value': 'mistral:7b'},
    ],
}

# ===================== AI 设置管理 =====================

def _load_ai_settings():
    global _ai_settings_cache
    if _ai_settings_cache is not None:
        return _ai_settings_cache
    os.makedirs(os.path.dirname(AI_SETTINGS_FILE), exist_ok=True)
    if os.path.exists(AI_SETTINGS_FILE):
        try:
            with open(AI_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                _ai_settings_cache = json.load(f)
        except Exception:
            _ai_settings_cache = {}
    else:
        _ai_settings_cache = {}
    return _ai_settings_cache


def _save_ai_settings(settings):
    global _ai_settings_cache
    os.makedirs(os.path.dirname(AI_SETTINGS_FILE), exist_ok=True)
    with open(AI_SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    _ai_settings_cache = settings


def get_ai_settings():
    s = _load_ai_settings()
    return {
        'provider': s.get('provider', 'deepseek'),
        'api_key': s.get('api_key', ''),
        'base_url': s.get('base_url', 'https://api.deepseek.com/v1'),
        'model': s.get('model', 'deepseek-chat'),
    }


def save_ai_settings(provider, api_key, base_url, model):
    settings = {
        'provider': provider,
        'api_key': api_key,
        'base_url': base_url,
        'model': model,
        'updated_at': datetime.now().isoformat()
    }
    _save_ai_settings(settings)
    return settings


# ===================== AI 调用 =====================

def call_ai(prompt, system_prompt="你是专业的A股市场分析师。"):
    """调用 AI 分析"""
    settings = get_ai_settings()
    if not settings['api_key']:
        return "⚠️ 请先在左侧配置 AI API Key"
    try:
        import requests
        resp = requests.post(
            f"{settings['base_url'].rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings['api_key']}",
                "Content-Type": "application/json"
            },
            json={
                "model": settings['model'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.6,
                "max_tokens": 600
            },
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            return data['choices'][0]['message']['content']
        else:
            logging.warning(f"AI 返回非200: {resp.status_code}")
            return f"AI 请求失败: HTTP {resp.status_code}"
    except Exception as e:
        logging.error(f"AI 调用异常: {e}")
        return f"AI 调用出错: {str(e)}"


# ===================== 数据采集 =====================

def fetch_market_data(start_date='20250101', end_date=None):
    """拉取主要指数 + 行业ETF数据（固定日线：日/周/月涨跌幅均按日收盘计算）"""
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    # 指数数据
    index_data = {}
    for key, info in INDEX_CODES.items():
        try:
            df = get_and_process_data(info['code'], start_date, end_date, freq='daily')
            if df is not None and len(df) >= 5:
                index_data[key] = df
            else:
                logging.warning(f"{info['name']} ({info['code']}) 数据为空")
        except Exception as e:
            logging.warning(f"获取 {info['name']} 失败: {e}")

    # 行业 ETF
    sector_data = {}
    for name, code in SECTOR_ETFS.items():
        try:
            df = get_and_process_data(code, start_date, end_date, freq='daily')
            if df is not None and len(df) >= 5:
                sector_data[name] = df
        except Exception:
            pass

    return index_data, sector_data


def compute_market_metrics(index_data, sector_data):
    """计算各类市场指标"""
    metrics = {}

    # 1. 指数涨跌幅
    index_returns = {}
    for key, df in index_data.items():
        if len(df) < 5:
            continue
        closes = df['close'].values
        # 日涨跌
        day_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
        week_chg = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
        month_chg = (closes[-1] / closes[-20] - 1) * 100 if len(closes) >= 20 else 0
        index_returns[key] = {
            'name': INDEX_CODES[key]['name'],
            'close': closes[-1],
            'day': round(day_chg, 2),
            'week': round(week_chg, 2),
            'month': round(month_chg, 2),
        }
    metrics['index_returns'] = index_returns

    # 2. 行业板块强弱
    sector_perf = {}
    for name, df in sector_data.items():
        if len(df) < 5:
            continue
        closes = df['close'].values
        day_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
        sector_perf[name] = round(day_chg, 2)
    metrics['sector_perf'] = sector_perf

    # 3. 成交量变化
    if 'shanghai' in index_data and len(index_data['shanghai']) >= 5:
        vols = index_data['shanghai']['vol'].values
        vol_ma5 = np.mean(vols[-5:])
        vol_ma20 = np.mean(vols[-20:]) if len(vols) >= 20 else vol_ma5
        vol_ratio = vol_ma5 / vol_ma20 if vol_ma20 > 0 else 1.0
        metrics['volume'] = {
            'latest': vols[-1],
            'ma5': int(vol_ma5),
            'ratio': round(vol_ratio, 2),
            'status': '放量' if vol_ratio > 1.2 else ('缩量' if vol_ratio < 0.8 else '平量')
        }

    # 4. 涨跌家数估算 (基于指数价格变动判断)
    up_count = sum(1 for v in index_returns.values() if v['day'] > 0)
    down_count = sum(1 for v in index_returns.values() if v['day'] < 0)
    metrics['breadth'] = {'up': up_count, 'down': down_count}

    # 5. 波动率 (基于沪深300 20日)
    if 'hs300' in index_data and len(index_data['hs300']) >= 20:
        closes_hs = index_data['hs300']['close'].values
        returns = np.diff(closes_hs[-21:]) / closes_hs[-21:-1]
        vol_20d = np.std(returns) * 100
        metrics['volatility'] = round(vol_20d, 2)

    # 6. 技术指标 (上证指数)
    if 'shanghai' in index_data and len(index_data['shanghai']) >= 30:
        df_sh = index_data['shanghai']
        rsi = rsi_calculation(df_sh)
        bb = bollinger_bands(df_sh)
        close_last = df_sh['close'].values[-1]
        metrics['technical'] = {
            'rsi': round(rsi[-1], 1) if len(rsi) > 0 else None,
            'bb_upper': round(bb['upper'][-1], 2) if len(bb['upper']) > 0 else None,
            'bb_lower': round(bb['lower'][-1], 2) if len(bb['lower']) > 0 else None,
            'close': close_last,
        }

    return metrics


# ===================== 财经新闻爬取 =====================

def fetch_financial_news():
    """从东方财富网爬取最新财经要闻，返回带来源的新闻列表"""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get('https://finance.eastmoney.com/a/czqyw.html',
                            headers=headers, timeout=10)
        resp.encoding = 'gbk'
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            titles = []
            for item in soup.select('.title a')[:12]:
                text = item.get_text(strip=True)
                if text and len(text) > 6:
                    titles.append({'title': text, 'source': '东方财富'})
            if titles:
                return titles
    except Exception as e:
        logging.warning(f"新闻爬取失败: {e}")
    # fallback: 尝试新浪财经
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get('https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2512&k=&num=10&r=0.5',
                            headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('result', {}).get('data'):
                return [{'title': item['title'], 'source': '新浪财经'} for item in data['result']['data'][:10] if item.get('title')]
    except Exception:
        pass
    return []


def fetch_international_news():
    """爬取国际市场相关新闻，返回带来源的新闻列表"""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        # 新浪财经国际频道
        resp = requests.get('https://finance.sina.com.cn/worldmac/', headers=headers, timeout=10)
        resp.encoding = 'utf-8'
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            titles = []
            for item in soup.select('.blk_01 a, .blk_02 a, .blk_03 a')[:8]:
                text = item.get_text(strip=True)
                if text and len(text) > 5:
                    titles.append({'title': text, 'source': '新浪国际'})
            if titles:
                return titles
    except Exception:
        pass
    # fallback: 用国内要闻做补充
    return []


# ===================== AI 调用封装 =====================

def _safe_ai_call(prompt, fallback_text=""):
    """AI 调用 + 失败时的本地兜底"""
    try:
        result = call_ai(prompt)
        if result and 'AI' not in result[:10] and '⚠️' not in result[:5]:
            return result
    except Exception as e:
        logging.warning(f"AI 调用失败: {e}")
    return fallback_text


# ===================== AI 五面分析 =====================

def analyze_policy(metrics):
    """政策面分析 — 结合真实财经新闻"""
    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    vol = metrics.get('volume', {})

    # 爬取新闻
    news = fetch_financial_news()
    news_text = "\n".join(f"· {t['title']}【来源：{t['source']}】" for t in news[:8]) if news else "(暂无新闻数据)"

    lines = ["=== 今日市场数据 ==="]
    for k, v in idx.items():
        lines.append(f"{v['name']}: {v['close']:.0f} 日{v['day']:+.2f}% 周{v['week']:+.2f}% 月{v['month']:+.2f}%")
    if sec:
        top5 = sorted(sec.items(), key=lambda x: -x[1])[:5]
        bot5 = sorted(sec.items(), key=lambda x: x[1])[:5]
        lines.append(f"\n强势行业: " + ", ".join(f"{k}{v:+.1f}%" for k, v in top5))
        lines.append(f"弱势行业: " + ", ".join(f"{k}{v:+.1f}%" for k, v in bot5))
    if vol:
        lines.append(f"\n成交量: {vol['status']} (量比{vol['ratio']})")

    lines.append(f"\n=== 最新财经要闻 ===")
    lines.append(news_text)

    prompt = "\n".join(lines) + "\n\n请结合上述市场数据与财经要闻，从政策面角度分析(200字内)：政策风向、监管动态、行业利好/利空、政策面对市场的影响预判。"

    fallback = _build_policy_fallback(metrics, news)
    return _safe_ai_call(prompt, fallback)


def _build_policy_fallback(metrics, news):
    """政策面本地兜底分析"""
    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    up_count = sum(1 for v in idx.values() if v['day'] > 0)
    lines = []
    lines.append(f"【政策面分析 (本地)】")
    if news:
        lines.append(f"\n今日要闻 ({len(news)}条):")
        for t in news[:5]:
            lines.append(f"  · {t['title']}【来源：{t['source']}】")
    if sec:
        top = sorted(sec.items(), key=lambda x: -x[1])[:3]
        bot = sorted(sec.items(), key=lambda x: x[1])[:3]
        lines.append(f"\n行业分化明显: {', '.join(f'{k}领涨' for k, v in top)} | {', '.join(f'{k}领跌' for k, v in bot)}")

    if up_count >= 4:
        lines.append(f"\n{up_count}个指数上涨 → 市场整体偏暖，政策面或有利好预期")
    elif up_count <= 2:
        lines.append(f"\n{up_count}个指数上涨 → 市场偏弱，关注政策托底信号")
    else:
        lines.append(f"\n指数涨跌分化 → 结构性行情，关注行业政策")
    return "\n".join(lines)


def analyze_news_sentiment(metrics):
    """消息面分析 — 国际市场行情 + 财经新闻"""
    idx = metrics.get('index_returns', {})
    vol = metrics.get('volatility', None)
    sec = metrics.get('sector_perf', {})
    vol_info = metrics.get('volume', {})

    # 爬取国内+国际新闻
    domestic_news = fetch_financial_news()
    intl_news = fetch_international_news()
    all_news = domestic_news[:5] + intl_news[:5]

    lines = ["=== 今日市场行情 ==="]
    for k, v in idx.items():
        lines.append(f"{v['name']}: {v['close']:.0f} 日{v['day']:+.2f}% 周{v['week']:+.2f}%")
    if vol is not None:
        lines.append(f"\n沪深300 20日波动率: {vol}%")
    if vol_info:
        lines.append(f"成交量: {vol_info.get('status', 'N/A')} (量比{vol_info.get('ratio', 'N/A')})")

    export_sectors = ['半导体', '光伏', '新能源']
    export_data = {k: v for k, v in sec.items() if k in export_sectors}
    if export_data:
        lines.append(f"\n出口链行业: " + ", ".join(f"{k}{v:+.1f}%" for k, v in export_data.items()))

    resource_sectors = ['有色金属']
    resource_data = {k: v for k, v in sec.items() if k in resource_sectors}
    if resource_data:
        lines.append(f"资源类行业: " + ", ".join(f"{k}{v:+.1f}%" for k, v in resource_data.items()))
        lines.append("(有色金属反映工业需求与通胀预期，为重要先行指标)")

    news_text = "\n".join(f"· {t['title']}【来源：{t['source']}】" for t in all_news) if all_news else "(暂无新闻)"
    lines.append(f"\n=== 最新财经消息 ===")
    lines.append(news_text)

    prompt = "\n".join(lines) + "\n\n请结合上述市场行情和财经消息，从消息面角度综合分析(200字内)：国内外重大财经事件、市场消息面情绪、外围市场对A股的影响、近期消息对市场走势的预判。"

    fallback = _build_news_fallback(metrics, all_news)
    return _safe_ai_call(prompt, fallback)


def _build_news_fallback(metrics, news):
    """消息面本地兜底"""
    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    lines = ["【消息面分析 (本地)】"]

    if news:
        lines.append(f"\n今日消息 ({len(news)}条):")
        for t in news[:6]:
            lines.append(f"  · {t['title']}【来源：{t['source']}】")

    export_sectors = {'半导体': '科技出口', '光伏': '新能源出口', '新能源': '新能源出口'}
    for name, label in export_sectors.items():
        if name in sec:
            direction = "走强" if sec[name] > 0 else "走弱"
            lines.append(f"· {name}({label}) {sec[name]:+.1f}% → {direction}，消息面{'利好' if sec[name] > 0 else '利空'}")

    resource_sectors = {'有色金属': '资源/商品'}
    for name, label in resource_sectors.items():
        if name in sec:
            direction = "走强" if sec[name] > 0 else "走弱"
            signal = "大宗需求旺盛，通胀预期升温" if sec[name] > 0 else "工业需求偏弱，避险情绪上升"
            lines.append(f"· {name}({label}) {sec[name]:+.1f}% → {direction}，{signal}")

    hs300 = idx.get('hs300', {})
    up_count = sum(1 for v in idx.values() if v['day'] > 0)
    if hs300:
        if hs300['month'] < -5:
            lines.append(f"· 沪深300月跌{hs300['month']}% → 消息面偏空，关注政策应对")
        elif hs300['month'] > 3:
            lines.append(f"· 沪深300月涨{hs300['month']}% → 市场情绪偏乐观")
    if up_count >= 4:
        lines.append(f"· {up_count}个指数上涨 → 消息面整体偏多")
    elif up_count <= 2:
        lines.append(f"· {up_count}个指数上涨 → 消息面整体偏空")

    lines.append("\n注意: 完整消息面需结合国际市场实时行情")
    return "\n".join(lines)


def analyze_sentiment(metrics):
    """情绪面分析"""
    vol = metrics.get('volume', {})
    bread = metrics.get('breadth', {})
    tech = metrics.get('technical', {})
    idx = metrics.get('index_returns', {})

    lines = [
        "=== 市场情绪指标 ===",
        f"成交量: {vol.get('status', 'N/A')} (量比 {vol.get('ratio', 'N/A')})",
        f"主要指数涨跌: ↑{bread.get('up', 0)} ↓{bread.get('down', 0)}",
        f"上证 RSI: {tech.get('rsi', 'N/A')}",
    ]
    if tech.get('rsi') is not None:
        if tech['rsi'] > 70:
            lines.append("⚠️ RSI 超买 (>70)")
        elif tech['rsi'] < 30:
            lines.append("💡 RSI 超卖 (<30)")
    # 加普涨/普跌判断
    up_count = bread.get('up', 0)
    total = up_count + bread.get('down', 1)
    if up_count / max(total, 1) >= 0.8:
        lines.append("普涨格局，市场情绪偏乐观")
    elif up_count / max(total, 1) <= 0.3:
        lines.append("普跌格局，市场情绪偏悲观")

    lines.append("\n请从市场情绪面分析(200字内)：当前市场情绪评估、多空力量对比、恐慌/贪婪倾向、短期情绪对走势的影响。")
    return call_ai("\n".join(lines))


def analyze_capital_flow(metrics):
    """资金面分析 — 结合行业资金流向"""
    vol = metrics.get('volume', {})
    sec = metrics.get('sector_perf', {})
    idx = metrics.get('index_returns', {})

    lines = [
        "=== 资金面数据 ===",
        f"成交量状态: {vol.get('status', 'N/A')} (MA5:{vol.get('ma5', 'N/A')}万手, 量比{vol.get('ratio', 'N/A')})",
    ]
    if sec:
        top3 = sorted(sec.items(), key=lambda x: -x[1])[:3]
        bot3 = sorted(sec.items(), key=lambda x: x[1])[:3]
        mid3 = sorted(sec.items(), key=lambda x: abs(x[1]))[:3]
        lines.append(f"\n资金流入: {', '.join(f'{k}{v:+.1f}%' for k, v in top3)}")
        lines.append(f"资金流出: {', '.join(f'{k}{v:+.1f}%' for k, v in bot3)}")
        # 资金净流入行业数
        inflow_count = sum(1 for v in sec.values() if v > 0)
        lines.append(f"净流入行业: {inflow_count}/{len(sec)}")

    # 量价配合
    if 'shanghai' in idx:
        sh = idx['shanghai']
        lines.append(f"\n上证: {sh['close']:.0f} (日{sh['day']:+.2f}%)")
        if sh['day'] > 0 and vol.get('ratio', 1) > 1.1:
            lines.append("量价配合: 放量上涨，资金主动流入")
        elif sh['day'] < 0 and vol.get('ratio', 1) > 1.1:
            lines.append("量价背离: 放量下跌，资金出逃明显")
        elif sh['day'] > 0 and vol.get('ratio', 1) < 0.9:
            lines.append("缩量反弹，追涨意愿不足")
        elif sh['day'] < 0 and vol.get('ratio', 1) < 0.9:
            lines.append("缩量回调，抛压有限")

    prompt = "\n".join(lines) + "\n\n请从资金面角度分析(200字内)：资金流向特征、主力/散户动向、融资融券趋势、增量资金判断、北向资金可能动向。如有色金属等资源板块异动请单独说明。"
    fallback = _build_capital_fallback(metrics)
    return _safe_ai_call(prompt, fallback)


def _build_capital_fallback(metrics):
    """资金面本地兜底"""
    vol = metrics.get('volume', {})
    sec = metrics.get('sector_perf', {})
    idx = metrics.get('index_returns', {})
    lines = ["【资金面分析 (本地)】"]

    if vol:
        lines.append(f"· 量能: {vol['status']} (量比{vol.get('ratio', 'N/A')})")
    if 'shanghai' in idx:
        sh = idx['shanghai']
        if sh['day'] > 0.5 and vol.get('ratio', 1) > 1.1:
            lines.append(f"· 上证+{sh['day']}% 放量 → 增量资金进场迹象")
        elif sh['day'] < -1:
            lines.append(f"· 上证{sh['day']}% → 主力或减仓，注意避险")

    if sec:
        inflow_count = sum(1 for v in sec.values() if v > 0)
        if inflow_count >= 6:
            lines.append(f"· {inflow_count}/{len(sec)}行业净流入 → 资金面偏宽松")
        elif inflow_count <= 3:
            lines.append(f"· 仅{inflow_count}/{len(sec)}行业净流入 → 资金面偏紧，存量博弈")

        # 有色金属: 商品/资源指标
        if '有色金属' in sec:
            ys_val = sec['有色金属']
            if ys_val > 1:
                lines.append(f"· 有色金属 {ys_val:+.1f}% → 资金涌入资源板块，通胀/需求预期偏强")
            elif ys_val < -1:
                lines.append(f"· 有色金属 {ys_val:+.1f}% → 资金撤出资源板块，避险情绪升温")

    lines.append("\n注意: 完整资金面需结合北向资金/融资融券实时数据")
    return "\n".join(lines)


def analyze_technical(metrics):
    """走势技术分析"""
    tech = metrics.get('technical', {})
    idx = metrics.get('index_returns', {})
    vol = metrics.get('volatility', None)

    lines = [
        "=== 技术面数据 ===",
        f"上证 RSI: {tech.get('rsi', 'N/A')}",
        f"布林上轨: {tech.get('bb_upper', 'N/A')} 下轨: {tech.get('bb_lower', 'N/A')} 现价: {tech.get('close', 'N/A')}",
    ]
    if vol is not None:
        lines.append(f"20日波动率: {vol}%")
    lines.append("\n请从技术走势面分析(150字内)：趋势判断、支撑阻力位、量价配合、短期走势预测。")
    return call_ai("\n".join(lines))


# ===================== 图表构建 =====================

def build_index_trend_chart(index_data):
    """主板走势图 - 多指数归一化对比"""
    fig = go.Figure()
    colors = {'shanghai': '#ef4444', 'shenzhen': '#3b82f6', 'chinext': '#f97316',
              'hs300': '#22c55e', 'zz500': '#8b5cf6', 'sz50': '#ec4899'}

    for key, df in index_data.items():
        if len(df) < 5:
            continue
        df = df.sort_values('trade_date').tail(60)
        closes = df['close'].values
        closes_norm = closes / closes[0] * 100
        fig.add_trace(go.Scatter(
            x=df['trade_date'], y=closes_norm,
            mode='lines',
            name=INDEX_CODES[key]['name'],
            line=dict(color=colors.get(key, '#94a3b8'), width=1.5)
        ))

    fig.update_layout(
        title=dict(text='主板指数走势 (60日归一化)', font=dict(size=14, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=380,
        xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='归一化 %', showgrid=True, gridcolor='#f1f5f9'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode='x unified'
    )
    return fig


def build_sector_heatmap(sector_perf):
    """行业板块热力图"""
    sorted_sectors = sorted(sector_perf.items(), key=lambda x: -x[1])
    names = [x[0] for x in sorted_sectors]
    values = [x[1] for x in sorted_sectors]
    colors_list = ['#22c55e' if v > 0 else '#ef4444' for v in values]
    colors_alpha = [min(abs(v) / 3 * 0.8 + 0.2, 1.0) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=values,
        marker_color=colors_list,
        marker_opacity=colors_alpha,
        text=[f'{v:+.1f}%' for v in values],
        textposition='outside',
        textfont=dict(size=11, color=['#166534' if v > 0 else '#991b1b' for v in values]),
        name='日涨跌'
    ))
    fig.add_hline(y=0, line_color='#94a3b8', line_width=1)
    fig.update_layout(
        title=dict(text='行业板块涨跌', font=dict(size=14, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=300,
        xaxis=dict(title=''),
        yaxis=dict(title='日涨跌 %', showgrid=True, gridcolor='#f1f5f9'),
        margin=dict(l=40, r=20, t=40, b=50),
        showlegend=False
    )
    return fig


def build_sentiment_gauge(rsi_val, vol_status):
    """情绪面仪表盘"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=('RSI 强弱', '成交量状态'),
                        specs=[[{'type': 'indicator'}, {'type': 'xy'}]])
    # RSI 仪表
    rsi = rsi_val if rsi_val is not None else 50
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=rsi,
        title={'text': 'RSI(14)', 'font': {'size': 12}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#ef4444' if rsi > 70 else '#22c55e' if rsi < 30 else '#8b5cf6'},
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34,197,94,0.15)'},
                {'range': [30, 70], 'color': 'rgba(139,92,246,0.10)'},
                {'range': [70, 100], 'color': 'rgba(239,68,68,0.15)'},
            ],
            'threshold': {'line': {'color': '#475569', 'width': 1}, 'value': rsi}
        },
        number={'suffix': ' ', 'font': {'size': 22}}
    ), row=1, col=1)

    # 成交量柱状
    fig.add_trace(go.Bar(x=['成交量'], y=[1], marker_color='#3b82f6', width=0.3), row=1, col=2)
    fig.add_annotation(x='成交量', y=1.05, text=vol_status, showarrow=False,
                        font=dict(size=14, color='#1e293b'), row=1, col=2)
    fig.update_layout(
        template='plotly_white', height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig


def build_volatility_chart(index_data):
    """波动率走势"""
    if 'hs300' not in index_data or len(index_data['hs300']) < 25:
        return go.Figure()

    df = index_data['hs300'].sort_values('trade_date')
    closes = df['close'].values[-90:]
    dates = df['trade_date'].values[-90:]
    returns = np.diff(np.log(closes))
    windows = [5, 20]
    fig = go.Figure()
    for w in windows:
        hist_vol = pd.Series(returns).rolling(w).std().values * np.sqrt(252) * 100
        fig.add_trace(go.Scatter(
            x=dates[-len(hist_vol):], y=hist_vol,
            mode='lines',
            name=f'{w}日波动率',
            line=dict(width=1.5)
        ))
    fig.update_layout(
        title=dict(text='沪深300 波动率', font=dict(size=14, color='#1e3a5f'), x=0.02),
        template='plotly_white', height=280,
        xaxis=dict(title='', showgrid=False),
        yaxis=dict(title='年化 %', showgrid=True, gridcolor='#f1f5f9'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode='x unified'
    )
    return fig


# ===================== 五面评分引擎 =====================

def compute_five_factor_scores(metrics):
    """计算五面评分 (0-100) 和综合持仓建议"""
    scores = {}

    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    vol = metrics.get('volume', {})
    tech = metrics.get('technical', {})
    bread = metrics.get('breadth', {})
    vola = metrics.get('volatility', None)

    # ── 1. 政策面 (基于新闻+强弱势行业) ──
    policy_score = 50
    up_count = sum(1 for v in idx.values() if v['day'] > 0)
    total = len(idx)
    if total >= 3:
        if up_count >= 4:
            policy_score += 20  # 普涨，政策面偏暖
        elif up_count <= 2:
            policy_score -= 15  # 普跌，政策面偏冷
    # 强势/弱势行业差
    if sec:
        top_avg = sum(sorted(sec.values(), reverse=True)[:3]) / 3 if len(sec) >= 3 else sum(sec.values()) / len(sec)
        bot_avg = sum(sorted(sec.values())[:3]) / 3 if len(sec) >= 3 else 0
        spread = top_avg - bot_avg
        policy_score += min(spread * 4, 20)  # 行业分化大 → 结构性强
    scores['政策面'] = max(5, min(95, policy_score))

    # ── 2. 消息面 (基于新闻+出口链行业+资源类+波动率) ──
    news_score = 50
    export_names = ['半导体', '光伏', '新能源']
    export_vals = [sec[n] for n in export_names if n in sec]
    if export_vals:
        export_avg = sum(export_vals) / len(export_vals)
        news_score += export_avg * 6  # 出口链强势 → 消息面友好
    # 有色金属: 商品/资源信号，反映通胀预期与工业需求
    resource_names = ['有色金属']
    resource_vals = [sec[n] for n in resource_names if n in sec]
    if resource_vals:
        resource_avg = sum(resource_vals) / len(resource_vals)
        news_score += resource_avg * 5  # 有色走强 → 经济预期向好，通胀温和利好资源
    if vola is not None:
        if vola > 0.4:
            news_score -= 15  # 高波动 → 消息面不确定
        elif vola < 0.15:
            news_score += 10
    scores['消息面'] = max(5, min(95, news_score))

    # ── 3. 情绪面 (RSI + 量比 + 涨跌比) ──
    sentiment_score = 50
    rsi_val = tech.get('rsi', 50) or 50
    if rsi_val < 30:
        sentiment_score += 15  # 超卖 → 情绪冰点 → 反弹预期
    elif rsi_val > 70:
        sentiment_score -= 10  # 超买 → 情绪过热
    sentiment_score += (rsi_val - 50) * 0.3  # RSI 微调
    # 涨跌比
    up = bread.get('up', 0)
    dn = bread.get('down', 1)
    if up + dn > 0:
        ratio = up / (up + dn)
        sentiment_score += (ratio - 0.5) * 30
    # 量能
    vol_ratio = vol.get('ratio', 1)
    if vol_ratio > 1.3:
        sentiment_score += 10  # 放量 → 情绪活跃
    elif vol_ratio < 0.7:
        sentiment_score -= 10
    scores['情绪面'] = max(5, min(95, sentiment_score))

    # ── 4. 资金面 (量价配合 + 行业流入) ──
    capital_score = 50
    if 'shanghai' in idx:
        sh_day = idx['shanghai']['day']
        vr = vol.get('ratio', 1)
        if sh_day > 0.3 and vr > 1.1:
            capital_score += 20  # 放量上涨
        elif sh_day > 0.3 and vr < 0.9:
            capital_score += 5   # 缩量反弹
        elif sh_day < -0.3 and vr > 1.1:
            capital_score -= 20  # 放量下跌
        elif sh_day < -0.3 and vr < 0.9:
            capital_score -= 10
    if sec:
        inflow = sum(1 for v in sec.values() if v > 0)
        inflow_pct = inflow / len(sec) * 100
        capital_score += (inflow_pct - 40) * 0.5
    scores['资金面'] = max(5, min(95, capital_score))

    # ── 5. 走势技术 (RSI + 波动率 + 月趋势) ──
    tech_score = 50
    if rsi_val is not None:
        if 40 <= rsi_val <= 60:
            tech_score += 10  # 中性区间
        tech_score += (rsi_val - 50) * 0.2
    if vola is not None:
        if vola > 0.5:
            tech_score -= 10
        elif vola < 0.15:
            tech_score += 5
    # 月趋势
    if 'hs300' in idx:
        mon = idx['hs300']['month']
        tech_score += mon * 1.5  # 月涨/跌影响
    scores['走势技术'] = max(5, min(95, tech_score))

    # ── 综合评分 & 持仓建议 ──
    composite = round(sum(scores.values()) / 5, 1)
    if composite >= 80:
        position = '🔴 重仓 (80%+)'
        advice = '五面共振偏多，可积极持仓'
    elif composite >= 65:
        position = '🟠 偏多 (60-80%)'
        advice = '多数维度偏多，适度加仓'
    elif composite >= 50:
        position = '🟡 中性 (40-60%)'
        advice = '多空交织，控制仓位观望'
    elif composite >= 35:
        position = '🟢 偏空 (20-40%)'
        advice = '多数维度偏空，降低仓位'
    else:
        position = '🔵 防御 (0-20%)'
        advice = '五面共振偏空，建议轻仓或空仓'

    return {
        'scores': scores,
        'composite': composite,
        'position': position,
        'advice': advice
    }


# ===================== 市场过冷过热提醒 =====================

def compute_market_temperature(metrics):
    """综合技术/资金/量能分析市场过冷过热程度"""
    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    vol = metrics.get('volume', {})
    tech = metrics.get('technical', {})
    bread = metrics.get('breadth', {})
    vola = metrics.get('volatility', None)

    temp_score = 50  # 0=极冷, 100=极热

    # 1. RSI 维度 (权重高)
    rsi_val = tech.get('rsi', 50) or 50
    if rsi_val >= 80:
        temp_score += 25
    elif rsi_val >= 70:
        temp_score += 15
    elif rsi_val >= 60:
        temp_score += 5
    elif rsi_val <= 20:
        temp_score -= 25
    elif rsi_val <= 30:
        temp_score -= 15
    elif rsi_val <= 40:
        temp_score -= 5

    # 2. 量比维度
    vol_ratio = vol.get('ratio', 1)
    if vol_ratio >= 2.0:
        temp_score += 15  # 天量 → 过热
    elif vol_ratio >= 1.5:
        temp_score += 8
    elif vol_ratio <= 0.4:
        temp_score -= 15  # 地量 → 过冷
    elif vol_ratio <= 0.6:
        temp_score -= 8

    # 3. 涨跌比维度
    up = bread.get('up', 0)
    dn = bread.get('down', 1)
    if up + dn > 0:
        breadth_ratio = up / (up + dn)
        if breadth_ratio >= 0.85:
            temp_score += 15  # 普涨 → 过热
        elif breadth_ratio >= 0.7:
            temp_score += 7
        elif breadth_ratio <= 0.15:
            temp_score -= 15  # 普跌 → 过冷
        elif breadth_ratio <= 0.3:
            temp_score -= 7

    # 4. 波动率维度
    if vola is not None:
        if vola >= 0.6:
            temp_score += 12  # 高波动(恐慌/亢奋) → 偏热
        elif vola >= 0.4:
            temp_score += 5
        elif vola <= 0.1:
            temp_score -= 10  # 极低波动 → 偏冷(无人交易)

    # 5. 短期涨幅维度
    if 'shanghai' in idx:
        sh = idx['shanghai']
        if sh['day'] >= 3:
            temp_score += 15  # 单日大涨
        elif sh['day'] >= 1.5:
            temp_score += 8
        elif sh['day'] <= -3:
            temp_score -= 15  # 单日大跌
        elif sh['day'] <= -1.5:
            temp_score -= 8
        # 周度趋势
        if sh['week'] >= 5:
            temp_score += 10
        elif sh['week'] <= -5:
            temp_score -= 10

    # 6. 行业资金流向维度
    if sec:
        inflow_count = sum(1 for v in sec.values() if v > 0)
        inflow_pct = inflow_count / max(len(sec), 1)
        if inflow_pct >= 0.85:
            temp_score += 12  # 资金全面涌入
        elif inflow_pct >= 0.7:
            temp_score += 6
        elif inflow_pct <= 0.15:
            temp_score -= 12  # 资金全面出逃
        elif inflow_pct <= 0.3:
            temp_score -= 6

    temp_score = max(0, min(100, temp_score))

    # 分级
    if temp_score >= 80:
        level = '🔥 市场过热'
        level_color = '#ef4444'
        desc = '多个维度显示市场情绪亢奋，追高风险较大，注意回调压力'
        suggestion = '建议减仓或轻仓观望，警惕高位回落'
    elif temp_score >= 65:
        level = '🟠 市场偏热'
        level_color = '#f97316'
        desc = '市场活跃度偏高，短期走势偏强，但需警惕过热指标'
        suggestion = '可持股但不宜追高，逐步锁定利润'
    elif temp_score >= 45:
        level = '🟢 市场正常'
        level_color = '#22c55e'
        desc = '市场温度适中，多空力量相对均衡，结构性机会为主'
        suggestion = '可正常仓位运作，精选个股'
    elif temp_score >= 25:
        level = '🔵 市场偏冷'
        level_color = '#3b82f6'
        desc = '市场交投清淡，情绪偏弱，多数资金处于观望状态'
        suggestion = '控制仓位，可关注超跌反弹机会'
    else:
        level = '❄️ 市场过冷'
        level_color = '#6b7280'
        desc = '多维度指向市场悲观/恐慌，风险释放充分，可能接近底部'
        suggestion = '不建议恐慌杀跌，可逐步低吸布局'

    return {
        'score': temp_score,
        'level': level,
        'level_color': level_color,
        'desc': desc,
        'suggestion': suggestion,
        'details': {
            'rsi': rsi_val,
            'vol_ratio': vol_ratio,
            'breadth_ratio': round(up / max(up + dn, 1), 2),
            'volatility': vola,
            'inflow_pct': round(sum(1 for v in sec.values() if v > 0) / max(len(sec), 1) * 100, 1) if sec else None,
        }
    }


def analyze_market_temperature(metrics):
    """AI 分析市场过冷过热状态"""
    temp = compute_market_temperature(metrics)
    idx = metrics.get('index_returns', {})
    sec = metrics.get('sector_perf', {})
    vol = metrics.get('volume', {})

    lines = [
        "=== 市场温度数据 ===",
        f"温度评分: {temp['score']}/100 ({temp['level']})",
        f"RSI(14): {temp['details']['rsi']}",
        f"量比: {temp['details']['vol_ratio']}",
        f"涨跌比: {temp['details']['breadth_ratio']}",
    ]
    if temp['details']['volatility'] is not None:
        lines.append(f"20日波动率: {temp['details']['volatility']}%")
    if temp['details']['inflow_pct'] is not None:
        lines.append(f"行业资金流入比例: {temp['details']['inflow_pct']}%")

    if 'shanghai' in idx:
        sh = idx['shanghai']
        lines.append(f"上证日涨跌: {sh['day']:+.2f}%, 周涨跌: {sh['week']:+.2f}%")

    if sec:
        top3 = sorted(sec.items(), key=lambda x: -x[1])[:3]
        bot3 = sorted(sec.items(), key=lambda x: x[1])[:3]
        lines.append(f"资金涌入: {', '.join(f'{k}{v:+.1f}%' for k, v in top3)}")
        lines.append(f"资金出逃: {', '.join(f'{k}{v:+.1f}%' for k, v in bot3)}")

    prompt = "\n".join(lines) + \
        f"\n\n当前市场温度评级为: {temp['level']}(评分{temp['score']}/100)。" \
        "请结合上述各项技术指标、资金流向和量能数据，从市场过冷过热角度进行综合分析(200字内)：" \
        "当前市场处于什么温度区间？有哪些风险信号？与历史极值相比如何？" \
        "应如何调整投资策略？"

    fallback = (f"【市场温度分析 (本地)】\n\n"
                f"{temp['level']} | 评分: {temp['score']}/100\n\n"
                f"{temp['desc']}\n\n"
                f"建议: {temp['suggestion']}\n\n"
                f"关键数据: RSI={temp['details']['rsi']}, 量比={temp['details']['vol_ratio']}, "
                f"涨跌比={temp['details']['breadth_ratio']}")
    return _safe_ai_call(prompt, fallback)


def build_five_factor_radar(scores_dict):
    """五面雷达图 + 综合仪表"""
    categories = ['政策面', '消息面', '情绪面', '资金面', '走势技术']
    values = [round(scores_dict.get(c, 50), 1) for c in categories]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('五面雷达', '综合评分'),
        specs=[[{'type': 'polar'}, {'type': 'indicator'}]],
        column_widths=[0.58, 0.42]
    )

    # 雷达图
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories,
        fill='toself', fillcolor='rgba(59,130,246,0.2)',
        line=dict(color='#3b82f6', width=2),
        name='当前评分',
        hovertemplate='%{theta}: %{r:.0f}分<extra></extra>'
    ), row=1, col=1)

    # 基准 50 分圆
    fig.add_trace(go.Scatterpolar(
        r=[50]*5, theta=categories,
        fill='none', line=dict(color='#94a3b8', width=1, dash='dash'),
        name='基准 50', showlegend=False
    ), row=1, col=1)

    fig.update_polars(
        radialaxis=dict(range=[0, 100], showline=False, gridcolor='#e2e8f0'),
        angularaxis=dict(gridcolor='#e2e8f0')
    )

    # 综合仪表
    composite = round(sum(values) / 5, 1)
    color = '#ef4444' if composite >= 80 else '#f97316' if composite >= 65 else '#eab308' if composite >= 50 else '#22c55e' if composite >= 35 else '#3b82f6'
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=composite,
        title={'text': '五面综合', 'font': {'size': 14, 'color': '#1e3a5f'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color, 'thickness': 0.2},
            'steps': [
                {'range': [0, 35], 'color': 'rgba(59,130,246,0.1)'},
                {'range': [35, 50], 'color': 'rgba(34,197,94,0.1)'},
                {'range': [50, 65], 'color': 'rgba(234,179,8,0.1)'},
                {'range': [65, 80], 'color': 'rgba(249,115,22,0.1)'},
                {'range': [80, 100], 'color': 'rgba(239,68,68,0.1)'},
            ],
            'threshold': {'line': {'color': '#1e293b', 'width': 2}, 'value': composite}
        },
        number={'font': {'size': 28}, 'suffix': '分'}
    ), row=1, col=2)

    fig.update_layout(
        template='plotly_white',
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.25)
    )
    return fig


# ===================== 页面布局 =====================

def market_overview_layout():
    ai_settings = get_ai_settings()
    sources = get_available_sources()

    return html.Div([
        html.Div([
            # 左侧边栏
            html.Div([
                html.H3("📊 五面分析", className="text-xl font-bold text-gray-800 mb-4 pb-2 border-b"),

                # 数据源
                html.Label("数据源:", className="block text-sm font-medium text-gray-700 mb-1"),
                dcc.Dropdown(
                    id='mo-data-source',
                    options=[{'label': s.capitalize(), 'value': s} for s in sources],
                    value='tushare', clearable=False, className="mb-3"
                ),

                # 日期
                html.Label("分析日期:", className="block text-sm font-medium text-gray-700 mb-1"),
                dcc.DatePickerSingle(
                    id='mo-date',
                    date=datetime.now().strftime('%Y-%m-%d'),
                    className="w-full mb-3"
                ),

                html.Button(
                    [html.I(className="fas fa-play mr-1"), "运行分析"],
                    id='mo-run-btn',
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2.5 px-4 rounded-lg font-medium transition-all text-sm mb-3"
                ),

                html.Button(
                    [html.I(className="fas fa-sun mr-1"), "刷新盘前提醒"],
                    id='mo-premarket-refresh-btn',
                    className="w-full bg-amber-500 hover:bg-amber-600 text-white py-1.5 px-3 rounded-lg font-medium transition-all text-xs mb-3"
                ),

                html.Div(id='mo-status', className="text-sm text-gray-500 mb-2"),
                html.Div(id='mo-error', className="text-sm text-red-500 mb-2"),
                html.Div(id='mo-summary', className="text-sm text-gray-600 mb-4"),

                # ─── AI 测试 ───
                html.Button(
                    [html.I(className="fas fa-plug mr-1"), "测试 AI 连接"],
                    id='mo-test-ai-btn',
                    className="w-full bg-orange-500 hover:bg-orange-600 text-white py-1.5 px-3 rounded-lg font-medium transition-all text-xs mb-3"
                ),
                html.Div(id='mo-test-ai-result', className="text-sm text-gray-600 mb-2"),

                # ─── AI 设置 ───
                html.Hr(className="my-3"),
                html.H4("🤖 AI 设置", className="text-base font-bold text-gray-700 mb-2"),

                html.Label("AI 提供商:", className="block text-sm text-gray-600 mb-0.5"),
                dcc.Dropdown(
                    id='mo-ai-provider',
                    options=[
                        {'label': 'DeepSeek', 'value': 'deepseek'},
                        {'label': '千问 (Qwen)', 'value': 'qwen'},
                        {'label': 'OpenAI', 'value': 'openai'},
                        {'label': 'Azure OpenAI', 'value': 'azure'},
                        {'label': '自定义 (Ollama等)', 'value': 'custom'},
                    ],
                    value=ai_settings['provider'],
                    clearable=False, className="mb-2"
                ),

                html.Label("Base URL:", className="block text-sm text-gray-600 mb-0.5"),
                dcc.Input(
                    id='mo-ai-base-url',
                    type='text',
                    value=ai_settings['base_url'],
                    placeholder='https://api.openai.com/v1',
                    className="w-full py-1.5 px-2 border border-gray-300 rounded text-sm mb-2"
                ),

                html.Label("Model:", className="block text-sm text-gray-600 mb-0.5"),
                dcc.Dropdown(
                    id='mo-ai-model',
                    options=MODEL_OPTIONS.get(ai_settings['provider'], MODEL_OPTIONS['custom']),
                    value=ai_settings['model'],
                    placeholder='选择或输入模型名...',
                    searchable=True,
                    clearable=False, className="mb-2"
                ),

                html.Label("API Key:", className="block text-sm text-gray-600 mb-0.5"),
                dcc.Input(
                    id='mo-ai-api-key',
                    type='password',
                    value=ai_settings['api_key'],
                    placeholder='sk-...',
                    className="w-full py-1.5 px-2 border border-gray-300 rounded text-sm mb-2"
                ),

                html.Button(
                    [html.I(className="fas fa-save mr-1"), "保存设置"],
                    id='mo-save-ai-btn',
                    className="w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded-lg font-medium transition-all text-sm mb-1"
                ),
                html.Div(id='mo-ai-save-status', className="text-sm text-green-600"),

            ], className="w-full md:w-56 lg:w-64 bg-white rounded-xl shadow-sm border border-gray-100 p-4 space-y-0.5 overflow-y-auto"),

            # 右侧主区域
            html.Div([
                # 盘前提醒（页面加载自动生成，数据全部来自真实数据源）
                html.Div(id='mo-premarket-card', className="mb-3"),
                # 5个分析模块卡片
                html.Div(id='mo-panels-container', className="space-y-3"),
                dcc.Store(id='mo-metrics-store', storage_type='memory'),
            ], className="w-full md:w-[calc(100%-14rem)] lg:w-[calc(100%-16rem)] pl-4")
        ], className="flex flex-col md:flex-row"),
    ], className="min-h-screen bg-gray-50 p-4")


def _build_panel(title, icon, color_class, analysis_id, chart_id):
    """构建分析面板"""
    return html.Div([
        html.Div([
            html.Span(f"{icon} {title}", className=f"text-base font-bold {color_class}"),
        ], className="pb-2 border-b border-gray-100"),
        html.Div(id=analysis_id, className="text-sm text-gray-700 leading-relaxed py-2 whitespace-pre-wrap"),
        html.Div(id=chart_id),
    ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-4")


# ===================== 回调注册 =====================

def _metrics_to_json(metrics):
    """将 metrics dict 转为 JSON-safe dict (numpy → float)"""
    import numpy as np

    def _convert(v):
        if isinstance(v, dict):
            return {k: _convert(v2) for k, v2 in v.items()}
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    return _convert(metrics)


def _json_to_metrics(data):
    """将 JSON dict 还原为 metrics (所有值已是 python 原生类型)"""
    return data or {}


def register_market_callbacks(app):

    # ─── 盘前提醒（页面加载即生成 / 手动刷新绕过缓存） ───
    from .premarket import register_premarket_callbacks
    register_premarket_callbacks(app)

    @app.callback(
        [Output('mo-status', 'children'),
         Output('mo-error', 'children'),
         Output('mo-summary', 'children'),
         Output('mo-panels-container', 'children'),
         Output('mo-metrics-store', 'data')],
        [Input('mo-run-btn', 'n_clicks')],
        [State('mo-data-source', 'value'),
         State('mo-date', 'date')]
    )
    def run_market_analysis(n_clicks, data_source, analysis_date):
        if not n_clicks:
            return '', '', '', [], None

        try:
            if analysis_date:
                end_date = pd.Timestamp(analysis_date).strftime('%Y%m%d')
                start_date = (pd.Timestamp(analysis_date) - pd.Timedelta(days=120)).strftime('%Y%m%d')
            else:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')

            set_data_source(data_source)
            status = f"⏳ 正在拉取数据 ({data_source}) ..."
            index_data, sector_data = fetch_market_data(start_date, end_date)
            metrics = compute_market_metrics(index_data, sector_data)

            idx_ret = metrics.get('index_returns', {})
            if not idx_ret:
                return '', '❌ 未获取到任何指数数据，请检查数据源', '', [], None

            status = f"✅ 数据加载完成 ({len(index_data)} 指数, {len(sector_data)} 行业)"

            # 汇总
            up_names = [v['name'] for v in idx_ret.values() if v['day'] > 0]
            summary = f"↑{len(up_names)} ↓{len(idx_ret) - len(up_names)} | "
            if 'shanghai' in idx_ret:
                summary += f"上证 {idx_ret['shanghai']['close']:.0f} ({idx_ret['shanghai']['day']:+.2f}%)"

            # 序列化 metrics 到 Store (numpy → python)
            store_data = _metrics_to_json(metrics)

            # 五面评分
            factor_result = compute_five_factor_scores(metrics)
            store_data['five_factor'] = factor_result  # 合并到 store

            # 市场过冷过热提醒
            temperature = compute_market_temperature(metrics)
            store_data['market_temperature'] = temperature

            # 生成面板 (AI 分析位先放占位符)
            panels = []

            # ★ 五面雷达图 + 持仓建议 (放最前面)
            radar_fig = build_five_factor_radar(factor_result['scores'])
            panels.append(html.Div([
                dcc.Graph(figure=radar_fig, config={'displayModeBar': False})
            ], className="mb-3"))
            panels.append(html.Div([
                html.Span(f"📊 推荐仓位: {factor_result['position']}",
                          className="text-base font-bold text-gray-800"),
                html.Span(f" | {factor_result['advice']}",
                          className="text-sm text-gray-600 ml-2"),
                html.Span(f" (综合 {factor_result['composite']} 分)",
                          className="text-sm text-gray-500 ml-1"),
            ], className="bg-white rounded-xl shadow-sm border border-gray-100 p-3 mb-3 text-center"))

            # 市场温度提醒面板
            temp_bg = f"bg-{'red' if temperature['score'] >= 80 else 'orange' if temperature['score'] >= 65 else 'green' if temperature['score'] >= 45 else 'blue' if temperature['score'] >= 25 else 'gray'}-50"
            temp_border = f"border-{'red' if temperature['score'] >= 80 else 'orange' if temperature['score'] >= 65 else 'green' if temperature['score'] >= 45 else 'blue' if temperature['score'] >= 25 else 'gray'}-300"
            panels.append(html.Div([
                html.Div([
                    html.Span(f"🌡️ {temperature['level']}",
                              style={'color': temperature['level_color']},
                              className="text-base font-bold"),
                    html.Span(f" 评分 {temperature['score']}/100",
                              className="text-sm text-gray-500 ml-2"),
                ], className="mb-1"),
                html.Div(f"{temperature['desc']}", className="text-sm text-gray-700 mb-1"),
                html.Div(f"💡 {temperature['suggestion']}", className="text-sm font-medium text-gray-800 mb-2"),
                html.Div([
                    html.Span(f"RSI:{temperature['details']['rsi']} | ", className="text-xs text-gray-500"),
                    html.Span(f"量比:{temperature['details']['vol_ratio']} | ", className="text-xs text-gray-500"),
                    html.Span(f"涨跌比:{temperature['details']['breadth_ratio']} | ", className="text-xs text-gray-500"),
                    html.Span(f"波动率:{temperature['details']['volatility']:.2f}%" if temperature['details']['volatility'] is not None else "波动率:N/A", className="text-xs text-gray-500"),
                    html.Span(f" | 资金流入:{temperature['details']['inflow_pct']}%" if temperature['details']['inflow_pct'] is not None else "", className="text-xs text-gray-500"),
                ]),
                html.Div(id='mo-temp-analysis', className="text-sm text-gray-700 leading-relaxed pt-2 mt-2 border-t border-gray-200 whitespace-pre-wrap"),
            ], className=f"bg-white rounded-xl shadow-sm border-2 border-l-4 p-4 mb-3", style={'borderLeftColor': temperature['level_color']}))

            tech_fig = build_index_trend_chart(index_data)
            panels.append(_build_panel("走势技术", "📈", "text-blue-700", 'mo-tech-analysis', 'mo-tech-chart'))
            panels.append(html.Div([dcc.Graph(figure=tech_fig, config={'displayModeBar': True, 'displaylogo': False})],
                                    className="mb-3"))

            if metrics.get('sector_perf'):
                sector_fig = build_sector_heatmap(metrics['sector_perf'])
                panels.append(html.Div([dcc.Graph(figure=sector_fig, config={'displayModeBar': False})],
                                        className="mb-3"))

            tech = metrics.get('technical', {})
            vol = metrics.get('volume', {})
            sentiment_fig = build_sentiment_gauge(tech.get('rsi'), vol.get('status', 'N/A'))
            panels.append(_build_panel("情绪面", "😊", "text-orange-600", 'mo-sentiment-analysis', 'mo-sentiment-chart'))
            panels.append(html.Div([dcc.Graph(figure=sentiment_fig, config={'displayModeBar': False})],
                                    className="mb-3"))

            vol_fig = build_volatility_chart(index_data)
            panels.append(html.Div([dcc.Graph(figure=vol_fig, config={'displayModeBar': False})],
                                    className="mb-3"))

            panels.append(_build_panel("政策面", "🏛️", "text-red-700", 'mo-policy-analysis', 'mo-policy-chart'))
            panels.append(_build_panel("消息面", "📰", "text-indigo-700", 'mo-international-analysis', 'mo-international-chart'))
            panels.append(_build_panel("资金面", "💰", "text-green-700", 'mo-capital-analysis', 'mo-capital-chart'))

            return status, '', summary, panels, store_data

        except Exception:
            traceback.print_exc()
            return '', f"❌ 分析出错: {traceback.format_exc()}", '', [], None

    # ─── AI 分析回调 (读取 Store，不重复取数据) ───
    def _load_metrics(data):
        if data is None:
            return {}
        return _json_to_metrics(data)

    @app.callback(
        Output('mo-tech-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def tech_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_technical(m)

    @app.callback(
        Output('mo-policy-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def policy_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_policy(m)

    @app.callback(
        Output('mo-international-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def news_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_news_sentiment(m)

    @app.callback(
        Output('mo-sentiment-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def sentiment_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_sentiment(m)

    @app.callback(
        Output('mo-capital-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def capital_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_capital_flow(m)

    @app.callback(
        Output('mo-temp-analysis', 'children'),
        [Input('mo-metrics-store', 'data')],
        prevent_initial_call=True
    )
    def temp_ai(data):
        m = _load_metrics(data)
        if not m:
            return ""
        return analyze_market_temperature(m)

    # ─── 测试 AI 连接 ───
    @app.callback(
        Output('mo-test-ai-result', 'children'),
        [Input('mo-test-ai-btn', 'n_clicks')],
        [State('mo-ai-provider', 'value'),
         State('mo-ai-base-url', 'value'),
         State('mo-ai-model', 'value'),
         State('mo-ai-api-key', 'value')],
        prevent_initial_call=True
    )
    def test_ai_connection(n, provider, base_url, model, api_key):
        if not api_key:
            return "❌ 请先填写 API Key"
        settings = {
            'provider': provider, 'api_key': api_key,
            'base_url': base_url, 'model': model
        }
        try:
            import requests
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": "回复：OK"}], "max_tokens": 5},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message']['content']
                return f"✅ 连接成功! 模型 {model} 返回: {content[:50]}"
            else:
                return f"❌ HTTP {resp.status_code}: {resp.text[:100]}"
        except Exception as e:
            return f"❌ 连接失败: {str(e)[:100]}"

    # ─── AI 设置保存 ───
    @app.callback(
        Output('mo-ai-save-status', 'children'),
        [Input('mo-save-ai-btn', 'n_clicks')],
        [State('mo-ai-provider', 'value'),
         State('mo-ai-base-url', 'value'),
         State('mo-ai-model', 'value'),
         State('mo-ai-api-key', 'value')],
        prevent_initial_call=True
    )
    def save_ai_callback(n, provider, base_url, model, api_key):
        if not n:
            return ''
        save_ai_settings(provider, api_key or '', base_url, model)
        return '✅ 设置已保存到 cache/ai_settings.json'

    # ─── 数据源切换 ───
    @app.callback(
        [Output('mo-status', 'children', allow_duplicate=True),
         Output('mo-error', 'children', allow_duplicate=True)],
        [Input('mo-data-source', 'value')],
        prevent_initial_call=True
    )
    def switch_mo_source(source_name):
        set_data_source(source_name)
        return f'🔄 数据源已切换为 {source_name}', ''

    # ─── AI provider 切换，自动填充 base_url 和 model ───
    @app.callback(
        [Output('mo-ai-base-url', 'value'),
         Output('mo-ai-model', 'options'),
         Output('mo-ai-model', 'value')],
        [Input('mo-ai-provider', 'value')],
        prevent_initial_call=True
    )
    def auto_fill_ai_config(provider):
        presets = {
            'deepseek': {'base_url': 'https://api.deepseek.com/v1', 'model': 'deepseek-chat'},
            'qwen': {'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'model': 'qwen-plus'},
            'openai': {'base_url': 'https://api.openai.com/v1', 'model': 'gpt-4o-mini'},
            'azure': {'base_url': 'https://YOUR_RESOURCE.openai.azure.com', 'model': 'gpt-4o'},
            'custom': {'base_url': 'http://localhost:11434/v1', 'model': 'qwen2.5:7b'},
        }
        preset = presets.get(provider, presets['custom'])
        options = MODEL_OPTIONS.get(provider, MODEL_OPTIONS['custom'])
        return preset['base_url'], options, preset['model']
