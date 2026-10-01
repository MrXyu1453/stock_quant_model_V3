# stock_quant_model_V3/app/data_manager.py
import tushare as ts
import pandas as pd
import logging
import os
import time
import io
import redis

from .paths import load_config

# 加载配置文件（缺 config.yaml 时回退模板/内置默认值）
config = load_config()

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 tushare token（环境变量 TUSHARE_TOKEN 优先于配置文件，便于不落盘共享凭据）
TOKEN = os.environ.get('TUSHARE_TOKEN') or (config.get('tushare', {}).get('token') or '').strip()
pro = None
if TOKEN:
    try:
        ts.set_token(TOKEN)
        pro = ts.pro_api()
        logging.info("成功设置 tushare token")
    except Exception as e:
        logging.error(f"设置 tushare token 时出错: {e}")
else:
    logging.warning("未配置 tushare token，tushare 数据源不可用，将回退到免费数据源 baostock。"
                    "如需使用 tushare: 在 https://tushare.pro 注册后于「个人主页→接口TOKEN」获取，"
                    "设置环境变量 TUSHARE_TOKEN 或填入 config/config.yaml")

# 连接 Redis（protocol=2 以兼容不支持 HELLO 命令的旧版 Redis 服务端）
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, protocol=2)
    redis_client.ping()
    logging.info("成功连接到 Redis 服务器")
except redis.exceptions.RedisError:
    logging.error("无法连接到 Redis 服务器，请确保 Redis 已启动")
    redis_client = None

# 缓存过期时间（秒）
CACHE_EXPIRATION = config.get('cache', {}).get('expiration', 21600)

# 请求频率控制 - 仅限制实际 API 调用（不影响缓存读取）
REQUEST_DELAY = config.get('tushare', {}).get('request_delay', 1.3)
_last_request_time = 0
import threading as _threading
_rate_lock = _threading.Lock()

def _rate_limit():
    global _last_request_time
    with _rate_lock:
        current_time = time.time()
        time_since_last_request = current_time - _last_request_time
        if time_since_last_request < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last_request)
        _last_request_time = time.time()

# ==================== K线周期管理 ====================
# 支持的周期: daily / weekly / monthly / 60min / 30min / 15min
# 分钟线: baostock 历史深度仅到 2020-01-02; tushare stk_mins 接口对低积分账户限 1次/分钟
FREQ_ALIASES = {
    'daily': 'daily', 'd': 'daily', 'day': 'daily',
    'weekly': 'weekly', 'w': 'weekly', 'week': 'weekly',
    'monthly': 'monthly', 'm': 'monthly', 'month': 'monthly',
    '60min': '60min', '60': '60min',
    '30min': '30min', '30': '30min',
    '15min': '15min', '15': '15min',
    '5min': '5min', '5': '5min',
}
MINUTE_FREQS = ('5min', '15min', '30min', '60min')

def _normalize_freq(freq):
    """归一化K线周期参数, 未知值回退为 daily"""
    if freq is None:
        return 'daily'
    key = str(freq).strip().lower()
    if key in FREQ_ALIASES:
        return FREQ_ALIASES[key]
    logging.warning(f"未知的K线周期: {freq}，回退为日线")
    return 'daily'


# ==================== 数据源管理 ====================
# 支持的数据源：tushare, baostock
_current_data_source = config.get('data_source', {}).get('default', 'tushare')
_data_source_lock = _threading.Lock()

def set_data_source(source_name):
    """设置当前数据源"""
    global _current_data_source
    available = config.get('data_source', {}).get('available', ['tushare'])
    if source_name not in available:
        logging.warning(f"不支持的数据源: {source_name}，可用: {available}")
        return False
    with _data_source_lock:
        _current_data_source = source_name
    logging.info(f"数据源已切换为: {source_name}")
    return True

def get_data_source():
    """获取当前数据源名称"""
    return _current_data_source

def get_available_sources():
    """获取所有可用数据源"""
    return config.get('data_source', {}).get('available', ['tushare'])


def _fetch_single(code, start_date, end_date, freq):
    return code, get_and_process_data(code, start_date, end_date, freq=freq)

# 批量并发获取数据（带进度回调）
def batch_get_data(stock_codes, start_date, end_date, progress_callback=None, freq='daily'):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    freq = _normalize_freq(freq)
    results = {}
    total = len(stock_codes)
    completed = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_single, code, start_date, end_date, freq): code for code in stock_codes}
        for future in as_completed(futures):
            completed += 1
            code = futures[future]
            try:
                _, data = future.result()
                results[code] = data
                if progress_callback:
                    progress_callback(completed, total, code)
            except Exception as e:
                logging.error(f"获取 {code} 数据失败: {e}")
                results[code] = None
                if progress_callback:
                    progress_callback(completed, total, code)
    return results

# 获取并处理实时数据，使用 Redis 缓存
def get_and_process_data(stock_code, start_date, end_date, source=None, freq='daily'):
    """
    获取股票K线数据，支持多数据源切换和多周期

    参数:
        stock_code: 股票代码 (tushare格式: 600519.SH)
        start_date: 开始日期 (YYYYMMDD格式)
        end_date: 结束日期 (YYYYMMDD格式)
        source: 数据源名称，None则使用当前设置的数据源
        freq: K线周期 daily/weekly/monthly/60min/30min/15min/5min

    返回:
        pandas DataFrame，统一格式: trade_date, open, high, low, close, vol, amount
        (分钟线的 trade_date 为具体时间戳)
    """
    if source is None:
        source = _current_data_source
    freq = _normalize_freq(freq)

    # Redis 缓存键包含数据源和周期名称，避免不同源/不同周期数据混用
    if redis_client:
        cache_key = f"{source}:{freq}:{stock_code}:{start_date}:{end_date}"
        cached_data = redis_client.get(cache_key)
        if cached_data:
            try:
                logging.info(f"从 Redis 缓存中获取 {stock_code} ({source}, {freq}) 的数据")
                cached_data = cached_data.decode('utf-8')
                # pandas>=2 的 read_json 需要用 StringIO 包裹 JSON 字符串
                data = pd.read_json(io.StringIO(cached_data), orient='records')
                if data is not None and not data.empty and 'trade_date' in data.columns:
                    data['trade_date'] = pd.to_datetime(data['trade_date'])
                    return data
                else:
                    logging.warning(f"从缓存读取 {stock_code} ({source}, {freq}) 数据格式无效，将重新获取")
                    redis_client.delete(cache_key)
            except Exception as e:
                logging.warning(f"从缓存读取 {stock_code} ({source}, {freq}) 数据失败，将重新获取: {e}")
                redis_client.delete(cache_key)

    # 根据数据源调用不同的获取函数
    # 分钟线固定优先使用 baostock: tushare stk_mins 接口对低积分账户限 1次/分钟, 批量场景不可用
    if freq in MINUTE_FREQS:
        _sources_to_try = ['baostock']
        if source == 'tushare':
            _sources_to_try.append('tushare')
        logging.info(f"分钟线数据优先使用 baostock 获取 (请求周期: {freq})")
    else:
        _sources_to_try = [source]
        alt_source = 'baostock' if source == 'tushare' else 'tushare'
        _sources_to_try.append(alt_source)

    last_error = None
    for src in _sources_to_try:
        try:
            logging.info(f"{src.capitalize()} 开始获取 {stock_code} 的{freq}数据...")
            if src == 'baostock':
                df = _fetch_from_baostock(stock_code, start_date, end_date, freq)
            else:
                df = _fetch_from_tushare(stock_code, start_date, end_date, freq)

            if df is not None and not df.empty:
                if src != _sources_to_try[0]:
                    logging.info(f"{_sources_to_try[0]} 失败，自动回退到 {src} 成功获取 {stock_code} 数据 ({len(df)} 行)")
                if redis_client:
                    cache_key = f"{src}:{freq}:{stock_code}:{start_date}:{end_date}"
                    redis_client.setex(cache_key, CACHE_EXPIRATION, df.to_json(orient='records', date_format='iso'))
                return df
            else:
                last_error = f"{src} 返回空数据"
                logging.warning(f"{src.capitalize()} {stock_code} 获取到的数据为空")
        except Exception as e:
            last_error = f"{src}: {str(e)}"
            logging.error(f"{src.capitalize()} {stock_code} 获取数据时发生错误: {e}")

    logging.error(f"所有数据源 ({', '.join(_sources_to_try)}) 均无法获取 {stock_code} 的{freq}数据: {last_error}")
    return None


def _fetch_from_tushare(stock_code, start_date, end_date, freq='daily'):
    """从 Tushare 获取K线数据 (daily/weekly/monthly 走对应接口, 分钟线走 stk_mins)"""
    if pro is None:
        logging.error("未配置 tushare token，无法使用 tushare 数据源"
                      "（设置环境变量 TUSHARE_TOKEN 或填入 config/config.yaml）")
        return None
    freq = _normalize_freq(freq)
    try:
        logging.info(f"Tushare 开始获取 {stock_code} 的{freq}数据...")
        _rate_limit()
        if freq == 'weekly':
            df = pro.weekly(ts_code=stock_code, start_date=start_date, end_date=end_date)
        elif freq == 'monthly':
            df = pro.monthly(ts_code=stock_code, start_date=start_date, end_date=end_date)
        elif freq in MINUTE_FREQS:
            # stk_mins 需要 'YYYY-MM-DD HH:MM:SS' 格式
            if len(start_date) == 8:
                start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} 09:00:00"
            else:
                start_dt = start_date
            if len(end_date) == 8:
                end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]} 15:30:00"
            else:
                end_dt = end_date
            df = pro.stk_mins(ts_code=stock_code, freq=freq, start_date=start_dt, end_date=end_dt)
            df = df.rename(columns={'trade_time': 'trade_date'})
        else:
            df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
        if df.empty:
            logging.warning(f"Tushare {stock_code} 获取到的数据为空")
            return None
        logging.info(f"Tushare {stock_code} 获取到的数据行数: {len(df)}")
        df = df.sort_values(by='trade_date')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        logging.info(f"Tushare {stock_code} 数据处理完成，数据行数: {len(df)}")
        return df
    except Exception as e:
        if "频率超限" in str(e) or "rate limit" in str(e).lower():
            logging.error(f"Tushare {stock_code} 接口频率超限: {e}")
        else:
            logging.error(f"Tushare {stock_code} 获取数据时发生错误: {e}")
        return None


def _fetch_from_baostock(stock_code, start_date, end_date, freq='daily'):
    """从 Baostock 获取K线数据"""
    try:
        from .baostock_fetcher import fetch_kline_data
        return fetch_kline_data(stock_code, start_date, end_date, freq=freq)
    except ImportError:
        logging.error("Baostock 模块未安装，请执行: pip install baostock")
        return None
    except Exception as e:
        logging.error(f"Baostock {stock_code} 获取数据时发生错误: {e}")
        return None

# 获取股票中文名称
def get_stock_name(ts_code, source=None):
    """获取股票名称，支持多数据源"""
    if source is None:
        source = _current_data_source

    if source == 'baostock':
        try:
            from .baostock_fetcher import get_stock_name_bs
            return get_stock_name_bs(ts_code)
        except ImportError:
            logging.warning("Baostock 模块未安装")
            return ''
        except Exception as e:
            logging.warning(f"Baostock 获取 {ts_code} 名称失败: {e}")
            return ''

    # 默认使用 tushare
    try:
        _rate_limit()
        df = pro.stock_basic(ts_code=ts_code, fields='name')
        if df is not None and not df.empty:
            name = df.iloc[0]['name']
            return name
    except Exception as e:
        logging.warning(f"获取 {ts_code} 名称失败: {e}")
    return ''

# 批量获取股票中文名称
def batch_get_stock_names(ts_codes):
    name_map = {}
    try:
        for code in ts_codes:
            name = get_stock_name(code)
            if name:
                name_map[code] = name
        return name_map
    except Exception as e:
        logging.error(f"批量获取股票名称失败: {e}")
    return name_map

# 验证股票或 ETF 代码
def validate_code(code):
    if len(code) == 9 and (code.endswith('.SH') or code.endswith('.SZ')):
        return True
    if code.isdigit():
        return True
    return False