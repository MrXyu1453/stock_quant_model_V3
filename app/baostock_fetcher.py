# stock_quant_model_V3/app/baostock_fetcher.py
"""
Baostock 数据源模块
负责从 Baostock API 获取股票K线数据（日线/周线/月线/分钟线），并统一为标准格式
"""
import baostock as bs
import pandas as pd
import logging
import time
import threading

# K线周期: 统一格式 -> baostock frequency 参数
# 注: 分钟线历史数据只到 2020-01-02, 之前无数据
FREQ_MAP = {
    'daily': 'd',
    'weekly': 'w',
    'monthly': 'm',
    '5min': '5',
    '15min': '15',
    '30min': '30',
    '60min': '60',
}

# 各周期可用字段不同: 日线最全; 周/月线不支持 preclose/turn/tradestatus/pctChg/isST; 分钟线多 time 字段
_DAY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
_WM_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag"
_MIN_FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"

# 请求频率控制
REQUEST_DELAY = 1.0
_last_request_time = 0
_rate_lock = threading.Lock()

# 登录状态
_logged_in = False
_login_lock = threading.Lock()

# baostock 查询锁: baostock 使用进程级全局 socket 收发数据(非线程安全),
# 并发查询会导致响应串扰(utf-8 解码错误), 所有查询必须串行执行
_query_lock = threading.Lock()

def _ensure_login():
    """确保 baostock 已登录"""
    global _logged_in
    with _login_lock:
        if not _logged_in:
            lg = bs.login()
            if lg.error_code == '0':
                _logged_in = True
                logging.info("Baostock 登录成功")
            else:
                logging.error(f"Baostock 登录失败: {lg.error_msg}")
                return False
    return True


def _rate_limit():
    """请求频率控制"""
    global _last_request_time
    with _rate_lock:
        current_time = time.time()
        time_since_last_request = current_time - _last_request_time
        if time_since_last_request < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - time_since_last_request)
        _last_request_time = time.time()


def _convert_ts_code_to_bs(ts_code):
    """
    将 tushare 格式代码转换为 baostock 格式
    600519.SH -> sh.600519
    000001.SZ -> sz.000001
    """
    if ts_code.endswith('.SH'):
        return f"sh.{ts_code[:-3]}"
    elif ts_code.endswith('.SZ'):
        return f"sz.{ts_code[:-3]}"
    else:
        return ts_code


def get_stock_name_bs(ts_code):
    """通过 baostock 获取股票名称"""
    if not _ensure_login():
        return ''
    try:
        bs_code = _convert_ts_code_to_bs(ts_code)
        _rate_limit()
        with _query_lock:
            rs = bs.query_stock_basic(code=bs_code)
            if rs.error_code == '0':
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    return data_list[0][1]
        return ''
    except Exception as e:
        logging.warning(f"Baostock 获取 {ts_code} 名称失败: {e}")
        return ''


def fetch_kline_data(ts_code, start_date, end_date, adjustflag="2", freq='daily'):
    """
    从 Baostock 获取股票K线数据并统一为标准格式

    参数:
        ts_code: tushare 格式代码 (如 600519.SH)
        start_date: 开始日期 (YYYYMMDD 格式)
        end_date: 结束日期 (YYYYMMDD 格式)
        adjustflag: 复权类型 1=后复权 2=前复权 3=不复权
        freq: K线周期 daily/weekly/monthly/60min/30min/15min/5min

    返回:
        pandas DataFrame，包含以下列:
        trade_date (datetime, 分钟线为具体时间戳), open, high, low, close, vol, amount
        如果获取失败返回 None
    """
    if freq not in FREQ_MAP:
        logging.error(f"Baostock 不支持的K线周期: {freq}")
        return None

    if not _ensure_login():
        return None

    bs_code = _convert_ts_code_to_bs(ts_code)
    bs_freq = FREQ_MAP[freq]
    if freq in ('5min', '15min', '30min', '60min'):
        fields = _MIN_FIELDS
    elif freq in ('weekly', 'monthly'):
        fields = _WM_FIELDS
    else:
        fields = _DAY_FIELDS

    # Baostock 需要 YYYY-MM-DD 格式，将 YYYYMMDD 转换为 YYYY-MM-DD
    if len(start_date) == 8:
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
    if len(end_date) == 8:
        end_date = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

    try:
        _rate_limit()
        logging.info(f"Baostock 开始获取 {ts_code} ({bs_code}) 的{freq}数据，日期: {start_date} ~ {end_date}...")

        # 查询与逐页迭代都必须持有查询锁 (rs.next() 会持续通过全局 socket 收发数据)
        with _query_lock:
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency=bs_freq,
                adjustflag=adjustflag  # 复权类型: 1=后复权 2=前复权 3=不复权
            )

            if rs is None or rs.error_code != '0':
                error_msg = rs.error_msg if rs else "查询返回空结果（日期格式可能不正确）"
                logging.error(f"Baostock 查询 {ts_code} 失败: {error_msg}")
                return None

            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

        if not data_list:
            logging.warning(f"Baostock {ts_code} 获取到的数据为空")
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 过滤掉空数据行
        df = df[df['volume'] != '']
        df = df[df['close'] != '']

        if df.empty:
            logging.warning(f"Baostock {ts_code} 过滤后数据为空")
            return None

        # 统一字段名和数据类型（与 tushare 格式对齐）
        df = df.rename(columns={
            'volume': 'vol',
        })

        # 分钟线的 time 字段为 YYYYMMDDHHMMSSmmm (K线结束时间戳), 合并为 trade_date
        if freq in ('5min', '15min', '30min', '60min'):
            df['trade_date'] = pd.to_datetime(df['time'].astype(str).str[:14], format='%Y%m%d%H%M%S')
        else:
            df['trade_date'] = pd.to_datetime(df['date'])

        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

        # 按日期排序
        df = df.sort_values(by='trade_date')

        # 只保留核心字段
        df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]

        logging.info(f"Baostock {ts_code} 获取成功，数据行数: {len(df)}")
        return df

    except Exception as e:
        logging.error(f"Baostock 获取 {ts_code} 数据时发生错误: {e}")
        return None


def fetch_daily_data(ts_code, start_date, end_date, adjustflag="2"):
    """兼容旧接口: 获取日线数据"""
    return fetch_kline_data(ts_code, start_date, end_date, adjustflag=adjustflag, freq='daily')


def fetch_trade_dates(start_date, end_date):
    """
    获取交易日历（baostock query_trade_dates，免积分无限频）

    参数:
        start_date / end_date: YYYYMMDD 或 YYYY-MM-DD 格式

    返回:
        pandas DataFrame (columns: calendar_date, is_trading_day)，is_trading_day 为 '1'/'0'；
        获取失败返回 None
    """
    if not _ensure_login():
        return None

    def _fmt(d):
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d

    try:
        with _query_lock:
            rs = bs.query_trade_dates(start_date=_fmt(start_date),
                                      end_date=_fmt(end_date))
            if rs is None or rs.error_code != '0':
                error_msg = rs.error_msg if rs else "查询返回空结果"
                logging.error(f"Baostock 交易日历查询失败: {error_msg}")
                return None
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
        if not rows:
            logging.warning("Baostock 交易日历获取到的数据为空")
            return None
        return pd.DataFrame(rows, columns=rs.fields)
    except Exception as e:
        logging.error(f"Baostock 获取交易日历时发生错误: {e}")
        return None


def batch_get_stock_names_bs(ts_codes):
    """批量获取股票名称"""
    name_map = {}
    for code in ts_codes:
        name = get_stock_name_bs(code)
        if name:
            name_map[code] = name
    return name_map


def logout():
    """登出 baostock"""
    global _logged_in
    with _login_lock:
        if _logged_in:
            bs.logout()
            _logged_in = False
            logging.info("Baostock 已登出")