"""
后台信号扫描器：定时用 baostock 拉取自选股日线，检测双均线买入信号，供页面提示。

- 数据源固定 baostock（用户指定），逐只拉取并礼貌限速
- 双均线买入信号定义：最新交易日处于金叉后的持仓状态（short MA > long MA），
  并记录最近一次金叉入场日期
- 同时用分析页保存的三重过滤设置（趋势/确认/ADX）标注该信号是否通过过滤，
  被过滤的信号单独标注——减少虚假信号打扰
- 扫描参数（均线窗口/过滤设置）取自分析页最近一次使用的设置（cache/analysis_scan_settings.json）
- 后台线程无 Flask 用户上下文，自选股列表直接读 stock_codes 表去重
"""
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

from .database_manager import normalize_stock_code
from .data_manager import get_and_process_data
from .signal_enhancer import enhance_signals
from .strategy_manager import double_moving_average_strategy

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETTINGS_PATH = os.path.join(_PROJECT_ROOT, 'cache', 'analysis_scan_settings.json')
_RESULTS_PATH = os.path.join(_PROJECT_ROOT, 'cache', 'signal_scan_results.json')
_LOCK_PATH = os.path.join(_PROJECT_ROOT, 'cache', 'signal_scanner.lock')

SCAN_INTERVAL_SECONDS = 300   # 每 5 分钟扫描一轮（数据走 Redis 缓存，重复扫描很快）
_FETCH_DELAY = 0.3            # baostock 逐只拉取间隔（秒）
_MAX_CODES = 300              # 单轮扫描的股票数上限
_LOCK_STALE_SECONDS = SCAN_INTERVAL_SECONDS * 4   # 锁超过该时长视为持有者已死，可抢占

_DEFAULT_SETTINGS = {'short_window': 5, 'long_window': 20, 'filter_options': [],
                     'confirm_days': 2, 'adx_threshold': 20, 'trend_ma': 60}

_lock = threading.Lock()
_scanner_started = False
_latest = {'scanned_at': None, 'results': [], 'scanning': False, 'last_error': None,
           'n_scanned': 0, 'total': 0, 'current_code': '', 'started_at': None}


# ==================== 扫描设置（由分析页回调写入） ====================

def save_scan_settings(settings):
    """记录分析页最近使用的双均线参数与过滤设置，供后台扫描使用"""
    try:
        os.makedirs(os.path.dirname(_SETTINGS_PATH), exist_ok=True)
        with open(_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"保存扫描设置失败: {e}")


def load_scan_settings():
    try:
        with open(_SETTINGS_PATH, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        settings = dict(_DEFAULT_SETTINGS)
        settings.update({k: v for k, v in saved.items() if v is not None})
        return settings
    except Exception:
        return dict(_DEFAULT_SETTINGS)


# ==================== 自选股列表（无用户上下文） ====================

def _db_path():
    return os.path.join(_PROJECT_ROOT, 'stock_codes.db')


def _get_watchlist_codes():
    """直接读 stock_codes 表去重（后台线程没有 Flask 用户会话）"""
    codes = []
    try:
        conn = sqlite3.connect(_db_path())
        rows = conn.execute('SELECT DISTINCT code FROM stock_codes').fetchall()
        conn.close()
        for (c,) in rows:
            c = (c or '').strip()
            if not c:
                continue
            code = normalize_stock_code(c)
            if code not in codes:
                codes.append(code)
    except Exception as e:
        logger.error(f"读取自选股列表失败: {e}")
    return codes[:_MAX_CODES]


# ==================== 信号判定 ====================

def evaluate_dma_signal(df, short_w=5, long_w=20, filter_opts=None,
                        confirm_days=2, adx_threshold=20, trend_ma=60):
    """判定一只股票当前是否存在双均线买入信号（金叉后持仓中）。

    返回 dict 或 None：
      code/signal_date/last_date/price/days_held/fresh(3日内新金叉)/filtered_active(三重过滤后是否仍持仓)
    """
    try:
        if df is None or len(df) < max(long_w, 30) + 3:
            return None
        df = df.sort_values('trade_date').reset_index(drop=True)
        raw = double_moving_average_strategy(df, short_w, long_w)
        if raw is None or raw.empty or raw['signal'].iloc[-1] <= 0:
            return None

        entry_positions = df.index[raw['positions'] == 1]
        if len(entry_positions) == 0:
            return None
        entry_idx = int(entry_positions[-1])
        last_idx = len(df) - 1

        filtered_active = None
        if filter_opts:
            try:
                filtered, _ = enhance_signals(
                    df, raw, 'dma',
                    use_trend='trend' in filter_opts, trend_ma=trend_ma,
                    use_confirm='confirm' in filter_opts, confirm_days=confirm_days,
                    use_adx='adx' in filter_opts, adx_threshold=adx_threshold)
                filtered_active = bool(filtered['signal'].iloc[-1] > 0)
            except Exception as e:
                logger.warning(f"过滤确认失败: {e}")
                filtered_active = None

        return {
            'signal_date': str(df['trade_date'].iloc[entry_idx])[:10],
            'last_date': str(df['trade_date'].iloc[-1])[:10],
            'price': float(df['close'].iloc[-1]),
            'days_held': int(last_idx - entry_idx),
            'fresh': bool(entry_idx >= last_idx - 2),
            'filtered_active': filtered_active,
        }
    except Exception as e:
        logger.warning(f"信号判定失败: {e}")
        return None


# ==================== 扫描线程（文件锁单例：调试模式会派生多个进程，只有持锁进程扫描） ====================

def _acquire_scanner_lock():
    """尝试获取扫描锁（O_CREAT|O_EXCL）；存活持有者定期刷新 mtime，陈旧锁可抢占"""
    try:
        if os.path.exists(_LOCK_PATH):
            if time.time() - os.path.getmtime(_LOCK_PATH) < _LOCK_STALE_SECONDS:
                return False
            os.remove(_LOCK_PATH)
        fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except Exception as e:
        logger.warning(f"扫描锁操作失败（不阻塞扫描）: {e}")
        return True


def _refresh_scanner_lock():
    try:
        if os.path.exists(_LOCK_PATH):
            os.utime(_LOCK_PATH, None)
    except Exception:
        pass


def _scanner_loop():
    while True:
        if not _acquire_scanner_lock():
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue
        try:
            _scan_once()
            _refresh_scanner_lock()
        except Exception as e:
            logger.error(f"信号扫描异常: {e}")
            with _lock:
                _latest.update({'scanning': False, 'last_error': str(e)})
        time.sleep(SCAN_INTERVAL_SECONDS)

def _scan_once():
    settings = load_scan_settings()
    short_w = int(settings.get('short_window', 5))
    long_w = int(settings.get('long_window', 20))
    filter_opts = settings.get('filter_options') or []
    confirm_days = int(settings.get('confirm_days', 2))
    adx_threshold = float(settings.get('adx_threshold', 20))
    trend_ma = int(settings.get('trend_ma', 60))

    codes = _get_watchlist_codes()
    with _lock:
        _latest.update({'scanning': True, 'n_scanned': 0, 'total': len(codes),
                        'current_code': '', 'started_at': time.time(),
                        'note': '' if codes else '自选股列表为空，请先在分析页添加自选股'})
    if not codes:
        with _lock:
            _latest.update({'scanning': False, 'scanned_at': datetime.now().strftime('%H:%M')})
        return

    ed = datetime.now().strftime('%Y%m%d')
    sd = (datetime.now() - timedelta(days=200)).strftime('%Y%m%d')
    hits, errors = [], 0
    for i, code in enumerate(codes):
        try:
            # baostock 为主源（用户指定）；走 get_and_process_data 以复用 Redis 缓存
            df = get_and_process_data(code, sd, ed, source='baostock', freq='daily')
        except Exception as e:
            logger.warning(f"扫描拉取 {code} 失败: {e}")
            df = None
        hit = evaluate_dma_signal(df, short_w, long_w, filter_opts, confirm_days, adx_threshold, trend_ma)
        if hit:
            hit['code'] = code
            hits.append(hit)
        if df is None:
            errors += 1
        with _lock:
            _latest['n_scanned'] = i + 1
            _latest['current_code'] = code
        time.sleep(_FETCH_DELAY)

    # 排序：信号日期新→旧（稳定排序先行），再按 新信号优先、过滤确认优先 排列
    hits.sort(key=lambda h: h['signal_date'], reverse=True)
    hits.sort(key=lambda h: (not h['fresh'],
                             0 if h['filtered_active'] else (1 if h['filtered_active'] is None else 2)))

    with _lock:
        _latest.update({
            'scanned_at': datetime.now().strftime('%m-%d %H:%M'),
            'results': hits,
            'scanning': False,
            'current_code': '',
            'started_at': None,
            'last_error': f'{errors} 只拉取失败' if errors else None,
        })
    try:
        with open(_RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump({'scanned_at': _latest['scanned_at'], 'results': hits}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"写入扫描结果文件失败: {e}")
    logger.info(f"信号扫描完成: {len(codes)} 只自选股, {len(hits)} 只存在双均线买入信号, {errors} 只拉取失败")


def start_background_scanner():
    """启动后台扫描线程（幂等；请在应用入口调用）"""
    global _scanner_started
    if _scanner_started:
        return
    with _lock:
        if _scanner_started:
            return
        _scanner_started = True
    t = threading.Thread(target=_scanner_loop, daemon=True, name='signal-scanner')
    t.start()
    logger.info(f"后台信号扫描器已启动（每 {SCAN_INTERVAL_SECONDS} 秒扫描一次自选股，数据源 baostock）")


def get_latest_scan():
    """读取最近一次扫描结果（页面提示用）"""
    with _lock:
        return dict(_latest)
