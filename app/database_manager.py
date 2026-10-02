import sqlite3
import json
import os
import logging

from .security import hash_password
from .paths import load_config, project_path

# 加载配置文件（缺 config.yaml 时回退模板/内置默认值）
config = load_config()

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据库路径（相对项目根目录解析，避免受启动目录影响）
DB_NAME = project_path(config.get('database', {}).get('name', 'stock_codes.db'))
# 旧版已标记股票文件的存放目录（init_db 时一次性迁移入数据库）
MARKED_FILES_DIR = project_path('.')

# 默认管理员账号（首次初始化时创建，接管历史遗留数据）
DEFAULT_ADMIN_USERNAME = 'admin'
DEFAULT_ADMIN_PASSWORD = 'admin123'


def normalize_stock_code(code):
    """将 6 位数字股票代码统一为带交易所后缀的标准格式
    5/6/9 开头 -> .SH, 0/1/2/3 开头 -> .SZ, 92/4/8 开头 -> .BJ, 其他原样返回"""
    code = (code or '').strip()
    if not code or not code.isdigit() or len(code) != 6:
        return code
    if code[:2] == '92' or code[0] in ('4', '8'):
        return code + '.BJ'
    if code[0] in ('5', '6', '9'):
        return code + '.SH'
    return code + '.SZ'


def get_current_user_id():
    """从 Flask session 获取当前登录用户 ID；无请求上下文或未登录返回 None"""
    try:
        from flask import session, has_request_context
        if has_request_context():
            return session.get('user_id')
    except Exception:
        pass
    return None


def get_current_username():
    """从 Flask session 获取当前登录用户名；无请求上下文或未登录返回 None"""
    try:
        from flask import session, has_request_context
        if has_request_context():
            return session.get('username')
    except Exception:
        pass
    return None


def _column_names(c, table):
    """返回指定表的所有列名"""
    return [row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]


def _ensure_user_id_column(c, table):
    """若表缺少 user_id 列则补齐"""
    if 'user_id' not in _column_names(c, table):
        c.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")


def _migrate_legacy_to_user(c, uid, legacy_capital):
    """将历史遗留（user_id 为 NULL）的数据归属到指定用户"""
    for t in ('trading_journal', 'capital_flows', 'stock_codes', 'watchlists', 'watchlist_stocks'):
        c.execute(f"UPDATE {t} SET user_id=? WHERE user_id IS NULL", (uid,))
    if legacy_capital is not None:
        c.execute("INSERT OR IGNORE INTO account (user_id, initial_capital) VALUES (?, ?)",
                  (uid, legacy_capital))


# 初始化数据库
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # 用户表
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT NOT NULL UNIQUE,
                      password_hash TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # 股票代码表
        c.execute('''CREATE TABLE IF NOT EXISTS stock_codes
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT, name TEXT DEFAULT '')''')
        try:
            c.execute("ALTER TABLE stock_codes ADD COLUMN name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        _ensure_user_id_column(c, 'stock_codes')

        # 自选股分组表（需要 user_id + (user_id, name) 唯一，旧表无 user_id 则重建）
        if 'user_id' not in _column_names(c, 'watchlists'):
            c.execute('''CREATE TABLE IF NOT EXISTS watchlists
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            c.execute("ALTER TABLE watchlists RENAME TO watchlists_legacy")
            c.execute('''CREATE TABLE watchlists
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT NOT NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          user_id INTEGER,
                          UNIQUE(user_id, name))''')
            c.execute("INSERT INTO watchlists (id, name, created_at, user_id) "
                      "SELECT id, name, created_at, NULL FROM watchlists_legacy")
            c.execute("DROP TABLE watchlists_legacy")
        else:
            c.execute('''CREATE TABLE IF NOT EXISTS watchlists
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          user_id INTEGER, UNIQUE(user_id, name))''')

        # 自选股分组-股票关联表
        c.execute('''CREATE TABLE IF NOT EXISTS watchlist_stocks
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, watchlist_id INTEGER, code TEXT,
                      FOREIGN KEY (watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE,
                      UNIQUE(watchlist_id, code))''')
        _ensure_user_id_column(c, 'watchlist_stocks')

        # 交易日志表
        c.execute('''CREATE TABLE IF NOT EXISTS trading_journal
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      stock_code TEXT NOT NULL,
                      stock_name TEXT DEFAULT '',
                      direction TEXT DEFAULT '买入',
                      trade_date TEXT DEFAULT '',
                      price REAL DEFAULT 0,
                      quantity INTEGER DEFAULT 0,
                      amount REAL DEFAULT 0,
                      trade_type TEXT DEFAULT '短线',
                      tags TEXT DEFAULT '',
                      status TEXT DEFAULT '持仓中',
                      profit_loss REAL DEFAULT 0,
                      profit_pct REAL DEFAULT 0,
                      reason TEXT DEFAULT '',
                      memo TEXT DEFAULT '',
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        try:
            c.execute("ALTER TABLE trading_journal ADD COLUMN fee REAL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            c.execute("ALTER TABLE trading_journal ADD COLUMN action TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        _ensure_user_id_column(c, 'trading_journal')

        # 账户资金表（每用户一行，user_id 主键）
        legacy_capital = None
        if 'user_id' not in _column_names(c, 'account'):
            try:
                row = c.execute("SELECT initial_capital FROM account WHERE id=1").fetchone()
                legacy_capital = row[0] if row else 0.0
            except Exception:
                legacy_capital = 0.0
            c.execute("DROP TABLE IF EXISTS account")
            c.execute('''CREATE TABLE account
                         (user_id INTEGER PRIMARY KEY,
                          initial_capital REAL DEFAULT 0,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        else:
            c.execute('''CREATE TABLE IF NOT EXISTS account
                         (user_id INTEGER PRIMARY KEY,
                          initial_capital REAL DEFAULT 0,
                          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # 出入金流水表
        c.execute('''CREATE TABLE IF NOT EXISTS capital_flows
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      flow_type TEXT DEFAULT '转入',
                      amount REAL DEFAULT 0,
                      flow_date TEXT DEFAULT '',
                      note TEXT DEFAULT '',
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        _ensure_user_id_column(c, 'capital_flows')

        # 模拟炒股战绩表
        c.execute('''CREATE TABLE IF NOT EXISTS game_records
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT DEFAULT '',
                      stock_code TEXT DEFAULT '',
                      stock_name TEXT DEFAULT '',
                      freq TEXT DEFAULT 'daily',
                      play_bars INTEGER DEFAULT 0,
                      initial_cash REAL DEFAULT 0,
                      final_value REAL DEFAULT 0,
                      return_pct REAL DEFAULT 0,
                      bench_return_pct REAL DEFAULT 0,
                      excess_pp REAL DEFAULT 0,
                      max_dd REAL DEFAULT 0,
                      n_buy INTEGER DEFAULT 0,
                      n_sell INTEGER DEFAULT 0,
                      win_rate REAL DEFAULT 0,
                      realized_pnl REAL DEFAULT 0,
                      fees REAL DEFAULT 0,
                      grade TEXT DEFAULT '',
                      comment TEXT DEFAULT '',
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        _ensure_user_id_column(c, 'game_records')

        # 已购买标记表（原 marked_stocks*.txt 文件，按用户隔离）
        marked_table_new = ('marked_stocks',) not in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        c.execute('''CREATE TABLE IF NOT EXISTS marked_stocks
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      code TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      UNIQUE(user_id, code))''')
        _ensure_user_id_column(c, 'marked_stocks')

        # 首次初始化：创建默认管理员并接管历史遗留数据
        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                      (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD)))
            admin_id = c.lastrowid
            _migrate_legacy_to_user(c, admin_id, legacy_capital)
            logging.info("已创建默认管理员账号 admin，并接管历史遗留数据")

        if marked_table_new:
            _migrate_marked_stocks_files(c)

        conn.commit()
        conn.close()
        logging.info("数据库初始化完成")
    except Exception as e:
        logging.error(f"数据库初始化出错: {e}")


def _migrate_marked_stocks_files(c):
    """将旧版 marked_stocks*.txt 文件一次性导入数据库（按文件名中的用户名归属）"""
    import glob
    import os
    imported = 0
    for path in glob.glob(os.path.join(MARKED_FILES_DIR, 'marked_stocks*.txt')):
        base = os.path.basename(path)                     # marked_stocks.txt / marked_stocks_admin.txt
        suffix = base[len('marked_stocks'):-len('.txt')]  # '' 或 '_admin'
        username = suffix[1:] if suffix.startswith('_') else None
        uid = None
        if username:
            row = c.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            uid = row[0] if row else None
        else:
            # 无后缀的旧共享文件归到第一个用户（与 _migrate_legacy_to_user 一致）
            row = c.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
            uid = row[0] if row else None
        if uid is None:
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                codes = {line.strip() for line in f if line.strip()}
            for code in sorted(codes):
                c.execute("INSERT OR IGNORE INTO marked_stocks (code, user_id) VALUES (?, ?)",
                          (code, uid))
                imported += 1
        except Exception as e:
            logging.warning(f"迁移标记文件 {base} 失败: {e}")
    if imported:
        logging.info(f"已迁移 {imported} 条已标记股票记录到数据库")


# ==================== 用户操作 ====================

def username_exists(username):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT 1 FROM users WHERE username=?", (username,))
        exists = c.fetchone() is not None
        conn.close()
        return exists
    except Exception as e:
        logging.error(f"检查用户名时出错: {e}")
        return False


def insert_user(username, password_hash):
    """新增用户，返回 user_id，失败返回 None"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                  (username, password_hash))
        conn.commit()
        uid = c.lastrowid
        conn.close()
        return uid
    except Exception as e:
        logging.error(f"新增用户时出错: {e}")
        return None


def get_user_by_username(username):
    """返回 (id, password_hash) 或 None"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
        row = c.fetchone()
        conn.close()
        return row
    except Exception as e:
        logging.error(f"查询用户时出错: {e}")
        return None


def get_user_by_id(uid):
    """返回 {'id':.., 'username':..} 或 None"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, username FROM users WHERE id=?", (uid,))
        row = c.fetchone()
        conn.close()
        return {'id': row[0], 'username': row[1]} if row else None
    except Exception as e:
        logging.error(f"查询用户时出错: {e}")
        return None


def update_user_password(uid, password_hash):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE users SET password_hash=? WHERE id=?", (password_hash, uid))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"更新密码时出错: {e}")
        return False


def is_default_admin_password():
    """判断 admin 是否仍在使用默认密码（用于登录页提示）"""
    try:
        row = get_user_by_username(DEFAULT_ADMIN_USERNAME)
        if not row:
            return False
        from .security import verify_password
        return verify_password(DEFAULT_ADMIN_PASSWORD, row[1])
    except Exception:
        return False


# ==================== 自选股分组操作 ====================

def create_watchlist(name):
    """创建自选股分组"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO watchlists (name, user_id) VALUES (?, ?)", (name, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"创建自选股分组 {name} 时出错: {e}")
        return False


def get_all_watchlists():
    """获取当前用户所有自选股分组"""
    user_id = get_current_user_id()
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, name FROM watchlists WHERE user_id=? ORDER BY id", (user_id,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"获取自选股分组时出错: {e}")
        return []


def delete_watchlist(watchlist_id):
    """删除自选股分组"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM watchlists WHERE id=? AND user_id=?", (watchlist_id, user_id))
        c.execute("DELETE FROM watchlist_stocks WHERE watchlist_id=?", (watchlist_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"删除自选股分组时出错: {e}")
        return False


def add_stock_to_watchlist(watchlist_id, code):
    """添加股票到分组（自动规范化为带交易所后缀格式）"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    code = normalize_stock_code(code)
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO watchlist_stocks (watchlist_id, code, user_id) VALUES (?, ?, ?)",
                  (watchlist_id, code, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"添加股票 {code} 到分组时出错: {e}")
        return False


def remove_stock_from_watchlist(watchlist_id, code):
    """从分组移除股票"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM watchlist_stocks WHERE watchlist_id=? AND code=? AND user_id=?",
                  (watchlist_id, code, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"从分组移除股票 {code} 时出错: {e}")
        return False


def get_watchlist_stocks(watchlist_id):
    """获取指定分组的股票代码列表"""
    user_id = get_current_user_id()
    if user_id is None:
        return set()
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT code FROM watchlist_stocks WHERE watchlist_id=? AND user_id=?", (watchlist_id, user_id))
        rows = c.fetchall()
        conn.close()
        return {row[0] for row in rows}
    except Exception as e:
        logging.error(f"获取分组股票时出错: {e}")
        return set()


def get_stock_watchlist_map():
    """获取当前用户所有股票->分组映射 (code -> [watchlist_names])"""
    user_id = get_current_user_id()
    if user_id is None:
        return {}
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""SELECT wl.name, ws.code FROM watchlist_stocks ws
                     JOIN watchlists wl ON ws.watchlist_id = wl.id
                     WHERE ws.user_id=?""", (user_id,))
        rows = c.fetchall()
        conn.close()
        result = {}
        for wl_name, code in rows:
            result.setdefault(code, []).append(wl_name)
        return result
    except Exception as e:
        logging.error(f"获取股票分组映射时出错: {e}")
        return {}


# 插入股票代码到数据库（自动规范化为带交易所后缀格式，重复则仅补全名称）
def insert_stock_code(code, name=''):
    """返回 True=新增成功，False=已存在或失败"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    code = normalize_stock_code(code)
    if not code:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, name FROM stock_codes WHERE code=? AND user_id=?", (code, user_id))
        existing = c.fetchone()
        if existing:
            # 已存在：若库内名称为空且本次传入了名称，则补全
            if name and not existing[1]:
                c.execute("UPDATE stock_codes SET name=? WHERE code=? AND user_id=?", (name, code, user_id))
            conn.commit()
            conn.close()
            return False
        c.execute("INSERT INTO stock_codes (code, name, user_id) VALUES (?, ?, ?)", (code, name, user_id))
        conn.commit()
        conn.close()
        logging.info(f"股票代码 {code} ({name}) 已插入数据库")
        return True
    except Exception as e:
        logging.error(f"插入股票代码 {code} 到数据库时出错: {e}")
        return False


# 更新股票名称
def update_stock_name(code, name):
    user_id = get_current_user_id()
    if user_id is None:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE stock_codes SET name=? WHERE code=? AND user_id=?", (name, code, user_id))
        conn.commit()
        conn.close()
        logging.info(f"股票 {code} 名称已更新为: {name}")
    except Exception as e:
        logging.error(f"更新股票名称 {code} 时出错: {e}")


# 更新股票代码和名称
def update_stock_code(old_code, new_code, new_name):
    user_id = get_current_user_id()
    if user_id is None:
        return False
    new_code = normalize_stock_code(new_code)
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("UPDATE stock_codes SET code=?, name=? WHERE code=? AND user_id=?",
                  (new_code, new_name, old_code, user_id))
        # 同步自选股分组关联表中的代码，避免分组关联因改码而失效
        c.execute("UPDATE watchlist_stocks SET code=? WHERE code=? AND user_id=?",
                  (new_code, old_code, user_id))
        conn.commit()
        conn.close()
        logging.info(f"股票 {old_code} 已更新为 {new_code} ({new_name})")
        return True
    except Exception as e:
        logging.error(f"更新股票代码 {old_code} 时出错: {e}")
        return False


# 从数据库获取当前用户所有股票代码和名称
def get_all_stock_codes():
    user_id = get_current_user_id()
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT code, name FROM stock_codes WHERE user_id=?", (user_id,))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"从数据库获取所有股票代码时出错: {e}")
        return []


# 获取股票名称映射字典
def get_stock_name_map():
    rows = get_all_stock_codes()
    return {row[0]: row[1] for row in rows}


# 读取缓存的股票或 ETF 代码
def read_cached_codes():
    stock_list = get_all_stock_codes()
    if stock_list:
        codes = [row[0] for row in stock_list[:5]]
        return ','.join(codes)
    return '600519.SH,000001.SZ'


# 保存股票或 ETF 代码，支持批量附带名称
def save_codes(codes, name_map=None):
    user_id = get_current_user_id()
    if user_id is None:
        return
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM stock_codes WHERE user_id=?", (user_id,))
        for code in codes.split(','):
            code = code.strip()
            if code:
                name = (name_map or {}).get(code, '')
                c.execute("INSERT INTO stock_codes (code, name, user_id) VALUES (?, ?, ?)", (code, name, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"保存股票代码到数据库时出错: {e}")


# 删除指定股票代码
def delete_stock_code(code):
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM stock_codes WHERE code=? AND user_id=?", (code, user_id))
        # 同步清理自选股分组关联，避免残留孤儿记录
        c.execute("DELETE FROM watchlist_stocks WHERE code=? AND user_id=?", (code, user_id))
        conn.commit()
        conn.close()
        logging.info(f"股票代码 {code} 已删除")
        return True
    except Exception as e:
        logging.error(f"删除股票代码 {code} 时出错: {e}")
        return False


# 按代码或名称搜索股票
def search_stock_codes(keyword):
    user_id = get_current_user_id()
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT code, name FROM stock_codes WHERE user_id=? AND (code LIKE ? OR name LIKE ?)",
                  (user_id, f'%{keyword}%', f'%{keyword}%'))
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logging.error(f"搜索股票代码时出错: {e}")
        return []


# ==================== 交易日志操作 ====================

_TRADE_FIELDS = ('stock_code', 'stock_name', 'direction', 'action', 'trade_date',
                 'price', 'quantity', 'amount', 'fee', 'trade_type', 'tags',
                 'reason', 'memo')


def _trade_row_to_dict(row):
    """将查询行转为字典（股票代码统一规范格式）"""
    cols = ('id',) + _TRADE_FIELDS + ('created_at', 'updated_at')
    d = dict(zip(cols, row))
    d['stock_code'] = normalize_stock_code(d.get('stock_code', ''))
    return d


def add_trade_entry(data):
    """新增交易日志，返回新纪录 id，失败返回 None"""
    user_id = get_current_user_id()
    if user_id is None:
        return None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        keys = list(_TRADE_FIELDS) + ['user_id']
        placeholders = ','.join('?' for _ in keys)
        cols = ','.join(keys)
        values = [data.get(k, '') for k in _TRADE_FIELDS] + [user_id]
        c.execute(f"INSERT INTO trading_journal ({cols}) VALUES ({placeholders})", values)
        conn.commit()
        new_id = c.lastrowid
        conn.close()
        logging.info(f"交易日志已新增: id={new_id} {data.get('stock_code', '')} {data.get('action', '')}")
        return new_id
    except Exception as e:
        logging.error(f"新增交易日志时出错: {e}")
        return None


def update_trade_entry(entry_id, data):
    """更新交易日志（仅限当前用户自己的记录）"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        keys = list(_TRADE_FIELDS)
        set_clause = ','.join(f"{k}=?" for k in keys)
        values = [data.get(k, '') for k in keys] + [entry_id, user_id]
        c.execute(f"UPDATE trading_journal SET {set_clause}, updated_at=CURRENT_TIMESTAMP WHERE id=? AND user_id=?", values)
        updated = c.rowcount > 0
        conn.commit()
        conn.close()
        if updated:
            logging.info(f"交易日志已更新: id={entry_id}")
        else:
            logging.warning(f"更新交易日志失败(记录不存在或不属于当前用户): id={entry_id}")
        return updated
    except Exception as e:
        logging.error(f"更新交易日志 id={entry_id} 时出错: {e}")
        return False


def delete_trade_entry(entry_id):
    """删除交易日志（仅限当前用户自己的记录）"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM trading_journal WHERE id=? AND user_id=?", (entry_id, user_id))
        deleted = c.rowcount > 0
        conn.commit()
        conn.close()
        if deleted:
            logging.info(f"交易日志已删除: id={entry_id}")
        else:
            logging.warning(f"删除交易日志失败(记录不存在或不属于当前用户): id={entry_id}")
        return deleted
    except Exception as e:
        logging.error(f"删除交易日志 id={entry_id} 时出错: {e}")
        return False


def clear_all_trade_data():
    """一键清除当前用户全部交易日志数据（交易记录 + 出入金流水 + 初始资金）"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM trading_journal WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM capital_flows WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM account WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        logging.info(f"已一键清除当前用户(user_id={user_id})的全部交易日志数据")
        return True
    except Exception as e:
        logging.error(f"一键清除交易日志数据时出错: {e}")
        return False


def get_trade_entries():
    """获取当前用户全部交易日志（按交易日期倒序）"""
    user_id = get_current_user_id()
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, stock_code, stock_name, direction, action, trade_date, price, quantity, amount, "
                  "fee, trade_type, tags, reason, memo, created_at, updated_at "
                  "FROM trading_journal WHERE user_id=? ORDER BY trade_date DESC, id DESC", (user_id,))
        rows = c.fetchall()
        conn.close()
        return [_trade_row_to_dict(r) for r in rows]
    except Exception as e:
        logging.error(f"获取交易日志时出错: {e}")
        return []


def get_trade_entry(entry_id):
    """获取单条交易日志"""
    user_id = get_current_user_id()
    if user_id is None:
        return None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, stock_code, stock_name, direction, action, trade_date, price, quantity, amount, "
                  "fee, trade_type, tags, reason, memo, created_at, updated_at "
                  "FROM trading_journal WHERE id=? AND user_id=?", (entry_id, user_id))
        row = c.fetchone()
        conn.close()
        return _trade_row_to_dict(row) if row else None
    except Exception as e:
        logging.error(f"获取交易日志 id={entry_id} 时出错: {e}")
        return None


# ==================== 账户资金操作 ====================

def get_account():
    """获取当前用户账户资金信息"""
    user_id = get_current_user_id()
    if user_id is None:
        return {'initial_capital': 0.0}
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT initial_capital FROM account WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        return {'initial_capital': row[0] if row else 0.0}
    except Exception as e:
        logging.error(f"获取账户资金时出错: {e}")
        return {'initial_capital': 0.0}


def set_initial_capital(value):
    """设置当前用户初始资金"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO account (user_id, initial_capital) VALUES (?, ?)", (user_id, value))
        c.execute("UPDATE account SET initial_capital=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?", (value, user_id))
        conn.commit()
        conn.close()
        logging.info(f"初始资金已设置为: {value}")
        return True
    except Exception as e:
        logging.error(f"设置初始资金时出错: {e}")
        return False


def add_capital_flow(flow_type, amount, flow_date, note):
    """新增出入金流水"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO capital_flows (flow_type, amount, flow_date, note, user_id) VALUES (?, ?, ?, ?, ?)",
                  (flow_type, amount, flow_date, note, user_id))
        conn.commit()
        conn.close()
        logging.info(f"出入金流水已新增: {flow_type} {amount}")
        return True
    except Exception as e:
        logging.error(f"新增出入金流水时出错: {e}")
        return False


def get_capital_flows():
    """获取当前用户全部出入金流水（按日期倒序）"""
    user_id = get_current_user_id()
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, flow_type, amount, flow_date, note, created_at "
                  "FROM capital_flows WHERE user_id=? ORDER BY flow_date DESC, id DESC", (user_id,))
        rows = c.fetchall()
        conn.close()
        cols = ('id', 'flow_type', 'amount', 'flow_date', 'note', 'created_at')
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logging.error(f"获取出入金流水时出错: {e}")
        return []


def delete_capital_flow(flow_id):
    """删除出入金流水"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM capital_flows WHERE id=? AND user_id=?", (flow_id, user_id))
        conn.commit()
        conn.close()
        logging.info(f"出入金流水已删除: id={flow_id}")
        return True
    except Exception as e:
        logging.error(f"删除出入金流水时出错: {e}")
        return False


# ===================== 模拟炒股战绩 =====================

_GAME_RECORD_COLS = ('id', 'username', 'stock_code', 'stock_name', 'freq', 'play_bars',
                     'initial_cash', 'final_value', 'return_pct', 'bench_return_pct',
                     'excess_pp', 'max_dd', 'n_buy', 'n_sell', 'win_rate',
                     'realized_pnl', 'fees', 'grade', 'comment', 'created_at')


def add_game_record(data):
    """新增模拟炒股战绩，返回新纪录 id；未登录或失败返回 None"""
    user_id = get_current_user_id()
    if user_id is None:
        return None
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        keys = [k for k in _GAME_RECORD_COLS if k not in ('id', 'created_at')]
        placeholders = ','.join('?' for _ in range(len(keys) + 1))
        values = [data.get(k, '') for k in keys] + [user_id]
        c.execute(f"INSERT INTO game_records ({','.join(keys)}, user_id) "
                  f"VALUES ({placeholders})", values)
        conn.commit()
        new_id = c.lastrowid
        conn.close()
        return new_id
    except Exception as e:
        logging.error(f"保存模拟炒股战绩时出错: {e}")
        return None


def _game_record_row_to_dict(row):
    return dict(zip(_GAME_RECORD_COLS, row))


def get_top_game_records(limit=10, min_trades=1):
    """排行榜：全用户按超额收益降序（至少有一次买入才计入）"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        rows = c.execute(
            "SELECT " + ','.join(_GAME_RECORD_COLS) +
            " FROM game_records WHERE n_buy >= ? "
            "ORDER BY excess_pp DESC, created_at DESC LIMIT ?",
            (min_trades, limit)).fetchall()
        conn.close()
        return [_game_record_row_to_dict(r) for r in rows]
    except Exception as e:
        logging.error(f"获取模拟炒股排行榜时出错: {e}")
        return []


def get_my_game_records(user_id, limit=20):
    """指定用户的最近战绩"""
    if user_id is None:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        rows = c.execute(
            "SELECT " + ','.join(_GAME_RECORD_COLS) +
            " FROM game_records WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
            (user_id, limit)).fetchall()
        conn.close()
        return [_game_record_row_to_dict(r) for r in rows]
    except Exception as e:
        logging.error(f"获取用户战绩时出错: {e}")
        return []


def get_game_user_summary(user_id):
    """指定用户的战绩汇总: 总局数/平均超额/正超额占比/最佳超额"""
    records = get_my_game_records(user_id, limit=1000)
    n = len(records)
    if n == 0:
        return {'games': 0, 'avg_excess': 0.0, 'positive_ratio': 0.0, 'best_excess': 0.0}
    excesses = [r['excess_pp'] for r in records]
    return {'games': n,
            'avg_excess': round(sum(excesses) / n, 2),
            'positive_ratio': round(sum(1 for e in excesses if e > 0) / n, 2),
            'best_excess': round(max(excesses), 2)}


# ==================== 已购买标记（按用户隔离） ====================

def get_marked_stocks():
    """当前用户已标记（已购买）的股票代码集合；未登录返回空集合"""
    user_id = get_current_user_id()
    if user_id is None:
        return set()
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        rows = c.execute("SELECT code FROM marked_stocks WHERE user_id=? ORDER BY code",
                         (user_id,)).fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception as e:
        logging.error(f"获取已标记股票时出错: {e}")
        return set()


def add_marked_stock(code):
    """为当前用户添加标记；未登录返回 False"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO marked_stocks (code, user_id) VALUES (?, ?)",
                  (code, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"添加已标记股票时出错: {e}")
        return False


def remove_marked_stock(code):
    """移除当前用户的标记；未登录返回 False"""
    user_id = get_current_user_id()
    if user_id is None:
        return False
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM marked_stocks WHERE code=? AND user_id=?", (code, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"移除已标记股票时出错: {e}")
        return False
