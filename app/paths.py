"""项目路径集中管理：所有文件读写都基于项目根目录的绝对路径，
使应用可以从任意工作目录启动（原来依赖 cwd 的相对路径在换目录运行时会崩溃）"""
import copy
import logging
import os

# 项目根目录 = app 包的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.yaml')
CONFIG_EXAMPLE_FILE = os.path.join(CONFIG_DIR, 'config.example.yaml')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

# config.yaml / config.example.yaml 都不存在时的内置默认值（baostock 免登录可用）
_DEFAULT_CONFIG = {
    'tushare': {'token': '', 'request_delay': 1.3},
    'database': {'name': 'stock_codes.db'},
    'cache': {'expiration': 21600},
    'data_source': {'default': 'tushare', 'available': ['tushare', 'baostock']},
}


def cache_path(*parts):
    """返回 cache 目录下的绝对路径，并确保目录存在"""
    path = os.path.join(CACHE_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def project_path(*parts):
    """返回项目根目录下的绝对路径"""
    return os.path.join(PROJECT_ROOT, *parts)


def _deep_merge(base, override):
    """递归合并 override 到 base（返回新 dict，不改入参）"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config():
    """加载应用配置。

    优先级: config/config.yaml（用户自建，含个人凭据，不入库）
          → config/config.example.yaml（仓库模板，token 留空）
          → 内置默认值。
    任意一层缺失的键由下一层补齐，保证全新克隆开箱即用。
    """
    import yaml
    base = copy.deepcopy(_DEFAULT_CONFIG)
    for path in (CONFIG_EXAMPLE_FILE, CONFIG_FILE):
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.error(f"读取配置文件失败 {path}: {e}")
            continue
        base = _deep_merge(base, data)
        if path == CONFIG_FILE:
            break
    if not os.path.exists(CONFIG_FILE):
        logging.warning("未找到 config/config.yaml，使用模板默认配置。"
                        "可复制 config/config.example.yaml 为 config/config.yaml 后自定义"
                        "（tushare token 建议通过环境变量 TUSHARE_TOKEN 提供）")
    return base
