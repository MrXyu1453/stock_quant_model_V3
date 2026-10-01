"""密码安全模块：PBKDF2 加盐哈希与校验、会话密钥管理"""
import hashlib
import os
import secrets

from .paths import cache_path

_ITERATIONS = 260000

# 会话签名密钥持久化文件（避免硬编码密钥，防止伪造 session 冒用账户）
_SECRET_KEY_FILE = cache_path('secret_key')


def load_or_create_secret_key():
    """获取会话签名密钥：优先环境变量，其次本地文件，最后随机生成并持久化"""
    env_key = os.environ.get('APP_SECRET_KEY')
    if env_key:
        return env_key
    try:
        if os.path.exists(_SECRET_KEY_FILE):
            with open(_SECRET_KEY_FILE, 'r', encoding='utf-8') as f:
                key = f.read().strip()
                if key:
                    return key
    except Exception:
        pass
    key = secrets.token_hex(32)
    try:
        os.makedirs(os.path.dirname(_SECRET_KEY_FILE), exist_ok=True)
        with open(_SECRET_KEY_FILE, 'w', encoding='utf-8') as f:
            f.write(key)
    except Exception:
        pass
    return key


def hash_password(password, salt=None):
    """生成带盐的 PBKDF2-SHA256 密码哈希，格式: iterations$salt$hexdigest"""
    if salt is None:
        salt = secrets.token_hex(16)
    password = (password or '').encode('utf-8')
    salt_bytes = salt.encode('utf-8')
    dk = hashlib.pbkdf2_hmac('sha256', password, salt_bytes, _ITERATIONS)
    return f"{_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password, stored):
    """校验明文密码与存储哈希是否匹配（恒定时间比较）"""
    try:
        iterations, salt, expected = (stored or '').split('$')
        dk = hashlib.pbkdf2_hmac('sha256', (password or '').encode('utf-8'),
                                 salt.encode('utf-8'), int(iterations))
        return secrets.compare_digest(dk.hex(), expected)
    except Exception:
        return False
