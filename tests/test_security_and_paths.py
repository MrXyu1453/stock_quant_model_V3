"""安全与路径模块单元测试（security / paths / data_manager.validate_code）"""
import os

import pytest

from app.security import hash_password, verify_password, load_or_create_secret_key
from app.paths import PROJECT_ROOT, CONFIG_FILE, CACHE_DIR, cache_path, project_path


# ==================== 密码哈希 ====================

class TestPasswordHashing:
    def test_roundtrip(self):
        h = hash_password('s3cret!')
        assert verify_password('s3cret!', h)
        assert not verify_password('wrong', h)

    def test_unique_salt(self):
        assert hash_password('abc') != hash_password('abc')

    def test_format(self):
        h = hash_password('x')
        iterations, salt, digest = h.split('$')
        assert iterations == '260000'
        assert len(salt) == 32  # 16 字节 hex
        assert len(digest) == 64

    def test_malformed_stored_hash_returns_false(self):
        assert not verify_password('x', 'not-a-valid-hash')
        assert not verify_password('x', '')
        assert not verify_password('x', None)

    def test_unicode_password(self):
        h = hash_password('密码测试123')
        assert verify_password('密码测试123', h)
        assert not verify_password('密码测试456', h)


# ==================== 会话密钥 ====================

class TestSecretKey:
    def test_load_or_create_returns_stable_key(self, tmp_path, monkeypatch):
        key_file = tmp_path / 'secret_key'
        monkeypatch.setattr('app.security._SECRET_KEY_FILE', str(key_file))
        k1 = load_or_create_secret_key()
        assert len(k1) == 64
        k2 = load_or_create_secret_key()
        assert k1 == k2

    def test_env_var_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.setenv('APP_SECRET_KEY', 'from-env')
        monkeypatch.setattr('app.security._SECRET_KEY_FILE', str(tmp_path / 's'))
        assert load_or_create_secret_key() == 'from-env'


# ==================== 路径模块 ====================

class TestPaths:
    def test_project_root_is_parent_of_app(self):
        assert os.path.basename(PROJECT_ROOT) == 'stock_quant_model_V3 - zcoder'
        assert os.path.isdir(os.path.join(PROJECT_ROOT, 'app'))

    def test_config_file_exists(self):
        assert os.path.isfile(CONFIG_FILE)

    def test_cache_path_creates_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr('app.paths.CACHE_DIR', str(tmp_path / 'cache'))
        p = cache_path('sub', 'file.json')
        assert os.path.isdir(os.path.dirname(p))
        assert p.endswith('file.json')

    def test_project_path(self):
        assert project_path('stock_codes.db').startswith(PROJECT_ROOT)


# ==================== 代码校验 ====================

class TestValidateCode:
    def test_valid_formats(self):
        from app.data_manager import validate_code
        assert validate_code('600519.SH')
        assert validate_code('000001.SZ')
        assert validate_code('600519')
        assert validate_code('159915')

    def test_invalid_formats(self):
        from app.data_manager import validate_code
        assert not validate_code('abcdefg')
        assert not validate_code('')
        assert not validate_code('60051.SH')
