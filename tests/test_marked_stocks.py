"""已标记股票数据隔离测试：按用户隔离 + 旧文件迁移"""
import pytest

from app import database_manager as dm


@pytest.fixture()
def marked_db(monkeypatch, tmp_path):
    """临时库 + 可切换的当前用户 + 迁移目录指向空临时目录（测试自带旧文件）"""
    state = {'uid': 1}
    monkeypatch.setattr(dm, 'DB_NAME', str(tmp_path / 'marked_test.db'))
    monkeypatch.setattr(dm, 'MARKED_FILES_DIR', str(tmp_path))
    monkeypatch.setattr(dm, 'get_current_user_id', lambda: state['uid'])
    dm.init_db()
    return state


def test_marked_stocks_isolated_per_user(marked_db):
    # 用户1标记两只
    marked_db['uid'] = 1
    assert dm.add_marked_stock('600519.SH') is True
    assert dm.add_marked_stock('000001.SZ') is True
    # 用户2标记另一只
    marked_db['uid'] = 2
    assert dm.add_marked_stock('300750.SZ') is True

    marked_db['uid'] = 1
    assert dm.get_marked_stocks() == {'600519.SH', '000001.SZ'}, '用户1只见自己的标记'
    marked_db['uid'] = 2
    assert dm.get_marked_stocks() == {'300750.SZ'}, '用户2只见自己的标记'


def test_marked_stocks_add_duplicate_and_remove(marked_db):
    marked_db['uid'] = 1
    dm.add_marked_stock('600519.SH')
    dm.add_marked_stock('600519.SH')  # 重复添加幂等
    assert dm.get_marked_stocks() == {'600519.SH'}
    assert dm.remove_marked_stock('600519.SH') is True
    assert dm.get_marked_stocks() == set()
    assert dm.remove_marked_stock('600519.SH') is True  # 重复删除不报错


def test_marked_stocks_requires_login(marked_db, monkeypatch):
    monkeypatch.setattr(dm, 'get_current_user_id', lambda: None)
    assert dm.get_marked_stocks() == set()
    assert dm.add_marked_stock('600519.SH') is False
    assert dm.remove_marked_stock('600519.SH') is False


def test_legacy_marked_file_migration(monkeypatch, tmp_path):
    """旧版 marked_stocks*.txt 应一次性导入对应账号（无后缀共享文件归第一个用户）"""
    (tmp_path / 'marked_stocks.txt').write_text('600519.SH\n000001.SZ\n', encoding='utf-8')
    (tmp_path / 'marked_stocks_bob.txt').write_text('300750.SZ\n', encoding='utf-8')

    monkeypatch.setattr(dm, 'DB_NAME', str(tmp_path / 'marked_mig.db'))
    monkeypatch.setattr(dm, 'MARKED_FILES_DIR', str(tmp_path))
    monkeypatch.setattr(dm, 'get_current_user_id', lambda: 1)
    dm.init_db()

    import sqlite3
    conn = sqlite3.connect(dm.DB_NAME)
    admin_id = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()[0]
    admin_codes = {r[0] for r in conn.execute(
        "SELECT code FROM marked_stocks WHERE user_id=?", (admin_id,)).fetchall()}
    bob_id = conn.execute("SELECT id FROM users WHERE username='bob'").fetchone()
    conn.close()
    assert admin_codes == {'600519.SH', '000001.SZ'}, '共享旧文件应归第一个用户(admin)'
    assert bob_id is None, '不存在的用户文件应跳过'
