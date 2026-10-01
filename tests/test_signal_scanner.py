"""后台信号扫描器测试：双均线信号判定、过滤标注、设置读写、结果排序"""
import numpy as np
import pandas as pd
import pytest

import app.signal_scanner as scanner
from app.signal_scanner import (evaluate_dma_signal, load_scan_settings,
                                save_scan_settings)


def _make_data(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': closes * 1.01, 'low': closes * 0.99,
        'close': closes, 'vol': 10000.0, 'amount': 1e7,
    }, index=idx)


def test_active_buy_signal_detected():
    # 下跌后强势上穿：末期金叉持仓 → 检出买入信号
    closes = list(np.linspace(30, 15, 60)) + list(np.linspace(15, 25, 40))
    df = _make_data(closes)
    hit = evaluate_dma_signal(df, short_w=5, long_w=20)
    assert hit is not None
    assert hit['price'] == pytest.approx(closes[-1])
    assert hit['days_held'] >= 0
    assert hit['signal_date'] <= hit['last_date']


def test_no_signal_in_downtrend():
    closes = list(np.linspace(30, 10, 120))
    df = _make_data(closes)
    assert evaluate_dma_signal(df, short_w=5, long_w=20) is None


def test_fresh_flag_and_filtered_flag():
    # 末段 3 根急拉 → 金叉恰好落在最后 3 根内 → fresh=True；ADX 阈值 150 → filtered_active=False
    closes = list(np.linspace(20, 12, 90)) + [12.0] * 12 + [13.0, 15.5, 18.0]
    df = _make_data(closes)
    hit = evaluate_dma_signal(df, short_w=5, long_w=20,
                              filter_opts=['trend', 'confirm', 'adx'], adx_threshold=150)
    assert hit is not None
    assert hit['fresh'] is True
    assert hit['filtered_active'] is False
    # 不启用过滤 → filtered_active 为 None
    hit2 = evaluate_dma_signal(df, short_w=5, long_w=20, filter_opts=None)
    assert hit2['filtered_active'] is None


def test_insufficient_data_returns_none():
    assert evaluate_dma_signal(_make_data([10, 11, 12]), 5, 20) is None
    assert evaluate_dma_signal(None, 5, 20) is None


def test_scan_settings_roundtrip(tmp_path):
    path = tmp_path / 'analysis_scan_settings.json'
    old = scanner._SETTINGS_PATH
    scanner._SETTINGS_PATH = str(path)
    try:
        assert load_scan_settings()['short_window'] == 5  # 默认值
        save_scan_settings({'short_window': 8, 'long_window': 25, 'filter_options': ['trend', 'adx']})
        s = load_scan_settings()
        assert s['short_window'] == 8 and s['long_window'] == 25
        assert s['filter_options'] == ['trend', 'adx']
        assert s['confirm_days'] == 2  # 未提供的键保留默认
    finally:
        scanner._SETTINGS_PATH = old


def test_watchlist_codes_dedup(tmp_path):
    import sqlite3
    db = tmp_path / 'stock_codes.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE stock_codes (code TEXT, name TEXT, user_id INTEGER)')
    conn.execute("INSERT INTO stock_codes VALUES ('600598', '北大荒', 1)")
    conn.execute("INSERT INTO stock_codes VALUES ('600598.SH', '北大荒', 1)")
    conn.execute("INSERT INTO stock_codes VALUES ('000001', '平安银行', 2)")
    conn.commit()
    conn.close()
    old = scanner._db_path
    scanner._db_path = lambda: str(db)
    try:
        codes = scanner._get_watchlist_codes()
        assert codes == ['600598.SH', '000001.SZ']
    finally:
        scanner._db_path = old
