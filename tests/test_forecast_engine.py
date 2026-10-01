"""量化走势预测引擎测试：模型点估计、区间单调性、概率与校准、状态识别、面板渲染"""
import numpy as np
import pandas as pd
import pytest

from app.forecast_engine import (barrier_probabilities, build_forecast_figure,
                                 build_forecast_panel, detect_regime_series,
                                 empirical_up_prob, forecast_ensemble, forecast_gbm,
                                 forecast_holt, forecast_ols, probability_up_gbm,
                                 run_forecast, summarize_regime, walk_forward_backtest)


def _make_data(closes, start='2024-01-01'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq='B')
    return pd.DataFrame({
        'trade_date': idx, 'open': closes, 'high': closes * 1.01, 'low': closes * 0.99,
        'close': closes, 'vol': 10000.0, 'amount': 1e7,
    }, index=idx)


def test_holt_and_ols_track_linear_trend():
    noise = np.random.default_rng(4).normal(0, 0.15, 200)
    truth = np.linspace(10, 30, 200) + noise
    true_future = np.polyval(np.polyfit(np.arange(200), np.linspace(10, 30, 200), 1), 200 + 5)
    assert abs(forecast_holt(truth, 5)['median'][-1] - true_future) < 0.8
    assert abs(forecast_ols(truth, 5)['median'][-1] - true_future) < 0.8


def test_gbm_band_monotonic_and_positive():
    rng = np.random.default_rng(7)
    closes = 20 * np.exp(rng.normal(0, 0.01, 250).cumsum())
    p5 = forecast_gbm(closes, 5)[0]
    p20 = forecast_gbm(closes, 20)[0]
    assert (p5['q10'] < p5['q25']).all() and (p5['q75'] < p5['q90']).all()
    assert (p20['q90'][-1] - p20['q10'][-1]) > (p5['q90'][-1] - p5['q10'][-1])
    assert (p5['q10'] > 0).all()
    assert p5['q10'][-1] < p5['median'][-1] < p5['q90'][-1]


def test_probability_up_gbm_directional():
    # 强正漂移 → P(涨) 应显著高于 0.5；零漂移 → 接近 0.5
    assert probability_up_gbm(mu=0.003, sigma=0.01, horizon=10) > 0.75
    assert abs(probability_up_gbm(mu=0.0, sigma=0.01, horizon=10) - 0.5) < 0.05


def test_barrier_probabilities_bounded_and_directional():
    rng = np.random.default_rng(3)
    closes = 20 * np.exp(rng.normal(0, 0.01, 250).cumsum())
    sim = np.random.default_rng(5).normal(0, 0.02, (3000, 10)).cumsum(axis=1) + np.log(20)
    prices = np.exp(sim)
    p_up, p_dn = barrier_probabilities(prices, last=20.0, up_pct=0.05, dn_pct=0.05)
    assert 0.0 <= p_up <= 1.0 and 0.0 <= p_dn <= 1.0
    # 更近的障碍位触及概率更高
    p_up_far, _ = barrier_probabilities(prices, last=20.0, up_pct=0.30, dn_pct=0.05)
    assert p_up_far < p_up


def test_ensemble_weighting():
    def _path(median, lo):
        return {'median': np.full(5, median, dtype=float),
                'q10': np.full(5, lo, dtype=float), 'q25': np.full(5, lo, dtype=float),
                'q75': np.full(5, median, dtype=float), 'q90': np.full(5, median, dtype=float)}

    paths = [_path(10.0, 9.0), _path(20.0, 19.0)]
    assert forecast_ensemble(paths, [0.1, 0.1])['median'][-1] == pytest.approx(15.0)
    assert 10.0 < forecast_ensemble(paths, [0.01, 0.2])['median'][-1] < 15.0


def test_regime_detection():
    up_data = _make_data(np.concatenate([np.linspace(10, 30, 200)]))
    reg = detect_regime_series(up_data)
    assert summarize_regime(reg)['state'] == 'up'
    dn_data = _make_data(np.linspace(30, 10, 200))
    reg_dn = detect_regime_series(dn_data)
    assert summarize_regime(reg_dn)['state'] == 'down'
    # 横盘毛刺数据 → 震荡
    chop_data = _make_data(np.tile([20.0, 20.8], 120))
    reg_chop = detect_regime_series(chop_data)
    assert summarize_regime(reg_chop)['state'] in ('chop', 'up')


def test_empirical_up_prob_uses_matching_regime_only():
    closes = np.linspace(10, 30, 200)
    regimes = np.array(['up'] * 100 + ['chop'] * 100)
    # 当前状态 'up'：只用前 100 根（全是上涨段）→ 上涨频率应为 1.0 附近
    p = empirical_up_prob(closes, regimes, 'up', horizon=5)
    assert p == pytest.approx(1.0)
    # 状态 'down'：无样本 → None
    assert empirical_up_prob(closes, regimes, 'down', horizon=5) is None


def test_walk_forward_returns_sane_metrics_with_calibration():
    rng = np.random.default_rng(11)
    data = _make_data(20 * np.exp(rng.normal(0.0005, 0.012, 300).cumsum()))
    bt = walk_forward_backtest(data['close'].values, horizon=10, data=data)
    assert bt, '足够长的数据应产生回测结果'
    for name, m in bt.items():
        assert 0.0 <= m['dir_rate'] <= 1.0
        assert 0.0 <= m['coverage'] <= 1.0
        assert m['coverage'] >= 0.3, f'{name} 区间覆盖异常偏低: {m["coverage"]}'
        assert 0.0 <= m['mape'] < 0.5
        assert m['dir_n'] >= 3
    # GBM/集成应有 P(涨) 校准指标
    for name in ('GBM', '集成'):
        assert name in bt and 'cal_gap' in bt[name]
        assert 0.0 <= bt[name]['cal_gap'] <= 1.0


def test_run_forecast_probability_outputs():
    rng = np.random.default_rng(13)
    data = _make_data(20 * np.exp(rng.normal(0.0008, 0.012, 300).cumsum()))
    res = run_forecast(data, 10, code='600000.SH', name='浦发银行')
    assert res['ok']
    assert 0.0 <= res['p_up'] <= 1.0
    assert 0.0 <= res['p_touch_up'] <= 1.0 and 0.0 <= res['p_touch_dn'] <= 1.0
    assert res['barrier_pct'] >= 0.02
    assert res['regime']['state'] in ('up', 'down', 'chop')
    assert res['code'] == '600000.SH' and res['name'] == '浦发银行'
    fig = build_forecast_figure(res)
    names = [t.name for t in fig.data]
    assert '历史收盘' in names and '集成中位预测' in names and '80%预测区间' in names
    assert '乐观情景' in names and '悲观情景' in names
    panel = build_forecast_panel(res)
    html_text = str(panel)
    assert '浦发银行' in html_text and '上涨概率' in html_text
    assert '止盈参考' in html_text and '止损参考' in html_text
    assert 'P(涨)校准差' in html_text and '准确性说明' in html_text
    assert '趋势' in html_text or '震荡' in html_text


def test_run_forecast_insufficient():
    res = run_forecast(_make_data([10, 11, 12]), 10)
    assert not res['ok'] and '不足' in res['reason']
