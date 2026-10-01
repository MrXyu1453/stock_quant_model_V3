"""因子引擎单元测试（预处理 / IC / 分层回测 / 合成得分 / 组合回测）"""
import numpy as np
import pandas as pd
import pytest

from app.factor_engine import (
    winsorize_mad,
    standardize_cross_section,
    preprocess_factor,
    calculate_rank_ic,
    calculate_ic_statistics,
    layered_backtest,
    compute_group_returns,
    compute_icir_weighted_score,
    portfolio_backtest,
)


def make_factor_panel(n_dates=40, n_stocks=15, seed=5, freq='B'):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n_dates, freq=freq)
    codes = [f'S{i:03d}' for i in range(n_stocks)]
    return pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=codes)


# ==================== 预处理 ====================

class TestWinsorizeMad:
    def test_normal_mad_clips_outliers(self):
        rng = np.random.default_rng(42)
        s = pd.Series(np.concatenate([rng.normal(0, 1, 98), [50.0, -50.0]]))
        clipped = winsorize_mad(s)
        assert clipped.abs().max() < 50.0

    def test_zero_mad_falls_back_to_std(self):
        # 现实场景：过半样本因子值相同（如停牌/一字板），残留一个 -9999 型缺失标记
        s = pd.Series([1.0] * 30 + np.linspace(0.9, 1.1, 19).tolist() + [-9999.0])
        clipped = winsorize_mad(s)
        assert clipped.min() > -9999.0  # 缺失标记被剪掉

    def test_extreme_minority_outliers_not_clipped_but_finite(self):
        # 离群值占比过高（2/22）时任何统计方法都无法剪除，只需保证不抛异常、结果有限
        s = pd.Series([1.0] * 20 + [1e9, -1e9])
        clipped = winsorize_mad(s)
        assert np.isfinite(clipped).all()

    def test_zero_mad_returns_original(self):
        s = pd.Series([5.0] * 10)
        pd.testing.assert_series_equal(winsorize_mad(s), s)


class TestStandardize:
    def test_row_mean_std(self):
        f = make_factor_panel()
        z = standardize_cross_section(f)
        assert z.mean(axis=1).abs().max() < 1e-9
        assert (z.std(axis=1) - 1).abs().max() < 1e-9


class TestPreprocess:
    def test_pipeline_changes_values(self):
        f = make_factor_panel()
        out = preprocess_factor(f)
        assert out.shape == f.dropna(how='all').shape
        # 截面标准化后每行均值接近 0
        assert out.mean(axis=1).abs().max() < 1e-9


# ==================== IC ====================

class TestRankIC:
    def test_perfect_factor_high_ic(self):
        f = make_factor_panel(seed=1)
        ret = f * 0.01  # 因子与未来收益完全一致
        ic = calculate_rank_ic(f, ret)
        assert len(ic) > 0
        assert ic.mean() > 0.95

    def test_random_factor_low_ic(self):
        f = make_factor_panel(seed=1)
        ret = make_factor_panel(seed=99) * 0.01
        ic = calculate_rank_ic(f, ret)
        assert abs(ic.mean()) < 0.5

    def test_stats_fields(self):
        f = make_factor_panel(seed=1)
        ret = f * 0.01
        ic = calculate_rank_ic(f, ret)
        stats = calculate_ic_statistics(ic)
        for key in ('ic_mean', 'ic_std', 'icir', 'ir', 'win_rate'):
            assert key in stats
        assert stats['win_rate'] > 0.9

    def test_empty_series_stats(self):
        stats = calculate_ic_statistics(pd.Series(dtype=float))
        assert stats['ic_mean'] == 0


# ==================== 分层回测 ====================

class TestLayeredBacktest:
    def test_top_group_beats_bottom(self):
        f = make_factor_panel(n_dates=60, n_stocks=30, seed=2)
        ret = f * 0.01  # 因子决定收益
        nav = layered_backtest(f, ret, n_groups=5)
        stats = compute_group_returns(nav, n_groups=5)
        top = stats[stats['group'] == 5]['annual_return'].iloc[0]
        bottom = stats[stats['group'] == 1]['annual_return'].iloc[0]
        assert top > bottom

    def test_nav_stays_positive(self):
        # 首日即使分组成功净值也不应被置 0（0 会被逐日相乘永久传播）
        f = make_factor_panel(n_dates=30, n_stocks=20)
        ret = f * 0.01
        nav = layered_backtest(f, ret, n_groups=5)
        for i in range(5):
            s = nav[i].dropna()
            assert (s > 0).all()
            assert np.isfinite(s).all()


# ==================== 合成得分 ====================

class TestIcirWeighted:
    def test_positive_icir_gets_weight(self):
        f1, f2 = make_factor_panel(seed=1), make_factor_panel(seed=2)
        ic_stats = {'f1': {'icir': 0.5}, 'f2': {'icir': 0.25}}
        score, weights = compute_icir_weighted_score({'f1': f1, 'f2': f2}, ic_stats)
        assert weights['f1'] == pytest.approx(2 / 3)
        assert weights['f2'] == pytest.approx(1 / 3)
        assert not score.empty

    def test_all_nonpositive_falls_back_equal(self):
        f1, f2 = make_factor_panel(seed=1), make_factor_panel(seed=2)
        ic_stats = {'f1': {'icir': -0.5}, 'f2': {'icir': 0.0}}
        score, weights = compute_icir_weighted_score({'f1': f1, 'f2': f2}, ic_stats)
        assert weights == {'f1': 0.5, 'f2': 0.5}


# ==================== 组合回测 ====================

class TestPortfolioBacktest:
    def test_nav_starts_and_grows(self):
        score = make_factor_panel(n_dates=50, n_stocks=20, seed=3)
        ret = score * 0.01  # 高分高收益
        nav, stats = portfolio_backtest(score, ret, top_pct=0.2)
        assert nav.dropna().iloc[0] == pytest.approx(1.0)
        assert stats['annual_return'] > 0
        assert stats['max_dd'] <= 0
        assert len(stats['dates']) == len(nav.dropna())

    def test_insufficient_scores_holds_nav(self):
        score = make_factor_panel(n_dates=20, n_stocks=3)  # 少于 5 只
        ret = score * 0.01
        nav, stats = portfolio_backtest(score, ret)
        # 全程持有现金，净值恒为 1
        assert (nav.dropna() == 1.0).all()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
