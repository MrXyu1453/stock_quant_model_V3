"""冒烟测试: lightgbm 可训练 + train_lgbm_rolling 端到端"""
import numpy as np
import pandas as pd


def test_lightgbm_basic_fit():
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
    X = np.random.rand(100, 3)
    y = X[:, 0] * 2
    m.fit(X, y)
    assert m.feature_importances_.sum() > 0


def test_train_lgbm_rolling_end_to_end():
    from app.factor_engine import train_lgbm_rolling
    rng = np.random.default_rng(7)
    dates = pd.date_range('2022-01-31', periods=24, freq='ME')
    codes = [f'S{i:03d}' for i in range(12)]
    f1 = pd.DataFrame(rng.normal(size=(24, 12)), index=dates, columns=codes)
    f2 = pd.DataFrame(rng.normal(size=(24, 12)), index=dates, columns=codes)
    ret = pd.DataFrame(0.5 * f1.values + rng.normal(scale=0.01, size=(24, 12)),
                       index=dates, columns=codes)
    pred, imp = train_lgbm_rolling({'f1': f1, 'f2': f2}, ret, train_months=12, step_months=1)
    assert pred is not None and not pred.empty
    assert set(imp.keys()) == {'f1', 'f2'}
    assert imp['f1'] > 0  # f1 与收益相关，重要度应更高方向上有信号


if __name__ == '__main__':
    test_lightgbm_basic_fit()
    print('test_lightgbm_basic_fit passed')
    test_train_lgbm_rolling_end_to_end()
    print('test_train_lgbm_rolling_end_to_end passed')
