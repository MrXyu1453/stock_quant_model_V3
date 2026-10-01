"""
因子计算引擎：复用 data_manager 数据读取，提供因子预处理、IC计算、分层回测、LGBM训练
"""
import numpy as np
import pandas as pd
import logging
import os
from scipy import stats
from .data_manager import get_and_process_data, batch_get_data, get_stock_name
from .metrics_calculator import calculate_sharpe_ratio, calculate_max_drawdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==================== 因子预处理 ====================

def winsorize_mad(series, n=5):
    """MAD 去极值：中位数 +/- n * MAD；MAD=0 时（过半样本同值）回退用标准差兜底"""
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        std = series.std()
        if std == 0 or np.isnan(std):
            return series
        return series.clip(median - n * std, median + n * std)
    upper = median + n * mad
    lower = median - n * mad
    return series.clip(lower, upper)


def standardize_cross_section(factor_df):
    """截面 Z-score 标准化 (index=date, columns=stock_code)"""
    return factor_df.subtract(factor_df.mean(axis=1), axis=0).div(factor_df.std(axis=1), axis=0)


def neutralize(factor_df, industry_df, mktcap_df):
    """行业+市值中性化：回归取残差"""
    # 对齐索引
    common_idx = factor_df.index.intersection(industry_df.index).intersection(mktcap_df.index)
    factor_df, industry_df, mktcap_df = factor_df.loc[common_idx], industry_df.loc[common_idx], mktcap_df.loc[common_idx]
    common_cols = factor_df.columns & industry_df.columns & mktcap_df.columns
    if len(common_cols) < 3:
        return factor_df
    factor_df = factor_df[common_cols]
    industry_dummies = pd.get_dummies(industry_df[common_cols].stack()).unstack()
    neutralized = pd.DataFrame(index=factor_df.index, columns=common_cols, dtype=float)
    for date in factor_df.index:
        y = factor_df.loc[date, common_cols]
        X = pd.DataFrame(index=common_cols)
        X['mktcap'] = mktcap_df.loc[date, common_cols]
        if date in industry_dummies.index.get_level_values(0):
            ind = industry_dummies.xs(date, level=0)
            X = pd.concat([X, ind.reindex(common_cols).fillna(0)], axis=1)
        X = X.dropna()
        valid = X.index.intersection(y.dropna().index)
        if len(valid) < 10:
            neutralized.loc[date, common_cols] = y
            continue
        X_v, y_v = X.loc[valid], y.loc[valid]
        try:
            from numpy.linalg import lstsq
            beta, _, _, _ = lstsq(X_v.values, y_v.values)
            residuals = y_v.values - X_v.values @ beta
            neutralized.loc[date, valid] = residuals
        except:
            neutralized.loc[date, valid] = y_v.values
    return neutralized


def preprocess_factor(factor_df, do_mad=True, do_zscore=True, industry_df=None, mktcap_df=None):
    """因子预处理流水线：去极值 -> Z标准化 -> 行业市值中性"""
    result = factor_df.copy()
    if do_mad:
        result = result.apply(lambda col: winsorize_mad(col.dropna()), axis=0)
    if do_zscore and len(result) > 1:
        result = standardize_cross_section(result)
    if industry_df is not None and mktcap_df is not None:
        result = neutralize(result, industry_df, mktcap_df)
    return result


# ==================== IC 计算 ====================

def calculate_rank_ic(factor_df, ret_df):
    """计算月度 Rank IC 序列
    factor_df: index=date, columns=stock_code
    ret_df:    index=date, columns=stock_code (未来收益)
    """
    common_dates = factor_df.index.intersection(ret_df.index)
    ic_series = pd.Series(index=common_dates, dtype=float)
    for date in common_dates:
        f = factor_df.loc[date].dropna()
        r = ret_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            ic_series[date] = np.nan
            continue
        ic_series[date] = stats.spearmanr(f[common], r[common])[0]
    return ic_series.dropna()


def calculate_ic_statistics(ic_series):
    """IC 统计：均值、标准差、ICIR、胜率"""
    if len(ic_series) == 0:
        return {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'ir': 0, 'win_rate': 0}
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    win_rate = (ic_series > 0).sum() / len(ic_series)
    ir = icir
    return {'ic_mean': round(ic_mean, 4), 'ic_std': round(ic_std, 4), 'icir': round(icir, 4),
            'ir': round(ir, 4), 'win_rate': round(win_rate, 4)}


# ==================== 分层回测 ====================

def layered_backtest(factor_df, ret_df, n_groups=10):
    """分层回测：按因子值分 N 组，返回各组累计收益序列
    返回: dict {group_idx: cumulative_return_series}, group_returns_df
    """
    common_dates = factor_df.index.intersection(ret_df.index)
    group_nav = {i: pd.Series(index=common_dates, dtype=float) for i in range(n_groups)}
    for idx, date in enumerate(common_dates):
        f = factor_df.loc[date].dropna()
        r = ret_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups:
            for i in range(n_groups):
                # 首日净值置 1.0，避免 0 被后续逐日相乘永久传播
                group_nav[i].iloc[idx] = 1.0 if idx == 0 else group_nav[i].iloc[idx - 1]
            continue
        labels = pd.qcut(f[common], n_groups, labels=False, duplicates='drop')
        for i in range(n_groups):
            mask = labels == i
            if mask.sum() == 0:
                group_nav[i].iloc[idx] = group_nav[i].iloc[idx - 1] if idx > 0 else 1.0
            else:
                avg_ret = r[common][mask].mean()
                prev = group_nav[i].iloc[idx - 1] if idx > 0 else 1.0
                group_nav[i].iloc[idx] = prev * (1 + avg_ret)

    return group_nav


def compute_group_returns(group_nav, n_groups=10):
    """计算分组的年化收益、夏普、最大回撤"""
    results = []
    for i in range(n_groups):
        nav = group_nav[i].dropna()
        if len(nav) < 2:
            results.append({'group': i + 1, 'annual_return': 0, 'sharpe': 0, 'max_dd': 0})
            continue
        returns = nav.pct_change().dropna()
        annual_return = returns.mean() * 252
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = (nav / nav.cummax() - 1).min()
        results.append({'group': i + 1, 'annual_return': round(annual_return, 4),
                        'sharpe': round(sharpe, 4), 'max_dd': round(max_dd, 4)})
    return pd.DataFrame(results)


# ==================== LGBM 多因子训练 ====================

def train_lgbm_rolling(factor_data, ret_data, train_months=60, step_months=1):
    """滚动时序训练（lightgbm 优先，缺失时回退 sklearn GradientBoosting）
    factor_data: dict {factor_name: DataFrame(index=date, columns=stock_code)}
    ret_data: DataFrame (index=date, columns=stock_code) 未来收益
    返回: pred_scores DataFrame, feature_importance dict
    """
    try:
        import lightgbm  # noqa: F401
        _lgbm_available = True
    except ImportError:
        _lgbm_available = False
        logging.warning("lightgbm 未安装，回退使用 sklearn GradientBoostingRegressor")

    def _make_model():
        if _lgbm_available:
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=100, max_depth=5, num_leaves=31,
                                     verbose=-1, random_state=42)
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

    # 合并所有因子 -> 长表
    all_dates = sorted(set().union(*[f.index for f in factor_data.values()]))
    if len(all_dates) < train_months + 1:
        logging.warning(f"数据不足: {len(all_dates)} < {train_months + 1}")
        return None, None

    factor_names = list(factor_data.keys())
    pred_scores = pd.DataFrame()
    all_importances = {name: [] for name in factor_names}

    for i in range(train_months, len(all_dates), step_months):
        train_dates = all_dates[i - train_months:i]
        test_dates = all_dates[i:i + step_months]
        if len(train_dates) < train_months or len(test_dates) == 0:
            continue

        X_train_rows, y_train_rows = [], []
        X_test_rows, test_indices = [], []
        for date in train_dates:
            for code in factor_data[factor_names[0]].columns:
                row = [factor_data[name].loc[date, code] if date in factor_data[name].index and code in factor_data[name].columns else np.nan for name in factor_names]
                label = ret_data.loc[date, code] if date in ret_data.index and code in ret_data.columns else np.nan
                if not np.isnan(row).any() and not np.isnan(label):
                    X_train_rows.append(row)
                    y_train_rows.append(label)

        for date in test_dates:
            for code in factor_data[factor_names[0]].columns:
                row = [factor_data[name].loc[date, code] if date in factor_data[name].index and code in factor_data[name].columns else np.nan for name in factor_names]
                if not np.isnan(row).any():
                    X_test_rows.append(row)
                    test_indices.append((date, code))

        if len(X_train_rows) < 50:
            continue

        model = _make_model()
        model.fit(np.array(X_train_rows), np.array(y_train_rows))

        for j, name in enumerate(factor_names):
            all_importances[name].append(model.feature_importances_[j])

        if len(X_test_rows) > 0:
            preds = model.predict(np.array(X_test_rows))
            for (date, code), pred in zip(test_indices, preds):
                if date not in pred_scores.index:
                    pred_scores.loc[date, code] = np.nan
                pred_scores.loc[date, code] = pred

    # 汇总特征重要度
    feat_importance = {name: round(float(np.mean(vals)), 4) if vals else 0 for name, vals in all_importances.items()}
    return pred_scores, feat_importance


# ==================== 合成得分 ====================

def compute_icir_weighted_score(factor_data, ic_stats):
    """ICIR 加权合成总分，ICIR 全 ≤ 0 时回退到等权"""
    _icirs = {name: stat['icir'] for name, stat in ic_stats.items()}
    total_weight = sum(max(v, 0) for v in _icirs.values())

    if total_weight > 0:
        weights = {name: max(v, 0) / total_weight for name, v in _icirs.items()}
    else:
        # 所有因子 ICIR ≤ 0：回退到等权，仍可继续验证
        n = len(factor_data)
        weights = {name: 1.0 / n for name in factor_data}

    composite = pd.DataFrame()
    for name, factor in factor_data.items():
        w = weights.get(name, 0)
        if w == 0:
            continue
        weighted = factor.fillna(0) * w
        if composite.empty:
            composite = weighted
        else:
            composite = composite.add(weighted, fill_value=0)
    return composite, weights


# ==================== 数据提供者 ====================

class FactorDataProvider:
    """因子数据提供者：批量加载、缓存、公式求值"""

    def __init__(self, stock_codes, start_date, end_date, source=None):
        self.stock_codes = stock_codes
        self.start_date = start_date
        self.end_date = end_date
        self.source = source
        self.data_cache = {}  # {code: DataFrame}
        self._loaded = False

    def load_data(self, progress_callback=None):
        """批量加载日线数据"""
        results = batch_get_data(self.stock_codes, self.start_date, self.end_date)
        for code, df in results.items():
            if df is not None and not df.empty:
                self.data_cache[code] = df.sort_values('trade_date')
                self.data_cache[code].set_index('trade_date', inplace=True)
        self._loaded = True
        if progress_callback:
            progress_callback(len(self.data_cache), len(self.stock_codes))

    def get_date_index(self):
        """获取统一日期索引"""
        all_dates = set()
        for df in self.data_cache.values():
            all_dates.update(df.index)
        return pd.DatetimeIndex(sorted(all_dates))

    def eval_expression(self, expr):
        """求值因子表达式，返回 (date, stock_code) 的宽表"""
        dates = self.get_date_index()
        result = pd.DataFrame(index=dates, columns=self.stock_codes, dtype=float)
        for code in self.stock_codes:
            if code not in self.data_cache:
                continue
            df = self.data_cache[code]
            # 构建 eval 命名空间：数据用 pd.Series 保留 shift/rolling 等方法
            ns = {}
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                if col in df.columns:
                    ns[col] = df[col]
            # MA 滚动均值辅助函数
            def _ma(s, win):
                return s.rolling(window=int(win)).mean()
            ns['MA'] = _ma
            ns['__builtins__'] = {'abs': abs}  # 允许 abs() 内置函数
            try:
                val = eval(expr, ns)
                if isinstance(val, (pd.Series, np.ndarray)):
                    result.loc[df.index, code] = val
                else:
                    result.loc[df.index, code] = val
            except Exception as e:
                logging.warning(f"公式求值失败 code={code}: {e}")
        return result.dropna(how='all')

    def compute_future_returns(self, horizon=5):
        """计算未来 N 日收益率 (日期对齐到因子日期)"""
        ret_df = pd.DataFrame(index=self.get_date_index(), columns=self.stock_codes, dtype=float)
        for code in self.stock_codes:
            if code not in self.data_cache:
                continue
            df = self.data_cache[code]
            if 'close' not in df.columns:
                continue
            fwd = df['close'].shift(-horizon) / df['close'] - 1
            ret_df.loc[df.index, code] = fwd.values
        return ret_df.dropna(how='all')


# ==================== 组合回测 ====================

def portfolio_backtest(score_df, ret_df, top_pct=0.2, single_stock_cap=0.1):
    """按得分选股回测
    score_df: index=date, columns=stock_code (分数)
    ret_df:   index=date, columns=stock_code (同期收益)
    返回: nav_series, stats_dict
    """
    common_dates = score_df.index.intersection(ret_df.index)
    nav_series = pd.Series(index=common_dates, dtype=float)
    nav_series.iloc[0] = 1.0
    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i - 1]
        scores = score_df.loc[date].dropna()
        if len(scores) < 5:
            nav_series.iloc[i] = nav_series.iloc[i - 1]
            continue
        n_select = max(int(len(scores) * top_pct), 5)
        selected = scores.nlargest(n_select).index
        n_stocks = len(selected)
        weight = min(1.0 / n_stocks, single_stock_cap) if n_stocks > 0 else 0

        rets = ret_df.loc[date, selected].dropna()
        if len(rets) == 0:
            nav_series.iloc[i] = nav_series.iloc[i - 1]
        else:
            avg_ret = rets.mean()
            nav_series.iloc[i] = nav_series.iloc[i - 1] * (1 + avg_ret)

    returns = nav_series.pct_change().dropna()
    if len(returns) < 2:
        return nav_series, {'annual_return': 0, 'sharpe': 0, 'max_dd': 0}

    annual_return = returns.mean() * 252
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    max_dd = (nav_series / nav_series.cummax() - 1).min()
    stats = {
        'annual_return': round(annual_return, 4),
        'sharpe': round(sharpe, 4),
        'max_dd': round(max_dd, 4),
        'nav': nav_series.tolist(),
        'dates': [str(d).split('T')[0] for d in nav_series.index.tolist()]
    }
    return nav_series, stats
