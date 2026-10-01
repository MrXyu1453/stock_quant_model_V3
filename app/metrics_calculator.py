import pandas as pd
import logging

# 计算夏普比率
def calculate_sharpe_ratio(data):
    try:
        if len(data) < 2:
            logging.warning("数据不足，无法计算夏普比率")
            return 0.0
        
        returns = data['close'].pct_change()
        risk_free_rate = 0.0  # 假设无风险利率为 0
        if returns.std() == 0:
            return 0.0
        sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
        return sharpe_ratio
    except Exception as e:
        logging.error(f"计算夏普比率时出错: {e}")
        return 0.0

# 计算最大回撤
def calculate_max_drawdown(data):
    try:
        if len(data) < 2:
            logging.warning("数据不足，无法计算最大回撤")
            return 0.0
        
        cumulative_returns = (1 + data['close'].pct_change()).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        return max_drawdown
    except Exception as e:
        logging.error(f"计算最大回撤时出错: {e}")
        return 0.0

# 模拟交易并计算累计收益率
def simulate_trades(data, signals, initial_capital=100000.0, transaction_fee=0.0005):
    try:
        if signals is None or len(signals) == 0:
            return 0.0
            
        # 处理数据和信号的缺失值
        data = data.dropna(subset=['close'])
        signals = signals.dropna()

        # 检查处理后的数据和信号是否为空
        if data.empty or signals.empty:
            logging.error("处理后的数据或信号为空，无法进行模拟交易。")
            return 0.0

        # 检查索引是否匹配
        if not data.index.equals(signals.index):
            common_index = data.index.intersection(signals.index)
            data = data.loc[common_index]
            signals = signals.loc[common_index]

            # 再次检查处理后的数据和信号是否为空
            if data.empty or signals.empty:
                logging.error("处理后的数据或信号为空，无法进行模拟交易。")
                return 0.0

        if len(data) == 0:
            return 0.0

        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions['stock'] = signals['signal']
        portfolio = positions.multiply(data['close'], axis=0)
        pos_diff = positions.diff()

        cash = initial_capital
        holdings = 0
        portfolio_values = []

        for i in range(len(data)):
            if i >= len(pos_diff):
                continue
            
            if pos_diff.iloc[i]['stock'] > 0:  # 买入
                if data.iloc[i]['close'] <= 0:
                    portfolio_values.append(cash + holdings * data.iloc[i]['close'] if i < len(data) else cash)
                    continue
                quantity = cash // (data.iloc[i]['close'] * (1 + transaction_fee))
                cost = quantity * data.iloc[i]['close'] * (1 + transaction_fee)
                cash -= cost
                holdings += quantity
            elif pos_diff.iloc[i]['stock'] < 0:  # 卖出
                if holdings > 0:
                    income = holdings * data.iloc[i]['close'] * (1 - transaction_fee)
                    cash += income
                    holdings = 0

            portfolio_value = cash + holdings * data.iloc[i]['close']
            portfolio_values.append(portfolio_value)

        portfolio['total'] = portfolio_values
        portfolio['returns'] = portfolio['total'].pct_change()

        # 检查是否存在除零错误
        if len(portfolio) == 0 or portfolio['total'].iloc[0] == 0:
            logging.error("初始总资产为零，无法计算累计收益率。")
            return 0.0

        if len(portfolio) == 0:
            return 0.0

        cumulative_return = (portfolio['total'].iloc[-1] / initial_capital) - 1
        return cumulative_return
    except IndexError as ie:
        logging.error(f"索引错误: {ie}")
    except ZeroDivisionError as zde:
        logging.error(f"除零错误: {zde}")
    except Exception as e:
        import traceback
        logging.error(f"模拟交易计算累计收益率时发生未知错误: {e}")
        logging.error(traceback.format_exc())
    return 0.0
