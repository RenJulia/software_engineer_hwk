# -*- coding: utf-8 -*-
"""
中证1000多因子策略 - 策略回测模块
基于合成信号进行多头选股回测
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 性能优化
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class StrategyBacktester:
    """策略回测器类"""
    
    def __init__(self, signal_data, return_data, benchmark_data=None, 
                 price_data=None, constituent_manager=None):
        """
        初始化策略回测器
        
        Parameters:
        -----------
        signal_data : dict, 信号数据字典，key为信号名，value为DataFrame（股票×日期）
        return_data : pd.DataFrame, 收益率数据（股票×日期）
        benchmark_data : pd.DataFrame, 基准指数数据（可选），包含日期和收益率
        price_data : pd.DataFrame, 价格数据（用于计算换手率等）
        constituent_manager : ConstituentManager, 成分股管理器
        """
        self.signal_data = signal_data
        self.return_data = return_data
        self.benchmark_data = benchmark_data
        self.price_data = price_data
        self.constituent_manager = constituent_manager
        
        # 对齐数据
        self._align_data()
    
    def _align_data(self):
        """对齐信号数据和收益率数据"""
        print("对齐信号数据和收益率数据...")
        
        # 获取所有信号的公共日期和股票
        all_dates = set(self.return_data.columns)
        all_stocks = set(self.return_data.index)
        
        for signal_name, signal_df in self.signal_data.items():
            if signal_df is None or signal_df.empty:
                continue
            all_dates = all_dates & set(signal_df.columns)
            all_stocks = all_stocks & set(signal_df.index)
        
        # 对齐收益率数据
        self.return_data = self.return_data.loc[list(all_stocks), list(sorted(all_dates))]
        
        # 对齐信号数据
        aligned_signals = {}
        for signal_name, signal_df in self.signal_data.items():
            if signal_df is None or signal_df.empty:
                aligned_signals[signal_name] = pd.DataFrame(
                    index=list(all_stocks),
                    columns=list(sorted(all_dates))
                )
                aligned_signals[signal_name][:] = np.nan
            else:
                aligned = signal_df.loc[list(all_stocks), list(sorted(all_dates))]
                aligned_signals[signal_name] = aligned
        
        self.signal_data = aligned_signals
        print(f"✅ 数据对齐完成，股票数: {len(all_stocks)}, 日期数: {len(all_dates)}")
    
    def analyze_single_signal(self, signal_name, save_dir='./results/'):
        """
        对单个信号进行单因子分析（调用factor_evaluation.py中的函数）
        
        Parameters:
        -----------
        signal_name : str, 信号名称
        save_dir : str, 结果保存目录
        
        Returns:
        --------
        dict : 分析结果
        """
        if signal_name not in self.signal_data:
            print(f"⚠️  信号 {signal_name} 不存在")
            return {}
        
        print(f"\n{'='*60}")
        print(f"单信号分析: {signal_name}")
        print(f"{'='*60}")
        
        # 使用factor_evaluation进行单因子分析
        from factor_evaluation import FactorEvaluator
        
        # 将信号作为因子进行分析
        factor_data = {signal_name: self.signal_data[signal_name]}
        
        evaluator = FactorEvaluator(
            factor_data=factor_data,
            return_data=self.return_data,
            price_data=self.price_data,
            constituent_manager=self.constituent_manager
        )
        
        # 1. 计算IC和IR
        print("\n1. 计算IC和IR...")
        ic_stats = evaluator.calculate_IR(signal_name, forward_period=1)
        rankic_stats = evaluator.calculate_RankIC(signal_name, forward_period=1)
        
        # 2. 分层回测
        print("\n2. 进行分层回测...")
        nav_df = evaluator.layer_backtest_single_factor(
            factor_name=signal_name,
            layers=5,
            freq=1,  # 日度调仓
            fee=0.002
        )
        
        # 2.5. 绘制并保存分层回测图
        if not nav_df.empty:
            os.makedirs(save_dir, exist_ok=True)
            layer_plot_path = os.path.join(save_dir, f'{signal_name}_layer_backtest.png')
            evaluator.plot_layer_returns(
                factor_name=signal_name,
                layers=5,
                freq=1,
                save_path=layer_plot_path
            )
        
        # 3. 检查单调性
        print("\n3. 检查单调性...")
        monotonicity_result = evaluator.check_monotonicity(nav_df, check_layers=[1, 3, 5])
        
        # 4. 计算多空收益
        print("\n4. 计算多空收益...")
        direction = monotonicity_result.get('direction', 'positive')
        long_short_result = evaluator.calculate_long_short_return(nav_df, direction=direction)
        
        # 5. 相关性分析（与其他信号）
        print("\n5. 相关性分析...")
        correlation_info = {}
        if len(self.signal_data) > 1:
            other_signals = [s for s in self.signal_data.keys() if s != signal_name]
            if len(other_signals) > 0:
                # 计算与其他信号的相关性
                all_factors = {signal_name: self.signal_data[signal_name]}
                for other_signal in other_signals[:3]:  # 只分析前3个其他信号
                    all_factors[other_signal] = self.signal_data[other_signal]
                
                evaluator_all = FactorEvaluator(
                    factor_data=all_factors,
                    return_data=self.return_data,
                    constituent_manager=self.constituent_manager
                )
                corr_matrix = evaluator_all.calculate_factor_correlation()
                
                if not corr_matrix.empty and signal_name in corr_matrix.index:
                    correlation_info = corr_matrix.loc[signal_name].to_dict()
        
        # 汇总结果
        result = {
            'IC_stats': ic_stats,
            'RankIC': rankic_stats,
            'nav_df': nav_df,
            'monotonicity': monotonicity_result,
            'long_short_result': long_short_result,
            'correlation': correlation_info
        }
        
        # 保存结果
        os.makedirs(save_dir, exist_ok=True)
        summary_file = os.path.join(save_dir, f'{signal_name}_signal_analysis_summary.csv')
        
        summary_data = {
            'IC_Mean': [ic_stats.get('IC_Mean', np.nan)],
            'IC_Std': [ic_stats.get('IC_Std', np.nan)],
            'IR': [ic_stats.get('IR', np.nan)],
            'IC_WinRate': [ic_stats.get('IC_WinRate', np.nan)],
            'Is_Monotonic': [monotonicity_result.get('is_monotonic', False)],
            'Direction': [monotonicity_result.get('direction', None)],
            'Long_Short_Return': [long_short_result.get('total_return', np.nan)],
            'Long_Short_Annual_Return': [long_short_result.get('annual_return', np.nan)],
            'Long_Short_Sharpe': [long_short_result.get('sharpe_ratio', np.nan)]
        }
        
        summary_df = pd.DataFrame(summary_data, index=[signal_name])
        summary_df.to_csv(summary_file, encoding='utf-8-sig')
        print(f"\n✅ 分析结果已保存: {summary_file}")
        
        return result
    
    def long_only_backtest(self, signal_name, top_pct=0.1, fee=0.002, 
                          slippage=0.001, rebalance_freq=1, date_range=None):
        """
        多头选股回测（日度调仓）
        
        Parameters:
        -----------
        signal_name : str, 信号名称
        top_pct : float, 选取头部股票的比例（例如0.1表示10%）
        fee : float, 交易费率（双边，默认0.002即0.2%）
        slippage : float, 滑点（默认0.001即0.1%）
        rebalance_freq : int, 调仓频率（默认1，即每日调仓）
        date_range : tuple, (start_date, end_date) 可选，指定回测日期范围
        
        Returns:
        --------
        dict : 回测结果
            - nav_series: pd.Series, 净值序列
            - weight_df: pd.DataFrame, 持仓权重（股票×日期）
            - turnover_series: pd.Series, 换手率序列
            - metrics: dict, 性能指标
        """
        if signal_name not in self.signal_data:
            print(f"⚠️  信号 {signal_name} 不存在")
            return {}
        
        signal_df = self.signal_data[signal_name]
        dates = sorted(signal_df.columns)
        
        # 如果指定了日期范围，只使用该范围内的日期
        if date_range is not None:
            start_date, end_date = date_range
            dates = [d for d in dates if start_date <= d <= end_date]
            if len(dates) == 0:
                print(f"⚠️  指定日期范围内没有数据")
                return {}
            signal_df = signal_df[dates]
        
        stocks = sorted(signal_df.index)
        
        # 初始化
        nav_series = pd.Series(index=dates, dtype=float)
        nav_series.iloc[0] = 1.0
        
        weight_df = pd.DataFrame(index=stocks, columns=dates, dtype=float)
        weight_df[:] = 0.0
        
        turnover_series = pd.Series(index=dates, dtype=float)
        turnover_series.iloc[0] = 0.0
        
        current_weight = pd.Series(0.0, index=stocks)
        
        # 日度回测
        # 注意：信号预测t+1日的收益率，所以应该：
        # - 在t日使用t日的信号值选择股票（建仓）
        # - 在t+1日使用t+1日的收益率计算收益
        iterator = tqdm(enumerate(dates), total=len(dates), desc=f"回测{signal_name}") if TQDM_AVAILABLE else enumerate(dates)
        
        for i, date in iterator:
            if i == 0:
                # 第一天：使用第0日的信号值选择股票，但还没有收益
                # 获取第0日的信号值
                signal_values = signal_df[date].dropna()
                
                # 如果提供了成分股管理器，只使用成分股
                if self.constituent_manager is not None:
                    constituents = self.constituent_manager.get_constituents_by_date(date)
                    constituents_set = set(constituents)
                    signal_values = signal_values[signal_values.index.isin(constituents_set)]
                
                if len(signal_values) > 0:
                    # 选取头部top_pct比例的股票
                    n_select = max(1, int(len(signal_values) * top_pct))
                    top_stocks = signal_values.nlargest(n_select).index.tolist()
                    
                    # 等权重分配
                    new_weight = pd.Series(0.0, index=stocks)
                    if len(top_stocks) > 0:
                        new_weight[top_stocks] = 1.0 / len(top_stocks)
                    
                    current_weight = new_weight
                    turnover_series.iloc[i] = 0.0  # 第一天没有换手
                else:
                    current_weight = pd.Series(0.0, index=stocks)
                    turnover_series.iloc[i] = 0.0
                
                weight_df[date] = current_weight
                nav_series.iloc[i] = 1.0  # 初始净值
                continue
            
            prev_date = dates[i-1]
            prev_nav = nav_series.iloc[i-1]
            
            # 计算当日收益（使用上日的持仓和当日的收益率）
            # 因为信号预测t+1日收益，所以t日建立的仓位在t+1日获得收益
            if date in self.return_data.columns:
                # 确保股票对齐
                aligned_return = self.return_data.loc[stocks, date].fillna(0)
                daily_return = (current_weight * aligned_return).sum()
                nav_series.iloc[i] = prev_nav * (1 + daily_return)
            else:
                nav_series.iloc[i] = prev_nav
            
            # 检查是否需要调仓（在获得收益后调仓）
            if i % rebalance_freq == 0:
                # 获取当期信号值（用于下一期的持仓）
                signal_values = signal_df[date].dropna()
                
                # 如果提供了成分股管理器，只使用成分股
                if self.constituent_manager is not None:
                    constituents = self.constituent_manager.get_constituents_by_date(date)
                    constituents_set = set(constituents)
                    signal_values = signal_values[signal_values.index.isin(constituents_set)]
                
                if len(signal_values) == 0:
                    # 如果没有可用股票，保持上期权重
                    new_weight = current_weight.copy()
                    turnover_series.iloc[i] = 0.0
                else:
                    # 选取头部top_pct比例的股票
                    n_select = max(1, int(len(signal_values) * top_pct))
                    top_stocks = signal_values.nlargest(n_select).index.tolist()
                    
                    # 等权重分配
                    new_weight = pd.Series(0.0, index=stocks)
                    if len(top_stocks) > 0:
                        new_weight[top_stocks] = 1.0 / len(top_stocks)
                    
                    # 计算换手率（双边）
                    turnover = (new_weight - current_weight).abs().sum()
                    turnover_series.iloc[i] = turnover
                    
                    # 扣除交易成本（在调仓日扣除，从当前净值扣除）
                    total_cost = (fee + slippage) * turnover
                    nav_series.iloc[i] = nav_series.iloc[i] * (1 - total_cost)
                
                current_weight = new_weight
            else:
                # 非调仓日，保持上期权重，换手率为0
                turnover_series.iloc[i] = 0.0
            
            weight_df[date] = current_weight
        
        # 准备基准数据（如果可用）
        benchmark_nav = None
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
        
        # 计算性能指标
        metrics = self._calculate_metrics(nav_series, turnover_series, benchmark_nav)
        
        return {
            'nav_series': nav_series,
            'weight_df': weight_df,
            'turnover_series': turnover_series,
            'metrics': metrics
        }
    
    def layer5_backtest(self, signal_name, fee=0.002, 
                        slippage=0.001, rebalance_freq=1, date_range=None):
        """
        基于分层回测第五层选股逻辑的多头选股回测
        
        第五层选股逻辑：选择因子值在80%-100%分位数区间的股票（即因子值最高的20%股票）
        这对应分层回测中第五层的选股方法，使得回测结果更接近分层回测中第五层的表现
        
        Parameters:
        -----------
        signal_name : str, 信号名称
        fee : float, 交易费率（双边，默认0.002即0.2%）
        slippage : float, 滑点（默认0.001即0.1%）
        rebalance_freq : int, 调仓频率（默认1，即每日调仓）
        date_range : tuple, (start_date, end_date) 可选，指定回测日期范围
        
        Returns:
        --------
        dict : 回测结果
            - nav_series: pd.Series, 净值序列
            - weight_df: pd.DataFrame, 持仓权重（股票×日期）
            - turnover_series: pd.Series, 换手率序列
            - metrics: dict, 性能指标
        """
        if signal_name not in self.signal_data:
            print(f"⚠️  信号 {signal_name} 不存在")
            return {}
        
        signal_df = self.signal_data[signal_name]
        dates = sorted(signal_df.columns)
        
        # 如果指定了日期范围，只使用该范围内的日期
        if date_range is not None:
            start_date, end_date = date_range
            dates = [d for d in dates if start_date <= d <= end_date]
            if len(dates) == 0:
                print(f"⚠️  指定日期范围内没有数据")
                return {}
            signal_df = signal_df[dates]
        
        stocks = sorted(signal_df.index)
        
        # 初始化
        nav_series = pd.Series(index=dates, dtype=float)
        nav_series.iloc[0] = 1.0
        
        weight_df = pd.DataFrame(index=stocks, columns=dates, dtype=float)
        weight_df[:] = 0.0
        
        turnover_series = pd.Series(index=dates, dtype=float)
        turnover_series.iloc[0] = 0.0
        
        current_weight = pd.Series(0.0, index=stocks)
        
        # 日度回测
        # 注意：信号预测t+1日的收益率，所以应该：
        # - 在t日使用t日的信号值选择股票（建仓）
        # - 在t+1日使用t+1日的收益率计算收益
        iterator = tqdm(enumerate(dates), total=len(dates), desc=f"回测{signal_name}(第五层选股)") if TQDM_AVAILABLE else enumerate(dates)
        
        for i, date in iterator:
            if i == 0:
                # 第一天：使用第0日的信号值选择股票，但还没有收益
                signal_values = signal_df[date].dropna()
                
                # 如果提供了成分股管理器，只使用成分股
                if self.constituent_manager is not None:
                    constituents = self.constituent_manager.get_constituents_by_date(date)
                    constituents_set = set(constituents)
                    signal_values = signal_values[signal_values.index.isin(constituents_set)]
                
                if len(signal_values) > 0:
                    # 第五层选股逻辑：选择80%-100%分位数区间的股票（即最高的20%）
                    threshold_low = np.nanquantile(signal_values.values, 0.8)  # 80%分位数
                    threshold_high = np.nanquantile(signal_values.values, 1.0)  # 100%分位数（最大值）
                    
                    # 选择在分位数区间内的股票
                    layer5_mask = (signal_values >= threshold_low) & (signal_values <= threshold_high)
                    layer5_stocks = signal_values.index[layer5_mask].tolist()
                    
                    # 等权重分配
                    new_weight = pd.Series(0.0, index=stocks)
                    if len(layer5_stocks) > 0:
                        new_weight[layer5_stocks] = 1.0 / len(layer5_stocks)
                    
                    current_weight = new_weight
                    turnover_series.iloc[i] = 0.0  # 第一天没有换手
                else:
                    current_weight = pd.Series(0.0, index=stocks)
                    turnover_series.iloc[i] = 0.0
                
                weight_df[date] = current_weight
                nav_series.iloc[i] = 1.0  # 初始净值
                continue
            
            prev_date = dates[i-1]
            prev_nav = nav_series.iloc[i-1]
            
            # 计算当日收益（使用上日的持仓和当日的收益率）
            # 因为信号预测t+1日收益，所以t日建立的仓位在t+1日获得收益
            if date in self.return_data.columns:
                # 确保股票对齐
                aligned_return = self.return_data.loc[stocks, date].fillna(0)
                daily_return = (current_weight * aligned_return).sum()
                nav_series.iloc[i] = prev_nav * (1 + daily_return)
            else:
                nav_series.iloc[i] = prev_nav
            
            # 检查是否需要调仓（在获得收益后调仓）
            if i % rebalance_freq == 0:
                # 获取当期信号值（用于下一期的持仓）
                signal_values = signal_df[date].dropna()
                
                # 如果提供了成分股管理器，只使用成分股
                if self.constituent_manager is not None:
                    constituents = self.constituent_manager.get_constituents_by_date(date)
                    constituents_set = set(constituents)
                    signal_values = signal_values[signal_values.index.isin(constituents_set)]
                
                if len(signal_values) == 0:
                    # 如果没有可用股票，保持上期权重
                    new_weight = current_weight.copy()
                    turnover_series.iloc[i] = 0.0
                else:
                    # 第五层选股逻辑：选择80%-100%分位数区间的股票（即最高的20%）
                    threshold_low = np.nanquantile(signal_values.values, 0.8)  # 80%分位数
                    threshold_high = np.nanquantile(signal_values.values, 1.0)  # 100%分位数（最大值）
                    
                    # 选择在分位数区间内的股票
                    layer5_mask = (signal_values >= threshold_low) & (signal_values <= threshold_high)
                    layer5_stocks = signal_values.index[layer5_mask].tolist()
                    
                    # 等权重分配
                    new_weight = pd.Series(0.0, index=stocks)
                    if len(layer5_stocks) > 0:
                        new_weight[layer5_stocks] = 1.0 / len(layer5_stocks)
                    
                    # 计算换手率（双边）
                    turnover = (new_weight - current_weight).abs().sum()
                    turnover_series.iloc[i] = turnover
                    
                    # 扣除交易成本（在调仓日扣除，从当前净值扣除）
                    total_cost = (fee + slippage) * turnover
                    nav_series.iloc[i] = nav_series.iloc[i] * (1 - total_cost)
                
                current_weight = new_weight
            else:
                # 非调仓日，保持上期权重，换手率为0
                turnover_series.iloc[i] = 0.0
            
            weight_df[date] = current_weight
        
        # 准备基准数据（如果可用）
        benchmark_nav = None
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
            if benchmark_nav is not None and date_range is not None:
                # 如果指定了日期范围，对齐基准数据
                start_date, end_date = date_range
                benchmark_nav = benchmark_nav[(benchmark_nav.index >= start_date) & (benchmark_nav.index <= end_date)]
        
        # 计算性能指标
        metrics = self._calculate_metrics(nav_series, turnover_series, benchmark_nav)
        
        return {
            'nav_series': nav_series,
            'weight_df': weight_df,
            'turnover_series': turnover_series,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, nav_series, turnover_series, benchmark_nav=None):
        """
        计算回测性能指标
        
        Parameters:
        -----------
        nav_series : pd.Series, 净值序列
        turnover_series : pd.Series, 换手率序列
        benchmark_nav : pd.Series, 基准净值序列（可选）
        
        Returns:
        --------
        dict : 性能指标
        """
        if len(nav_series) < 2:
            return {}
        
        # 计算收益率序列
        returns = nav_series.pct_change().dropna()
        
        # 基本指标
        total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
        
        # 年化收益率（假设252个交易日）
        days = len(nav_series)
        if days > 1:
            annual_return = (np.power(nav_series.iloc[-1] / nav_series.iloc[0], 252 / days) - 1) * 100
        else:
            annual_return = np.nan
        
        # 年化波动率
        if len(returns) > 1:
            annual_volatility = returns.std() * np.sqrt(252) * 100
        else:
            annual_volatility = np.nan
        
        # 夏普比率（假设无风险利率为0）
        if annual_volatility > 0:
            sharpe_ratio = annual_return / annual_volatility
        else:
            sharpe_ratio = np.nan
        
        # 最大回撤
        max_drawdown, max_drawdown_duration = self._calculate_max_drawdown(nav_series)
        
        # Calmar比率
        if max_drawdown > 0:
            calmar_ratio = annual_return / (max_drawdown * 100)
        else:
            calmar_ratio = np.nan
        
        # 年化换手率
        if len(turnover_series) > 1:
            annual_turnover = turnover_series.sum() / (days / 252)
        else:
            annual_turnover = np.nan
        
        # 胜率（正收益天数比例）
        if len(returns) > 0:
            win_rate = (returns > 0).sum() / len(returns) * 100
        else:
            win_rate = np.nan
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'annual_turnover': annual_turnover,
            'win_rate': win_rate,
            'max_drawdown_duration': max_drawdown_duration
        }
        
        # 如果有基准，计算超额收益
        if benchmark_nav is not None and len(benchmark_nav) > 0:
            excess_metrics = self._calculate_excess_metrics(nav_series, benchmark_nav)
            metrics.update(excess_metrics)
        
        return metrics
    
    def _calculate_max_drawdown(self, nav_series):
        """
        计算最大回撤
        
        Parameters:
        -----------
        nav_series : pd.Series, 净值序列
        
        Returns:
        --------
        float : 最大回撤（比例）
        int : 最大回撤持续时间（交易日）
        """
        if len(nav_series) < 2:
            return 0.0, 0
        
        # 计算累计最高净值
        cummax = nav_series.expanding().max()
        
        # 计算回撤
        drawdown = (nav_series - cummax) / cummax
        
        max_drawdown = abs(drawdown.min())
        
        # 计算最大回撤持续时间
        max_dd_idx = drawdown.idxmin()
        max_dd_date = nav_series.index.get_loc(max_dd_idx)
        
        # 找到回撤开始的位置（净值最高点）
        peak_idx = cummax.loc[:max_dd_idx].idxmax()
        peak_date = nav_series.index.get_loc(peak_idx)
        
        max_drawdown_duration = max_dd_date - peak_date
        
        return max_drawdown, max_drawdown_duration
    
    def _calculate_excess_metrics(self, nav_series, benchmark_nav):
        """
        计算相对基准的超额收益指标
        
        Parameters:
        -----------
        nav_series : pd.Series, 策略净值序列
        benchmark_nav : pd.Series, 基准净值序列
        
        Returns:
        --------
        dict : 超额收益指标
        """
        # 对齐日期
        common_dates = sorted(set(nav_series.index) & set(benchmark_nav.index))
        if len(common_dates) < 2:
            return {}
        
        nav_aligned = nav_series.loc[common_dates]
        benchmark_aligned = benchmark_nav.loc[common_dates]
        
        # 计算超额收益序列
        excess_returns = nav_aligned.pct_change() - benchmark_aligned.pct_change()
        excess_returns = excess_returns.dropna()
        
        # 计算超额收益净值
        excess_nav = (1 + excess_returns).cumprod()
        
        # 年化超额收益
        days = len(excess_nav)
        if days > 1:
            annual_excess_return = (np.power(excess_nav.iloc[-1] / excess_nav.iloc[0], 252 / days) - 1) * 100
        else:
            annual_excess_return = np.nan
        
        # 跟踪误差（年化）
        if len(excess_returns) > 1:
            tracking_error = excess_returns.std() * np.sqrt(252) * 100
        else:
            tracking_error = np.nan
        
        # 信息比率
        if tracking_error > 0:
            information_ratio = annual_excess_return / tracking_error
        else:
            information_ratio = np.nan
        
        # 超额收益最大回撤
        excess_max_drawdown, _ = self._calculate_max_drawdown(excess_nav)
        
        # 月度胜率
        monthly_win_rate = self._calculate_monthly_win_rate(excess_returns, common_dates)
        
        return {
            'annual_excess_return': annual_excess_return,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'excess_max_drawdown': excess_max_drawdown * 100,
            'monthly_win_rate': monthly_win_rate
        }
    
    def _calculate_monthly_win_rate(self, excess_returns, dates):
        """计算月度超额收益胜率"""
        if len(excess_returns) == 0:
            return np.nan
        
        # 转换为DataFrame以便按月份分组
        excess_df = pd.DataFrame({
            'date': pd.to_datetime(dates[1:len(excess_returns)+1]),
            'excess_return': excess_returns.values
        })
        
        excess_df['year_month'] = excess_df['date'].dt.to_period('M')
        
        # 计算月度超额收益
        monthly_excess = excess_df.groupby('year_month')['excess_return'].apply(
            lambda x: (1 + x).prod() - 1
        )
        
        if len(monthly_excess) > 0:
            win_rate = (monthly_excess > 0).sum() / len(monthly_excess) * 100
            return win_rate
        else:
            return np.nan
    
    def backtest_all_signals(self, top_pcts=[0.05, 0.1, 0.2], fee=0.002, 
                            slippage=0.001, rebalance_freq=1, save_dir='./results/'):
        """
        对所有信号进行回测（不同头部比例）
        
        Parameters:
        -----------
        top_pcts : list, 头部股票比例列表（默认[0.05, 0.1, 0.2]）
        fee : float, 交易费率
        slippage : float, 滑点
        rebalance_freq : int, 调仓频率
        save_dir : str, 结果保存目录
        
        Returns:
        --------
        dict : 所有回测结果
        """
        print("\n" + "="*80)
        print("开始策略回测")
        print("="*80)
        
        all_results = {}
        
        # 准备基准数据（中证1000指数）
        benchmark_nav = None
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
        
        # 对每个信号进行回测
        for signal_name in self.signal_data.keys():
            print(f"\n回测信号: {signal_name}")
            print("-"*80)
            
            signal_results = {}
            
            for top_pct in top_pcts:
                print(f"\n头部比例: {top_pct*100:.0f}%")
                
                result = self.long_only_backtest(
                    signal_name=signal_name,
                    top_pct=top_pct,
                    fee=fee,
                    slippage=slippage,
                    rebalance_freq=rebalance_freq
                )
                
                # 如果有基准，计算超额收益
                if benchmark_nav is not None:
                    excess_metrics = self._calculate_excess_metrics(
                        result['nav_series'], 
                        benchmark_nav
                    )
                    result['metrics'].update(excess_metrics)
                
                signal_results[f'top_{int(top_pct*100)}pct'] = result
                
                # 打印关键指标
                metrics = result['metrics']
                print(f"  总收益率: {metrics.get('total_return', np.nan):.2f}%")
                print(f"  年化收益率: {metrics.get('annual_return', np.nan):.2f}%")
                print(f"  年化波动率: {metrics.get('annual_volatility', np.nan):.2f}%")
                print(f"  夏普比率: {metrics.get('sharpe_ratio', np.nan):.4f}")
                print(f"  最大回撤: {metrics.get('max_drawdown', np.nan):.2f}%")
                if 'annual_excess_return' in metrics:
                    print(f"  年化超额收益: {metrics.get('annual_excess_return', np.nan):.2f}%")
                    print(f"  信息比率: {metrics.get('information_ratio', np.nan):.4f}")
            
            all_results[signal_name] = signal_results
        
        # 保存结果
        self._save_backtest_results(all_results, save_dir)
        
        return all_results
    
    def backtest_all_signals_layer5(self, fee=0.002, 
                                     slippage=0.001, rebalance_freq=1, save_dir='./results/'):
        """
        基于第五层选股逻辑对所有信号进行回测
        
        第五层选股逻辑：选择因子值在80%-100%分位数区间的股票（即因子值最高的20%股票）
        这对应分层回测中第五层的选股方法
        
        Parameters:
        -----------
        fee : float, 交易费率（双边，默认0.002即0.2%）
        slippage : float, 滑点（默认0.001即0.1%）
        rebalance_freq : int, 调仓频率（默认1，即每日调仓）
        save_dir : str, 结果保存目录
        
        Returns:
        --------
        dict : 所有回测结果（格式：{signal_name: result_dict}）
        """
        print("\n" + "="*80)
        print("开始策略回测（第五层选股逻辑）")
        print("="*80)
        
        all_results = {}
        
        # 准备基准数据（中证1000指数）
        benchmark_nav = None
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
        
        # 对每个信号进行回测
        for signal_name in self.signal_data.keys():
            print(f"\n回测信号: {signal_name}")
            print("-"*80)
            
            result = self.layer5_backtest(
                signal_name=signal_name,
                fee=fee,
                slippage=slippage,
                rebalance_freq=rebalance_freq
            )
            
            if not result:
                continue
            
            # 如果有基准，计算超额收益
            if benchmark_nav is not None:
                excess_metrics = self._calculate_excess_metrics(
                    result['nav_series'], 
                    benchmark_nav
                )
                result['metrics'].update(excess_metrics)
            
            all_results[signal_name] = result
            
            # 打印关键指标
            metrics = result['metrics']
            print(f"  总收益率: {metrics.get('total_return', np.nan):.2f}%")
            print(f"  年化收益率: {metrics.get('annual_return', np.nan):.2f}%")
            print(f"  年化波动率: {metrics.get('annual_volatility', np.nan):.2f}%")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', np.nan):.4f}")
            print(f"  最大回撤: {metrics.get('max_drawdown', np.nan):.2f}%")
            if 'annual_excess_return' in metrics:
                print(f"  年化超额收益: {metrics.get('annual_excess_return', np.nan):.2f}%")
                print(f"  信息比率: {metrics.get('information_ratio', np.nan):.4f}")
        
        # 保存结果
        self._save_layer5_backtest_results(all_results, save_dir)
        
        return all_results
    
    def find_optimal_top_pct(self, all_results, metric='sharpe_ratio'):
        """
        从回测结果中找到每个信号的最优头部比例
        
        Parameters:
        -----------
        all_results : dict, 回测结果字典
        metric : str, 用于选择最优比例的指标（默认'sharpe_ratio'）
        
        Returns:
        --------
        dict : {signal_name: optimal_top_pct}
        """
        optimal_pcts = {}
        
        for signal_name, signal_results in all_results.items():
            best_pct = None
            best_value = -np.inf
            
            for top_pct_key, result in signal_results.items():
                metrics = result['metrics']
                value = metrics.get(metric, -np.inf)
                
                if value > best_value:
                    best_value = value
                    # 从top_pct_key中提取比例值，例如'top_10pct' -> 0.1
                    pct_str = top_pct_key.replace('top_', '').replace('pct', '')
                    best_pct = float(pct_str) / 100.0
            
            if best_pct is not None:
                optimal_pcts[signal_name] = best_pct
                print(f"  {signal_name}: 最优头部比例 = {best_pct*100:.0f}% ({metric} = {best_value:.4f})")
        
        return optimal_pcts
    
    def out_of_sample_backtest(self, all_results, train_ratio=0.8, fee=0.002,
                              slippage=0.001, rebalance_freq=1, save_dir='./results/',
                              metric='sharpe_ratio'):
        """
        纯袋外回测：在测试集（后20%数据）上使用最优头部比例进行回测
        
        Parameters:
        -----------
        all_results : dict, 全样本回测结果，用于确定最优头部比例
        train_ratio : float, 训练集比例（默认0.8，即后20%为测试集）
        fee : float, 交易费率
        slippage : float, 滑点
        rebalance_freq : int, 调仓频率
        save_dir : str, 结果保存目录
        metric : str, 用于选择最优比例的指标（默认'sharpe_ratio'）
        
        Returns:
        --------
        dict : 袋外回测结果
        """
        print("\n" + "="*80)
        print("纯袋外回测（测试集：后20%数据）")
        print("="*80)
        
        # 1. 找到每个信号的最优头部比例
        print("\n【步骤1】确定各信号的最优头部比例（基于全样本回测）")
        print("-"*80)
        optimal_pcts = self.find_optimal_top_pct(all_results, metric=metric)
        
        if len(optimal_pcts) == 0:
            print("⚠️  无法确定最优头部比例，袋外回测终止")
            return {}
        
        # 2. 确定测试集日期范围（后20%数据）
        print("\n【步骤2】确定测试集日期范围")
        print("-"*80)
        
        # 获取所有信号的公共日期
        all_dates = set()
        for signal_df in self.signal_data.values():
            if signal_df is not None and not signal_df.empty:
                all_dates.update(signal_df.columns)
        
        all_dates = sorted(list(all_dates))
        
        if len(all_dates) == 0:
            print("⚠️  没有可用的日期数据")
            return {}
        
        # 计算训练集和测试集的分割点
        train_end_idx = int(len(all_dates) * train_ratio)
        test_dates = all_dates[train_end_idx:]
        
        if len(test_dates) == 0:
            print("⚠️  测试集为空，袋外回测终止")
            return {}
        
        print(f"  总日期数: {len(all_dates)}")
        print(f"  训练集日期: {all_dates[0]} 至 {all_dates[train_end_idx-1]} ({train_end_idx} 个)")
        print(f"  测试集日期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 个)")
        
        # 3. 准备基准数据
        benchmark_nav = None
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
            if benchmark_nav is not None:
                # 只保留测试集日期的基准数据
                benchmark_nav = benchmark_nav[benchmark_nav.index.isin(test_dates)]
        
        # 4. 在测试集上进行回测
        print("\n【步骤3】在测试集上进行回测（使用最优头部比例）")
        print("-"*80)
        
        oos_results = {}
        
        for signal_name in self.signal_data.keys():
            if signal_name not in optimal_pcts:
                print(f"⚠️  跳过信号 {signal_name}（未找到最优比例）")
                continue
            
            optimal_pct = optimal_pcts[signal_name]
            print(f"\n回测信号: {signal_name} (头部比例: {optimal_pct*100:.0f}%)")
            
            # 使用测试集日期范围进行回测
            result = self.long_only_backtest(
                signal_name=signal_name,
                top_pct=optimal_pct,
                fee=fee,
                slippage=slippage,
                rebalance_freq=rebalance_freq,
                date_range=(test_dates[0], test_dates[-1])
            )
            
            if not result:
                continue
            
            # 如果有基准，计算超额收益
            if benchmark_nav is not None and len(benchmark_nav) > 0:
                excess_metrics = self._calculate_excess_metrics(
                    result['nav_series'],
                    benchmark_nav
                )
                result['metrics'].update(excess_metrics)
            
            oos_results[signal_name] = result
            
            # 打印关键指标
            metrics = result['metrics']
            print(f"  总收益率: {metrics.get('total_return', np.nan):.2f}%")
            print(f"  年化收益率: {metrics.get('annual_return', np.nan):.2f}%")
            print(f"  年化波动率: {metrics.get('annual_volatility', np.nan):.2f}%")
            print(f"  夏普比率: {metrics.get('sharpe_ratio', np.nan):.4f}")
            print(f"  最大回撤: {metrics.get('max_drawdown', np.nan):.2f}%")
            if 'annual_excess_return' in metrics:
                print(f"  年化超额收益: {metrics.get('annual_excess_return', np.nan):.2f}%")
                print(f"  信息比率: {metrics.get('information_ratio', np.nan):.4f}")
        
        # 5. 保存结果
        print("\n【步骤4】保存袋外回测结果")
        print("-"*80)
        self._save_oos_results(oos_results, optimal_pcts, save_dir)
        
        return oos_results
    
    def _save_oos_results(self, oos_results, optimal_pcts, save_dir):
        """保存袋外回测结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"保存袋外回测结果到: {save_dir}")
        
        # 汇总所有指标
        summary_list = []
        
        for signal_name, result in oos_results.items():
            metrics = result['metrics']
            summary_list.append({
                'signal': signal_name,
                'optimal_top_pct': f"{optimal_pcts.get(signal_name, np.nan)*100:.0f}%",
                **metrics
            })
            
            # 保存净值序列
            nav_file = os.path.join(save_dir, f'{signal_name}_oos_nav.csv')
            result['nav_series'].to_csv(nav_file, encoding='utf-8-sig')
            print(f"  ✅ {signal_name} 净值序列: {nav_file}")
        
        # 保存汇总表
        summary_df = pd.DataFrame(summary_list)
        summary_file = os.path.join(save_dir, 'oos_backtest_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✅ 袋外回测汇总已保存: {summary_file}")
    
    def _prepare_benchmark_nav(self):
        """准备基准净值序列"""
        if self.benchmark_data is None or self.benchmark_data.empty:
            # 如果没有基准数据，使用等权重组合作为基准
            return self._calculate_equal_weight_benchmark()
        
        # 假设benchmark_data包含日期和收益率
        if 'RETURN' in self.benchmark_data.columns:
            returns = self.benchmark_data['RETURN']
            # 确保按日期排序
            returns = returns.sort_index()
            nav = (1 + returns).cumprod()
            # 将第一个日期归一化为1.0
            if len(nav) > 0:
                first_value = nav.iloc[0]
                if first_value > 0:
                    nav = nav / first_value
                nav.iloc[0] = 1.0
            return nav
        elif len(self.benchmark_data.columns) == 1:
            # 如果只有一列，假设是收益率
            returns = self.benchmark_data.iloc[:, 0]
            # 确保按日期排序
            if isinstance(returns, pd.Series):
                returns = returns.sort_index()
            nav = (1 + returns).cumprod()
            # 将第一个日期归一化为1.0
            if len(nav) > 0:
                first_value = nav.iloc[0]
                if first_value > 0:
                    nav = nav / first_value
                nav.iloc[0] = 1.0
            return nav
        else:
            # 使用等权重组合作为基准
            return self._calculate_equal_weight_benchmark()
    
    def _calculate_equal_weight_benchmark(self):
        """计算等权重组合作为基准（中证1000等权重）"""
        if self.return_data.empty:
            return None
        
        dates = sorted(self.return_data.columns)
        stocks = sorted(self.return_data.index)
        
        nav_series = pd.Series(index=dates, dtype=float)
        nav_series.iloc[0] = 1.0
        
        for i, date in enumerate(dates[1:], start=1):
            prev_nav = nav_series.iloc[i-1]
            
            # 等权重组合：所有成分股等权重
            if self.constituent_manager is not None:
                constituents = self.constituent_manager.get_constituents_by_date(date)
                constituents_set = set(constituents) & set(stocks)
                if len(constituents_set) > 0:
                    equal_weight = 1.0 / len(constituents_set)
                    daily_return = self.return_data.loc[list(constituents_set), date].mean()
                else:
                    daily_return = 0.0
            else:
                # 如果没有成分股管理器，使用所有股票
                daily_return = self.return_data[date].mean()
            
            nav_series.iloc[i] = prev_nav * (1 + daily_return)
        
        return nav_series
    
    def _save_backtest_results(self, all_results, save_dir):
        """保存回测结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n保存回测结果到: {save_dir}")
        
        # 汇总所有指标
        summary_list = []
        
        for signal_name, signal_results in all_results.items():
            for top_pct_key, result in signal_results.items():
                metrics = result['metrics']
                summary_list.append({
                    'signal': signal_name,
                    'top_pct': top_pct_key,
                    **metrics
                })
                
                # 保存净值序列
                nav_file = os.path.join(save_dir, f'{signal_name}_{top_pct_key}_nav.csv')
                result['nav_series'].to_csv(nav_file, encoding='utf-8-sig')
        
        # 保存汇总表
        summary_df = pd.DataFrame(summary_list)
        summary_file = os.path.join(save_dir, 'backtest_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✅ 回测汇总已保存: {summary_file}")
    
    def _save_layer5_backtest_results(self, all_results, save_dir):
        """保存第五层选股逻辑的回测结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n保存回测结果到: {save_dir}")
        
        # 汇总所有指标
        summary_list = []
        
        for signal_name, result in all_results.items():
            metrics = result['metrics']
            summary_list.append({
                'signal': signal_name,
                **metrics
            })
            
            # 保存净值序列
            nav_file = os.path.join(save_dir, f'{signal_name}_layer5_nav.csv')
            result['nav_series'].to_csv(nav_file, encoding='utf-8-sig')
            print(f"  ✅ {signal_name} 净值序列: {nav_file}")
        
        # 保存汇总表
        summary_df = pd.DataFrame(summary_list)
        summary_file = os.path.join(save_dir, 'layer5_backtest_summary.csv')
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"✅ 回测汇总已保存: {summary_file}")
    
    def visualize_results(self, all_results, save_dir='./results/'):
        """
        可视化回测结果
        
        Parameters:
        -----------
        all_results : dict, 所有回测结果
        save_dir : str, 结果保存目录
        """
        print("\n生成可视化图表...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 净值曲线对比图
        self._plot_nav_curves(all_results, save_dir)
        
        # 2. 月度收益统计柱状图
        self._plot_monthly_returns(all_results, save_dir)
        
        # 3. 性能指标对比图
        self._plot_metrics_comparison(all_results, save_dir)
        
        print("✅ 可视化完成")
    
    def visualize_layer5_results(self, all_results, save_dir='./results/'):
        """
        可视化第五层选股逻辑的回测结果
        
        Parameters:
        -----------
        all_results : dict, 所有回测结果（格式：{signal_name: result_dict}）
        save_dir : str, 结果保存目录
        """
        print("\n生成可视化图表（第五层选股）...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 净值曲线对比图
        self._plot_layer5_nav_curves(all_results, save_dir)
        
        # 2. 月度收益统计柱状图
        self._plot_layer5_monthly_returns(all_results, save_dir)
        
        # 3. 性能指标对比图
        self._plot_layer5_metrics_comparison(all_results, save_dir)
        
        print("✅ 可视化完成")
    
    def _plot_layer5_nav_curves(self, all_results, save_dir):
        """绘制第五层选股逻辑的净值曲线对比图"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for signal_name, result in all_results.items():
            nav_series = result['nav_series']
            ax.plot(nav_series.index, nav_series.values, label=signal_name, linewidth=2, alpha=0.8)
        
        # 如果有基准，绘制基准
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
            if benchmark_nav is not None:
                # 获取所有信号的公共日期
                all_dates = set()
                for result in all_results.values():
                    all_dates.update(result['nav_series'].index)
                common_dates = sorted(set(all_dates) & set(benchmark_nav.index))
                if len(common_dates) > 0:
                    benchmark_aligned = benchmark_nav.loc[common_dates]
                    # 归一化基准净值，使其在公共日期范围的起始点为1.0
                    if len(benchmark_aligned) > 0:
                        first_value = benchmark_aligned.iloc[0]
                        if first_value > 0:
                            benchmark_aligned = benchmark_aligned / first_value
                        ax.plot(benchmark_aligned.index, benchmark_aligned.values, 
                               label='中证1000', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title('第五层选股策略 - 净值曲线对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'layer5_nav_curves_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 净值曲线图: {save_path}")
        plt.close()
    
    def _plot_layer5_monthly_returns(self, all_results, save_dir):
        """绘制第五层选股逻辑的月度收益统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for signal_name, result in all_results.items():
            if plot_idx >= 4:
                break
            
            ax = axes[plot_idx]
            nav_series = result['nav_series']
            
            # 计算月度收益
            monthly_returns = self._calculate_monthly_returns(nav_series)
            
            # 绘制柱状图
            colors = ['green' if x >= 0 else 'red' for x in monthly_returns.values]
            ax.bar(range(len(monthly_returns)), monthly_returns.values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xticks(range(len(monthly_returns)))
            ax.set_xticklabels([f'{idx.year}-{idx.month:02d}' for idx in monthly_returns.index], 
                              rotation=45, ha='right')
            ax.set_title(f'{signal_name} - 月度收益率', fontsize=12, fontweight='bold')
            ax.set_xlabel('月份', fontsize=10)
            ax.set_ylabel('收益率 (%)', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            plot_idx += 1
        
        # 隐藏多余的子图
        for idx in range(plot_idx, 4):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'layer5_monthly_returns.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 月度收益统计图: {save_path}")
        plt.close()
    
    def _plot_layer5_metrics_comparison(self, all_results, save_dir):
        """绘制第五层选股逻辑的性能指标对比图"""
        # 提取关键指标
        metrics_list = []
        signal_names = []
        
        for signal_name, result in all_results.items():
            metrics = result['metrics']
            metrics_list.append({
                '年化收益率 (%)': metrics.get('annual_return', np.nan),
                '年化波动率 (%)': metrics.get('annual_volatility', np.nan),
                '夏普比率': metrics.get('sharpe_ratio', np.nan),
                '最大回撤 (%)': metrics.get('max_drawdown', np.nan),
            })
            if 'annual_excess_return' in metrics:
                metrics_list[-1]['年化超额收益 (%)'] = metrics.get('annual_excess_return', np.nan)
                metrics_list[-1]['信息比率'] = metrics.get('information_ratio', np.nan)
            signal_names.append(signal_name)
        
        metrics_df = pd.DataFrame(metrics_list, index=signal_names)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(metrics_df.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                   ax=ax, cbar_kws={'label': '指标值'})
        ax.set_title('第五层选股策略 - 性能指标对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('信号', fontsize=12)
        ax.set_ylabel('指标', fontsize=12)
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'layer5_metrics_comparison_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 指标对比热力图: {save_path}")
        plt.close()
        
        # 绘制柱状图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        key_metrics = ['年化收益率 (%)', '夏普比率', '最大回撤 (%)', '年化超额收益 (%)']
        for idx, metric in enumerate(key_metrics):
            if metric not in metrics_df.columns:
                axes[idx].axis('off')
                continue
            
            ax = axes[idx]
            values = metrics_df[metric]
            colors = ['green' if x >= 0 else 'red' for x in values]
            if metric == '最大回撤 (%)':
                colors = ['red' if x <= 0 else 'green' for x in values]  # 回撤越小越好
            
            ax.bar(range(len(values)), values, color=colors, alpha=0.7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values.index, rotation=45, ha='right')
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('指标值', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'layer5_metrics_comparison_bar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ 指标对比柱状图: {save_path}")
        plt.close()
    
    def _plot_nav_curves(self, all_results, save_dir):
        """绘制净值曲线对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for signal_name, signal_results in all_results.items():
            if plot_idx >= 4:
                break
            
            ax = axes[plot_idx]
            
            for top_pct_key, result in signal_results.items():
                nav_series = result['nav_series']
                top_pct = top_pct_key.replace('top_', '').replace('pct', '%')
                ax.plot(nav_series.index, nav_series.values, label=f'{top_pct}', linewidth=1.5)
            
            # 如果有基准，绘制基准
            if self.benchmark_data is not None:
                benchmark_nav = self._prepare_benchmark_nav()
                if benchmark_nav is not None:
                    common_dates = sorted(set(nav_series.index) & set(benchmark_nav.index))
                    if len(common_dates) > 0:
                        benchmark_aligned = benchmark_nav.loc[common_dates]
                        # 归一化基准净值，使其在公共日期范围的起始点为1.0
                        if len(benchmark_aligned) > 0:
                            first_value = benchmark_aligned.iloc[0]
                            if first_value > 0:
                                benchmark_aligned = benchmark_aligned / first_value
                            ax.plot(benchmark_aligned.index, benchmark_aligned.values, 
                                   label='中证1000', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax.set_title(f'{signal_name} - 净值曲线', fontsize=12, fontweight='bold')
            ax.set_xlabel('日期', fontsize=10)
            ax.set_ylabel('净值', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # 隐藏多余的子图
        for i in range(plot_idx, 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'nav_curves_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 净值曲线图: {save_path}")
    
    def _plot_monthly_returns(self, all_results, save_dir):
        """绘制月度收益统计柱状图（增强版）"""
        # 1. 为每个信号绘制单独的月度收益图
        for signal_name, signal_results in all_results.items():
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            for idx, (top_pct_key, result) in enumerate(signal_results.items()):
                ax = axes[idx]
                nav_series = result['nav_series']
                
                # 计算月度收益率
                monthly_returns = self._calculate_monthly_returns(nav_series)
                
                if len(monthly_returns) > 0:
                    # 按月份分组（索引已经是Timestamp类型，无需再次转换）
                    monthly_returns = monthly_returns.sort_index()
                    
                    # 绘制柱状图
                    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_returns.values]
                    bars = ax.bar(range(len(monthly_returns)), monthly_returns.values * 100, 
                                 color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # 添加数值标签
                    for i, (bar, ret) in enumerate(zip(bars, monthly_returns.values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{ret*100:.1f}%',
                               ha='center', va='bottom' if height > 0 else 'top',
                               fontsize=8)
                    
                    ax.set_title(f'{signal_name} - {top_pct_key.replace("_", " ").title()} 月度收益率', 
                               fontsize=12, fontweight='bold')
                    ax.set_xlabel('月份', fontsize=10)
                    ax.set_ylabel('收益率 (%)', fontsize=10)
                    ax.set_xticks(range(len(monthly_returns)))
                    ax.set_xticklabels([f"{d.strftime('%Y-%m')}" for d in monthly_returns.index], 
                                      rotation=45, ha='right')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # 添加统计信息
                    avg_return = monthly_returns.mean() * 100
                    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
                    ax.text(0.02, 0.98, f'平均收益率: {avg_return:.2f}%\n胜率: {win_rate:.1f}%',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{signal_name}_monthly_returns.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ 月度收益图: {save_path}")
        
        # 2. 绘制所有信号的月度收益对比图（按月份汇总）
        self._plot_monthly_returns_comparison(all_results, save_dir)
    
    def _calculate_monthly_returns(self, nav_series):
        """计算月度收益率"""
        nav_df = pd.DataFrame({'nav': nav_series})
        nav_df.index = pd.to_datetime(nav_df.index)
        nav_df['year_month'] = nav_df.index.to_period('M')
        
        # 计算每月最后一天的净值
        monthly_nav = nav_df.groupby('year_month')['nav'].last()
        
        # 计算月度收益率
        monthly_returns = monthly_nav.pct_change().dropna()
        
        # 将Period索引转换为Timestamp索引
        monthly_returns.index = monthly_returns.index.to_timestamp()
        
        return monthly_returns
    
    def _plot_monthly_returns_comparison(self, all_results, save_dir):
        """绘制所有信号的月度收益对比图（按月份）"""
        # 收集所有信号的月度收益数据
        monthly_data = {}
        
        for signal_name, signal_results in all_results.items():
            for top_pct_key, result in signal_results.items():
                nav_series = result['nav_series']
                monthly_returns = self._calculate_monthly_returns(nav_series)
                
                if len(monthly_returns) > 0:
                    key = f'{signal_name}_{top_pct_key}'
                    monthly_data[key] = monthly_returns
        
        if len(monthly_data) == 0:
            return
        
        # 找到所有月份
        all_months = set()
        for monthly_returns in monthly_data.values():
            all_months.update(monthly_returns.index)
        all_months = sorted(list(all_months))
        
        # 创建对比数据
        comparison_df = pd.DataFrame(index=all_months)
        for key, monthly_returns in monthly_data.items():
            comparison_df[key] = monthly_returns
        
        comparison_df = comparison_df.fillna(0) * 100  # 转换为百分比
        
        # 绘制分组柱状图
        fig, ax = plt.subplots(figsize=(20, 8))
        
        x = np.arange(len(all_months))
        width = 0.8 / len(comparison_df.columns)
        
        for i, col in enumerate(comparison_df.columns):
            offset = (i - len(comparison_df.columns) / 2) * width + width / 2
            values = comparison_df[col].values
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in values]
            ax.bar(x + offset, values, width, label=col.replace('_', ' '), 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
        
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('收益率 (%)', fontsize=12)
        ax.set_title('所有信号月度收益率对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'all_signals_monthly_returns_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 月度收益对比图: {save_path}")
    
    def _plot_metrics_comparison(self, all_results, save_dir):
        """绘制性能指标对比图（增强版）"""
        # 汇总所有指标
        metrics_list = []
        
        for signal_name, signal_results in all_results.items():
            for top_pct_key, result in signal_results.items():
                metrics = result['metrics']
                metrics_list.append({
                    'signal': signal_name,
                    'top_pct': top_pct_key,
                    **metrics
                })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # 选择关键指标进行对比
        key_metrics = {
            'annual_return': '年化收益率 (%)',
            'sharpe_ratio': '夏普比率',
            'max_drawdown': '最大回撤 (%)',
            'annual_excess_return': '年化超额收益 (%)',
            'information_ratio': '信息比率',
            'calmar_ratio': 'Calmar比率'
        }
        
        available_metrics = {k: v for k, v in key_metrics.items() if k in metrics_df.columns}
        
        if len(available_metrics) == 0:
            return
        
        # 1. 热力图对比
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, (metric, label) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # 创建透视表
            pivot_data = metrics_df.pivot(index='signal', columns='top_pct', values=metric)
            
            # 绘制热力图
            if len(pivot_data) > 0 and len(pivot_data.columns) > 0:
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                           center=0, ax=ax, cbar_kws={'label': label},
                           linewidths=0.5, linecolor='gray')
            
            ax.set_title(f'{label}', fontsize=11, fontweight='bold')
            ax.set_xlabel('头部比例', fontsize=9)
            ax.set_ylabel('信号', fontsize=9)
        
        # 隐藏多余的子图
        for i in range(len(available_metrics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'metrics_comparison_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 指标对比热力图: {save_path}")
        
        # 2. 柱状图对比（按信号分组）
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        top_metrics = list(available_metrics.items())[:4]
        
        for idx, (metric, label) in enumerate(top_metrics):
            ax = axes[idx]
            
            # 按信号分组绘制柱状图
            pivot_data = metrics_df.pivot(index='signal', columns='top_pct', values=metric)
            
            if len(pivot_data) > 0:
                pivot_data.plot(kind='bar', ax=ax, width=0.8, alpha=0.8, edgecolor='black')
            
            ax.set_title(f'{label}', fontsize=11, fontweight='bold')
            ax.set_xlabel('信号', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.legend(title='头部比例', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'metrics_comparison_bar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 指标对比柱状图: {save_path}")
    
    def visualize_oos_results(self, oos_results, save_dir='./results/'):
        """
        可视化袋外回测结果
        
        Parameters:
        -----------
        oos_results : dict, 袋外回测结果
        save_dir : str, 结果保存目录
        """
        print("\n生成袋外回测可视化图表...")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 净值曲线对比图
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for signal_name, result in oos_results.items():
            nav_series = result['nav_series']
            ax.plot(nav_series.index, nav_series.values, label=signal_name, linewidth=2)
        
        # 如果有基准，绘制基准
        if self.benchmark_data is not None:
            benchmark_nav = self._prepare_benchmark_nav()
            if benchmark_nav is not None:
                # 只保留袋外回测的日期范围
                if len(oos_results) > 0:
                    first_result = list(oos_results.values())[0]
                    test_dates = first_result['nav_series'].index
                    benchmark_aligned = benchmark_nav[benchmark_nav.index.isin(test_dates)]
                    if len(benchmark_aligned) > 0:
                        # 归一化基准净值，使其在测试集日期范围的起始点为1.0
                        first_value = benchmark_aligned.iloc[0]
                        if first_value > 0:
                            benchmark_aligned = benchmark_aligned / first_value
                        ax.plot(benchmark_aligned.index, benchmark_aligned.values,
                               label='中证1000', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title('袋外回测 - 净值曲线对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('净值', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'oos_nav_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 袋外回测净值曲线图: {save_path}")
        
        # 2. 性能指标对比图
        metrics_list = []
        for signal_name, result in oos_results.items():
            metrics = result['metrics']
            metrics_list.append({
                'signal': signal_name,
                **metrics
            })
        
        if len(metrics_list) > 0:
            metrics_df = pd.DataFrame(metrics_list)
            
            # 选择关键指标
            key_metrics = {
                'annual_return': '年化收益率 (%)',
                'sharpe_ratio': '夏普比率',
                'max_drawdown': '最大回撤 (%)',
                'annual_excess_return': '年化超额收益 (%)',
                'information_ratio': '信息比率'
            }
            
            available_metrics = {k: v for k, v in key_metrics.items() if k in metrics_df.columns}
            
            if len(available_metrics) > 0:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                for idx, (metric, label) in enumerate(list(available_metrics.items())[:6]):
                    ax = axes[idx]
                    
                    values = metrics_df[metric].values
                    signals = metrics_df['signal'].values
                    
                    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
                    bars = ax.bar(range(len(signals)), values, color=colors[:len(signals)], 
                                 alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # 添加数值标签
                    for bar, val in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{val:.2f}',
                               ha='center', va='bottom' if height > 0 else 'top',
                               fontsize=10)
                    
                    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
                    ax.set_xlabel('信号', fontsize=10)
                    ax.set_ylabel(label, fontsize=10)
                    ax.set_xticks(range(len(signals)))
                    ax.set_xticklabels(signals, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
                # 隐藏多余的子图
                for i in range(len(available_metrics), 6):
                    axes[i].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, 'oos_metrics_comparison.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✅ 袋外回测指标对比图: {save_path}")
        
        print("✅ 袋外回测可视化完成")


def load_signals_from_directory(signal_path='./signal/'):
    """
    从signal目录加载所有信号数据
    
    Parameters:
    -----------
    signal_path : str, 信号文件目录路径
    
    Returns:
    --------
    dict : 信号数据字典，key为信号名，value为DataFrame（股票×日期）
    """
    signals = {}
    signal_path = os.path.abspath(signal_path)
    
    if not os.path.exists(signal_path):
        print(f"❌ 信号目录不存在: {signal_path}")
        return signals
    
    print(f"📂 从目录加载信号数据: {signal_path}")
    
    # 信号文件列表
    signal_files = [
        'model1_factor_selection_prediction.csv',
        'model2_mlp_prediction.csv',
        'model3_xgboost_prediction.csv',
        'final_signal.csv'
    ]
    
    signal_names = {
        'model1_factor_selection_prediction.csv': '模型1_因子筛选',
        'model2_mlp_prediction.csv': '模型2_MLP',
        'model3_xgboost_prediction.csv': '模型3_XGBoost',
        'final_signal.csv': '最终信号'
    }
    
    for signal_file in signal_files:
        signal_name = signal_names.get(signal_file, signal_file.replace('.csv', ''))
        signal_file_path = os.path.join(signal_path, signal_file)
        
        if os.path.exists(signal_file_path):
            try:
                signal_df = pd.read_csv(signal_file_path, index_col=0, encoding='utf-8-sig')
                signal_df.columns = pd.to_datetime(signal_df.columns)
                signals[signal_name] = signal_df
                print(f"   ✅ {signal_name}: {signal_df.shape}")
            except Exception as e:
                print(f"   ⚠️  {signal_name}: 加载失败 - {e}")
        else:
            print(f"   ⚠️  {signal_name}: 文件不存在")
    
    print(f"\n✅ 共加载 {len(signals)} 个信号")
    return signals


def load_benchmark_data(data_path='./data/'):
    """
    加载中证1000指数数据作为基准
    
    Parameters:
    -----------
    data_path : str, 数据文件目录路径
    
    Returns:
    --------
    pd.DataFrame : 基准数据（包含日期和收益率）
    """
    data_path = os.path.abspath(data_path)
    market_file = os.path.join(data_path, 'market_index_data.csv')
    
    if not os.path.exists(market_file):
        print(f"⚠️  市场指数数据文件不存在: {market_file}")
        print("   将使用价格数据计算中证1000指数收益率（等权重）")
        return None  # 返回None，后续可以从价格数据计算
    
    try:
        print(f"📂 加载市场指数数据: {market_file}")
        market_data = pd.read_csv(market_file, encoding='utf-8-sig')
        
        # 尝试不同的字段名
        date_col = None
        return_col = None
        
        # 查找日期列
        for col in market_data.columns:
            if 'DATE' in col.upper() or 'DT' in col.upper() or '日期' in col:
                date_col = col
                break
        
        # 查找收益率列
        for col in market_data.columns:
            if 'RETURN' in col.upper() or '收益率' in col or 'MARKET_RETURN' in col.upper():
                return_col = col
                break
        
        # 如果没有找到收益率列，尝试从价格计算
        if date_col and not return_col:
            # 查找价格列
            price_col = None
            for col in market_data.columns:
                if 'CLOSE' in col.upper() or 'PRICE' in col.upper() or 'INDEX' in col.upper():
                    price_col = col
                    break
            
            if price_col:
                market_data = market_data.sort_values(date_col)
                market_data['RETURN'] = market_data[price_col].pct_change()
                return_col = 'RETURN'
        
        if date_col and return_col:
            market_data[date_col] = pd.to_datetime(market_data[date_col])
            benchmark_df = market_data[[date_col, return_col]].set_index(date_col)
            benchmark_df.columns = ['RETURN']
            benchmark_df = benchmark_df.dropna()
            print(f"✅ 基准数据加载完成，形状: {benchmark_df.shape}")
            return benchmark_df
        else:
            print("⚠️  市场指数数据格式不符合要求，将使用等权重组合作为基准")
            return None
    except Exception as e:
        print(f"⚠️  加载基准数据时出错: {e}")
        print("   将使用等权重组合作为基准")
        return None


def main(signal_path='./signal/', data_path='./data/', results_dir='./results/'):
    """
    主函数：加载信号数据，进行单因子分析和策略回测
    
    Parameters:
    -----------
    signal_path : str, 信号文件目录路径
    data_path : str, 数据文件目录路径
    results_dir : str, 结果保存目录路径
    """
    print("="*80)
    print("中证1000多因子策略 - 策略回测")
    print("="*80)
    print()
    
    # 1. 加载信号数据
    print("【步骤1】加载信号数据")
    print("-"*80)
    signal_data = load_signals_from_directory(signal_path)
    
    if len(signal_data) == 0:
        print("❌ 未加载到任何信号数据，程序终止")
        return
    
    # 2. 加载收益率数据
    print("\n【步骤2】加载收益率数据")
    print("-"*80)
    from factor_combination import load_price_data_and_calculate_returns
    return_data = load_price_data_and_calculate_returns(data_path)
    
    if return_data.empty:
        print("❌ 未加载到收益率数据，程序终止")
        return
    
    # 3. 加载价格数据（用于计算换手率等）
    print("\n【步骤3】加载价格数据")
    print("-"*80)
    from data_collection import load_data_from_csv
    data = load_data_from_csv(data_path)
    price_data = data.get('price_data', None)
    
    # 4. 加载成分股管理器
    print("\n【步骤4】加载成分股管理器")
    print("-"*80)
    from factor_combination import load_constituent_manager
    constituent_manager = load_constituent_manager(data_path)
    
    # 5. 加载基准数据
    print("\n【步骤5】加载基准数据")
    print("-"*80)
    benchmark_data = load_benchmark_data(data_path)
    
    # 6. 初始化回测器
    print("\n【步骤6】初始化策略回测器")
    print("-"*80)
    backtester = StrategyBacktester(
        signal_data=signal_data,
        return_data=return_data,
        benchmark_data=benchmark_data,
        price_data=price_data,
        constituent_manager=constituent_manager
    )
    
    # 7. 单信号分析（IC、IR、分层回测、相关性分析）
    print("\n【步骤7】单信号分析")
    print("-"*80)
    analysis_results = {}
    
    for signal_name in signal_data.keys():
        print(f"\n分析信号: {signal_name}")
        
        # 检查是否已存在分析结果
        summary_file = os.path.join(results_dir, f'{signal_name}_signal_analysis_summary.csv')
        if os.path.exists(summary_file):
            print(f"⏭️  检测到已存在的分析结果: {summary_file}")
            print("   直接加载，跳过计算...")
            try:
                summary_df = pd.read_csv(summary_file, index_col=0, encoding='utf-8-sig')
                analysis_results[signal_name] = {'summary': summary_df}
                print(f"✅ 分析结果加载完成")
                continue
            except Exception as e:
                print(f"⚠️  加载失败，将重新计算: {e}")
        
        try:
            result = backtester.analyze_single_signal(signal_name, save_dir=results_dir)
            analysis_results[signal_name] = result
        except Exception as e:
            print(f"⚠️  信号 {signal_name} 分析失败: {e}")
    
    # 7.5 信号间相关性分析
    print("\n【步骤7.5】信号间相关性分析")
    print("-"*80)
    
    corr_file = os.path.join(results_dir, 'signal_correlation_matrix.csv')
    corr_heatmap_file = os.path.join(results_dir, 'signal_correlation_heatmap.png')
    
    if os.path.exists(corr_file) and os.path.exists(corr_heatmap_file):
        print(f"⏭️  检测到已存在的相关性分析结果")
        print("   直接加载，跳过计算...")
        try:
            corr_matrix = pd.read_csv(corr_file, index_col=0, encoding='utf-8-sig')
            print(f"\n信号相关性矩阵:")
            print(corr_matrix)
            print(f"✅ 相关性分析结果已加载")
        except Exception as e:
            print(f"⚠️  加载失败，将重新计算: {e}")
            corr_matrix = None
    else:
        corr_matrix = None
    
    if corr_matrix is None and len(signal_data) > 1:
        from factor_evaluation import FactorEvaluator
        evaluator = FactorEvaluator(
            factor_data=signal_data,
            return_data=return_data,
            constituent_manager=constituent_manager
        )
        
        corr_matrix = evaluator.calculate_factor_correlation()
        if not corr_matrix.empty:
            print(f"\n信号相关性矩阵:")
            print(corr_matrix)
            
            # 保存相关性矩阵
            corr_matrix.to_csv(corr_file, encoding='utf-8-sig')
            print(f"✅ 相关性矩阵已保存: {corr_file}")
            
            # 绘制相关性热力图
            evaluator.plot_correlation_heatmap(save_path=corr_heatmap_file)
    
    # 8. 策略回测（不同头部比例）
    print("\n【步骤8】策略回测（多头选股）")
    print("-"*80)
    
    # 检查是否已存在回测汇总结果
    backtest_summary_file = os.path.join(results_dir, 'backtest_summary.csv')
    backtest_results = {}
    
    if os.path.exists(backtest_summary_file):
        print(f"⏭️  检测到已存在的回测汇总: {backtest_summary_file}")
        print("   尝试加载已存在的回测结果...")
        try:
            summary_df = pd.read_csv(backtest_summary_file, encoding='utf-8-sig')
            
            # 尝试加载净值序列
            loaded_count = 0
            for _, row in summary_df.iterrows():
                signal_name = row['signal']
                top_pct_key = row['top_pct']
                nav_file = os.path.join(results_dir, f'{signal_name}_{top_pct_key}_nav.csv')
                
                if os.path.exists(nav_file):
                    nav_series = pd.read_csv(nav_file, index_col=0, encoding='utf-8-sig')
                    nav_series.index = pd.to_datetime(nav_series.index)
                    nav_series = nav_series.iloc[:, 0]  # 取第一列
                    
                    if signal_name not in backtest_results:
                        backtest_results[signal_name] = {}
                    
                    backtest_results[signal_name][top_pct_key] = {
                        'nav_series': nav_series,
                        'metrics': row.to_dict()
                    }
                    loaded_count += 1
            
            if loaded_count > 0:
                print(f"✅ 成功加载 {loaded_count} 组回测结果")
                print("   如需重新计算，请删除 backtest_summary.csv 文件")
            else:
                print("⚠️  未找到净值序列文件，将重新计算")
                backtest_results = {}
        except Exception as e:
            print(f"⚠️  加载失败，将重新计算: {e}")
            backtest_results = {}
    
    # 如果没有加载到结果，进行回测
    if not backtest_results:
        backtest_results = backtester.backtest_all_signals(
            top_pcts=[0.05, 0.1, 0.2],
            fee=0.002,
            slippage=0.001,
            rebalance_freq=1,  # 日度调仓
            save_dir=results_dir
        )
    
    # 9. 可视化
    print("\n【步骤9】生成可视化图表")
    print("-"*80)
    backtester.visualize_results(backtest_results, save_dir=results_dir)
    
    # 10. 纯袋外回测（测试集：后20%数据）
    print("\n【步骤10】纯袋外回测（测试集：后20%数据）")
    print("-"*80)
    
    # 检查是否已存在袋外回测结果
    oos_summary_file = os.path.join(results_dir, 'oos_backtest_summary.csv')
    oos_results = {}
    
    if os.path.exists(oos_summary_file):
        print(f"⏭️  检测到已存在的袋外回测汇总: {oos_summary_file}")
        print("   尝试加载已存在的袋外回测结果...")
        try:
            oos_summary_df = pd.read_csv(oos_summary_file, encoding='utf-8-sig')
            
            # 尝试加载净值序列
            loaded_count = 0
            for _, row in oos_summary_df.iterrows():
                signal_name = row['signal']
                nav_file = os.path.join(results_dir, f'{signal_name}_oos_nav.csv')
                
                if os.path.exists(nav_file):
                    nav_series = pd.read_csv(nav_file, index_col=0, encoding='utf-8-sig')
                    nav_series.index = pd.to_datetime(nav_series.index)
                    nav_series = nav_series.iloc[:, 0]  # 取第一列
                    
                    oos_results[signal_name] = {
                        'nav_series': nav_series,
                        'metrics': row.to_dict()
                    }
                    loaded_count += 1
            
            if loaded_count > 0:
                print(f"✅ 成功加载 {loaded_count} 组袋外回测结果")
                print("   如需重新计算，请删除 oos_backtest_summary.csv 文件")
            else:
                print("⚠️  未找到净值序列文件，将重新计算")
                oos_results = {}
        except Exception as e:
            print(f"⚠️  加载失败，将重新计算: {e}")
            oos_results = {}
    
    # 如果没有加载到结果，进行袋外回测
    if not oos_results and backtest_results:
        oos_results = backtester.out_of_sample_backtest(
            all_results=backtest_results,
            train_ratio=0.8,  # 前80%训练，后20%测试
            fee=0.002,
            slippage=0.001,
            rebalance_freq=1,
            save_dir=results_dir,
            metric='sharpe_ratio'  # 使用夏普比率选择最优比例
        )
    
    # 11. 袋外回测可视化
    if len(oos_results) > 0:
        print("\n【步骤11】生成袋外回测可视化图表")
        print("-"*80)
        backtester.visualize_oos_results(oos_results, save_dir=results_dir)
    
    print("\n" + "="*80)
    print("✅ 策略回测完成！")
    print("="*80)
    
    return {
        'analysis_results': analysis_results,
        'backtest_results': backtest_results,
        'oos_results': oos_results
    }


if __name__ == '__main__':
    import sys
    
    # 设置默认路径（可根据实际情况修改）
    signal_path = 'd:/programme/vscode_c/courses/Software Enginerring/signal/'
    data_path = 'd:/programme/vscode_c/courses/Software Enginerring/data/'
    results_dir = 'd:/programme/vscode_c/courses/Software Enginerring/backtest_results/'
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        signal_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    if len(sys.argv) > 3:
        results_dir = sys.argv[3]
    
    main(signal_path=signal_path, data_path=data_path, results_dir=results_dir)
