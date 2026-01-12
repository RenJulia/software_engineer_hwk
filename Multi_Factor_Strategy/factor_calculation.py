# -*- coding: utf-8 -*-
"""
中证1000多因子策略 - 因子计算模块
实现20个因子的计算函数
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
import os
from pathlib import Path
from functools import partial
import time
warnings.filterwarnings('ignore')

# 尝试导入并行计算和numba加速库
try:
    from numba import jit, prange, float64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 定义占位函数
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

class FactorCalculator:
    """因子计算器类"""
    
    def __init__(self, price_data, mv_data=None, turnover_data=None, market_data=None, constituent_manager=None):
        """
        初始化因子计算器
        
        Parameters:
        -----------
        price_data : pd.DataFrame, 价格数据，必须包含列：S_INFO_WINDCODE, TRADE_DT, CLOSE_PRICE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_ADJ, VOLUME, AMOUNT
        mv_data : pd.DataFrame, 市值数据
        turnover_data : pd.DataFrame, 换手率数据
        market_data : pd.DataFrame, 市场指数数据
        constituent_manager : ConstituentManager, 成分股管理器（可选），用于在计算因子时过滤成分股
        """
        self.price_data = price_data.copy()
        self.mv_data = mv_data
        self.turnover_data = turnover_data
        self.market_data = market_data
        self.constituent_manager = constituent_manager
        
        # 数据预处理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """数据预处理：计算收益率等基础指标"""
        # 按股票代码和日期排序
        self.price_data = self.price_data.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
        
        # 确保CLOSE_ADJ存在（如果不存在，使用CLOSE_PRICE）
        if 'CLOSE_ADJ' not in self.price_data.columns:
            if 'CLOSE_PRICE' in self.price_data.columns:
                self.price_data['CLOSE_ADJ'] = self.price_data['CLOSE_PRICE']
                print("⚠️  CLOSE_ADJ字段不存在，使用CLOSE_PRICE代替（未复权）")
            else:
                raise ValueError("价格数据中缺少CLOSE_PRICE或CLOSE_ADJ字段")
        
        # 计算收益率（使用复权收盘价）
        self.price_data['RETURN'] = self.price_data.groupby('S_INFO_WINDCODE')['CLOSE_ADJ'].pct_change()
        
        # 计算对数收益率（某些因子需要）
        self.price_data['LOG_RETURN'] = np.log(self.price_data['CLOSE_ADJ'] / self.price_data.groupby('S_INFO_WINDCODE')['CLOSE_ADJ'].shift(1))
        
        # 使用VWAP字段（如果存在），否则用成交金额/成交量计算，再否则用收盘价
        if 'VWAP' not in self.price_data.columns or self.price_data['VWAP'].isna().all():
            if 'AMOUNT' in self.price_data.columns and 'VOLUME' in self.price_data.columns:
                self.price_data['VWAP'] = self.price_data['AMOUNT'] / (self.price_data['VOLUME'] + 1e-10)
                print("⚠️  VWAP字段不存在，使用成交金额/成交量计算")
            else:
                self.price_data['VWAP'] = self.price_data['CLOSE_PRICE']
                print("⚠️  VWAP字段不存在，使用收盘价代替")
        else:
            # 如果VWAP字段存在但有缺失值，用成交金额/成交量填充
            if self.price_data['VWAP'].isna().any():
                mask = self.price_data['VWAP'].isna()
                if 'AMOUNT' in self.price_data.columns and 'VOLUME' in self.price_data.columns:
                    self.price_data.loc[mask, 'VWAP'] = (
                        self.price_data.loc[mask, 'AMOUNT'] / 
                        (self.price_data.loc[mask, 'VOLUME'] + 1e-10)
                    )
                else:
                    self.price_data.loc[mask, 'VWAP'] = self.price_data.loc[mask, 'CLOSE_PRICE']
        
        # 如果市场数据存在，合并
        if self.market_data is not None:
            self.price_data = self.price_data.merge(
                self.market_data[['TRADE_DT', 'MARKET_RETURN']], 
                on='TRADE_DT', 
                how='left'
            )
        
        # 合并市值数据
        if self.mv_data is not None:
            self.price_data = self.price_data.merge(
                self.mv_data[['S_INFO_WINDCODE', 'TRADE_DT', 'TOTAL_MV']],
                on=['S_INFO_WINDCODE', 'TRADE_DT'],
                how='left'
            )
        
        # 合并换手率数据
        if self.turnover_data is not None:
            # 检查换手率数据中是否有TURNOVER_RATE字段
            if 'TURNOVER_RATE' in self.turnover_data.columns:
                self.price_data = self.price_data.merge(
                    self.turnover_data[['S_INFO_WINDCODE', 'TRADE_DT', 'TURNOVER_RATE']],
                    on=['S_INFO_WINDCODE', 'TRADE_DT'],
                    how='left'
                )
            elif 'AMOUNT' in self.turnover_data.columns:
                # 如果没有TURNOVER_RATE，合并成交金额，后续可以结合市值计算换手率
                self.price_data = self.price_data.merge(
                    self.turnover_data[['S_INFO_WINDCODE', 'TRADE_DT', 'AMOUNT']],
                    on=['S_INFO_WINDCODE', 'TRADE_DT'],
                    how='left',
                    suffixes=('', '_turnover')
                )
        
        # 如果换手率不存在，尝试从成交金额和市值计算
        if 'TURNOVER_RATE' not in self.price_data.columns or self.price_data['TURNOVER_RATE'].isna().all():
            if 'AMOUNT' in self.price_data.columns and 'TOTAL_MV' in self.price_data.columns:
                self.price_data['TURNOVER_RATE'] = self.price_data['AMOUNT'] / (self.price_data['TOTAL_MV'] + 1e8)
                print("⚠️  换手率字段不存在，使用成交金额/市值计算")
            elif 'AMOUNT' in self.price_data.columns:
                # 如果没有市值数据，暂时用0填充，后续需要市值数据
                self.price_data['TURNOVER_RATE'] = 0.0
                print("⚠️  换手率字段不存在且缺少市值数据，暂时设为0")
            else:
                self.price_data['TURNOVER_RATE'] = 0.05  # 默认值
                print("⚠️  换手率字段不存在，使用默认值0.05")
    
    def pivot_to_wide_format(self, df, value_col, date_col='TRADE_DT', stock_col='S_INFO_WINDCODE'):
        """
        将长格式数据转换为宽格式（股票为行，日期为列）
        
        Parameters:
        -----------
        df : pd.DataFrame
        value_col : str, 值列名
        date_col : str, 日期列名
        stock_col : str, 股票代码列名
        
        Returns:
        --------
        pd.DataFrame : 宽格式数据，index为股票代码，columns为日期
        """
        # 使用pivot而不是pivot_table，避免聚合
        df_wide = df.pivot(
            index=stock_col,
            columns=date_col,
            values=value_col
        )
        return df_wide
    
    # ==================== Factor 1: SCC ====================
    def calculate_SCC(self, window=252):
        """
        Factor 1: Spatial Centrality Centrality (SCC)
        基于股票间相关系数的空间中心性
        使用numba加速优化
        """
        print("计算因子1: SCC...")
        returns_wide = self.pivot_to_wide_format(
            self.price_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RETURN']].dropna(),
            'RETURN'
        )
        
        scc_factors = []
        dates = returns_wide.columns
        returns_matrix = returns_wide.values  # 转换为numpy数组
        
        for i, date in enumerate(dates):
            if i < window:
                continue
            
            # 获取过去window天的收益率数据
            start_idx = max(0, i - window + 1)
            returns_window = returns_matrix[:, start_idx:i+1]
            
            # 使用优化的相关系数计算
            if NUMBA_AVAILABLE and returns_window.shape[1] >= window // 2:
                # 使用numba加速版本
                try:
                    avg_corr_array = _calculate_scc_optimized(returns_window)
                    avg_corr = pd.Series(avg_corr_array, index=returns_wide.index)
                except Exception as e:
                    # 回退到pandas版本
                    returns_window_df = pd.DataFrame(
                        returns_window, 
                        index=returns_wide.index,
                        columns=dates[start_idx:i+1]
                    )
                    correlations = returns_window_df.T.corr()
                    avg_corr = correlations.mean(axis=1) - 1 / len(correlations)
                    avg_corr = avg_corr * len(correlations) / (len(correlations) - 1)
            else:
                # 使用pandas版本
                returns_window_df = pd.DataFrame(
                    returns_window, 
                    index=returns_wide.index,
                    columns=dates[start_idx:i+1]
                )
                correlations = returns_window_df.T.corr()
                avg_corr = correlations.mean(axis=1) - 1 / len(correlations)
                avg_corr = avg_corr * len(correlations) / (len(correlations) - 1)
            
            scc_series = pd.Series(avg_corr.values, index=returns_wide.index, name=date)
            scc_factors.append(scc_series)
        
        scc_df = pd.DataFrame(scc_factors).T
        print(f"✅ SCC因子计算完成，形状: {scc_df.shape}")
        return scc_df
    
    # ==================== Factor 2: TCC ====================
    def calculate_TCC(self, window=252):
        """
        Factor 2: Temporal Centrality Centrality (TCC)
        时间维度上的中心性，衡量股票收益率相对市场平均的稳定性
        """
        print("计算因子2: TCC...")
        
        if 'MARKET_RETURN' not in self.price_data.columns:
            print("⚠️  缺少市场收益率数据，使用截面均值代替")
            self.price_data['MARKET_RETURN'] = self.price_data.groupby('TRADE_DT')['RETURN'].transform('mean')
        
        # 计算标准化偏差
        self.price_data['Z_SCORE'] = self.price_data.groupby('TRADE_DT').apply(
            lambda x: (x['RETURN'] - x['RETURN'].mean()) / (x['RETURN'].std() + 1e-8)
        ).reset_index(level=0, drop=True)
        
        self.price_data['Z_SQUARED'] = self.price_data['Z_SCORE'] ** 2
        
        # 滚动窗口计算E[z²]的倒数
        tcc_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT')
            stock_data['TCC'] = 1 / (stock_data['Z_SQUARED'].rolling(window=window, min_periods=60).mean() + 1e-8)
            tcc_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'TCC']])
        
        tcc_df = pd.concat(tcc_list, ignore_index=True)
        tcc_wide = self.pivot_to_wide_format(tcc_df, 'TCC')
        
        print(f"✅ TCC因子计算完成，形状: {tcc_wide.shape}")
        return tcc_wide
    
    # ==================== Factor 3: APB ====================
    def calculate_APB(self, window=20):
        """
        Factor 3: Average Price Bias (APB)
        平均价格偏差，衡量买卖压力
        """
        print("计算因子3: APB...")
        
        apb_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算等权重平均价格
            stock_data['EW_PRICE'] = stock_data['CLOSE_PRICE'].rolling(window=window).mean()
            
            # 计算成交量加权平均价格（VWAP）
            stock_data['VWAP_WINDOW'] = (
                (stock_data['CLOSE_PRICE'] * stock_data['VOLUME']).rolling(window=window).sum() /
                stock_data['VOLUME'].rolling(window=window).sum()
            )
            
            # APB = log(等权重平均价格 / VWAP)
            stock_data['APB'] = np.log(stock_data['EW_PRICE'] / (stock_data['VWAP_WINDOW'] + 1e-8))
            
            apb_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'APB']])
        
        apb_df = pd.concat(apb_list, ignore_index=True)
        apb_wide = self.pivot_to_wide_format(apb_df, 'APB')
        
        print(f"✅ APB因子计算完成，形状: {apb_wide.shape}")
        return apb_wide
    
    # ==================== Factor 4-7: ARC/VRC/SRC/KRC ====================
    def calculate_relative_cost_moments(self, max_lookback=252):
        """
        Factor 4-7: ARC/VRC/SRC/KRC (Average/Variance/Skewness/Kurtosis of Relative Cost)
        相对成本的各阶矩
        使用numba加速优化
        """
        print("计算因子4-7: ARC/VRC/SRC/KRC...")
        
        if 'TURNOVER_RATE' not in self.price_data.columns:
            print("⚠️  缺少换手率数据，使用默认值")
            self.price_data['TURNOVER_RATE'] = 0.05
        
        # 使用并行计算处理多只股票
        if JOBLIB_AVAILABLE:
            import multiprocessing
            n_jobs = min(multiprocessing.cpu_count(), 4)  # 限制最大并行数，避免内存问题
            print(f"   使用并行计算（{n_jobs}个核心）...")
            
            stocks = self.price_data['S_INFO_WINDCODE'].unique().tolist()
            results_list = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(self._calculate_relative_cost_single_stock)(stock, max_lookback)
                for stock in stocks
            )
            
            # 合并结果
            results = {'ARC': [], 'VRC': [], 'SRC': [], 'KRC': []}
            for stock_results in results_list:
                for col in ['ARC', 'VRC', 'SRC', 'KRC']:
                    if stock_results[col] is not None:
                        results[col].append(stock_results[col])
        else:
            # 串行计算
            results = {}
            stocks = self.price_data['S_INFO_WINDCODE'].unique()
            for idx, stock in enumerate(stocks):
                if (idx + 1) % 100 == 0:
                    print(f"   处理进度: {idx + 1}/{len(stocks)}")
                stock_result = self._calculate_relative_cost_single_stock(stock, max_lookback)
                for col in ['ARC', 'VRC', 'SRC', 'KRC']:
                    if col not in results:
                        results[col] = []
                    if stock_result[col] is not None:
                        results[col].append(stock_result[col])
        
        factor_dfs = {}
        for factor_name, factor_list in results.items():
            if len(factor_list) > 0:
                factor_df = pd.concat(factor_list, ignore_index=True)
                factor_wide = self.pivot_to_wide_format(factor_df, factor_name)
                factor_dfs[factor_name] = factor_wide
        
        print(f"✅ ARC/VRC/SRC/KRC因子计算完成")
        return factor_dfs
    
    def _calculate_relative_cost_single_stock(self, stock, max_lookback):
        """计算单只股票的相对成本矩（用于并行计算）"""
        stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
        
        if len(stock_data) <= max_lookback:
            return {'ARC': None, 'VRC': None, 'SRC': None, 'KRC': None}
        
        arc_list, vrc_list, src_list, krc_list = [], [], [], []
        prices = stock_data['CLOSE_ADJ'].values
        turnovers = stock_data['TURNOVER_RATE'].fillna(0.05).values
        
        for i in range(len(stock_data)):
            if i < max_lookback:
                arc_list.append(np.nan)
                vrc_list.append(np.nan)
                src_list.append(np.nan)
                krc_list.append(np.nan)
                continue
            
            # 获取历史数据窗口
            start_idx = max(0, i - max_lookback + 1)
            hist_prices = prices[start_idx:i+1]
            hist_turnovers = turnovers[start_idx:i+1]
            current_price = prices[i]
            
            # 计算相对收益率
            relative_returns = np.full(len(hist_prices), np.nan)
            for j in range(1, len(hist_prices)):
                if hist_prices[j-1] > 0:
                    relative_returns[j] = (current_price / hist_prices[j-1] - 1)
            relative_returns[0] = 0.0
            
            # 计算换手率权重
            turnover_weights = np.zeros(len(hist_prices))
            for j in range(len(hist_prices)):
                days_ago = len(hist_prices) - 1 - j
                turnover = hist_turnovers[j] if not np.isnan(hist_turnovers[j]) else 0.05
                
                # 计算生存概率
                if days_ago == 0:
                    survival_prob = 1.0
                else:
                    avg_turnover = np.nanmean(hist_turnovers[:j+1])
                    if np.isnan(avg_turnover):
                        avg_turnover = 0.05
                    survival_prob = (1 - avg_turnover) ** days_ago
                
                turnover_weights[j] = turnover * survival_prob
            
            # 归一化权重
            weight_sum = np.nansum(turnover_weights)
            if weight_sum > 1e-10:
                turnover_weights = turnover_weights / weight_sum
            else:
                turnover_weights = np.ones(len(hist_prices)) / len(hist_prices)
            
            # 使用numba加速的加权矩计算
            valid_mask = ~np.isnan(relative_returns)
            if valid_mask.sum() < 10:  # 至少需要10个有效数据点
                arc_list.append(np.nan)
                vrc_list.append(np.nan)
                src_list.append(np.nan)
                krc_list.append(np.nan)
                continue
            
            valid_returns = relative_returns[valid_mask]
            valid_weights = turnover_weights[valid_mask]
            valid_weights = valid_weights / np.sum(valid_weights)  # 重新归一化
            
            # 计算加权矩
            arc, vrc, src, krc = _calculate_weighted_moments_numba(valid_returns, valid_weights)
            
            arc_list.append(arc if not np.isnan(arc) else 0.0)
            vrc_list.append(vrc if not np.isnan(vrc) else 0.0)
            src_list.append(src if not np.isnan(src) else 0.0)
            krc_list.append(krc if not np.isnan(krc) else 0.0)
        
        stock_data['ARC'] = arc_list
        stock_data['VRC'] = vrc_list
        stock_data['SRC'] = src_list
        stock_data['KRC'] = krc_list
        
        return {
            'ARC': stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'ARC']],
            'VRC': stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'VRC']],
            'SRC': stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'SRC']],
            'KRC': stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'KRC']]
        }
    
    # ==================== Factor 8: 20-day Price Bias ====================
    def calculate_BIAS(self, window=20):
        """
        Factor 8: 20-day Price Bias (BIAS)
        价格偏离20日均线的程度
        """
        print("计算因子8: BIAS...")
        
        bias_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算20日移动平均
            stock_data['MA20'] = stock_data['CLOSE_PRICE'].rolling(window=window).mean()
            
            # BIAS = (当前价格 - MA20) / MA20
            stock_data['BIAS'] = (stock_data['CLOSE_PRICE'] - stock_data['MA20']) / (stock_data['MA20'] + 1e-8)
            
            bias_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'BIAS']])
        
        bias_df = pd.concat(bias_list, ignore_index=True)
        bias_wide = self.pivot_to_wide_format(bias_df, 'BIAS')
        
        print(f"✅ BIAS因子计算完成，形状: {bias_wide.shape}")
        return bias_wide
    
    # ==================== Factor 9: 20-day Turnover Bias ====================
    def calculate_TurnoverBias(self, window=20):
        """
        Factor 9: 20-day Turnover Bias
        换手率偏离20日均值的程度
        """
        print("计算因子9: TurnoverBias...")
        
        if 'TURNOVER_RATE' not in self.price_data.columns:
            print("⚠️  缺少换手率数据，使用成交量/市值近似")
            self.price_data['TURNOVER_RATE'] = self.price_data['AMOUNT'] / (self.price_data['TOTAL_MV'] + 1e8)
        
        turnover_bias_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算20日换手率均值
            stock_data['TURNOVER_MA20'] = stock_data['TURNOVER_RATE'].rolling(window=window).mean()
            
            # Turnover Bias = (当前换手率 - MA20) / MA20
            stock_data['TURNOVER_BIAS'] = (stock_data['TURNOVER_RATE'] - stock_data['TURNOVER_MA20']) / (stock_data['TURNOVER_MA20'] + 1e-8)
            
            turnover_bias_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'TURNOVER_BIAS']])
        
        turnover_bias_df = pd.concat(turnover_bias_list, ignore_index=True)
        turnover_bias_wide = self.pivot_to_wide_format(turnover_bias_df, 'TURNOVER_BIAS')
        
        print(f"✅ TurnoverBias因子计算完成，形状: {turnover_bias_wide.shape}")
        return turnover_bias_wide
    
    # ==================== Factor 10: Ratio of New High Days ====================
    def calculate_NewHighRatio(self, window=20, lookback=20):
        """
        Factor 10: Ratio of New High Days (20-day)
        过去20天内创新高的天数比例
        """
        print("计算因子10: NewHighRatio...")
        
        newhigh_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算滚动20日最高价
            stock_data['ROLLING_HIGH'] = stock_data['HIGH_PRICE'].rolling(window=window).max()
            
            # 判断每日是否创新高
            stock_data['IS_NEW_HIGH'] = (stock_data['HIGH_PRICE'] >= stock_data['ROLLING_HIGH']).astype(int)
            
            # 计算过去lookback天内创新高的天数比例
            stock_data['NEW_HIGH_RATIO'] = stock_data['IS_NEW_HIGH'].rolling(window=lookback).sum() / lookback
            
            newhigh_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'NEW_HIGH_RATIO']])
        
        newhigh_df = pd.concat(newhigh_list, ignore_index=True)
        newhigh_wide = self.pivot_to_wide_format(newhigh_df, 'NEW_HIGH_RATIO')
        
        print(f"✅ NewHighRatio因子计算完成，形状: {newhigh_wide.shape}")
        return newhigh_wide
    
    # ==================== Factor 11: Volatility Factor ====================
    def calculate_VolatilityFactor(self, window=20):
        """
        Factor 11: Volatility Factor (ID_Vol, ID_Vol_deCorr)
        特质波动率因子，使用Fama-French三因子模型残差
        """
        print("计算因子11: VolatilityFactor...")
        
        # 简化版本：直接使用收益率的标准差作为特质波动率
        # 完整版本需要使用Fama-French三因子模型回归
        
        vol_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算滚动窗口的特质波动率（简化：使用收益率标准差）
            stock_data['ID_VOL'] = stock_data['RETURN'].rolling(window=window).std() * np.sqrt(252)  # 年化
            
            vol_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'ID_VOL']])
        
        vol_df = pd.concat(vol_list, ignore_index=True)
        vol_wide = self.pivot_to_wide_format(vol_df, 'ID_VOL')
        
        # 月度截面去相关（简化处理）
        vol_decorr = vol_wide.copy()
        # 这里应该进行月度截面回归，简化处理略过
        
        print(f"✅ VolatilityFactor因子计算完成，形状: {vol_wide.shape}")
        return vol_wide, vol_decorr
    
    # ==================== Factor 12: Turnover Rate Factor ====================
    def calculate_TurnoverFactor(self, window=20):
        """
        Factor 12: Turnover Rate Factor (Turn20)
        过去20天的平均换手率，市值调整
        """
        print("计算因子12: TurnoverFactor...")
        
        if 'TURNOVER_RATE' not in self.price_data.columns:
            print("⚠️  缺少换手率数据，使用成交量/市值近似")
            self.price_data['TURNOVER_RATE'] = self.price_data['AMOUNT'] / (self.price_data['TOTAL_MV'] + 1e8)
        
        turnover_list = []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算过去20天平均换手率
            stock_data['TURN20'] = stock_data['TURNOVER_RATE'].rolling(window=window).mean()
            
            turnover_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'TURN20']])
        
        turnover_df = pd.concat(turnover_list, ignore_index=True)
        turnover_wide = self.pivot_to_wide_format(turnover_df, 'TURN20')
        
        # 市值调整（在截面标准化）
        # 这里简化处理，实际应该按市值分组调整
        
        print(f"✅ TurnoverFactor因子计算完成，形状: {turnover_wide.shape}")
        return turnover_wide
    
    # ==================== Factor 13-14: CGO & RCGO ====================
    def calculate_CGO_RCGO(self, window=252):
        """
        Factor 13-14: Capital Gain Overhang (CGO) & Residual CGO (RCGO)
        资本收益悬置因子
        """
        print("计算因子13-14: CGO & RCGO...")
        
        if 'TURNOVER_RATE' not in self.price_data.columns:
            print("⚠️  缺少换手率数据，使用默认值")
            self.price_data['TURNOVER_RATE'] = 0.05
        
        cgo_list, rcgo_list = [], []
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            cgo_values, rcgo_values = [], []
            
            for i in range(len(stock_data)):
                if i < window:
                    cgo_values.append(np.nan)
                    rcgo_values.append(np.nan)
                    continue
                
                hist_data = stock_data.iloc[max(0, i-window+1):i+1].copy()
                hist_data = hist_data.reset_index(drop=True)
                
                current_price = hist_data.iloc[-1]['CLOSE_ADJ']
                
                # 计算平均成本（简化版本）
                # 完整版本需要根据换手率和价格历史计算
                avg_cost = hist_data['CLOSE_ADJ'].mean()
                
                # CGO = (当前价格 - 平均成本) / 平均成本
                cgo = (current_price - avg_cost) / (avg_cost + 1e-8)
                cgo_values.append(cgo)
                
                # RCGO需要通过回归获得残差（简化处理）
                # 这里简化：使用CGO减去市场均值
                rcgo = cgo  # 实际应该回归去相关
                rcgo_values.append(rcgo)
            
            stock_data['CGO'] = cgo_values
            stock_data['RCGO'] = rcgo_values
            
            cgo_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'CGO']])
            rcgo_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RCGO']])
        
        cgo_df = pd.concat(cgo_list, ignore_index=True)
        cgo_wide = self.pivot_to_wide_format(cgo_df, 'CGO')
        
        rcgo_df = pd.concat(rcgo_list, ignore_index=True)
        rcgo_wide = self.pivot_to_wide_format(rcgo_df, 'RCGO')
        
        print(f"✅ CGO & RCGO因子计算完成")
        return cgo_wide, rcgo_wide
    
    # ==================== Factor 15: SUE ====================
    def calculate_SUE(self):
        """
        Factor 15: Standardized Unexpected Earnings (SUE)
        标准化意外收益，需要财务数据（EPS）
        这里提供框架，实际需要EPS数据
        """
        print("计算因子15: SUE...")
        print("⚠️  SUE因子需要EPS财务数据，当前数据集中不包含，返回空DataFrame")
        
        # 创建空的DataFrame结构
        dates = self.price_data['TRADE_DT'].unique()
        stocks = self.price_data['S_INFO_WINDCODE'].unique()
        sue_wide = pd.DataFrame(index=stocks, columns=dates)
        sue_wide[:] = np.nan
        
        print("✅ SUE因子框架已创建（需要EPS数据填充）")
        return sue_wide
    
    # ==================== Factor 16-17: Candle Shadow Factors ====================
    def calculate_CandleShadowFactors(self, window=20, norm_window=5):
        """
        Factor 16-17: CandleAbove & CandleBelow shadow factors
        K线图上影线和下影线因子
        """
        print("计算因子16-17: CandleShadowFactors...")
        
        above_mean_list, above_std_list = [], []
        below_mean_list, below_std_list = [], []
        
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 计算上影线长度 = max(开盘价, 收盘价) - 最高价的绝对值
            stock_data['UPPER_SHADOW'] = np.abs(stock_data[['OPEN_PRICE', 'CLOSE_PRICE']].max(axis=1) - stock_data['HIGH_PRICE'])
            
            # 计算下影线长度 = 最低价 - min(开盘价, 收盘价)
            stock_data['LOWER_SHADOW'] = stock_data['LOW_PRICE'] - stock_data[['OPEN_PRICE', 'CLOSE_PRICE']].min(axis=1)
            
            # 5日均值标准化
            stock_data['UPPER_SHADOW_NORM'] = stock_data['UPPER_SHADOW'] / (stock_data['UPPER_SHADOW'].rolling(window=norm_window).mean() + 1e-8)
            stock_data['LOWER_SHADOW_NORM'] = stock_data['LOWER_SHADOW'] / (stock_data['LOWER_SHADOW'].rolling(window=norm_window).mean() + 1e-8)
            
            # 20日均值和标准差
            stock_data['CANDLE_ABOVE_MEAN'] = stock_data['UPPER_SHADOW_NORM'].rolling(window=window).mean()
            stock_data['CANDLE_ABOVE_STD'] = stock_data['UPPER_SHADOW_NORM'].rolling(window=window).std()
            stock_data['CANDLE_BELOW_MEAN'] = stock_data['LOWER_SHADOW_NORM'].rolling(window=window).mean()
            stock_data['CANDLE_BELOW_STD'] = stock_data['LOWER_SHADOW_NORM'].rolling(window=window).std()
            
            above_mean_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'CANDLE_ABOVE_MEAN']])
            above_std_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'CANDLE_ABOVE_STD']])
            below_mean_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'CANDLE_BELOW_MEAN']])
            below_std_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'CANDLE_BELOW_STD']])
        
        factors = {}
        for name, data_list in [
            ('CANDLE_ABOVE_MEAN', above_mean_list),
            ('CANDLE_ABOVE_STD', above_std_list),
            ('CANDLE_BELOW_MEAN', below_mean_list),
            ('CANDLE_BELOW_STD', below_std_list)
        ]:
            factor_df = pd.concat(data_list, ignore_index=True)
            factors[name] = self.pivot_to_wide_format(factor_df, name)
        
        print(f"✅ CandleShadowFactors因子计算完成")
        return factors
    
    # ==================== Factor 18-19: Williams Shadow Factors ====================
    def calculate_WilliamsShadowFactors(self, window=20, norm_window=5):
        """
        Factor 18-19: WilliamsAbove & WilliamsBelow shadow factors
        基于收盘价的Williams上影线和下影线因子
        """
        print("计算因子18-19: WilliamsShadowFactors...")
        
        williams_above_mean_list, williams_above_std_list = [], []
        williams_below_mean_list, williams_below_std_list = [], []
        
        for stock in self.price_data['S_INFO_WINDCODE'].unique():
            stock_data = self.price_data[self.price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT').copy()
            
            # 基于收盘价重新定义上影线和下影线
            stock_data['WILLIAMS_UPPER_SHADOW'] = np.abs(stock_data['CLOSE_PRICE'] - stock_data['HIGH_PRICE'])
            stock_data['WILLIAMS_LOWER_SHADOW'] = stock_data['CLOSE_PRICE'] - stock_data['LOW_PRICE']
            
            # 5日均值标准化
            stock_data['WILLIAMS_UPPER_SHADOW_NORM'] = stock_data['WILLIAMS_UPPER_SHADOW'] / (stock_data['WILLIAMS_UPPER_SHADOW'].rolling(window=norm_window).mean() + 1e-8)
            stock_data['WILLIAMS_LOWER_SHADOW_NORM'] = stock_data['WILLIAMS_LOWER_SHADOW'] / (stock_data['WILLIAMS_LOWER_SHADOW'].rolling(window=norm_window).mean() + 1e-8)
            
            # 20日均值和标准差
            stock_data['WILLIAMS_ABOVE_MEAN'] = stock_data['WILLIAMS_UPPER_SHADOW_NORM'].rolling(window=window).mean()
            stock_data['WILLIAMS_ABOVE_STD'] = stock_data['WILLIAMS_UPPER_SHADOW_NORM'].rolling(window=window).std()
            stock_data['WILLIAMS_BELOW_MEAN'] = stock_data['WILLIAMS_LOWER_SHADOW_NORM'].rolling(window=window).mean()
            stock_data['WILLIAMS_BELOW_STD'] = stock_data['WILLIAMS_LOWER_SHADOW_NORM'].rolling(window=window).std()
            
            williams_above_mean_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'WILLIAMS_ABOVE_MEAN']])
            williams_above_std_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'WILLIAMS_ABOVE_STD']])
            williams_below_mean_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'WILLIAMS_BELOW_MEAN']])
            williams_below_std_list.append(stock_data[['S_INFO_WINDCODE', 'TRADE_DT', 'WILLIAMS_BELOW_STD']])
        
        factors = {}
        for name, data_list in [
            ('WILLIAMS_ABOVE_MEAN', williams_above_mean_list),
            ('WILLIAMS_ABOVE_STD', williams_above_std_list),
            ('WILLIAMS_BELOW_MEAN', williams_below_mean_list),
            ('WILLIAMS_BELOW_STD', williams_below_std_list)
        ]:
            factor_df = pd.concat(data_list, ignore_index=True)
            factors[name] = self.pivot_to_wide_format(factor_df, name)
        
        print(f"✅ WilliamsShadowFactors因子计算完成")
        return factors
    
    # ==================== Factor 20: UBL ====================
    def calculate_UBL(self, candle_above_std, williams_below_mean):
        """
        Factor 20: UBL (Up & Bottom Line) factor
        综合上影线和下影线因子
        """
        print("计算因子20: UBL...")
        
        # 对齐日期和股票
        common_dates = set(candle_above_std.columns) & set(williams_below_mean.columns)
        common_stocks = set(candle_above_std.index) & set(williams_below_mean.index)
        
        candle_above_std_aligned = candle_above_std.loc[list(common_stocks), list(common_dates)]
        williams_below_mean_aligned = williams_below_mean.loc[list(common_stocks), list(common_dates)]
        
        # 市值中性化（简化处理：截面标准化）
        # 实际应该按市值分组进行中性化
        candle_above_std_neutral = candle_above_std_aligned.sub(candle_above_std_aligned.mean(axis=0), axis=1)
        williams_below_mean_neutral = williams_below_mean_aligned.sub(williams_below_mean_aligned.mean(axis=0), axis=1)
        
        # 截面标准化
        candle_above_std_std = candle_above_std_neutral.div(candle_above_std_neutral.std(axis=0) + 1e-8, axis=1)
        williams_below_mean_std = williams_below_mean_neutral.div(williams_below_mean_neutral.std(axis=0) + 1e-8, axis=1)
        
        # 等权线性组合
        ubl = (candle_above_std_std + williams_below_mean_std) / 2
        
        print(f"✅ UBL因子计算完成，形状: {ubl.shape}")
        return ubl
    
    # ==================== 计算所有因子 ====================
    def calculate_all_factors(self, filter_by_constituents=False):
        """
        计算所有20个因子
        
        Parameters:
        -----------
        filter_by_constituents : bool
            是否在计算完成后过滤非成分股的因子值。
            False（默认）：保留所有股票的因子值（推荐，因为回测阶段会过滤）
            True：只保留成分股的因子值（适用于某些需要截面数据的因子）
        
        Returns:
        --------
        dict : 所有因子数据
        """
        print("\n" + "="*60)
        print("开始计算所有因子...")
        print("="*60 + "\n")
        
        all_factors = {}
        
        # Factor 1-2
        all_factors['SCC'] = self.calculate_SCC()
        all_factors['TCC'] = self.calculate_TCC()
        
        # Factor 3
        all_factors['APB'] = self.calculate_APB()
        
        # Factor 4-7
        relative_cost_factors = self.calculate_relative_cost_moments()
        all_factors.update(relative_cost_factors)
        
        # Factor 8-10
        all_factors['BIAS'] = self.calculate_BIAS()
        all_factors['TURNOVER_BIAS'] = self.calculate_TurnoverBias()
        all_factors['NEW_HIGH_RATIO'] = self.calculate_NewHighRatio()
        
        # Factor 11
        vol_factor, vol_decorr = self.calculate_VolatilityFactor()
        all_factors['ID_VOL'] = vol_factor
        all_factors['ID_VOL_DECORR'] = vol_decorr
        
        # Factor 12
        all_factors['TURN20'] = self.calculate_TurnoverFactor()
        
        # Factor 13-14
        cgo, rcgo = self.calculate_CGO_RCGO()
        all_factors['CGO'] = cgo
        all_factors['RCGO'] = rcgo
        
        # Factor 15
        all_factors['SUE'] = self.calculate_SUE()
        
        # Factor 16-17
        candle_factors = self.calculate_CandleShadowFactors()
        all_factors.update(candle_factors)
        
        # Factor 18-19
        williams_factors = self.calculate_WilliamsShadowFactors()
        all_factors.update(williams_factors)
        
        # Factor 20
        all_factors['UBL'] = self.calculate_UBL(
            all_factors['CANDLE_ABOVE_STD'],
            all_factors['WILLIAMS_BELOW_MEAN']
        )
        
        # 可选：根据成分股过滤因子值（如果需要）
        # 注意：默认不过滤，因为回测阶段会正确过滤。但如果需要只保留成分股的因子值，可以启用
        if filter_by_constituents and self.constituent_manager is not None:
            print("\n根据成分股过滤因子值...")
            for factor_name, factor_df in all_factors.items():
                if factor_df is None or factor_df.empty:
                    continue
                # 对每个日期，只保留该日期的成分股
                filtered_factor = factor_df.copy()
                for date in factor_df.columns:
                    constituents = self.constituent_manager.get_constituents_by_date(date)
                    constituents_set = set(constituents)
                    # 将非成分股的因子值设为NaN
                    mask = ~factor_df.index.isin(constituents_set)
                    filtered_factor.loc[mask, date] = np.nan
                all_factors[factor_name] = filtered_factor
            print("✅ 因子值已根据成分股过滤")
        
        print("\n" + "="*60)
        print("✅ 所有因子计算完成！")
        print("="*60 + "\n")
        
        return all_factors


# ==================== 并行计算辅助函数 ====================

# ==================== Numba加速函数 ====================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _calculate_scc_optimized(returns_window):
        """
        使用numba加速的SCC计算
        returns_window: numpy数组 (n_stocks, n_dates)
        """
        n_stocks, n_dates = returns_window.shape
        avg_corr = np.full(n_stocks, np.nan, dtype=np.float64)
        
        # 计算每只股票的平均相关系数（简化版本，避免使用列表）
        for i in range(n_stocks):
            stock_i = returns_window[i, :]
            if np.isnan(stock_i).all():
                continue
            
            # 计算与所有其他股票的相关性总和和计数
            corr_sum = 0.0
            corr_count = 0
            
            for j in range(n_stocks):
                if i == j:
                    continue
                stock_j = returns_window[j, :]
                
                # 找到同时有效的日期
                valid_count = 0
                sum_i = 0.0
                sum_j = 0.0
                sum_ij = 0.0
                sum_i2 = 0.0
                sum_j2 = 0.0
                
                for k in range(n_dates):
                    if not (np.isnan(stock_i[k]) or np.isnan(stock_j[k])):
                        valid_count += 1
                        sum_i += stock_i[k]
                        sum_j += stock_j[k]
                        sum_ij += stock_i[k] * stock_j[k]
                        sum_i2 += stock_i[k] * stock_i[k]
                        sum_j2 += stock_j[k] * stock_j[k]
                
                if valid_count < n_dates // 2:  # 至少需要一半的有效数据
                    continue
                
                # 计算均值和标准差
                mean_i = sum_i / valid_count
                mean_j = sum_j / valid_count
                
                # 计算方差
                var_i = (sum_i2 / valid_count) - (mean_i * mean_i)
                var_j = (sum_j2 / valid_count) - (mean_j * mean_j)
                
                if var_i > 1e-8 and var_j > 1e-8:
                    # 计算协方差
                    cov_ij = (sum_ij / valid_count) - (mean_i * mean_j)
                    # 计算相关系数
                    corr = cov_ij / (np.sqrt(var_i) * np.sqrt(var_j))
                    if not np.isnan(corr) and not np.isinf(corr):
                        corr_sum += corr
                        corr_count += 1
            
            if corr_count > 0:
                avg_corr[i] = corr_sum / corr_count
        
        return avg_corr

    @jit(nopython=True)
    def _calculate_weighted_moments_numba(values, weights):
        """使用numba加速的加权矩计算"""
        n = len(values)
        if n == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        # 去除NaN
        valid_values = np.empty(n, dtype=np.float64)
        valid_weights = np.empty(n, dtype=np.float64)
        valid_count = 0
        
        weight_sum = 0.0
        for i in range(n):
            if not np.isnan(values[i]) and not np.isnan(weights[i]) and weights[i] > 0:
                valid_values[valid_count] = values[i]
                valid_weights[valid_count] = weights[i]
                weight_sum += weights[i]
                valid_count += 1
        
        if valid_count == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        # 归一化权重
        for i in range(valid_count):
            valid_weights[i] = valid_weights[i] / weight_sum
        
        # 一阶矩（均值）
        mean_val = 0.0
        for i in range(valid_count):
            mean_val += valid_values[i] * valid_weights[i]
        
        # 二阶矩（方差）
        variance = 0.0
        for i in range(valid_count):
            variance += ((valid_values[i] - mean_val) ** 2) * valid_weights[i]
        
        # 三阶标准化矩（偏度）
        if variance > 1e-8:
            skewness = 0.0
            for i in range(valid_count):
                skewness += ((valid_values[i] - mean_val) ** 3) * valid_weights[i]
            skewness = skewness / (variance ** 1.5)
        else:
            skewness = 0.0
        
        # 四阶标准化矩（峰度）
        if variance > 1e-8:
            kurt = 0.0
            for i in range(valid_count):
                kurt += ((valid_values[i] - mean_val) ** 4) * valid_weights[i]
            kurt = kurt / (variance ** 2) - 3.0
        else:
            kurt = 0.0
        
        return mean_val, variance, skewness, kurt
    
    @jit(nopython=True)
    def _rolling_mean_std_numba(values, window):
        """使用numba加速的滚动均值和标准差计算"""
        n = len(values)
        means = np.full(n, np.nan, dtype=np.float64)
        stds = np.full(n, np.nan, dtype=np.float64)
        
        for i in range(window - 1, n):
            window_data = values[i - window + 1:i + 1]
            valid_mask = ~np.isnan(window_data)
            
            if valid_mask.sum() >= window // 2:  # 至少需要一半的有效数据
                valid_data = window_data[valid_mask]
                means[i] = np.mean(valid_data)
                stds[i] = np.std(valid_data)
        
        return means, stds
else:
    # 如果numba不可用，提供回退实现
    def _calculate_scc_optimized(returns_window):
        """回退到numpy实现"""
        return np.full(returns_window.shape[0], np.nan)
    
    def _calculate_weighted_moments_numba(values, weights):
        """回退到numpy实现"""
        valid_mask = ~(np.isnan(values) | np.isnan(weights))
        if valid_mask.sum() == 0:
            return np.nan, np.nan, np.nan, np.nan
        
        valid_values = values[valid_mask]
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)
        
        mean_val = np.sum(valid_values * valid_weights)
        variance = np.sum(((valid_values - mean_val) ** 2) * valid_weights)
        
        if variance > 1e-8:
            skewness = np.sum(((valid_values - mean_val) ** 3) * valid_weights) / (variance ** 1.5)
            kurt = np.sum(((valid_values - mean_val) ** 4) * valid_weights) / (variance ** 2) - 3.0
        else:
            skewness = 0.0
            kurt = 0.0
        
        return mean_val, variance, skewness, kurt
    
    def _rolling_mean_std_numba(values, window):
        """回退到numpy实现"""
        means = pd.Series(values).rolling(window=window).mean().values
        stds = pd.Series(values).rolling(window=window).std().values
        return means, stds


# ==================== 主函数 ====================

def load_data_from_local(data_path='./data/'):
    """
    从本地CSV文件加载数据
    
    Parameters:
    -----------
    data_path : str, 数据文件路径
    
    Returns:
    --------
    dict : 包含所有数据的字典
    """
    data = {}
    data_path = Path(data_path)
    
    try:
        # 加载价格数据
        price_file = data_path / 'stock_price_data.csv'
        if price_file.exists():
            print(f"📂 加载价格数据: {price_file}")
            data['price_data'] = pd.read_csv(price_file, encoding='utf-8-sig')
            data['price_data']['TRADE_DT'] = pd.to_datetime(data['price_data']['TRADE_DT'])
            print(f"   ✅ {len(data['price_data'])} 条记录")
        else:
            print(f"   ❌ 未找到文件: {price_file}")
        
        # 加载市值数据
        mv_file = data_path / 'market_value_data.csv'
        if mv_file.exists():
            print(f"📂 加载市值数据: {mv_file}")
            data['mv_data'] = pd.read_csv(mv_file, encoding='utf-8-sig')
            data['mv_data']['TRADE_DT'] = pd.to_datetime(data['mv_data']['TRADE_DT'])
            print(f"   ✅ {len(data['mv_data'])} 条记录")
        else:
            print(f"   ⚠️  未找到文件: {mv_file}")
        
        # 加载换手率数据
        turnover_file = data_path / 'turnover_rate_data.csv'
        if turnover_file.exists():
            print(f"📂 加载换手率数据: {turnover_file}")
            data['turnover_data'] = pd.read_csv(turnover_file, encoding='utf-8-sig')
            data['turnover_data']['TRADE_DT'] = pd.to_datetime(data['turnover_data']['TRADE_DT'])
            print(f"   ✅ {len(data['turnover_data'])} 条记录")
        else:
            print(f"   ⚠️  未找到文件: {turnover_file}")
        
        # 加载市场数据
        market_file = data_path / 'market_index_data.csv'
        if market_file.exists():
            print(f"📂 加载市场数据: {market_file}")
            data['market_data'] = pd.read_csv(market_file, encoding='utf-8-sig')
            data['market_data']['TRADE_DT'] = pd.to_datetime(data['market_data']['TRADE_DT'])
            print(f"   ✅ {len(data['market_data'])} 条记录")
        else:
            print(f"   ⚠️  未找到文件: {market_file}")
        
        # 加载成分股历史数据（可选）
        constituents_file = data_path / 'csi1000_constituents_history.csv'
        if constituents_file.exists():
            print(f"📂 加载成分股历史数据: {constituents_file}")
            data['constituents_history'] = pd.read_csv(constituents_file, encoding='utf-8-sig')
            data['constituents_history']['S_CON_INDATE'] = pd.to_datetime(
                data['constituents_history']['S_CON_INDATE'], errors='coerce'
            )
            data['constituents_history']['S_CON_OUTDATE'] = pd.to_datetime(
                data['constituents_history']['S_CON_OUTDATE'], errors='coerce'
            )
            print(f"   ✅ {len(data['constituents_history'])} 条记录")
        else:
            print(f"   ⚠️  未找到文件: {constituents_file}（成分股过滤将不可用）")
        
    except Exception as e:
        print(f"❌ 加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return data


def save_factors_to_local(all_factors, factors_path='./factors/'):
    """
    保存因子数据到本地CSV文件
    
    Parameters:
    -----------
    all_factors : dict, 因子数据字典
    factors_path : str, 因子保存路径
    """
    factors_path = Path(factors_path)
    factors_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    failed_count = 0
    
    print(f"\n💾 保存因子数据到: {factors_path.absolute()}")
    
    for factor_name, factor_df in all_factors.items():
        if factor_df is None or factor_df.empty:
            print(f"   ⚠️  {factor_name}: 数据为空，跳过")
            failed_count += 1
            continue
        
        try:
            factor_file = factors_path / f'{factor_name}.csv'
            factor_df.to_csv(factor_file, encoding='utf-8-sig')
            print(f"   ✅ {factor_name}: {factor_df.shape} -> {factor_file.name}")
            saved_count += 1
        except Exception as e:
            print(f"   ❌ {factor_name}: 保存失败 - {e}")
            failed_count += 1
    
    print(f"\n✅ 保存完成: {saved_count} 个因子成功，{failed_count} 个失败")


def main(data_path='./data/', factors_path='./factors/', use_parallel=True, n_jobs=-1):
    """
    主函数：从本地数据计算因子并保存
    
    Parameters:
    -----------
    data_path : str, 数据文件路径
    factors_path : str, 因子保存路径
    use_parallel : bool, 是否使用并行计算（如果可用）
    n_jobs : int, 并行任务数，-1表示使用所有CPU核心
    """
    print("="*80)
    print("中证1000多因子计算 - 本地数据版本")
    print("="*80 + "\n")
    
    # 1. 加载数据
    print("【步骤1】加载本地数据")
    print("-"*80)
    data = load_data_from_local(data_path)
    
    if 'price_data' not in data or data['price_data'].empty:
        print("❌ 价格数据缺失，程序终止")
        return None
    
    # 2. 初始化成分股管理器（如果可用）
    constituent_manager = None
    if 'constituents_history' in data and not data['constituents_history'].empty:
        try:
            from constituent_manager import ConstituentManager
            constituent_manager = ConstituentManager(data['constituents_history'])
            print(f"\n✅ 成分股管理器已初始化")
        except ImportError:
            print(f"\n⚠️  无法导入ConstituentManager，跳过成分股管理")
    
    # 3. 初始化因子计算器
    print("\n【步骤2】初始化因子计算器")
    print("-"*80)
    calculator = FactorCalculator(
        price_data=data['price_data'],
        mv_data=data.get('mv_data'),
        turnover_data=data.get('turnover_data'),
        market_data=data.get('market_data'),
        constituent_manager=constituent_manager
    )
    print(f"✅ 价格数据预处理完成: {len(calculator.price_data)} 条记录")
    print(f"   股票数量: {calculator.price_data['S_INFO_WINDCODE'].nunique()}")
    print(f"   日期范围: {calculator.price_data['TRADE_DT'].min()} 至 {calculator.price_data['TRADE_DT'].max()}")
    
    # 4. 计算所有因子
    print("\n【步骤3】计算所有因子")
    print("-"*80)
    if NUMBA_AVAILABLE:
        print(f"✅ 使用numba JIT加速（如果可用）")
    else:
        print(f"⚠️  numba不可用，使用标准numpy计算")
    
    start_time = time.time()
    
    all_factors = calculator.calculate_all_factors(filter_by_constituents=False)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  因子计算耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    
    # 5. 保存因子数据
    print("\n【步骤4】保存因子数据")
    print("-"*80)
    save_factors_to_local(all_factors, factors_path)
    
    # 6. 生成因子汇总信息
    print("\n【步骤5】因子汇总信息")
    print("-"*80)
    factor_summary = []
    for factor_name, factor_df in all_factors.items():
        if factor_df is not None and not factor_df.empty:
            factor_summary.append({
                '因子名称': factor_name,
                '股票数': factor_df.shape[0],
                '日期数': factor_df.shape[1],
                '缺失率': (factor_df.isna().sum().sum() / (factor_df.shape[0] * factor_df.shape[1]) * 100)
            })
    
    summary_df = pd.DataFrame(factor_summary)
    summary_file = Path(factors_path) / 'factor_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✅ 因子汇总已保存: {summary_file}")
    print(f"\n因子统计:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ 因子计算完成！")
    print("="*80)
    
    return all_factors


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='中证1000多因子计算')
    parser.add_argument('--data_path', type=str, default='d:/programme/vscode_c/courses/Software Enginerring/data/',
                        help='数据文件路径（默认: ./data/）')
    parser.add_argument('--factors_path', type=str, default='d:/programme/vscode_c/courses/Software Enginerring/factors/',
                        help='因子保存路径（默认: ./factors/）')
    parser.add_argument('--no_parallel', action='store_true',
                        help='禁用并行计算')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='并行任务数（-1表示使用所有CPU核心）')
    
    args = parser.parse_args()
    
    try:
        all_factors = main(
            data_path=args.data_path,
            factors_path=args.factors_path,
            use_parallel=not args.no_parallel,
            n_jobs=args.n_jobs
        )
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
