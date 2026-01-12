# -*- coding: utf-8 -*-
"""
中证1000多因子策略 - 因子评估模块
包括单因子分析和多因子分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class FactorEvaluator:
    """因子评估器类"""
    
    def __init__(self, factor_data, return_data, price_data=None, constituent_manager=None):
        """
        初始化因子评估器
        
        Parameters:
        -----------
        factor_data : dict, 因子数据字典，key为因子名，value为DataFrame（股票×日期）
        return_data : pd.DataFrame, 收益率数据，格式同factor_data
        price_data : pd.DataFrame, 价格数据，用于分层回测
        constituent_manager : ConstituentManager, 成分股管理器，用于过滤成分股
        """
        self.factor_data = factor_data
        self.return_data = return_data
        self.price_data = price_data
        self.constituent_manager = constituent_manager
        
        # 对齐数据：确保所有因子和收益率数据的日期和股票一致
        self._align_data()
        
    def _align_data(self):
        """对齐因子数据和收益率数据的日期和股票"""
        print("对齐因子数据和收益率数据...")
        
        # 获取所有因子的公共日期和股票
        all_dates = set(self.return_data.columns)
        all_stocks = set(self.return_data.index)
        
        for factor_name, factor_df in self.factor_data.items():
            if factor_df is None or factor_df.empty:
                continue
            all_dates = all_dates & set(factor_df.columns)
            all_stocks = all_stocks & set(factor_df.index)
        
        # 对齐收益率数据
        self.return_data = self.return_data.loc[list(all_stocks), list(sorted(all_dates))]
        
        # 对齐因子数据
        aligned_factors = {}
        for factor_name, factor_df in self.factor_data.items():
            if factor_df is None or factor_df.empty:
                aligned_factors[factor_name] = pd.DataFrame(index=list(all_stocks), columns=list(sorted(all_dates)))
                aligned_factors[factor_name][:] = np.nan
            else:
                aligned = factor_df.loc[list(all_stocks), list(sorted(all_dates))]
                aligned_factors[factor_name] = aligned
        
        self.factor_data = aligned_factors
        print(f"✅ 数据对齐完成，股票数: {len(all_stocks)}, 日期数: {len(all_dates)}")
    
    # ==================== 单因子分析 ====================
    
    def calculate_IC(self, factor_name, forward_period=1):
        """
        计算IC (Information Coefficient) - 皮尔逊相关系数
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        forward_period : int, 前瞻期数（默认1，即下期收益率）
        
        Returns:
        --------
        pd.Series : 各日期IC值
        """
        if factor_name not in self.factor_data:
            print(f"⚠️  因子 {factor_name} 不存在")
            return pd.Series(dtype=float)
        
        factor_df = self.factor_data[factor_name]
        dates = sorted(set(factor_df.columns) & set(self.return_data.columns))
        
        ic_values = []
        ic_dates = []
        
        for i, date in enumerate(dates):
            if i + forward_period >= len(dates):
                break
            
            # 当期因子值
            factor_values = factor_df[date]
            
            # 未来收益率（forward_period期后）
            future_date = dates[i + forward_period]
            future_returns = self.return_data[future_date]
            
            # 对齐数据
            common_stocks = set(factor_values.index) & set(future_returns.index)
            factor_aligned = factor_values.loc[list(common_stocks)]
            return_aligned = future_returns.loc[list(common_stocks)]
            
            # 如果提供了成分股管理器，只使用成分股计算IC
            if self.constituent_manager is not None:
                constituents = self.constituent_manager.get_constituents_by_date(date)
                constituents_set = set(constituents)
                # 只保留成分股
                factor_aligned = factor_aligned[factor_aligned.index.isin(constituents_set)]
                return_aligned = return_aligned[return_aligned.index.isin(constituents_set)]
            
            # 去除NaN
            valid_mask = ~(factor_aligned.isna() | return_aligned.isna())
            if valid_mask.sum() < 10:  # 至少需要10个有效数据点
                continue
            
            factor_clean = factor_aligned[valid_mask]
            return_clean = return_aligned[valid_mask]
            
            # 计算皮尔逊相关系数（IC）
            if len(factor_clean) > 1 and factor_clean.std() > 1e-8:
                ic = np.corrcoef(factor_clean, return_clean)[0, 1]
                if not np.isnan(ic):
                    ic_values.append(ic)
                    ic_dates.append(date)
        
        ic_series = pd.Series(ic_values, index=ic_dates, name=f'IC_{forward_period}D')
        return ic_series
    
    def calculate_RankIC(self, factor_name, forward_period=1):
        """
        计算RankIC - 斯皮尔曼秩相关系数
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        forward_period : int, 前瞻期数
        
        Returns:
        --------
        pd.Series : 各日期RankIC值
        """
        if factor_name not in self.factor_data:
            print(f"⚠️  因子 {factor_name} 不存在")
            return pd.Series(dtype=float)
        
        factor_df = self.factor_data[factor_name]
        dates = sorted(set(factor_df.columns) & set(self.return_data.columns))
        
        rankic_values = []
        rankic_dates = []
        
        for i, date in enumerate(dates):
            if i + forward_period >= len(dates):
                break
            
            # 当期因子值
            factor_values = factor_df[date]
            
            # 未来收益率
            future_date = dates[i + forward_period]
            future_returns = self.return_data[future_date]
            
            # 对齐数据
            common_stocks = set(factor_values.index) & set(future_returns.index)
            factor_aligned = factor_values.loc[list(common_stocks)]
            return_aligned = future_returns.loc[list(common_stocks)]
            
            # 如果提供了成分股管理器，只使用成分股计算RankIC
            if self.constituent_manager is not None:
                constituents = self.constituent_manager.get_constituents_by_date(date)
                constituents_set = set(constituents)
                # 只保留成分股
                factor_aligned = factor_aligned[factor_aligned.index.isin(constituents_set)]
                return_aligned = return_aligned[return_aligned.index.isin(constituents_set)]
            
            # 去除NaN
            valid_mask = ~(factor_aligned.isna() | return_aligned.isna())
            if valid_mask.sum() < 10:
                continue
            
            factor_clean = factor_aligned[valid_mask]
            return_clean = return_aligned[valid_mask]
            
            # 计算斯皮尔曼秩相关系数（RankIC）
            if len(factor_clean) > 1:
                rankic = stats.spearmanr(factor_clean, return_clean)[0]
                if not np.isnan(rankic):
                    rankic_values.append(rankic)
                    rankic_dates.append(date)
        
        rankic_series = pd.Series(rankic_values, index=rankic_dates, name=f'RankIC_{forward_period}D')
        return rankic_series
    
    def calculate_IR(self, factor_name, forward_period=1):
        """
        计算IR (Information Ratio) = IC均值 / IC标准差
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        forward_period : int, 前瞻期数
        
        Returns:
        --------
        dict : 包含IC均值、IC标准差、IR、IC胜率等指标
        """
        ic_series = self.calculate_IC(factor_name, forward_period)
        
        if len(ic_series) == 0:
            return {
                'IC_Mean': np.nan,
                'IC_Std': np.nan,
                'IR': np.nan,
                'IC_WinRate': np.nan,
                'IC_Skewness': np.nan,
                'IC_Count': 0
            }
        
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / (ic_std + 1e-8)
        ic_winrate = (ic_series > 0).sum() / len(ic_series)
        ic_skewness = ic_series.skew()
        
        return {
            'IC_Mean': ic_mean,
            'IC_Std': ic_std,
            'IR': ir,
            'IC_WinRate': ic_winrate,
            'IC_Skewness': ic_skewness,
            'IC_Count': len(ic_series)
        }
    
    def plot_IC_trend(self, factor_name, forward_period=1, save_path=None):
        """
        绘制IC趋势图
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        forward_period : int, 前瞻期数
        save_path : str, 保存路径
        """
        ic_series = self.calculate_IC(factor_name, forward_period)
        rankic_series = self.calculate_RankIC(factor_name, forward_period)
        
        if len(ic_series) == 0:
            print(f"⚠️  因子 {factor_name} 没有有效的IC数据")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # IC趋势图
        axes[0].plot(ic_series.index, ic_series.values, label='IC', alpha=0.7, linewidth=1.5)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=ic_series.mean(), color='g', linestyle='--', alpha=0.5, label=f'IC均值: {ic_series.mean():.4f}')
        axes[0].set_title(f'{factor_name} - IC趋势图 ({forward_period}日)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('日期', fontsize=12)
        axes[0].set_ylabel('IC值', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # RankIC趋势图
        axes[1].plot(rankic_series.index, rankic_series.values, label='RankIC', alpha=0.7, linewidth=1.5, color='orange')
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=rankic_series.mean(), color='g', linestyle='--', alpha=0.5, label=f'RankIC均值: {rankic_series.mean():.4f}')
        axes[1].set_title(f'{factor_name} - RankIC趋势图 ({forward_period}日)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('日期', fontsize=12)
        axes[1].set_ylabel('RankIC值', fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ IC趋势图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def layer_backtest_single_factor(self, factor_name, layers=5, freq=5, fee=0.002):
        """
        单因子分层回测
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        layers : int, 分层数（默认5层）
        freq : int, 调仓频率（交易日，默认5日）
        fee : float, 交易费率（默认0.002）
        
        Returns:
        --------
        pd.DataFrame : 各层净值序列
        """
        if factor_name not in self.factor_data:
            print(f"⚠️  因子 {factor_name} 不存在")
            return pd.DataFrame()
        
        factor_df = self.factor_data[factor_name]
        return_df = self.return_data
        
        # 对齐数据
        common_dates = sorted(set(factor_df.columns) & set(return_df.columns))
        common_stocks = set(factor_df.index) & set(return_df.index)
        
        factor_aligned = factor_df.loc[list(common_stocks), common_dates]
        return_aligned = return_df.loc[list(common_stocks), common_dates]
        
        # 生成分层权重
        weights = self._gen_layer_weights(factor_aligned, layers=layers, freq=freq)
        
        # 回测
        nav_dict = {}
        for layer_idx in range(layers):
            layer_name = f'第{layer_idx+1}层'
            weight_df = weights[layer_idx]
            
            # 计算各层净值
            nav = self._calculate_layer_nav(weight_df, return_aligned, fee=fee)
            nav_dict[layer_name] = nav
        
        nav_df = pd.DataFrame(nav_dict)
        return nav_df
    
    def _gen_layer_weights(self, factor_df, layers=5, freq=5):
        """
        生成分层权重（参考layer_backtest.py中的gen_layers函数）
        
        重要：如果提供了constituent_manager，将只对成分股进行分层
        
        Parameters:
        -----------
        factor_df : pd.DataFrame, 因子数据（股票×日期）
        layers : int, 分层数
        freq : int, 调仓频率
        
        Returns:
        --------
        list : 各层的权重DataFrame列表
        """
        weights = [pd.DataFrame(index=factor_df.index) for _ in range(layers)]
        dates = sorted(factor_df.columns)
        
        for i in range(freq, len(dates), freq):
            last_date = dates[i - freq]
            date = dates[i]
            
            # 获取上期因子值（用于本期调仓）
            factor_values = factor_df[last_date].dropna()
            
            # 如果提供了成分股管理器，只对成分股进行分层
            if self.constituent_manager is not None:
                # 获取last_date的成分股列表
                constituents = self.constituent_manager.get_constituents_by_date(last_date)
                constituents_set = set(constituents)
                # 只保留成分股的因子值
                factor_values = factor_values[factor_values.index.isin(constituents_set)]
            
            if len(factor_values) < layers:
                continue
            
            # 计算分位数阈值（基于成分股的因子值）
            for layer_idx in range(layers):
                threshold_low = np.nanquantile(factor_values, layer_idx / layers)
                threshold_high = np.nanquantile(factor_values, (layer_idx + 1) / layers)
                
                # 生成权重：该层的股票等权重（只包括成分股）
                weight = pd.Series(0.0, index=factor_df.index)
                layer_mask = (factor_values >= threshold_low) & (factor_values <= threshold_high)
                # 只对满足条件的股票分配权重
                layer_stocks = factor_values.index[layer_mask]
                weight[layer_stocks] = 1.0
                
                # 归一化
                if weight.sum() > 0:
                    weight = weight / weight.sum()
                
                weights[layer_idx][date] = weight
        
        return weights
    
    def _calculate_layer_nav(self, weight_df, return_df, fee=0.002):
        """
        计算单层净值序列（简化版分层回测）
        
        Parameters:
        -----------
        weight_df : pd.DataFrame, 权重数据（股票×日期），列名为调仓日期
        return_df : pd.DataFrame, 收益率数据（股票×日期）
        fee : float, 交易费率
        
        Returns:
        --------
        pd.Series : 净值序列，索引为日期
        """
        # 对齐日期和股票
        all_dates = sorted(set(return_df.columns))
        rebalance_dates = sorted(set(weight_df.columns))
        
        if len(all_dates) == 0 or len(rebalance_dates) == 0:
            return pd.Series(dtype=float)
        
        # 初始化净值序列
        nav_series = pd.Series(1.0, index=all_dates)
        
        # 初始化权重
        current_weight = pd.Series(0.0, index=weight_df.index)
        
        # 找到第一个调仓日
        first_rebalance_idx = 0
        for i, date in enumerate(all_dates):
            if date in rebalance_dates:
                first_rebalance_idx = i
                break
        
        # 从第一个调仓日开始回测
        for i, date in enumerate(all_dates):
            if i < first_rebalance_idx:
                continue
            
            # 检查是否需要调仓
            if date in rebalance_dates:
                # 调仓日：更新权重并扣除交易成本
                prev_weight = current_weight.copy()
                current_weight = weight_df[date].fillna(0)
                
                # 计算换手率（双边）
                turnover = (current_weight - prev_weight).abs().sum()
                
                # 扣除交易成本（在调仓日扣除）
                if i > 0:
                    nav_before = nav_series.iloc[i-1]
                else:
                    nav_before = 1.0
                
                nav_series.iloc[i] = nav_before * (1 - turnover * fee)
            
            # 计算当日收益（使用当前权重）
            if date in return_df.columns and current_weight.sum() > 0:
                daily_return = (current_weight * return_df[date]).fillna(0).sum()
                
                # 如果不是调仓日，根据收益更新净值
                if date not in rebalance_dates and i > 0:
                    nav_series.iloc[i] = nav_series.iloc[i-1] * (1 + daily_return)
                elif date in rebalance_dates:
                    # 调仓日：净值已经在扣除交易成本后更新，再叠加当日收益
                    nav_series.iloc[i] = nav_series.iloc[i] * (1 + daily_return)
            else:
                # 如果没有收益率数据，保持上期净值
                if i > 0:
                    nav_series.iloc[i] = nav_series.iloc[i-1]
        
        return nav_series
    
    def plot_layer_returns(self, factor_name, layers=5, freq=5, save_path=None):
        """
        绘制分层收益曲线
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        layers : int, 分层数
        freq : int, 调仓频率
        save_path : str, 保存路径
        """
        nav_df = self.layer_backtest_single_factor(factor_name, layers=layers, freq=freq)
        
        if nav_df.empty:
            print(f"⚠️  因子 {factor_name} 分层回测失败")
            return
        
        plt.figure(figsize=(14, 8))
        
        for col in nav_df.columns:
            plt.plot(nav_df.index, nav_df[col].values, label=col, linewidth=2, alpha=0.8)
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        plt.title(f'{factor_name} - 分层回测收益曲线 ({layers}层)', fontsize=14, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值', fontsize=12)
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 分层收益曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def single_factor_analysis(self, factor_name, forward_period=1, layers=5, freq=5, save_dir='./results/'):
        """
        单因子完整分析
        
        Parameters:
        -----------
        factor_name : str, 因子名称
        forward_period : int, 前瞻期数
        layers : int, 分层数
        freq : int, 调仓频率
        save_dir : str, 结果保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"单因子分析: {factor_name}")
        print(f"{'='*60}\n")
        
        # 1. 计算IC指标
        print("1. 计算IC指标...")
        ic_stats = self.calculate_IR(factor_name, forward_period)
        print(f"   IC均值: {ic_stats['IC_Mean']:.4f}")
        print(f"   IC标准差: {ic_stats['IC_Std']:.4f}")
        print(f"   IR: {ic_stats['IR']:.4f}")
        print(f"   IC胜率: {ic_stats['IC_WinRate']:.4f}")
        
        # 2. 绘制IC趋势图
        print("\n2. 绘制IC趋势图...")
        self.plot_IC_trend(
            factor_name, 
            forward_period=forward_period,
            save_path=f'{save_dir}{factor_name}_IC_trend.png'
        )
        
        # 3. 分层回测
        print("\n3. 进行分层回测...")
        nav_df = self.layer_backtest_single_factor(factor_name, layers=layers, freq=freq)
        if not nav_df.empty:
            print(f"   最终净值:")
            for col in nav_df.columns:
                final_nav = nav_df[col].iloc[-1]
                total_return = (final_nav - 1) * 100
                print(f"     {col}: {final_nav:.4f} ({total_return:.2f}%)")
            
            # 保存净值数据
            nav_df.to_csv(f'{save_dir}{factor_name}_layer_nav.csv', encoding='utf-8-sig')
        
        # 4. 绘制分层收益曲线
        print("\n4. 绘制分层收益曲线...")
        self.plot_layer_returns(
            factor_name,
            layers=layers,
            freq=freq,
            save_path=f'{save_dir}{factor_name}_layer_returns.png'
        )
        
        # 5. 保存IC统计结果
        ic_stats_df = pd.DataFrame([ic_stats], index=[factor_name])
        ic_stats_df.to_csv(f'{save_dir}{factor_name}_IC_stats.csv', encoding='utf-8-sig')
        
        print(f"\n✅ {factor_name} 单因子分析完成！\n")
        
        return {
            'IC_stats': ic_stats,
            'nav_df': nav_df
        }
    
    # ==================== 多因子分析 ====================
    
    def calculate_factor_correlation(self, factor_list=None):
        """
        计算因子间的相关性矩阵
        
        Parameters:
        -----------
        factor_list : list, 因子名称列表，如果为None则使用所有因子
        
        Returns:
        --------
        pd.DataFrame : 相关性矩阵
        """
        if factor_list is None:
            factor_list = list(self.factor_data.keys())
        
        # 获取公共日期和股票
        all_dates = set(self.return_data.columns)
        all_stocks = set(self.return_data.index)
        
        for factor_name in factor_list:
            if factor_name in self.factor_data:
                factor_df = self.factor_data[factor_name]
                if factor_df is not None and not factor_df.empty:
                    all_dates = all_dates & set(factor_df.columns)
                    all_stocks = all_stocks & set(factor_df.index)
        
        # 将因子数据展平为一维序列并计算相关性
        factor_values_dict = {}
        
        for factor_name in factor_list:
            if factor_name not in self.factor_data:
                continue
            
            factor_df = self.factor_data[factor_name]
            if factor_df is None or factor_df.empty:
                continue
            
            # 提取公共区域的数据并展平
            factor_aligned = factor_df.loc[list(all_stocks), list(sorted(all_dates))]
            factor_flat = factor_aligned.values.flatten()
            
            # 去除NaN
            valid_mask = ~np.isnan(factor_flat)
            if valid_mask.sum() > 0:
                factor_values_dict[factor_name] = factor_flat[valid_mask]
        
        # 对齐长度（取最短的）
        if len(factor_values_dict) == 0:
            return pd.DataFrame()
        
        min_len = min(len(v) for v in factor_values_dict.values())
        factor_aligned_dict = {}
        
        for factor_name, factor_values in factor_values_dict.items():
            factor_aligned_dict[factor_name] = factor_values[:min_len]
        
        # 计算相关性矩阵
        factor_df = pd.DataFrame(factor_aligned_dict)
        corr_matrix = factor_df.corr()
        
        return corr_matrix
    
    def plot_correlation_heatmap(self, factor_list=None, save_path=None):
        """
        绘制因子相关性热力图
        
        Parameters:
        -----------
        factor_list : list, 因子名称列表
        save_path : str, 保存路径
        """
        corr_matrix = self.calculate_factor_correlation(factor_list)
        
        if corr_matrix.empty:
            print("⚠️  无法计算相关性矩阵")
            return
        
        plt.figure(figsize=(max(12, len(corr_matrix) * 0.8), max(10, len(corr_matrix) * 0.7)))
        
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('因子相关性热力图', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 相关性热力图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def filter_high_correlation_factors(self, threshold=0.7, factor_list=None):
        """
        筛选高相关性因子对
        
        Parameters:
        -----------
        threshold : float, 相关性阈值（默认0.7）
        factor_list : list, 因子名称列表
        
        Returns:
        --------
        list : 需要剔除的因子列表（保留因子中相关性较高的一个）
        """
        corr_matrix = self.calculate_factor_correlation(factor_list)
        
        if corr_matrix.empty:
            return []
        
        # 找到高相关性因子对
        high_corr_pairs = []
        factors_to_remove = set()
        
        for i, factor1 in enumerate(corr_matrix.index):
            for j, factor2 in enumerate(corr_matrix.columns):
                if i < j:  # 只检查上三角矩阵
                    corr_value = abs(corr_matrix.loc[factor1, factor2])
                    if corr_value >= threshold:
                        high_corr_pairs.append((factor1, factor2, corr_value))
        
        # 按相关性排序，优先剔除相关性高的因子对中的一个
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心策略：对于每个高相关性对，保留其中一个
        kept_factors = set()
        
        for factor1, factor2, corr_value in high_corr_pairs:
            if factor1 not in kept_factors and factor2 not in kept_factors:
                # 如果两个都没保留，选择保留一个（可以基于IC或其他指标选择）
                # 这里简化处理：保留第一个
                kept_factors.add(factor1)
                factors_to_remove.add(factor2)
            elif factor1 in kept_factors:
                factors_to_remove.add(factor2)
            elif factor2 in kept_factors:
                factors_to_remove.add(factor1)
        
        factors_to_remove = list(factors_to_remove)
        
        print(f"\n找到 {len(high_corr_pairs)} 对高相关性因子（阈值: {threshold}）")
        print(f"建议剔除 {len(factors_to_remove)} 个因子: {factors_to_remove}\n")
        
        # 打印高相关性因子对
        if high_corr_pairs:
            print("高相关性因子对详情:")
            for factor1, factor2, corr_value in high_corr_pairs[:10]:  # 只显示前10对
                print(f"  {factor1} <-> {factor2}: {corr_value:.4f}")
        
        return factors_to_remove
    
    def multi_factor_analysis(self, correlation_threshold=0.7, save_dir='./results/'):
        """
        多因子完整分析
        
        Parameters:
        -----------
        correlation_threshold : float, 相关性阈值
        save_dir : str, 结果保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("多因子分析")
        print(f"{'='*60}\n")
        
        # 1. 计算相关性矩阵
        print("1. 计算因子相关性矩阵...")
        corr_matrix = self.calculate_factor_correlation()
        if not corr_matrix.empty:
            corr_matrix.to_csv(f'{save_dir}factor_correlation_matrix.csv', encoding='utf-8-sig')
            print(f"   相关性矩阵已保存，形状: {corr_matrix.shape}")
        
        # 2. 绘制相关性热力图
        print("\n2. 绘制相关性热力图...")
        self.plot_correlation_heatmap(save_path=f'{save_dir}factor_correlation_heatmap.png')
        
        # 3. 筛选高相关性因子
        print("\n3. 筛选高相关性因子...")
        factors_to_remove = self.filter_high_correlation_factors(threshold=correlation_threshold)
        
        # 4. 保存筛选结果
        if factors_to_remove:
            removal_df = pd.DataFrame({
                'Factor_Name': factors_to_remove,
                'Reason': 'High Correlation'
            })
            removal_df.to_csv(f'{save_dir}factors_to_remove.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 多因子分析完成！\n")
        
        return {
            'correlation_matrix': corr_matrix,
            'factors_to_remove': factors_to_remove
        }
    
    def evaluate_all_factors(self, forward_period=1, layers=5, freq=5, correlation_threshold=0.7, save_dir='./results/'):
        """
        评估所有因子
        
        Parameters:
        -----------
        forward_period : int, 前瞻期数
        layers : int, 分层数
        freq : int, 调仓频率
        correlation_threshold : float, 相关性阈值
        save_dir : str, 结果保存目录
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("开始评估所有因子")
        print(f"{'='*80}\n")
        
        # 单因子分析
        all_ic_stats = {}
        valid_factors = []
        
        for factor_name in self.factor_data.keys():
            if self.factor_data[factor_name] is None or self.factor_data[factor_name].empty:
                continue
            
            try:
                result = self.single_factor_analysis(
                    factor_name,
                    forward_period=forward_period,
                    layers=layers,
                    freq=freq,
                    save_dir=save_dir
                )
                all_ic_stats[factor_name] = result['IC_stats']
                valid_factors.append(factor_name)
            except Exception as e:
                print(f"⚠️  因子 {factor_name} 分析失败: {e}")
                continue
        
        # 汇总单因子IC统计
        if all_ic_stats:
            ic_summary = pd.DataFrame(all_ic_stats).T
            ic_summary = ic_summary.sort_values('IR', ascending=False)
            ic_summary.to_csv(f'{save_dir}all_factors_IC_summary.csv', encoding='utf-8-sig')
            print(f"\n✅ IC统计汇总已保存")
            print(f"\nIC统计汇总（按IR排序）:")
            print(ic_summary[['IC_Mean', 'IC_Std', 'IR', 'IC_WinRate']].head(10))
        
        # 多因子分析
        if len(valid_factors) > 1:
            multi_result = self.multi_factor_analysis(
                correlation_threshold=correlation_threshold,
                save_dir=save_dir
            )
        else:
            print("⚠️  有效因子数量不足，跳过多因子分析")
            multi_result = {}
        
        print(f"\n{'='*80}")
        print("✅ 所有因子评估完成！")
        print(f"{'='*80}\n")
        
        return {
            'ic_summary': ic_summary if all_ic_stats else None,
            'multi_factor_result': multi_result
        }
