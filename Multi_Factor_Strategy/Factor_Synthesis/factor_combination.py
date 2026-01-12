# -*- coding: utf-8 -*-
"""
中证1000多因子策略 - 因子合成模块
实现多模型集成的因子合成方法
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 性能优化：并行处理和进度条
try:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    JOBLIB_AVAILABLE = True
    TQDM_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    TQDM_AVAILABLE = False
    # 定义占位函数
    def tqdm(iterable, *args, **kwargs):
        return iterable

# 检查GPU可用性
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️  使用CPU进行训练")


class CCCLoss(nn.Module):
    """
    一致性相关系数（Concordance Correlation Coefficient）损失函数
    CCC = (2 * rho * sigma_x * sigma_y) / (sigma_x^2 + sigma_y^2 + (mu_x - mu_y)^2)
    其中 rho 是皮尔逊相关系数
    """
    def __init__(self):
        super(CCCLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        """
        计算CCC损失（返回1-CCC，因为我们要最小化损失）
        
        Parameters:
        -----------
        y_pred : torch.Tensor, 预测值
        y_true : torch.Tensor, 真实值
        
        Returns:
        --------
        torch.Tensor : CCC损失值
        """
        # 去除NaN值
        mask = ~(torch.isnan(y_pred) | torch.isnan(y_true))
        if mask.sum() == 0:
            return torch.tensor(1.0, requires_grad=True)
        
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        
        if len(y_pred_clean) < 2:
            return torch.tensor(1.0, requires_grad=True)
        
        # 计算均值
        mu_pred = torch.mean(y_pred_clean)
        mu_true = torch.mean(y_true_clean)
        
        # 计算标准差
        sigma_pred = torch.std(y_pred_clean)
        sigma_true = torch.std(y_true_clean)
        
        # 计算协方差
        cov = torch.mean((y_pred_clean - mu_pred) * (y_true_clean - mu_true))
        
        # 计算CCC
        denominator = sigma_pred ** 2 + sigma_true ** 2 + (mu_pred - mu_true) ** 2
        if denominator < 1e-8:
            return torch.tensor(1.0, requires_grad=True)
        
        ccc = 2 * cov / denominator
        
        # 返回1-CCC作为损失（因为我们要最大化CCC，即最小化1-CCC）
        return 1 - ccc


def calculate_ccc_numpy(y_pred, y_true):
    """
    使用numpy计算CCC（用于XGBoost等非PyTorch模型）
    
    Parameters:
    -----------
    y_pred : np.ndarray, 预测值
    y_true : np.ndarray, 真实值
    
    Returns:
    --------
    float : CCC值
    """
    # 去除NaN值
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 2:
        return 0.0
    
    y_pred_clean = y_pred[mask]
    y_true_clean = y_true[mask]
    
    # 计算均值
    mu_pred = np.mean(y_pred_clean)
    mu_true = np.mean(y_true_clean)
    
    # 计算标准差
    sigma_pred = np.std(y_pred_clean)
    sigma_true = np.std(y_true_clean)
    
    # 计算协方差
    cov = np.mean((y_pred_clean - mu_pred) * (y_true_clean - mu_true))
    
    # 计算CCC
    denominator = sigma_pred ** 2 + sigma_true ** 2 + (mu_pred - mu_true) ** 2
    if denominator < 1e-8:
        return 0.0
    
    ccc = 2 * cov / denominator
    return ccc


class FactorDataset(Dataset):
    """因子数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPModel(nn.Module):
    """多层感知机模型"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class FactorCombiner:
    """因子合成器类"""
    
    def __init__(self, factor_data, return_data, constituent_manager=None):
        """
        初始化因子合成器
        
        Parameters:
        -----------
        factor_data : dict, 因子数据字典，key为因子名，value为DataFrame（股票×日期）
        return_data : pd.DataFrame, 收益率数据，格式同factor_data（股票×日期）
        constituent_manager : ConstituentManager, 成分股管理器，用于过滤成分股
        """
        self.factor_data = factor_data
        self.return_data = return_data
        self.constituent_manager = constituent_manager
        
        # 对齐数据
        self._align_data()
        
        # 存储模型预测结果
        self.model1_prediction = None  # 因子筛选等权合成
        self.model2_prediction = None  # MLP模型
        self.model3_prediction = None  # XGBoost模型
        self.final_signal = None  # 最终信号
    
    def _align_data(self):
        """对齐因子数据和收益率数据"""
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
                aligned_factors[factor_name] = pd.DataFrame(
                    index=list(all_stocks), 
                    columns=list(sorted(all_dates))
                )
                aligned_factors[factor_name][:] = np.nan
            else:
                aligned = factor_df.loc[list(all_stocks), list(sorted(all_dates))]
                aligned_factors[factor_name] = aligned
        
        self.factor_data = aligned_factors
        print(f"✅ 数据对齐完成，股票数: {len(all_stocks)}, 日期数: {len(all_dates)}")
    
    def _prepare_training_data(self, forward_period=1):
        """
        准备训练数据，确保防止未来信息泄露
        
        Parameters:
        -----------
        forward_period : int, 前瞻期数（默认1，即预测下期收益率）
        
        Returns:
        --------
        X : np.ndarray, 特征矩阵 (样本数 × 因子数)
        y : np.ndarray, 目标值 (样本数,)
        stock_info : list, 每个样本对应的股票和日期信息
        """
        dates = sorted(self.return_data.columns)
        all_factors = list(self.factor_data.keys())
        
        X_list = []
        y_list = []
        stock_info = []
        
        # 预计算成分股（如果可用）
        constituents_cache = {}
        if self.constituent_manager is not None:
            all_dates_set = set(dates)
            for date in all_dates_set:
                constituents_cache[date] = set(self.constituent_manager.get_constituents_by_date(date))
        
        # 向量化处理：批量处理日期
        iterator = tqdm(enumerate(dates), total=len(dates), desc="准备训练数据") if TQDM_AVAILABLE else enumerate(dates)
        
        for i, date in iterator:
            if i + forward_period >= len(dates):
                break
            
            # 当期因子值（t日）
            factor_values_dict = {}
            for factor_name in all_factors:
                if factor_name in self.factor_data:
                    factor_df = self.factor_data[factor_name]
                    if date in factor_df.columns:
                        factor_values_dict[factor_name] = factor_df[date]
            
            # 未来收益率（t+forward_period日）
            future_date = dates[i + forward_period]
            future_returns = self.return_data[future_date]
            
            # 获取公共股票（向量化操作）
            common_stocks = set(future_returns.index)
            for factor_values in factor_values_dict.values():
                common_stocks = common_stocks & set(factor_values.index)
            
            # 如果提供了成分股管理器，只使用成分股
            if self.constituent_manager is not None:
                if date in constituents_cache:
                    common_stocks = common_stocks & constituents_cache[date]
            
            if len(common_stocks) == 0:
                continue
            
            common_stocks = sorted(list(common_stocks))
            
            # 构建特征矩阵
            # 对于每个日期，只使用在该日期有值的因子
            available_factors = list(factor_values_dict.keys())
            
            if len(available_factors) == 0:
                continue
            
            # 向量化计算因子均值
            factor_means = {}
            common_stocks_list = list(common_stocks)
            for factor_name in all_factors:
                if factor_name in factor_values_dict:
                    factor_series = factor_values_dict[factor_name]
                    # 向量化提取和计算均值
                    common_values = factor_series.loc[common_stocks_list]
                    valid_values = common_values.dropna()
                    if len(valid_values) > 0:
                        factor_means[factor_name] = valid_values.mean()
                    else:
                        factor_means[factor_name] = 0.0
                else:
                    factor_means[factor_name] = 0.0
            
            # 批量构建特征矩阵（向量化）
            min_valid_ratio = 0.3
            min_valid_count = int(len(all_factors) * min_valid_ratio)
            
            # 预分配数组
            date_factor_matrix = np.full((len(common_stocks), len(all_factors)), np.nan)
            date_returns = np.full(len(common_stocks), np.nan)
            
            for idx, stock in enumerate(common_stocks):
                # 提取因子值
                for j, factor_name in enumerate(all_factors):
                    if factor_name in factor_values_dict:
                        value = factor_values_dict[factor_name].loc[stock]
                        if pd.isna(value):
                            date_factor_matrix[idx, j] = factor_means[factor_name]
                        else:
                            date_factor_matrix[idx, j] = value
                    else:
                        date_factor_matrix[idx, j] = 0.0
                
                # 提取收益率
                return_value = future_returns.loc[stock]
                if not pd.isna(return_value):
                    date_returns[idx] = return_value
            
            # 批量过滤有效样本
            valid_factor_counts = (~np.isnan(date_factor_matrix)).sum(axis=1)
            valid_return_mask = ~np.isnan(date_returns)
            valid_mask = (valid_factor_counts >= min_valid_count) & valid_return_mask
            
            if valid_mask.sum() > 0:
                valid_factors = date_factor_matrix[valid_mask]
                valid_returns = date_returns[valid_mask]
                valid_stocks = [common_stocks[i] for i in range(len(common_stocks)) if valid_mask[i]]
                
                X_list.extend(valid_factors.tolist())
                y_list.extend(valid_returns.tolist())
                stock_info.extend([
                    {'stock': stock, 'date': date, 'future_date': future_date}
                    for stock in valid_stocks
                ])
        
        if len(X_list) == 0:
            print("⚠️  警告：没有可用的训练数据")
            print(f"   因子数量: {len(all_factors)}")
            print(f"   日期数量: {len(dates)}")
            print("   可能原因：")
            print("   1. 因子数据和收益率数据的日期/股票不匹配")
            print("   2. 所有因子的值都缺失")
            print("   3. 成分股过滤后没有剩余的股票")
            print("   4. 有效因子比例过低（要求至少30%的因子有值）")
            
            # 添加诊断信息
            if len(dates) > 0:
                sample_date = dates[0]
                print(f"\n   诊断信息（以日期 {sample_date} 为例）：")
                if sample_date in self.return_data.columns:
                    return_stocks = set(self.return_data[sample_date].dropna().index)
                    print(f"     收益率数据有效股票数: {len(return_stocks)}")
                    
                    factor_stocks_sets = []
                    for factor_name in all_factors[:5]:  # 只检查前5个因子
                        if factor_name in self.factor_data:
                            factor_df = self.factor_data[factor_name]
                            if sample_date in factor_df.columns:
                                factor_stocks = set(factor_df[sample_date].dropna().index)
                                factor_stocks_sets.append(factor_stocks)
                                print(f"     {factor_name} 有效股票数: {len(factor_stocks)}")
                    
                    if factor_stocks_sets:
                        common_stocks = return_stocks
                        for fs in factor_stocks_sets:
                            common_stocks = common_stocks & fs
                        print(f"     前5个因子与收益率的公共股票数: {len(common_stocks)}")
                        
                        if self.constituent_manager is not None:
                            constituents = self.constituent_manager.get_constituents_by_date(sample_date)
                            final_stocks = common_stocks & set(constituents)
                            print(f"     成分股过滤后剩余股票数: {len(final_stocks)}")
            
            return np.array([]).reshape(0, len(all_factors)), np.array([]), []
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 检查X的维度
        if len(X.shape) == 1:
            # 如果是一维数组，说明所有样本的因子数量不一致，这是不应该发生的
            print(f"⚠️  警告：数据格式异常，X的形状为 {X.shape}")
            print(f"   预期的特征数: {len(all_factors)}")
            return np.array([]).reshape(0, len(all_factors)), np.array([]), []
        
        if X.shape[1] != len(all_factors):
            print(f"⚠️  警告：特征数不匹配，预期 {len(all_factors)}，实际 {X.shape[1]}")
        
        print(f"✅ 数据准备完成，样本数: {len(X)}, 特征数: {X.shape[1]}")
        return X, y, stock_info
    
    def model1_factor_selection_equal_weight(self, 
                                             ml_min_factors=2, ml_max_factors=3,
                                             price_min_factors=3, price_max_factors=5,
                                             ic_threshold=0.02, correlation_threshold=0.7,
                                             train_ratio=0.8):
        """
        模型1：因子筛选等权合成法（分类筛选）
        
        Parameters:
        -----------
        ml_min_factors : int, 机器学习因子最少保留数（默认2）
        ml_max_factors : int, 机器学习因子最多保留数（默认3）
        price_min_factors : int, 量价因子最少保留数（默认3）
        price_max_factors : int, 量价因子最多保留数（默认5）
        ic_threshold : float, IC阈值（绝对值），低于此值的因子将被剔除
        correlation_threshold : float, 相关性阈值，高于此值的因子对将被剔除其中一个
        train_ratio : float, 训练集比例（默认0.8，前80%用于训练，后20%用于测试）
        
        Returns:
        --------
        pd.DataFrame : 模型1的预测值（股票×日期）
        """
        print("\n" + "="*60)
        print("模型1：因子筛选等权合成法（分类筛选）")
        print("="*60)
        
        # 分类因子：机器学习因子 vs 量价因子
        ml_keywords = ['GRU', 'TRANSFORMER', 'SVM', 'LIGHTGBM', 'LGB', 'RF', 'RANDOM']
        ml_factors = []
        price_factors = []
        
        for factor_name in self.factor_data.keys():
            is_ml = any(keyword in factor_name.upper() for keyword in ml_keywords)
            if is_ml:
                ml_factors.append(factor_name)
            else:
                price_factors.append(factor_name)
        
        print(f"\n因子分类：")
        print(f"  机器学习因子: {len(ml_factors)} 个")
        print(f"  量价因子: {len(price_factors)} 个")
        
        # 计算所有因子的IC（只使用训练集数据，并行优化）
        print("\n计算因子IC（使用训练集数据）...")
        dates = sorted(self.return_data.columns)
        train_end_idx = int(len(dates) * train_ratio)
        train_dates = dates[:train_end_idx]
        
        # 临时保存原始return_data，使用训练集数据计算IC
        original_return_data = self.return_data
        self.return_data = self.return_data[train_dates]
        
        ic_stats = {}
        factor_names = list(self.factor_data.keys())
        
        if JOBLIB_AVAILABLE and len(factor_names) > 5:
            # 并行计算IC
            ic_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(self._calculate_IC_simple)(factor_name)
                for factor_name in tqdm(factor_names, desc="计算IC")
            )
            
            for factor_name, ic_series in zip(factor_names, ic_results):
                if len(ic_series) > 0:
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    ir = ic_mean / (ic_std + 1e-8) if ic_std > 1e-8 else 0
                    ic_stats[factor_name] = {
                        'IC_Mean': ic_mean,
                        'IC_Std': ic_std,
                        'IR': ir,
                        'IC_AbsMean': abs(ic_mean)
                    }
        else:
            # 串行计算IC
            for factor_name in tqdm(factor_names, desc="计算IC") if TQDM_AVAILABLE else factor_names:
                ic_series = self._calculate_IC_simple(factor_name)
                if len(ic_series) > 0:
                    ic_mean = ic_series.mean()
                    ic_std = ic_series.std()
                    ir = ic_mean / (ic_std + 1e-8) if ic_std > 1e-8 else 0
                    ic_stats[factor_name] = {
                        'IC_Mean': ic_mean,
                        'IC_Std': ic_std,
                        'IR': ir,
                        'IC_AbsMean': abs(ic_mean)
                    }
        
        # 恢复原始return_data
        self.return_data = original_return_data
        
        # 分别筛选机器学习因子和量价因子
        def select_factors_by_category(factor_list, min_count, max_count, category_name):
            """从指定类别中筛选因子"""
            category_factors = [f for f in factor_list if f in ic_stats]
            if len(category_factors) == 0:
                print(f"\n  {category_name}: 没有可用的因子")
                return []
            
            # 按IC绝对值排序
            sorted_category = sorted(
                [(f, ic_stats[f]) for f in category_factors],
                key=lambda x: x[1]['IC_AbsMean'],
                reverse=True
            )
            
            print(f"\n  {category_name} IC统计（按IC绝对值排序）:")
            for i, (factor_name, stats) in enumerate(sorted_category[:min(10, len(sorted_category))]):
                print(f"    {i+1}. {factor_name}: IC均值={stats['IC_Mean']:.4f}, IR={stats['IR']:.4f}")
            
            # 初步筛选：剔除IC较低的因子
            selected = []
            for factor_name, stats in sorted_category:
                if abs(stats['IC_Mean']) >= ic_threshold:
                    selected.append(factor_name)
            
            print(f"    初步筛选后: {len(selected)} 个")
            
            # 剔除高相关性因子对
            if len(selected) > 1:
                corr_matrix = self._calculate_factor_correlation(selected)
                factors_to_remove = set()
                
                for i, factor1 in enumerate(selected):
                    if factor1 in factors_to_remove:
                        continue
                    for j, factor2 in enumerate(selected[i+1:], start=i+1):
                        if factor2 in factors_to_remove:
                            continue
                        if factor1 in corr_matrix.index and factor2 in corr_matrix.columns:
                            corr = abs(corr_matrix.loc[factor1, factor2])
                            if corr > correlation_threshold:
                                ic1 = ic_stats[factor1]['IC_AbsMean']
                                ic2 = ic_stats[factor2]['IC_AbsMean']
                                if ic1 < ic2:
                                    factors_to_remove.add(factor1)
                                else:
                                    factors_to_remove.add(factor2)
                
                selected = [f for f in selected if f not in factors_to_remove]
                print(f"    剔除高相关性后: {len(selected)} 个")
            
            # 确保数量在合理范围内
            if len(selected) > max_count:
                selected = [f[0] for f in sorted_category if f[0] in selected][:max_count]
            elif len(selected) < min_count:
                # 如果筛选后因子太少，至少保留IC绝对值最大的min_count个
                selected = [f[0] for f in sorted_category[:min_count]]
            
            return selected
        
        # 筛选机器学习因子
        selected_ml_factors = select_factors_by_category(
            ml_factors, ml_min_factors, ml_max_factors, "机器学习因子"
        )
        
        # 筛选量价因子
        selected_price_factors = select_factors_by_category(
            price_factors, price_min_factors, price_max_factors, "量价因子"
        )
        
        # 合并选中的因子
        selected_factors = selected_ml_factors + selected_price_factors
        
        print(f"\n最终选择的因子（{len(selected_factors)}个）:")
        if selected_ml_factors:
            print(f"  机器学习因子（{len(selected_ml_factors)}个）:")
            for i, factor_name in enumerate(selected_ml_factors, 1):
                print(f"    {i}. {factor_name}")
        if selected_price_factors:
            print(f"  量价因子（{len(selected_price_factors)}个）:")
            for i, factor_name in enumerate(selected_price_factors, 1):
                print(f"    {i}. {factor_name}")
        
        # 等权合成（只对训练集日期进行预测，测试集留空用于纯袋外观测）
        print("\n进行等权合成...")
        dates = sorted(self.return_data.columns)
        train_end_idx = int(len(dates) * train_ratio)
        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:]
        
        print(f"  训练集日期: {len(train_dates)} 个 ({train_dates[0]} 至 {train_dates[-1]})")
        print(f"  测试集日期: {len(test_dates)} 个 ({test_dates[0]} 至 {test_dates[-1]}) - 纯袋外观测")
        
        stocks = sorted(self.return_data.index)
        
        prediction = pd.DataFrame(index=stocks, columns=dates)
        prediction[:] = np.nan
        
        # 只对训练集日期进行预测（向量化优化）
        date_iterator = tqdm(train_dates, desc="等权合成") if TQDM_AVAILABLE else train_dates
        
        # 预计算成分股（如果可用）
        constituents_cache = {}
        if self.constituent_manager is not None:
            for date in train_dates:
                constituents_cache[date] = set(self.constituent_manager.get_constituents_by_date(date))
        
        for date in date_iterator:
            # 获取所有选中因子在该日期的值
            factor_values_list = []
            for factor_name in selected_factors:
                if factor_name in self.factor_data:
                    factor_df = self.factor_data[factor_name]
                    if date in factor_df.columns:
                        factor_values_list.append(factor_df[date])
            
            if len(factor_values_list) == 0:
                continue
            
            # 对齐所有因子（向量化）
            common_stocks = set(stocks)
            for factor_values in factor_values_list:
                common_stocks = common_stocks & set(factor_values.index)
            
            # 如果提供了成分股管理器，只使用成分股
            if self.constituent_manager is not None:
                if date in constituents_cache:
                    common_stocks = common_stocks & constituents_cache[date]
            
            if len(common_stocks) == 0:
                continue
            
            common_stocks = sorted(list(common_stocks))
            
            # 向量化构建因子矩阵
            factor_matrix = np.full((len(common_stocks), len(factor_values_list)), np.nan)
            for j, factor_values in enumerate(factor_values_list):
                factor_matrix[:, j] = factor_values.loc[common_stocks].values
            
            # 过滤有效样本（所有因子都有值）
            valid_mask = ~np.isnan(factor_matrix).any(axis=1)
            
            if valid_mask.sum() == 0:
                continue
            
            factor_matrix_valid = factor_matrix[valid_mask]
            valid_stocks = [common_stocks[i] for i in range(len(common_stocks)) if valid_mask[i]]
            
            # 截面标准化（向量化）
            factor_matrix_std = (factor_matrix_valid - factor_matrix_valid.mean(axis=0)) / (factor_matrix_valid.std(axis=0) + 1e-8)
            
            # 等权合成（向量化）
            combined_signal = factor_matrix_std.mean(axis=1)
            
            # 批量保存预测值
            prediction.loc[valid_stocks, date] = combined_signal
        
        self.model1_prediction = prediction
        print(f"✅ 模型1完成，预测值形状: {prediction.shape}")
        return prediction
    
    def model2_mlp(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                   hidden_dims=[128, 64, 32], dropout=0.3, 
                   batch_size=256, epochs=100, patience=10, lr=0.001):
        """
        模型2：MLP深度神经网络模型
        
        Parameters:
        -----------
        train_ratio : float, 训练集比例（默认0.6，即3:1:1中的3/5）
        val_ratio : float, 验证集比例（默认0.2）
        test_ratio : float, 测试集比例（默认0.2）
        hidden_dims : list, 隐藏层维度
        dropout : float, Dropout比例
        batch_size : int, 批次大小
        epochs : int, 训练轮数
        patience : int, 早停耐心值
        lr : float, 学习率
        
        Returns:
        --------
        pd.DataFrame : 模型2的预测值（股票×日期）
        """
        print("\n" + "="*60)
        print("模型2：MLP深度神经网络模型")
        print("="*60)
        
        # 准备数据
        X, y, stock_info = self._prepare_training_data(forward_period=1)
        
        if len(X) == 0:
            print("⚠️  没有可用的训练数据")
            return pd.DataFrame()
        
        # 按时间顺序划分数据集（3:1:1）
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        stock_info_test = stock_info[n_train+n_val:]
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        
        # 标准化
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 创建数据集和数据加载器
        train_dataset = FactorDataset(X_train_scaled, y_train_scaled)
        val_dataset = FactorDataset(X_val_scaled, y_val_scaled)
        test_dataset = FactorDataset(X_test_scaled, y_test_scaled)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型（使用GPU如果可用）
        input_dim = X_train.shape[1]
        model = MLPModel(input_dim, hidden_dims, dropout).to(DEVICE)
        criterion = CCCLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 训练模型
        print(f"\n开始训练MLP模型（使用 {DEVICE}）...")
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        epoch_iterator = tqdm(range(epochs), desc="训练进度") if TQDM_AVAILABLE else range(epochs)
        
        for epoch in epoch_iterator:
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if TQDM_AVAILABLE:
                epoch_iterator.set_postfix({
                    'Train Loss': f'{train_loss:.6f}',
                    'Val Loss': f'{val_loss:.6f}'
                })
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if TQDM_AVAILABLE:
                        epoch_iterator.close()
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 预测（使用GPU如果可用）
        print("\n进行预测...")
        model.eval()
        predictions = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(DEVICE)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # 转换为因子格式
        dates = sorted(self.return_data.columns)
        stocks = sorted(self.return_data.index)
        prediction_df = pd.DataFrame(index=stocks, columns=dates)
        prediction_df[:] = np.nan
        
        for i, info in enumerate(stock_info_test):
            if i < len(predictions) and not np.isnan(predictions[i]):
                stock = info['stock']
                date = info['date']
                if stock in prediction_df.index and date in prediction_df.columns:
                    prediction_df.loc[stock, date] = predictions[i]
        
        self.model2_prediction = prediction_df
        print(f"✅ 模型2完成，预测值形状: {prediction_df.shape}")
        return prediction_df
    
    def model3_xgboost(self, train_ratio=0.8, test_ratio=0.2,
                      n_estimators=200, max_depth=6, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8):
        """
        模型3：XGBoost模型
        
        Parameters:
        -----------
        train_ratio : float, 训练集比例（默认0.8，即4:1中的4/5）
        test_ratio : float, 测试集比例（默认0.2）
        n_estimators : int, 树的数量
        max_depth : int, 树的最大深度
        learning_rate : float, 学习率
        subsample : float, 样本采样比例
        colsample_bytree : float, 特征采样比例
        
        Returns:
        --------
        pd.DataFrame : 模型3的预测值（股票×日期）
        """
        print("\n" + "="*60)
        print("模型3：XGBoost模型")
        print("="*60)
        
        # 准备数据
        X, y, stock_info = self._prepare_training_data(forward_period=1)
        
        if len(X) == 0:
            print("⚠️  没有可用的训练数据")
            return pd.DataFrame()
        
        # 按时间顺序划分数据集（前80%训练，后20%测试）
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        stock_info_train = stock_info[:n_train]
        stock_info_test = stock_info[n_train:]
        
        print(f"训练集: {len(X_train)}, 测试集: {len(X_test)} (纯袋外观测)")
        
        # 标准化（只使用训练集数据）
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # 创建XGBoost模型（优化参数）
        print("\n开始训练XGBoost模型（仅使用训练集）...")
        
        # 使用GPU如果可用
        tree_method = 'gpu_hist' if torch.cuda.is_available() else 'hist'
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=-1,
            tree_method=tree_method,
            predictor='gpu_predictor' if torch.cuda.is_available() else 'cpu_predictor'
        )
        
        # 只使用训练集进行训练（显示进度）
        model.fit(X_train_scaled, y_train, 
                 eval_set=[(X_train_scaled, y_train)],
                 verbose=False)
        
        # 对测试集进行预测（纯袋外观测）
        print("\n对测试集进行预测（纯袋外观测）...")
        predictions_test = model.predict(X_test_scaled)
        
        # 计算测试集CCC
        ccc_test = calculate_ccc_numpy(predictions_test, y_test)
        print(f"测试集CCC: {ccc_test:.6f}")
        
        # 对训练集也进行预测（用于完整性）
        predictions_train = model.predict(X_train_scaled)
        ccc_train = calculate_ccc_numpy(predictions_train, y_train)
        print(f"训练集CCC: {ccc_train:.6f}")
        
        # 转换为因子格式（包含训练集和测试集的预测）
        dates = sorted(self.return_data.columns)
        stocks = sorted(self.return_data.index)
        prediction_df = pd.DataFrame(index=stocks, columns=dates)
        prediction_df[:] = np.nan
        
        # 保存训练集预测
        for i, info in enumerate(stock_info_train):
            if i < len(predictions_train) and not np.isnan(predictions_train[i]):
                stock = info['stock']
                date = info['date']
                if stock in prediction_df.index and date in prediction_df.columns:
                    prediction_df.loc[stock, date] = predictions_train[i]
        
        # 保存测试集预测（纯袋外观测）
        for i, info in enumerate(stock_info_test):
            if i < len(predictions_test) and not np.isnan(predictions_test[i]):
                stock = info['stock']
                date = info['date']
                if stock in prediction_df.index and date in prediction_df.columns:
                    prediction_df.loc[stock, date] = predictions_test[i]
        
        self.model3_prediction = prediction_df
        print(f"✅ 模型3完成，预测值形状: {prediction_df.shape}")
        return prediction_df
    
    def combine_models(self):
        """
        等权合成三个模型的预测值，得到最终交易信号
        
        Returns:
        --------
        pd.DataFrame : 最终交易信号（股票×日期）
        """
        print("\n" + "="*60)
        print("等权合成三个模型")
        print("="*60)
        
        if self.model1_prediction is None or self.model2_prediction is None or self.model3_prediction is None:
            print("⚠️  请先运行三个模型")
            return pd.DataFrame()
        
        # 对齐三个模型的预测值
        dates = sorted(set(self.model1_prediction.columns) & 
                      set(self.model2_prediction.columns) & 
                      set(self.model3_prediction.columns))
        stocks = sorted(set(self.model1_prediction.index) & 
                       set(self.model2_prediction.index) & 
                       set(self.model3_prediction.index))
        
        final_signal = pd.DataFrame(index=stocks, columns=dates)
        final_signal[:] = np.nan
        
        # 向量化合成（批量处理）
        pred1_aligned = self.model1_prediction.loc[stocks, dates]
        pred2_aligned = self.model2_prediction.loc[stocks, dates]
        pred3_aligned = self.model3_prediction.loc[stocks, dates]
        
        # 向量化计算
        valid_mask = ~(pred1_aligned.isna() | pred2_aligned.isna() | pred3_aligned.isna())
        combined = (pred1_aligned + pred2_aligned + pred3_aligned) / 3
        final_signal.loc[stocks, dates] = combined
        final_signal.loc[stocks, dates] = final_signal.loc[stocks, dates].where(valid_mask, np.nan)
        
        self.final_signal = final_signal
        print(f"✅ 最终信号生成完成，形状: {final_signal.shape}")
        return final_signal
    
    def save_predictions(self, save_dir='./results/'):
        """
        保存三个模型的预测值和最终信号
        
        Parameters:
        -----------
        save_dir : str, 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n保存预测结果到: {save_dir}")
        
        if self.model1_prediction is not None:
            path1 = os.path.join(save_dir, 'model1_factor_selection_prediction.csv')
            self.model1_prediction.to_csv(path1, encoding='utf-8-sig')
            print(f"  ✅ 模型1预测值: {path1}")
        
        if self.model2_prediction is not None:
            path2 = os.path.join(save_dir, 'model2_mlp_prediction.csv')
            self.model2_prediction.to_csv(path2, encoding='utf-8-sig')
            print(f"  ✅ 模型2预测值: {path2}")
        
        if self.model3_prediction is not None:
            path3 = os.path.join(save_dir, 'model3_xgboost_prediction.csv')
            self.model3_prediction.to_csv(path3, encoding='utf-8-sig')
            print(f"  ✅ 模型3预测值: {path3}")
        
        if self.final_signal is not None:
            path_final = os.path.join(save_dir, 'final_signal.csv')
            self.final_signal.to_csv(path_final, encoding='utf-8-sig')
            print(f"  ✅ 最终信号: {path_final}")
    
    def _calculate_IC_simple(self, factor_name, forward_period=1):
        """简化的IC计算（用于因子筛选）"""
        if factor_name not in self.factor_data:
            return pd.Series(dtype=float)
        
        factor_df = self.factor_data[factor_name]
        dates = sorted(set(factor_df.columns) & set(self.return_data.columns))
        
        ic_values = []
        ic_dates = []
        
        for i, date in enumerate(dates):
            if i + forward_period >= len(dates):
                break
            
            factor_values = factor_df[date]
            future_date = dates[i + forward_period]
            future_returns = self.return_data[future_date]
            
            common_stocks = set(factor_values.index) & set(future_returns.index)
            
            if self.constituent_manager is not None:
                constituents = self.constituent_manager.get_constituents_by_date(date)
                common_stocks = common_stocks & set(constituents)
            
            factor_aligned = factor_values.loc[list(common_stocks)]
            return_aligned = future_returns.loc[list(common_stocks)]
            
            valid_mask = ~(factor_aligned.isna() | return_aligned.isna())
            if valid_mask.sum() < 10:
                continue
            
            factor_clean = factor_aligned[valid_mask]
            return_clean = return_aligned[valid_mask]
            
            if len(factor_clean) > 1 and factor_clean.std() > 1e-8:
                ic = np.corrcoef(factor_clean, return_clean)[0, 1]
                if not np.isnan(ic):
                    ic_values.append(ic)
                    ic_dates.append(date)
        
        return pd.Series(ic_values, index=ic_dates)
    
    def _calculate_factor_correlation(self, factor_list):
        """计算因子相关性矩阵"""
        all_dates = set(self.return_data.columns)
        all_stocks = set(self.return_data.index)
        
        for factor_name in factor_list:
            if factor_name in self.factor_data:
                factor_df = self.factor_data[factor_name]
                if factor_df is not None and not factor_df.empty:
                    all_dates = all_dates & set(factor_df.columns)
                    all_stocks = all_stocks & set(factor_df.index)
        
        factor_values_dict = {}
        for factor_name in factor_list:
            if factor_name not in self.factor_data:
                continue
            factor_df = self.factor_data[factor_name]
            if factor_df is None or factor_df.empty:
                continue
            
            factor_aligned = factor_df.loc[list(all_stocks), list(sorted(all_dates))]
            factor_flat = factor_aligned.values.flatten()
            valid_mask = ~np.isnan(factor_flat)
            if valid_mask.sum() > 0:
                factor_values_dict[factor_name] = factor_flat[valid_mask]
        
        if len(factor_values_dict) == 0:
            return pd.DataFrame()
        
        min_len = min(len(v) for v in factor_values_dict.values())
        factor_aligned_dict = {}
        for factor_name, factor_values in factor_values_dict.items():
            factor_aligned_dict[factor_name] = factor_values[:min_len]
        
        factor_df = pd.DataFrame(factor_aligned_dict)
        return factor_df.corr()


# ==================== 主函数 ====================

def _load_single_factor(csv_file, factors_path):
    """加载单个因子文件（用于并行处理）"""
    factor_name = csv_file.replace('.csv', '')
    factor_path = os.path.join(factors_path, csv_file)
    
    try:
        factor_df = pd.read_csv(factor_path, index_col=0, encoding='utf-8')
        factor_df.columns = pd.to_datetime(factor_df.columns)
        return factor_name, factor_df, None
    except Exception as e:
        return factor_name, None, str(e)


def load_factors_from_directory(factors_path='./factors/'):
    """
    从factors目录加载所有因子数据（并行优化）
    
    Parameters:
    -----------
    factors_path : str, 因子文件目录路径
    
    Returns:
    --------
    dict : 因子数据字典，key为因子名，value为DataFrame（股票×日期）
    """
    factors = {}
    factors_path = os.path.abspath(factors_path)
    
    if not os.path.exists(factors_path):
        print(f"❌ 因子目录不存在: {factors_path}")
        return factors
    
    print(f"📂 从目录加载因子数据: {factors_path}")
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(factors_path) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"   找到 {len(csv_files)} 个因子文件")
    
    # 并行加载（如果可用）
    if JOBLIB_AVAILABLE and len(csv_files) > 5:
        print("   使用并行加载...")
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(_load_single_factor)(csv_file, factors_path) 
            for csv_file in tqdm(csv_files, desc="加载因子")
        )
        
        for factor_name, factor_df, error in results:
            if error is None:
                factors[factor_name] = factor_df
                print(f"   ✅ {factor_name}: {factor_df.shape}")
            else:
                print(f"   ⚠️  {factor_name}: 加载失败 - {error}")
    else:
        # 串行加载
        for csv_file in tqdm(csv_files, desc="加载因子"):
            factor_name, factor_df, error = _load_single_factor(csv_file, factors_path)
            if error is None:
                factors[factor_name] = factor_df
                print(f"   ✅ {factor_name}: {factor_df.shape}")
            else:
                print(f"   ⚠️  {factor_name}: 加载失败 - {error}")
    
    print(f"\n✅ 共加载 {len(factors)} 个因子")
    return factors


def load_price_data_and_calculate_returns(data_path='./data/'):
    """
    加载价格数据并计算收益率
    
    Parameters:
    -----------
    data_path : str, 数据文件目录路径
    
    Returns:
    --------
    pd.DataFrame : 收益率数据（股票×日期）
    """
    data_path = os.path.abspath(data_path)
    price_file = os.path.join(data_path, 'stock_price_data.csv')
    
    if not os.path.exists(price_file):
        print(f"❌ 价格数据文件不存在: {price_file}")
        return pd.DataFrame()
    
    print(f"📂 加载价格数据: {price_file}")
    
    try:
        price_data = pd.read_csv(price_file, encoding='utf-8-sig')
        price_data['TRADE_DT'] = pd.to_datetime(price_data['TRADE_DT'])
        
        # 确保CLOSE_ADJ存在
        if 'CLOSE_ADJ' not in price_data.columns:
            if 'CLOSE_PRICE' in price_data.columns:
                price_data['CLOSE_ADJ'] = price_data['CLOSE_PRICE']
            else:
                print("❌ 价格数据中缺少CLOSE_PRICE或CLOSE_ADJ字段")
                return pd.DataFrame()
        
        # 按股票代码和日期排序
        price_data = price_data.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
        
        # 计算收益率
        price_data['RETURN'] = price_data.groupby('S_INFO_WINDCODE')['CLOSE_ADJ'].pct_change()
        
        # 转换为宽格式（股票×日期）
        return_df = price_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RETURN']].dropna()
        return_wide = return_df.pivot(
            index='S_INFO_WINDCODE',
            columns='TRADE_DT',
            values='RETURN'
        )
        
        print(f"✅ 收益率数据加载完成，形状: {return_wide.shape}")
        return return_wide
    
    except Exception as e:
        print(f"❌ 加载价格数据时出错: {e}")
        return pd.DataFrame()


def load_constituent_manager(data_path='./data/'):
    """
    加载成分股管理器
    
    Parameters:
    -----------
    data_path : str, 数据文件目录路径
    
    Returns:
    --------
    ConstituentManager or None
    """
    data_path = os.path.abspath(data_path)
    constituents_file = os.path.join(data_path, 'csi1000_constituents_history.csv')
    
    if not os.path.exists(constituents_file):
        print(f"⚠️  成分股历史数据文件不存在: {constituents_file}")
        return None
    
    try:
        from constituent_manager import ConstituentManager
        
        print(f"📂 加载成分股历史数据: {constituents_file}")
        constituents_history = pd.read_csv(constituents_file, encoding='utf-8-sig')
        constituents_history['S_CON_INDATE'] = pd.to_datetime(constituents_history['S_CON_INDATE'], errors='coerce')
        constituents_history['S_CON_OUTDATE'] = pd.to_datetime(constituents_history['S_CON_OUTDATE'], errors='coerce')
        
        constituent_manager = ConstituentManager(constituents_history)
        print(f"✅ 成分股管理器初始化完成")
        return constituent_manager
    except Exception as e:
        print(f"⚠️  加载成分股管理器时出错: {e}")
        return None


def main(factors_path='./factors/', data_path='./data/', signal_path='./signal/'):
    """
    主函数：加载因子数据，运行三个模型，保存预测结果
    
    Parameters:
    -----------
    factors_path : str, 因子文件目录路径
    data_path : str, 数据文件目录路径
    signal_path : str, 信号保存目录路径
    """
    print("="*80)
    print("中证1000多因子合成 - 因子合成模型")
    print("="*80)
    print()
    
    # 1. 加载因子数据
    print("【步骤1】加载因子数据")
    print("-"*80)
    factor_data = load_factors_from_directory(factors_path)
    
    if len(factor_data) == 0:
        print("❌ 未加载到任何因子数据，程序终止")
        return
    
    # 2. 加载收益率数据
    print("\n【步骤2】加载收益率数据")
    print("-"*80)
    return_data = load_price_data_and_calculate_returns(data_path)
    
    if return_data.empty:
        print("❌ 未加载到收益率数据，程序终止")
        return
    
    # 3. 加载成分股管理器
    print("\n【步骤3】加载成分股管理器")
    print("-"*80)
    constituent_manager = load_constituent_manager(data_path)
    
    # 4. 初始化因子合成器
    print("\n【步骤4】初始化因子合成器")
    print("-"*80)
    combiner = FactorCombiner(
        factor_data=factor_data,
        return_data=return_data,
        constituent_manager=constituent_manager
    )
    
    # 5. 运行模型1：因子筛选等权合成法（分类筛选）
    print("\n【步骤5】运行模型1：因子筛选等权合成法（分类筛选）")
    print("-"*80)
    model1_pred = combiner.model1_factor_selection_equal_weight(
        ml_min_factors=2,
        ml_max_factors=3,
        price_min_factors=3,
        price_max_factors=5,
        ic_threshold=0.02,
        correlation_threshold=0.7,
        train_ratio=0.8
    )
    
    # 6. 运行模型2：MLP深度神经网络模型
    print("\n【步骤6】运行模型2：MLP深度神经网络模型")
    print("-"*80)
    model2_pred = combiner.model2_mlp(
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        batch_size=256,
        epochs=100,
        patience=10,
        lr=0.001
    )
    
    # 7. 运行模型3：XGBoost模型
    print("\n【步骤7】运行模型3：XGBoost模型")
    print("-"*80)
    model3_pred = combiner.model3_xgboost(
        train_ratio=0.8,
        test_ratio=0.2,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # 8. 等权合成三个模型
    print("\n【步骤8】等权合成三个模型")
    print("-"*80)
    final_signal = combiner.combine_models()
    
    # 9. 保存预测结果
    print("\n【步骤9】保存预测结果")
    print("-"*80)
    os.makedirs(signal_path, exist_ok=True)
    combiner.save_predictions(save_dir=signal_path)
    
    print("\n" + "="*80)
    print("✅ 因子合成完成！")
    print("="*80)


if __name__ == '__main__':
    import sys
    
    # 设置默认路径（可根据实际情况修改）
    factors_path = 'd:/programme/vscode_c/courses/Software Enginerring/factors/'
    data_path = 'd:/programme/vscode_c/courses/Software Enginerring/data/'
    signal_path = 'd:/programme/vscode_c/courses/Software Enginerring/signal/'
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        factors_path = sys.argv[1]
    if len(sys.argv) > 2:
        data_path = sys.argv[2]
    if len(sys.argv) > 3:
        signal_path = sys.argv[3]
    
    main(factors_path=factors_path, data_path=data_path, signal_path=signal_path)
