# -*- coding: utf-8 -*-
"""
机器学习因子生成 - 工具函数模块
包含IC损失函数、数据标准化等辅助函数
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import stats


def pearson_r(y_true, y_pred):
    """
    计算皮尔逊相关系数（用于IC计算）
    
    Parameters:
    -----------
    y_true : torch.Tensor, 真实值
    y_pred : torch.Tensor, 预测值
    
    Returns:
    --------
    torch.Tensor : 皮尔逊相关系数
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # 去除NaN
    mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
    if mask.sum() < 2:
        return torch.tensor(0.0, device=y_true.device)
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # 计算均值
    mean_true = torch.mean(y_true_clean)
    mean_pred = torch.mean(y_pred_clean)
    
    # 计算协方差和标准差
    numerator = torch.sum((y_true_clean - mean_true) * (y_pred_clean - mean_pred))
    std_true = torch.std(y_true_clean)
    std_pred = torch.std(y_pred_clean)
    
    if std_true < 1e-8 or std_pred < 1e-8:
        return torch.tensor(0.0, device=y_true.device)
    
    corr = numerator / (std_true * std_pred * len(y_true_clean))
    return corr


def pearson_r_loss(y_true, y_pred):
    """
    IC损失函数：负的皮尔逊相关系数（用于最大化IC）
    
    Parameters:
    -----------
    y_true : torch.Tensor, 真实值
    y_pred : torch.Tensor, 预测值
    
    Returns:
    --------
    torch.Tensor : 损失值（负的IC）
    """
    return -pearson_r(y_true, y_pred)


def CCC(y_true, y_pred):
    """
    Concordance Correlation Coefficient (CCC)
    
    Parameters:
    -----------
    y_true : torch.Tensor, 真实值
    y_pred : torch.Tensor, 预测值
    
    Returns:
    --------
    torch.Tensor : CCC值
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mask = ~(torch.isnan(y_true) | torch.isnan(y_pred))
    if mask.sum() < 2:
        return torch.tensor(0.0, device=y_true.device)
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mean_true = torch.mean(y_true_clean)
    mean_pred = torch.mean(y_pred_clean)
    
    numerator = 2 * torch.sum((y_true_clean - mean_true) * (y_pred_clean - mean_pred))
    denominator = torch.sum((y_true_clean - mean_true) ** 2) + torch.sum((y_pred_clean - mean_pred) ** 2) + len(y_true_clean) * (mean_true - mean_pred) ** 2
    
    if denominator < 1e-8:
        return torch.tensor(0.0, device=y_true.device)
    
    ccc = numerator / denominator
    return ccc


def standardize_and_weight(y, mean=None, std=None):
    """
    标准化并计算权重（用于截面标准化）
    
    Parameters:
    -----------
    y : np.ndarray, 截面数据
    mean : float, 可选，如果提供则使用该均值进行标准化
    std : float, 可选，如果提供则使用该标准差进行标准化
    
    Returns:
    --------
    tuple : (标准化后的y, 权重, mean, std) 如果mean和std为None，返回计算得到的统计量
    """
    y = np.array(y).flatten()
    
    # 去除NaN
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        if mean is None or std is None:
            return y, np.ones_like(y), 0.0, 1.0
        else:
            return y, np.ones_like(y)
    
    y_valid = y[valid_mask]
    
    # 使用提供的统计量，或计算新的统计量
    if mean is None:
        mean_y = np.mean(y_valid)
    else:
        mean_y = mean
    
    if std is None:
        std_y = np.std(y_valid)
    else:
        std_y = std
    
    if std_y < 1e-8:
        y_std = np.zeros_like(y)
        y_std[valid_mask] = 0.0
    else:
        y_std = np.full_like(y, np.nan)
        y_std[valid_mask] = (y_valid - mean_y) / std_y
    
    # 计算权重（基于标准化后的值的绝对值，用于加权IC）
    weights = np.ones_like(y)
    weights[valid_mask] = np.abs(y_std[valid_mask])
    weights = weights / (np.sum(weights[valid_mask]) + 1e-8)
    
    if mean is None or std is None:
        return y_std, weights, mean_y, std_y
    else:
        return y_std, weights


def calculate_monthly_returns(daily_returns, dates):
    """
    计算月度收益率
    
    Parameters:
    -----------
    daily_returns : pd.DataFrame, 日度收益率数据（股票×日期）
    dates : list, 日期列表
    
    Returns:
    --------
    pd.DataFrame : 月度收益率数据（股票×日期，只在月末有值）
    """
    monthly_returns = daily_returns.copy()
    monthly_returns[:] = np.nan
    
    # 将日期转换为月份
    if isinstance(dates[0], str):
        date_series = pd.to_datetime(dates)
    else:
        date_series = pd.Series(dates)
    
    months = date_series.dt.to_period('M')
    
    # 找到每个月的最后一天
    last_day_of_month = {}
    for i, month in enumerate(months):
        if month not in last_day_of_month or i > last_day_of_month[month]:
            last_day_of_month[month] = i
    
    # 计算月度收益率（月末相对于月初）
    for month, end_idx in last_day_of_month.items():
        # 找到该月的第一天
        month_mask = months == month
        if month_mask.sum() == 0:
            continue
        
        start_idx = np.where(month_mask)[0][0]
        
        if start_idx == end_idx:
            continue
        
        # 计算月度收益率：(1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        month_returns = daily_returns.iloc[:, start_idx:end_idx+1]
        # 处理NaN值：如果某天是NaN，则月度收益率也为NaN
        monthly_ret = (1 + month_returns).prod(axis=1) - 1
        
        # 如果该月有任何一天是NaN，则月度收益率也为NaN
        has_nan = month_returns.isna().any(axis=1)
        monthly_ret[has_nan] = np.nan
        
        monthly_returns.iloc[:, end_idx] = monthly_ret
    
    return monthly_returns


def prepare_sequence_data(price_data, sequence_length=40, step=5):
    """
    准备序列数据用于模型训练
    
    Parameters:
    -----------
    price_data : pd.DataFrame, 价格数据，包含HIGH_PRICE, OPEN_PRICE, LOW_PRICE, CLOSE_PRICE, VOLUME, VWAP
    sequence_length : int, 序列长度（默认40天）
    step : int, 采样步长（默认5天，用于训练时减少样本数；预测时应使用step=1）
    
    Returns:
    --------
    dict : 包含X（特征序列）和stock_info（股票代码和日期信息）
    """
    # 提取6维特征
    features = ['HIGH_PRICE', 'OPEN_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'VWAP']
    
    # 检查必需的列是否存在
    missing_cols = [col for col in features if col not in price_data.columns]
    if missing_cols:
        raise ValueError(f"价格数据中缺少必需的列: {missing_cols}。"
                        f"可用列: {list(price_data.columns)}")
    
    # 检查必需的索引列
    if 'S_INFO_WINDCODE' not in price_data.columns:
        raise ValueError("价格数据中缺少必需的列: S_INFO_WINDCODE")
    if 'TRADE_DT' not in price_data.columns:
        raise ValueError("价格数据中缺少必需的列: TRADE_DT")
    
    # 按股票分组
    stocks = price_data['S_INFO_WINDCODE'].unique()
    dates = sorted(price_data['TRADE_DT'].unique())
    
    X_list = []
    stock_info_list = []
    
    for stock in stocks:
        stock_data = price_data[price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT')
        
        if len(stock_data) < sequence_length:
            continue
        
        # 沿着时间轴采样
        for j in range(0, len(stock_data) - sequence_length + 1, step):
            # 提取序列
            end_idx = j + sequence_length
            
            # 确保索引有效
            if end_idx > len(stock_data):
                continue
            
            # 使用更安全的方式：先选择行，再选择列
            try:
                # 选择行切片
                seq_slice = stock_data.iloc[j:end_idx]
                # 确保列存在后再选择
                available_features = [f for f in features if f in seq_slice.columns]
                if len(available_features) != len(features):
                    # 缺少必需的列，跳过
                    continue
                # 选择列
                seq_data = seq_slice[features].values
            except (KeyError, IndexError, ValueError, TypeError) as e:
                # 如果失败，跳过这个样本
                continue
            
            # 检查数据形状
            if seq_data.shape[0] != sequence_length or seq_data.shape[1] != len(features):
                continue
            
            # 检查是否有NaN或零值
            if np.isnan(seq_data).any() or (seq_data[-1, :] == 0).any():
                continue
            
            # 归一化：除以最后一天的值（相对价格）
            seq_data_last = np.tile(seq_data[-1, :], (seq_data.shape[0], 1))
            seq_data_norm = seq_data / (seq_data_last + 1e-8)
            
            X_list.append(seq_data_norm)
            stock_info_list.append({
                'stock': stock,
                'date': stock_data.iloc[j+sequence_length-1]['TRADE_DT'],
                'date_idx': j + sequence_length - 1
            })
    
    if len(X_list) == 0:
        return {'X': np.array([]), 'stock_info': []}
    
    X = np.array(X_list, dtype=np.float32)
    
    return {
        'X': X,
        'stock_info': stock_info_list
    }


def prepare_prediction_data_all_dates(price_data, sequence_length=40, med_x=None, mad_x=None):
    """
    为所有交易日准备预测数据（用于生成日频因子）
    注意：这个函数使用step=1，确保每个交易日都有预测值
    
    Parameters:
    -----------
    price_data : pd.DataFrame, 价格数据
    sequence_length : int, 序列长度
    med_x : np.ndarray, 中位数统计量（用于标准化）
    mad_x : np.ndarray, MAD统计量（用于标准化）
    
    Returns:
    --------
    dict : 包含X（特征序列）和stock_info（股票代码和日期信息）
    """
    # 提取6维特征
    features = ['HIGH_PRICE', 'OPEN_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'VWAP']
    
    # 检查必需的列是否存在
    missing_cols = [col for col in features if col not in price_data.columns]
    if missing_cols:
        raise ValueError(f"价格数据中缺少必需的列: {missing_cols}。"
                        f"可用列: {list(price_data.columns)}")
    
    # 检查必需的索引列
    if 'S_INFO_WINDCODE' not in price_data.columns:
        raise ValueError("价格数据中缺少必需的列: S_INFO_WINDCODE")
    if 'TRADE_DT' not in price_data.columns:
        raise ValueError("价格数据中缺少必需的列: TRADE_DT")
    
    # 按股票分组
    stocks = price_data['S_INFO_WINDCODE'].unique()
    
    X_list = []
    stock_info_list = []
    
    for stock in stocks:
        stock_data = price_data[price_data['S_INFO_WINDCODE'] == stock].sort_values('TRADE_DT')
        
        if len(stock_data) < sequence_length:
            continue
        
        # 使用step=1，确保每个交易日都有数据
        for j in range(sequence_length - 1, len(stock_data)):
            # 提取序列（从j-sequence_length+1到j）
            start_idx = j - sequence_length + 1
            end_idx = j + 1
            
            # 确保索引有效
            if start_idx < 0 or end_idx > len(stock_data):
                continue
            
            # 使用更安全的方式：先选择行，再选择列
            try:
                # 选择行切片
                seq_slice = stock_data.iloc[start_idx:end_idx]
                # 确保列存在后再选择
                available_features = [f for f in features if f in seq_slice.columns]
                if len(available_features) != len(features):
                    # 缺少必需的列，跳过
                    continue
                # 选择列
                seq_data = seq_slice[features].values
            except (KeyError, IndexError, ValueError, TypeError) as e:
                # 如果失败，跳过这个样本
                continue
            
            # 检查数据形状
            if seq_data.shape[0] != sequence_length or seq_data.shape[1] != len(features):
                continue
            
            # 检查是否有NaN或零值
            if np.isnan(seq_data).any() or (seq_data[-1, :] == 0).any():
                continue
            
            # 归一化：除以最后一天的值（相对价格）
            seq_data_last = np.tile(seq_data[-1, :], (seq_data.shape[0], 1))
            seq_data_norm = seq_data / (seq_data_last + 1e-8)
            
            X_list.append(seq_data_norm)
            stock_info_list.append({
                'stock': stock,
                'date': stock_data.iloc[j]['TRADE_DT'],
                'date_idx': j
            })
    
    if len(X_list) == 0:
        return {'X': np.array([]), 'stock_info': []}
    
    X = np.array(X_list, dtype=np.float32)
    
    return {
        'X': X,
        'stock_info': stock_info_list
    }


def calculate_med_mad_stats(X):
    """
    计算中位数和MAD（Median Absolute Deviation）统计量，用于数据标准化
    
    Parameters:
    -----------
    X : np.ndarray, 特征数据 (n_samples, seq_length, n_features)
    
    Returns:
    --------
    tuple : (med_x, mad_x) 中位数和MAD数组
    """
    n_samples, seq_length, n_features = X.shape
    
    med_x = np.zeros((seq_length, n_features))
    mad_x = np.zeros((seq_length, n_features))
    
    for k in range(seq_length):
        for s in range(n_features):
            values = X[:, k, s]
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                med_x[k, s] = np.median(valid_values)
                mad_x[k, s] = np.median(np.abs(valid_values - med_x[k, s])) + 1e-8
            else:
                med_x[k, s] = 0.0
                mad_x[k, s] = 1.0
    
    return med_x, mad_x


def normalize_features(X, med_x, mad_x):
    """
    使用中位数和MAD标准化特征
    
    Parameters:
    -----------
    X : np.ndarray, 特征数据
    med_x : np.ndarray, 中位数数组
    mad_x : np.ndarray, MAD数组
    
    Returns:
    --------
    np.ndarray : 标准化后的特征数据
    """
    X_norm = X.copy()
    n_samples, seq_length, n_features = X.shape
    
    for k in range(seq_length):
        for s in range(n_features):
            X_norm[:, k, s] = (X[:, k, s] - med_x[k, s]) / mad_x[k, s]
    
    return X_norm
