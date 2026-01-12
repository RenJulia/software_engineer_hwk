# -*- coding: utf-8 -*-
"""
机器学习因子生成模块
使用GRU、Transformer、LightGBM、SVM、随机森林5种模型
对日度和月度收益率进行预测，生成10个新因子
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
import warnings
import time
import multiprocessing as mp
from functools import partial
warnings.filterwarnings('ignore')

# 尝试导入并行计算库
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# 导入工具函数
from ml_utils import (
    pearson_r, pearson_r_loss, standardize_and_weight,
    calculate_monthly_returns, prepare_sequence_data, prepare_prediction_data_all_dates,
    calculate_med_mad_stats, normalize_features
)

# 导入数据加载函数
from data_collection import load_data_from_csv
from factor_calculation import FactorCalculator

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ==================== 模型定义 ====================

class GRUModel(nn.Module):
    """GRU模型"""
    def __init__(self, input_dim=6, hidden_dim=30, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.bn(out)
        # Dropout只在训练时生效，eval()时会自动关闭
        out = self.dropout(out)  # 添加Dropout防止过拟合
        out = self.linear(out)
        return out


class TransformerModel(nn.Module):
    """Transformer模型"""
    def __init__(self, input_dim=6, hidden_dim=36, nhead=6, num_layers=6, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=4*hidden_dim, 
            batch_first=True,
            dropout=dropout  # 添加Dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        hidden = self.embedding(x)
        hidden = self.transformer_encoder(hidden)
        hidden = hidden[:, -1, :]  # 取最后一个时间步
        hidden = self.bn(hidden)
        # Dropout只在训练时生效，eval()时会自动关闭
        hidden = self.dropout(hidden)  # 添加Dropout防止过拟合
        output = self.linear(hidden)
        return output


class StockDataset(Dataset):
    """股票数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== 训练函数 ====================

def train_neural_network(
    model, 
    train_loader, 
    val_loader, 
    epochs=50, 
    lr=0.005, 
    patience=10,
    min_delta=1e-5,
    overfit_threshold=0.1,
    weight_decay=1e-5,
    grad_clip=1.0,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5
):
    """
    训练神经网络模型（使用IC损失，增强防过拟合机制）
    
    Parameters:
    -----------
    model : nn.Module, 模型
    train_loader : DataLoader, 训练数据加载器
    val_loader : DataLoader, 验证数据加载器
    epochs : int, 训练轮数
    lr : float, 学习率
    patience : int, 早停耐心值
    min_delta : float, 验证IC的最小改进阈值（小于此值不视为改进）
    overfit_threshold : float, 过拟合阈值（训练IC - 验证IC > 此值时触发早停）
    weight_decay : float, L2正则化系数（权重衰减）
    grad_clip : float, 梯度裁剪阈值
    lr_scheduler_patience : int, 学习率衰减的耐心值
    lr_scheduler_factor : float, 学习率衰减因子
    
    Returns:
    --------
    model : 训练好的模型
    """
    model = model.to(device)
    # 添加L2正则化（权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器（当验证IC不再提升时降低学习率）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # 最大化IC
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        verbose=True,
        min_lr=1e-6
    )
    
    best_ic = -np.inf
    best_epoch = 0
    patience_counter = 0
    train_ic_history = []
    val_ic_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_ic_list = []
        train_loss_list = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = pearson_r_loss(batch_y, pred)
            
            # 梯度裁剪，防止梯度爆炸
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            with torch.no_grad():
                ic = pearson_r(batch_y, pred).item()
                train_ic_list.append(ic)
                train_loss_list.append(loss.item())
        
        avg_train_ic = np.mean(train_ic_list)
        avg_train_loss = np.mean(train_loss_list)
        train_ic_history.append(avg_train_ic)
        
        # 验证阶段
        model.eval()
        val_ic_list = []
        val_loss_list = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_X)
                ic = pearson_r(batch_y, pred).item()
                loss = pearson_r_loss(batch_y, pred).item()
                val_ic_list.append(ic)
                val_loss_list.append(loss)
        
        avg_val_ic = np.mean(val_ic_list)
        avg_val_loss = np.mean(val_loss_list)
        val_ic_history.append(avg_val_ic)
        
        # 更新学习率
        scheduler.step(avg_val_ic)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算训练IC和验证IC的差距（过拟合指标）
        ic_gap = avg_train_ic - avg_val_ic
        
        # 检查是否有显著改进（考虑min_delta）
        improved = False
        if avg_val_ic > best_ic + min_delta:
            best_ic = avg_val_ic
            best_epoch = epoch
            patience_counter = 0
            improved = True
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 每5轮输出一次（更频繁的监控）
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train IC={avg_train_ic:.4f}, Val IC={avg_val_ic:.4f}, "
                  f"IC Gap={ic_gap:.4f}, "
                  f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                  f"LR={current_lr:.6f}")
        
        # 早停条件1：验证IC不再提升
        if patience_counter >= patience:
            print(f"早停于第 {epoch+1} 轮（验证IC不再提升）")
            print(f"  最佳验证IC: {best_ic:.4f} (第 {best_epoch+1} 轮)")
            print(f"  当前验证IC: {avg_val_ic:.4f}")
            break
        
        # 早停条件2：过拟合检测（训练IC远高于验证IC）
        if ic_gap > overfit_threshold and epoch > 10:  # 至少训练10轮后再检查
            print(f"早停于第 {epoch+1} 轮（检测到过拟合）")
            print(f"  训练IC: {avg_train_ic:.4f}, 验证IC: {avg_val_ic:.4f}")
            print(f"  IC差距: {ic_gap:.4f} > 阈值 {overfit_threshold:.4f}")
            print(f"  最佳验证IC: {best_ic:.4f} (第 {best_epoch+1} 轮)")
            break
        
        # 早停条件3：学习率过小（说明已经收敛）
        if current_lr < 1e-6:
            print(f"早停于第 {epoch+1} 轮（学习率过小，已收敛）")
            print(f"  当前学习率: {current_lr:.6f}")
            print(f"  最佳验证IC: {best_ic:.4f} (第 {best_epoch+1} 轮)")
            break
    
    # 加载最佳模型
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
        print(f"\n✅ 使用最佳模型（第 {best_epoch+1} 轮，验证IC: {best_ic:.4f}）")
    else:
        print(f"\n⚠️  未找到最佳模型，使用当前模型")
    
    return model


def train_sklearn_model(model, X_train, y_train, X_val, y_val):
    """
    训练sklearn模型（LightGBM、SVM、随机森林）
    
    Parameters:
    -----------
    model : sklearn模型
    X_train : np.ndarray, 训练特征
    y_train : np.ndarray, 训练标签
    X_val : np.ndarray, 验证特征
    y_val : np.ndarray, 验证标签
    
    Returns:
    --------
    model : 训练好的模型
    """
    # 将序列数据展平为特征向量
    if len(X_train.shape) == 3:  # (n_samples, seq_length, n_features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
    
    # 去除NaN
    train_mask = ~(np.isnan(X_train_flat).any(axis=1) | np.isnan(y_train))
    val_mask = ~(np.isnan(X_val_flat).any(axis=1) | np.isnan(y_val))
    
    X_train_clean = X_train_flat[train_mask]
    y_train_clean = y_train[train_mask]
    X_val_clean = X_val_flat[val_mask]
    y_val_clean = y_val[val_mask]
    
    # 训练模型
    model.fit(X_train_clean, y_train_clean)
    
    # 计算验证IC
    y_pred_val = model.predict(X_val_clean)
    from scipy import stats
    if len(y_val_clean) > 1:
        val_ic = stats.pearsonr(y_val_clean, y_pred_val)[0]
        print(f"验证IC: {val_ic:.4f}")
    
    return model


def train_svm_optimized(X_train, y_train, X_val, y_val, max_samples=10000, use_linear=True):
    """
    优化的SVM训练函数
    
    Parameters:
    -----------
    X_train : np.ndarray, 训练特征
    y_train : np.ndarray, 训练标签
    X_val : np.ndarray, 验证特征
    y_val : np.ndarray, 验证标签
    max_samples : int, 最大训练样本数（如果数据量太大则采样）
    use_linear : bool, 是否使用LinearSVR（推荐，更快）
    
    Returns:
    --------
    model : 训练好的SVM模型
    """
    from sklearn.svm import SVR, LinearSVR
    from scipy import stats
    
    # 将序列数据展平为特征向量
    if len(X_train.shape) == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val
    
    # 去除NaN
    train_mask = ~(np.isnan(X_train_flat).any(axis=1) | np.isnan(y_train))
    val_mask = ~(np.isnan(X_val_flat).any(axis=1) | np.isnan(y_val))
    
    X_train_clean = X_train_flat[train_mask]
    y_train_clean = y_train[train_mask]
    X_val_clean = X_val_flat[val_mask]
    y_val_clean = y_val[val_mask]
    
    # 如果数据量太大，进行采样
    if len(X_train_clean) > max_samples:
        print(f"  训练样本数: {len(X_train_clean)}，超过最大限制 {max_samples}，进行随机采样...")
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_train_clean), max_samples, replace=False)
        X_train_clean = X_train_clean[sample_indices]
        y_train_clean = y_train_clean[sample_indices]
        print(f"  采样后训练样本数: {len(X_train_clean)}")
    
    # 选择SVM模型
    if use_linear:
        # 使用LinearSVR（线性核，速度快10-100倍）
        # 注意：当dual=False时，必须使用squared_epsilon_insensitive损失函数
        # 当n_samples > n_features时，dual=False更快
        n_samples, n_features = X_train_clean.shape
        if n_samples > n_features:
            # 样本数大于特征数，使用dual=False（更快）
            # 必须使用squared_epsilon_insensitive，因为epsilon_insensitive不支持dual=False
            model = LinearSVR(
                C=1.0,
                epsilon=0.1,
                loss='squared_epsilon_insensitive',  # dual=False时必须使用此损失函数
                max_iter=1000,
                random_state=42,
                dual=False
            )
        else:
            # 样本数小于等于特征数，使用dual=True
            model = LinearSVR(
                C=1.0,
                epsilon=0.1,
                loss='epsilon_insensitive',  # dual=True时可以使用此损失函数
                max_iter=1000,
                random_state=42,
                dual=True
            )
    else:
        # 使用SVR with optimized parameters（如果需要非线性）
        model = SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            max_iter=500,  # 限制迭代次数
            cache_size=500,  # 增大缓存
            tol=1e-3  # 提前停止容差
        )
    
    # 训练模型
    print(f"  开始训练（样本数: {len(X_train_clean)}, 特征数: {X_train_clean.shape[1]}）...")
    start_time = time.time()
    model.fit(X_train_clean, y_train_clean)
    train_time = time.time() - start_time
    print(f"  训练完成，耗时: {train_time:.2f}秒")
    
    # 计算验证IC
    y_pred_val = model.predict(X_val_clean)
    if len(y_val_clean) > 1:
        val_ic = stats.pearsonr(y_val_clean, y_pred_val)[0]
        print(f"  验证IC: {val_ic:.4f}")
    
    return model


# ==================== 单因子训练和保存函数 ====================

def train_and_save_single_factor(factor_name, train_data, factors_path, save_immediately=True):
    """
    训练单个因子模型并保存
    
    Parameters:
    -----------
    factor_name : str, 因子名称
    train_data : dict, 训练数据字典
    factors_path : str, 因子保存路径
    save_immediately : bool, 是否立即保存
    
    Returns:
    --------
    pd.DataFrame : 因子数据
    """
    import time
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"训练因子: {factor_name}")
    print(f"{'='*60}")
    
    try:
        # 根据因子名称选择数据和模型
        if 'DAILY' in factor_name:
            y_train_std = train_data['y_daily_train_std']
            y_val_std = train_data['y_daily_val_std']
            if 'GRU' in factor_name or 'TRANSFORMER' in factor_name:
                train_loader = train_data['train_loader_daily']
                val_loader = train_data['val_loader_daily']
        else:  # MONTHLY
            y_train_std = train_data['y_monthly_train_std']
            y_val_std = train_data['y_monthly_val_std']
            if 'GRU' in factor_name or 'TRANSFORMER' in factor_name:
                train_loader = train_data['train_loader_monthly']
                val_loader = train_data['val_loader_monthly']
        
        X_train_final = train_data['X_train_final']
        X_val = train_data['X_val']
        X_all_dates = train_data['X_all_dates']  # 使用所有日期的数据
        stock_info_all_dates = train_data['stock_info_all_dates']  # 使用所有日期的信息
        returns_daily = train_data['returns_daily']
        
        # 训练模型（使用训练集数据）
        print(f"  训练模型（使用训练集数据，增强防过拟合机制）...")
        if 'GRU' in factor_name:
            model = GRUModel(input_dim=6, hidden_dim=30, dropout=0.3)  # 增加dropout
            model = train_neural_network(
                model, train_loader, val_loader, 
                epochs=100,  # 增加最大轮数，让早停机制发挥作用
                lr=0.001,  # 降低初始学习率
                patience=15,  # 增加耐心值
                min_delta=1e-4,  # 最小改进阈值
                overfit_threshold=0.15,  # 过拟合阈值
                weight_decay=1e-4,  # L2正则化
                grad_clip=1.0,  # 梯度裁剪
                lr_scheduler_patience=5,  # 学习率衰减耐心值
                lr_scheduler_factor=0.5  # 学习率衰减因子
            )
            # 对所有日期进行预测
            predictions = predict_neural_network(model, X_all_dates)
        elif 'TRANSFORMER' in factor_name:
            model = TransformerModel(input_dim=6, hidden_dim=36, nhead=6, num_layers=6, dropout=0.3)  # 增加dropout
            model = train_neural_network(
                model, train_loader, val_loader,
                epochs=100,  # 增加最大轮数
                lr=0.001,  # 降低初始学习率
                patience=15,  # 增加耐心值
                min_delta=1e-4,  # 最小改进阈值
                overfit_threshold=0.15,  # 过拟合阈值
                weight_decay=1e-4,  # L2正则化
                grad_clip=1.0,  # 梯度裁剪
                lr_scheduler_patience=5,  # 学习率衰减耐心值
                lr_scheduler_factor=0.5  # 学习率衰减因子
            )
            # 对所有日期进行预测
            predictions = predict_neural_network(model, X_all_dates)
        elif 'LIGHTGBM' in factor_name:
            import lightgbm as lgb
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
            model = train_sklearn_model(model, X_train_final, y_train_std, X_val, y_val_std)
            # 对所有日期进行预测
            predictions = predict_sklearn_model(model, X_all_dates)
        elif 'SVM' in factor_name:
            # train_svm_optimized返回的是模型，需要预测
            model = train_svm_optimized(X_train_final, y_train_std, X_val, y_val_std)
            # 对所有日期进行预测
            predictions = predict_sklearn_model(model, X_all_dates)
        elif 'RF' in factor_name:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            model = train_sklearn_model(model, X_train_final, y_train_std, X_val, y_val_std)
            # 对所有日期进行预测
            predictions = predict_sklearn_model(model, X_all_dates)
        else:
            raise ValueError(f"未知的因子类型: {factor_name}")
        
        print(f"  对所有日期进行预测（共{len(predictions)}个样本）...")
        
        # 转换为因子格式（所有日期）
        stocks = returns_daily.index.tolist()
        dates = returns_daily.columns.tolist()
        factor_df = pd.DataFrame(index=stocks, columns=dates)
        factor_df[:] = np.nan
        
        # 填充所有日期的预测值
        for i, info in enumerate(stock_info_all_dates):
            if i < len(predictions) and not np.isnan(predictions[i]):
                stock = info['stock']
                date = info['date']
                if stock in factor_df.index and date in factor_df.columns:
                    factor_df.loc[stock, date] = predictions[i]
        
        # 立即保存
        if save_immediately:
            factor_file = os.path.join(factors_path, f'{factor_name}.csv')
            factor_df.to_csv(factor_file, encoding='utf-8-sig')
            elapsed_time = time.time() - start_time
            non_null_count = (~factor_df.isna()).sum().sum()
            print(f"  ✅ {factor_name}: 计算完成并已保存")
            print(f"     耗时: {elapsed_time:.2f}秒, 非空值: {non_null_count}")
        else:
            elapsed_time = time.time() - start_time
            non_null_count = (~factor_df.isna()).sum().sum()
            print(f"  ✅ {factor_name}: 计算完成")
            print(f"     耗时: {elapsed_time:.2f}秒, 非空值: {non_null_count}")
        
        return factor_df
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ❌ {factor_name}: 计算失败（耗时: {elapsed_time:.2f}秒）")
        print(f"     错误: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== 预测函数 ====================

def predict_neural_network(model, X_test):
    """
    使用神经网络模型进行预测
    
    Parameters:
    -----------
    model : nn.Module, 模型
    X_test : np.ndarray, 测试特征
    
    Returns:
    --------
    np.ndarray : 预测值
    """
    model.eval()
    dataset = StockDataset(X_test, np.zeros(len(X_test)))
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for batch_X, _ in loader:
            batch_X = batch_X.to(device)
            pred = model(batch_X)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0).flatten()


def predict_sklearn_model(model, X_test):
    """
    使用sklearn模型进行预测
    
    Parameters:
    -----------
    model : sklearn模型
    X_test : np.ndarray, 测试特征
    
    Returns:
    --------
    np.ndarray : 预测值
    """
    if len(X_test.shape) == 3:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_test_flat = X_test
    
    # 去除NaN
    test_mask = ~np.isnan(X_test_flat).any(axis=1)
    X_test_clean = X_test_flat[test_mask]
    
    if len(X_test_clean) == 0:
        return np.full(len(X_test), np.nan)
    
    predictions = model.predict(X_test_clean)
    
    # 填充NaN位置
    pred_full = np.full(len(X_test), np.nan)
    pred_full[test_mask] = predictions
    
    return pred_full


# ==================== 主函数 ====================

def generate_ml_factors(
    data_path='d:/programme/vscode_c/courses/Software Enginerring/data/',
    factors_path='d:/programme/vscode_c/courses/Software Enginerring/factors/',
    sequence_length=40,
    train_test_split=0.8,
    retrain_period=120,
    min_train_days=500,
    factors_to_compute=None,
    n_jobs=None,
    save_immediately=True
):
    """
    生成机器学习因子
    
    Parameters:
    -----------
    data_path : str, 数据路径
    factors_path : str, 因子保存路径
    sequence_length : int, 序列长度
    train_test_split : float, 训练集比例
    retrain_period : int, 重新训练周期（天数）
    min_train_days : int, 最小训练天数
    factors_to_compute : list, 要计算的因子列表，None表示计算所有因子
                        可选值: ['GRU_DAILY', 'GRU_MONTHLY', 'TRANSFORMER_DAILY', 
                                'TRANSFORMER_MONTHLY', 'LIGHTGBM_DAILY', 'LIGHTGBM_MONTHLY',
                                'SVM_DAILY', 'SVM_MONTHLY', 'RF_DAILY', 'RF_MONTHLY']
    n_jobs : int, 并行任务数，None表示自动选择（神经网络用1，其他模型并行）
    save_immediately : bool, 是否在计算完每个因子后立即保存
    """
    print("="*80)
    print("机器学习因子生成")
    print("="*80 + "\n")
    
    # 设置并行任务数
    if n_jobs is None:
        # 自动选择：神经网络串行，sklearn模型并行
        if JOBLIB_AVAILABLE:
            n_jobs = min(mp.cpu_count(), 4)  # 限制最大并行数，避免内存问题
        else:
            n_jobs = 1
    elif n_jobs == -1:
        n_jobs = mp.cpu_count() if JOBLIB_AVAILABLE else 1
    
    print(f"并行设置: n_jobs={n_jobs}, joblib可用={JOBLIB_AVAILABLE}")
    
    # 1. 加载数据
    print("\n【步骤1】加载数据...")
    data = load_data_from_csv(data_path)
    
    if 'price_data' not in data or data['price_data'].empty:
        print("❌ 价格数据缺失")
        return None
    
    price_data = data['price_data'].copy()
    
    # 确保有必要的字段
    required_fields = ['HIGH_PRICE', 'OPEN_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME', 'VWAP']
    for field in required_fields:
        if field not in price_data.columns:
            print(f"❌ 缺少字段: {field}")
            return None
    
    # 计算收益率
    calculator = FactorCalculator(price_data=price_data)
    price_data = calculator.price_data.copy()
    
    # 准备收益率数据（日度和月度）
    print("\n【步骤2】准备收益率数据...")
    returns_daily = calculator.pivot_to_wide_format(
        price_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RETURN']].dropna(),
        'RETURN'
    )
    
    # 计算月度收益率
    dates = sorted(returns_daily.columns)
    returns_monthly = calculate_monthly_returns(returns_daily, dates)
    
    # 数据对齐检查
    if list(returns_daily.columns) != dates:
        print("⚠️  警告：returns_daily的列顺序与dates不一致，已重新对齐")
        returns_daily = returns_daily[dates]
    if list(returns_monthly.columns) != dates:
        print("⚠️  警告：returns_monthly的列顺序与dates不一致，已重新对齐")
        returns_monthly = returns_monthly[dates]
    
    print(f"日度收益率数据形状: {returns_daily.shape}")
    print(f"月度收益率数据形状: {returns_monthly.shape}")
    
    # 2. 准备特征数据（用于训练，使用step=5减少样本数）
    print("\n【步骤3】准备训练特征数据...")
    seq_data = prepare_sequence_data(price_data, sequence_length=sequence_length, step=5)
    
    if len(seq_data['X']) == 0:
        print("❌ 无法准备序列数据")
        return None
    
    X = seq_data['X']
    stock_info = seq_data['stock_info']
    
    print(f"训练特征数据形状: {X.shape}")
    print(f"训练样本数量: {len(stock_info)}")
    print("注意：训练数据使用step=5采样，以减少训练样本数量")
    
    # 计算统计量用于标准化（基于训练数据）
    med_x, mad_x = calculate_med_mad_stats(X)
    X_norm = normalize_features(X, med_x, mad_x)
    
    # 3. 准备标签数据
    print("\n【步骤4】准备标签数据...")
    
    # 创建股票代码到索引的映射
    stock_to_idx = {stock: idx for idx, stock in enumerate(returns_daily.index)}
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    
    # 为每个样本找到对应的收益率
    y_daily_list = []
    y_monthly_list = []
    valid_indices = []
    
    for i, info in enumerate(stock_info):
        stock = info['stock']
        date = info['date']
        
        if stock in stock_to_idx and date in date_to_idx:
            stock_idx = stock_to_idx[stock]
            date_idx = date_to_idx[date]
            
            # 日度收益率（下一天）
            # 注意：特征数据是到date_idx这一天为止，预测的是date_idx+1的收益率
            # 这确保了没有未来信息泄露
            if date_idx + 1 < len(dates):
                y_daily = returns_daily.iloc[stock_idx, date_idx + 1]
            else:
                y_daily = np.nan
            
            # 月度收益率（下一个月末）
            # 找到下一个月的月末（从当前日期之后开始查找）
            y_monthly = np.nan
            if date_idx + 1 < len(dates):
                # 获取当前日期所在的月份
                current_month = pd.to_datetime(dates[date_idx]).to_period('M')
                # 从下一天开始查找，直到找到下一个月的月末
                for j in range(date_idx + 1, len(dates)):
                    check_date = pd.to_datetime(dates[j])
                    check_month = check_date.to_period('M')
                    # 如果到了新的月份，检查是否是月末（有月度收益率值）
                    if check_month > current_month:
                        monthly_ret = returns_monthly.iloc[stock_idx, j]
                        if not np.isnan(monthly_ret):
                            y_monthly = monthly_ret
                            break
                    # 如果仍在同一个月，继续查找
                    elif check_month == current_month:
                        # 如果是当前月的月末，也可以使用（但从date_idx+1开始，所以不应该是当前月）
                        continue
            
            # 至少有一个标签有效就保留样本
            if not (np.isnan(y_daily) and np.isnan(y_monthly)):
                y_daily_list.append(y_daily)
                y_monthly_list.append(y_monthly)
                valid_indices.append(i)
    
    X_valid = X_norm[valid_indices]
    y_daily = np.array(y_daily_list)
    y_monthly = np.array(y_monthly_list)
    stock_info_valid = [stock_info[i] for i in valid_indices]
    
    print(f"有效样本数: {len(X_valid)}")
    print(f"日度收益率有效数: {(~np.isnan(y_daily)).sum()}")
    print(f"月度收益率有效数: {(~np.isnan(y_monthly)).sum()}")
    
    # 4. 训练和预测
    print("\n【步骤5】训练模型并生成预测...")
    
    # 按日期划分训练集和测试集
    dates_valid = [info['date'] for info in stock_info_valid]
    unique_dates = sorted(set(dates_valid))
    split_idx = int(len(unique_dates) * train_test_split)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    
    train_mask = np.array([date in train_dates for date in dates_valid])
    test_mask = np.array([date in test_dates for date in dates_valid])
    
    X_train = X_valid[train_mask]
    y_daily_train = y_daily[train_mask]
    y_monthly_train = y_monthly[train_mask]
    X_test = X_valid[test_mask]
    y_daily_test = y_daily[test_mask]
    y_monthly_test = y_monthly[test_mask]
    stock_info_test = [stock_info_valid[i] for i in range(len(stock_info_valid)) if test_mask[i]]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 进一步划分训练集和验证集
    val_split = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split]
    X_val = X_train[val_split:]
    y_daily_train_final = y_daily_train[:val_split]
    y_daily_val = y_daily_train[val_split:]
    y_monthly_train_final = y_monthly_train[:val_split]
    y_monthly_val = y_monthly_train[val_split:]
    
    # 标准化标签（使用训练集的统计量标准化所有数据，避免数据泄露）
    y_daily_train_std, _, daily_mean, daily_std = standardize_and_weight(y_daily_train_final)
    y_daily_val_std, _ = standardize_and_weight(y_daily_val, mean=daily_mean, std=daily_std)
    y_monthly_train_std, _, monthly_mean, monthly_std = standardize_and_weight(y_monthly_train_final)
    y_monthly_val_std, _ = standardize_and_weight(y_monthly_val, mean=monthly_mean, std=monthly_std)
    
    # 创建数据加载器（用于神经网络）
    train_dataset_daily = StockDataset(X_train_final, y_daily_train_std)
    val_dataset_daily = StockDataset(X_val, y_daily_val_std)
    train_loader_daily = DataLoader(train_dataset_daily, batch_size=5000, shuffle=True)
    val_loader_daily = DataLoader(val_dataset_daily, batch_size=5000, shuffle=False)
    
    train_dataset_monthly = StockDataset(X_train_final, y_monthly_train_std)
    val_dataset_monthly = StockDataset(X_val, y_monthly_val_std)
    train_loader_monthly = DataLoader(train_dataset_monthly, batch_size=5000, shuffle=True)
    val_loader_monthly = DataLoader(val_dataset_monthly, batch_size=5000, shuffle=False)
    
    # 5. 定义所有要计算的因子任务
    print("\n【步骤5】准备因子计算任务...")
    
    # 定义所有可能的因子
    all_possible_factors = [
        'GRU_DAILY', 'GRU_MONTHLY',
        'TRANSFORMER_DAILY', 'TRANSFORMER_MONTHLY',
        'LIGHTGBM_DAILY', 'LIGHTGBM_MONTHLY',
        'SVM_DAILY', 'SVM_MONTHLY',
        'RF_DAILY', 'RF_MONTHLY'
    ]
    
    # 如果指定了要计算的因子，则只计算这些因子
    if factors_to_compute is None:
        factors_to_compute = all_possible_factors
    else:
        # 验证因子名称
        invalid_factors = [f for f in factors_to_compute if f not in all_possible_factors]
        if invalid_factors:
            print(f"⚠️  警告：以下因子名称无效，将被忽略: {invalid_factors}")
        factors_to_compute = [f for f in factors_to_compute if f in all_possible_factors]
    
    # 检查已存在的因子文件，跳过已计算的因子
    os.makedirs(factors_path, exist_ok=True)
    existing_factors = []
    for factor_name in factors_to_compute:
        factor_file = os.path.join(factors_path, f'{factor_name}.csv')
        if os.path.exists(factor_file):
            try:
                # 检查文件是否有效（非空）
                test_df = pd.read_csv(factor_file, index_col=0, nrows=1)
                if not test_df.empty:
                    existing_factors.append(factor_name)
                    print(f"  ⏭️  {factor_name}: 已存在，跳过计算")
            except:
                pass
    
    # 需要计算的因子
    factors_to_compute = [f for f in factors_to_compute if f not in existing_factors]
    
    if not factors_to_compute:
        print("\n✅ 所有因子已计算完成，无需重新计算")
        # 加载已存在的因子
        factors_dict = {}
        for factor_name in all_possible_factors:
            factor_file = os.path.join(factors_path, f'{factor_name}.csv')
            if os.path.exists(factor_file):
                try:
                    factors_dict[factor_name] = pd.read_csv(factor_file, index_col=0, encoding='utf-8-sig')
                    factors_dict[factor_name].columns = pd.to_datetime(factors_dict[factor_name].columns)
                except:
                    pass
        return factors_dict
    
    print(f"\n需要计算的因子: {factors_to_compute}")
    print(f"已存在的因子: {existing_factors if existing_factors else '无'}")
    
    # 准备所有日期的预测数据（用于生成日频因子）
    print("\n【步骤5.5】准备所有日期的预测数据（日频，step=1）...")
    pred_data_all = prepare_prediction_data_all_dates(
        price_data, 
        sequence_length=sequence_length,
        med_x=None,  # 不在这里标准化，稍后统一标准化
        mad_x=None
    )
    X_all_dates_raw = pred_data_all['X']
    stock_info_all_dates = pred_data_all['stock_info']
    
    # 使用训练数据的统计量标准化预测数据
    if len(X_all_dates_raw) > 0:
        X_all_dates = normalize_features(X_all_dates_raw, med_x, mad_x)
    else:
        X_all_dates = X_all_dates_raw
    
    print(f"所有日期预测数据: {X_all_dates.shape}")
    print(f"预测样本数: {len(stock_info_all_dates)}")
    print("注意：预测数据使用step=1，确保每个交易日都有因子值")
    
    # 准备训练数据（所有模型共享）
    train_data = {
        'X_train_final': X_train_final,
        'X_val': X_val,
        'X_test': X_test,
        'X_all_dates': X_all_dates,  # 新增：所有日期的预测数据
        'stock_info_all_dates': stock_info_all_dates,  # 新增：所有日期的股票信息
        'y_daily_train_std': y_daily_train_std,
        'y_daily_val_std': y_daily_val_std,
        'y_monthly_train_std': y_monthly_train_std,
        'y_monthly_val_std': y_monthly_val_std,
        'train_loader_daily': train_loader_daily,
        'val_loader_daily': val_loader_daily,
        'train_loader_monthly': train_loader_monthly,
        'val_loader_monthly': val_loader_monthly,
        'stock_info_test': stock_info_test,
        'returns_daily': returns_daily,
        'med_x': med_x,  # 新增：用于标准化
        'mad_x': mad_x   # 新增：用于标准化
    }
    
    # 6. 训练模型并生成因子（支持并行和立即保存）
    print("\n【步骤6】训练模型并生成因子...")
    
    # 分离神经网络模型和sklearn模型（神经网络需要串行，sklearn可以并行）
    neural_network_factors = [f for f in factors_to_compute if 'GRU' in f or 'TRANSFORMER' in f]
    sklearn_factors = [f for f in factors_to_compute if f not in neural_network_factors]
    
    factors_dict = {}
    
    # 先训练神经网络模型（串行，因为需要GPU或共享资源）
    for factor_name in neural_network_factors:
        try:
            factor_df = train_and_save_single_factor(
                factor_name, train_data, factors_path, save_immediately=save_immediately
            )
            if factor_df is not None:
                factors_dict[factor_name] = factor_df
        except Exception as e:
            print(f"  ❌ {factor_name}: 计算失败 - {e}")
            import traceback
            traceback.print_exc()
    
    # 然后并行训练sklearn模型
    if sklearn_factors:
        if JOBLIB_AVAILABLE and n_jobs is not None and n_jobs > 1:
            # 并行计算
            print(f"\n使用并行计算训练 {len(sklearn_factors)} 个sklearn模型（n_jobs={n_jobs}）...")
            results = Parallel(n_jobs=n_jobs, backend='threading', verbose=1)(
                delayed(train_and_save_single_factor)(
                    factor_name, train_data, factors_path, save_immediately=save_immediately
                )
                for factor_name in sklearn_factors
            )
            for factor_name, factor_df in zip(sklearn_factors, results):
                if factor_df is not None:
                    factors_dict[factor_name] = factor_df
        else:
            # 串行计算
            for factor_name in sklearn_factors:
                try:
                    factor_df = train_and_save_single_factor(
                        factor_name, train_data, factors_path, save_immediately=save_immediately
                    )
                    if factor_df is not None:
                        factors_dict[factor_name] = factor_df
                except Exception as e:
                    print(f"  ❌ {factor_name}: 计算失败 - {e}")
                    import traceback
                    traceback.print_exc()
    
    # 加载已存在的因子
    for factor_name in existing_factors:
        factor_file = os.path.join(factors_path, f'{factor_name}.csv')
        try:
            factors_dict[factor_name] = pd.read_csv(factor_file, index_col=0, encoding='utf-8-sig')
            factors_dict[factor_name].columns = pd.to_datetime(factors_dict[factor_name].columns)
        except:
            pass
    
    print("\n" + "="*80)
    print("✅ 机器学习因子生成完成！")
    print("="*80)
    print(f"\n生成的因子:")
    for factor_name in sorted(factors_dict.keys()):
        non_null_count = (~factors_dict[factor_name].isna()).sum().sum()
        print(f"  - {factor_name}: {factors_dict[factor_name].shape}, 非空值: {non_null_count}")
    
    return factors_dict


if __name__ == '__main__':
    # 设置路径
    data_path = '/mnt/SE/data/'
    factors_path = '/mnt/SE/factors/'
    
    # 生成因子
    # 可以指定要计算的因子，例如: factors_to_compute=['GRU_DAILY', 'LIGHTGBM_DAILY']
    # None表示计算所有因子
    factors = generate_ml_factors(
        data_path=data_path,
        factors_path=factors_path,
        sequence_length=40,
        train_test_split=0.8,
        factors_to_compute=None,  # 可以指定要计算的因子列表
        n_jobs=mp.cpu_count() if JOBLIB_AVAILABLE else 1,  # 并行任务数
        save_immediately=True  # 每个因子计算完成后立即保存
    )
