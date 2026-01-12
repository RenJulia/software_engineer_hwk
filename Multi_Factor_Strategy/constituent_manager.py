# -*- coding: utf-8 -*-
"""
成分股管理器 - 处理中证1000成分股的时间序列变动
"""

import pandas as pd
import numpy as np
from datetime import datetime

class ConstituentManager:
    """成分股时间序列管理器"""
    
    def __init__(self, constituents_history_df):
        """
        初始化成分股管理器
        
        Parameters:
        -----------
        constituents_history_df : pd.DataFrame
            成分股历史变动记录，必须包含列：
            - S_INFO_WINDCODE: 股票代码
            - S_CON_INDATE: 纳入日期 (datetime)
            - S_CON_OUTDATE: 剔除日期 (datetime, 可能为NaT)
        """
        self.constituents_df = constituents_history_df.copy()
        
        # 确保日期字段是datetime类型
        if 'S_CON_INDATE' in self.constituents_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.constituents_df['S_CON_INDATE']):
                self.constituents_df['S_CON_INDATE'] = pd.to_datetime(self.constituents_df['S_CON_INDATE'], errors='coerce')
        
        if 'S_CON_OUTDATE' in self.constituents_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.constituents_df['S_CON_OUTDATE']):
                self.constituents_df['S_CON_OUTDATE'] = pd.to_datetime(self.constituents_df['S_CON_OUTDATE'], errors='coerce')
        
        print(f"✅ 成分股管理器初始化完成，共 {len(self.constituents_df)} 条变动记录")
    
    def get_constituents_by_date(self, date):
        """
        获取指定日期的成分股列表
        
        Parameters:
        -----------
        date : str or datetime
            日期，格式'YYYYMMDD'或datetime对象
        
        Returns:
        --------
        list : 该日期的成分股代码列表
        """
        if isinstance(date, str):
            date = pd.to_datetime(date, format='%Y%m%d')
        elif isinstance(date, pd.Timestamp):
            pass
        else:
            date = pd.to_datetime(date)
        
        # 筛选条件：纳入日期 <= 指定日期 AND (剔除日期 IS NULL OR 剔除日期 >= 指定日期)
        mask = (
            (self.constituents_df['S_CON_INDATE'] <= date) &
            (
                self.constituents_df['S_CON_OUTDATE'].isna() |
                (self.constituents_df['S_CON_OUTDATE'] >= date)
            )
        )
        
        constituents = self.constituents_df.loc[mask, 'S_INFO_WINDCODE'].unique().tolist()
        return constituents
    
    def get_constituents_mask(self, stock_list, date):
        """
        获取指定日期成分股的布尔掩码
        
        Parameters:
        -----------
        stock_list : list or pd.Index
            股票代码列表
        date : str or datetime
            日期
        
        Returns:
        --------
        pd.Series : 布尔掩码，True表示该股票在指定日期是成分股
        """
        constituents = self.get_constituents_by_date(date)
        constituents_set = set(constituents)
        mask = pd.Series([stock in constituents_set for stock in stock_list], index=stock_list)
        return mask
    
    def get_all_constituents(self, start_date=None, end_date=None):
        """
        获取日期范围内所有曾经是成分股的股票列表
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            开始日期
        end_date : str or datetime, optional
            结束日期
        
        Returns:
        --------
        list : 所有曾经是成分股的股票代码列表
        """
        mask = pd.Series(True, index=self.constituents_df.index)
        
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date, format='%Y%m%d')
            mask = mask & (self.constituents_df['S_CON_INDATE'] <= start_date) & (
                self.constituents_df['S_CON_OUTDATE'].isna() | 
                (self.constituents_df['S_CON_OUTDATE'] >= start_date)
            )
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date, format='%Y%m%d')
            mask = mask & (
                (self.constituents_df['S_CON_INDATE'] <= end_date) &
                (
                    self.constituents_df['S_CON_OUTDATE'].isna() |
                    (self.constituents_df['S_CON_OUTDATE'] >= end_date)
                )
            )
        
        all_stocks = self.constituents_df.loc[mask, 'S_INFO_WINDCODE'].unique().tolist()
        return all_stocks
    
    def filter_by_constituents(self, df, date, stock_col='S_INFO_WINDCODE', date_col='TRADE_DT'):
        """
        根据成分股列表过滤DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            要过滤的DataFrame
        date : str or datetime
            日期
        stock_col : str
            股票代码列名
        date_col : str
            日期列名
        
        Returns:
        --------
        pd.DataFrame : 过滤后的DataFrame
        """
        constituents = self.get_constituents_by_date(date)
        constituents_set = set(constituents)
        
        if date_col in df.columns:
            # 如果DataFrame包含日期列，同时过滤日期和股票
            if isinstance(date, str):
                date = pd.to_datetime(date, format='%Y%m%d')
            
            mask = (df[stock_col].isin(constituents_set)) & (df[date_col] == date)
        else:
            # 如果DataFrame不包含日期列，只过滤股票
            mask = df[stock_col].isin(constituents_set)
        
        return df.loc[mask].copy()
    
    def build_constituent_matrix(self, dates):
        """
        构建成分股矩阵（股票×日期），值为True表示该股票在该日期是成分股
        
        Parameters:
        -----------
        dates : list
            日期列表（datetime或字符串）
        
        Returns:
        --------
        pd.DataFrame : 成分股矩阵，index为股票代码，columns为日期，值为布尔值
        """
        all_stocks = self.constituents_df['S_INFO_WINDCODE'].unique()
        
        # 转换为datetime格式
        dates_dt = [pd.to_datetime(d, format='%Y%m%d') if isinstance(d, str) else pd.to_datetime(d) for d in dates]
        
        # 构建矩阵
        matrix = pd.DataFrame(False, index=all_stocks, columns=dates_dt)
        
        for date in dates_dt:
            constituents = self.get_constituents_by_date(date)
            matrix.loc[constituents, date] = True
        
        return matrix
