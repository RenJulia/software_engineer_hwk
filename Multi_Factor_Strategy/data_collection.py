# -*- coding: utf-8 -*-
"""
中证1000多因子策略 - 数据获取模块
从Oracle数据库获取A股量价数据
"""

import oracledb
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Oracle数据库配置
lib_dir = os.path.expanduser("D:\\Software\\Oracle\\instantclient_23_0")
db_config = {
    "user": "student2501212302",
    "password": "student2501212302",
    "dsn": "219.223.208.52/orcl"
}

def init_oracle_client():
    """初始化Oracle客户端"""
    try:
        oracledb.init_oracle_client(lib_dir=lib_dir)
        print("✅ Instant Client 驱动加载成功")
    except Exception as e:
        print(f"⚠️  驱动加载提示: {e}")
        # 如果已经初始化过，会报错但可以继续使用

def get_csi1000_constituents_history(cursor):
    """
    获取中证1000成分股的所有历史变动记录
    
    根据AIndexMembers数据字典，获取所有历史成分股的纳入和剔除日期记录。
    这将用于构建成分股时间序列，确定每个日期的成分股列表。
    
    Parameters:
    -----------
    cursor : Oracle cursor
    
    Returns:
    --------
    pd.DataFrame : 所有历史成分股变动记录，包含S_INFO_WINDCODE, S_CON_INDATE, S_CON_OUTDATE等
    """
    sql = """
    SELECT 
        S_CON_WINDCODE as S_INFO_WINDCODE,
        S_CON_INDATE,
        S_CON_OUTDATE,
        CUR_SIGN
    FROM FILESYNC.AINDEXMEMBERS
    WHERE S_INFO_WINDCODE = '000852.SH'
    ORDER BY S_CON_INDATE, S_INFO_WINDCODE
    """
    
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    
    if rows and len(rows) > 0:
        df = pd.DataFrame(rows, columns=columns)
        # 从S_INFO_WINDCODE提取S_INFO_CODE（去掉后缀部分）
        if 'S_INFO_WINDCODE' in df.columns:
            df['S_INFO_CODE'] = df['S_INFO_WINDCODE'].str.split('.').str[0]
        
        # 处理日期字段：转换为datetime格式
        df['S_CON_INDATE'] = pd.to_datetime(df['S_CON_INDATE'], format='%Y%m%d', errors='coerce')
        df['S_CON_OUTDATE'] = pd.to_datetime(df['S_CON_OUTDATE'], format='%Y%m%d', errors='coerce')
        
        print(f"✅ 获取到 {len(df)} 条中证1000成分股历史变动记录")
        print(f"   涉及股票数: {df['S_INFO_WINDCODE'].nunique()}")
        print(f"   日期范围: {df['S_CON_INDATE'].min().strftime('%Y-%m-%d')} 至 {df['S_CON_OUTDATE'].max().strftime('%Y-%m-%d') if df['S_CON_OUTDATE'].notna().any() else '至今'}")
        return df
    else:
        print("⚠️  未获取到成分股历史数据")
        return pd.DataFrame()

def get_csi1000_constituents_by_date(cursor, date):
    """
    获取指定日期的中证1000成分股列表
    
    Parameters:
    -----------
    cursor : Oracle cursor
    date : str, 日期，格式'YYYYMMDD'
    
    Returns:
    --------
    pd.DataFrame : 指定日期的成分股列表
    """
    sql = f"""
    SELECT DISTINCT 
        S_CON_WINDCODE as S_INFO_WINDCODE,
        S_CON_INDATE,
        S_CON_OUTDATE,
        CUR_SIGN
    FROM FILESYNC.AINDEXMEMBERS
    WHERE S_INFO_WINDCODE = '000852.SH'
      AND S_CON_INDATE <= '{date}'
      AND (S_CON_OUTDATE IS NULL OR S_CON_OUTDATE = '' OR S_CON_OUTDATE >= '{date}')
    """
    
    cursor.execute(sql)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    
    if rows and len(rows) > 0:
        df = pd.DataFrame(rows, columns=columns)
        if 'S_INFO_WINDCODE' in df.columns:
            df['S_INFO_CODE'] = df['S_INFO_WINDCODE'].str.split('.').str[0]
        return df
    else:
        return pd.DataFrame()

def get_csi1000_constituents(cursor, date=None):
    """
    获取中证1000成分股列表（兼容旧接口）
    
    Parameters:
    -----------
    cursor : Oracle cursor
    date : str, 日期，格式'YYYYMMDD'，如果为None则获取最新成分股
    
    Returns:
    --------
    pd.DataFrame : 成分股列表
    """
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    
    df = get_csi1000_constituents_by_date(cursor, date)
    if len(df) > 0:
        print(f"✅ 获取到 {len(df)} 只中证1000成分股（日期: {date}）")
    else:
        print("⚠️  未获取到成分股数据")
    return df

def get_stock_price_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=None):
    """
    获取股票价格数据（OHLC）
    
    根据实际测试结果，使用以下字段：
    - S_DQ_ADJCLOSE: 后复权收盘价（作为CLOSE_ADJ）
    - S_DQ_AVGPRICE: 均价(VWAP)
    - TRADE_DT: VARCHAR2(8)格式，直接字符串比较
    
    Parameters:
    -----------
    cursor : Oracle cursor
    stock_list : list, 股票代码列表（Wind代码格式，如'000001.SZ'）
    start_date : str, 开始日期 'YYYYMMDD'
    end_date : str, 结束日期 'YYYYMMDD'
    batch_size : int, 批次大小（默认100，减小以降低超时风险）
    connection : Oracle connection, 用于重连（可选）
    
    Returns:
    --------
    pd.DataFrame : 价格数据，列包括S_INFO_WINDCODE, TRADE_DT, CLOSE_PRICE, CLOSE_ADJ等
    """
    if len(stock_list) == 0:
        return pd.DataFrame()
    
    import time
    
    # 减小批次大小，避免查询超时
    total_batches = (len(stock_list) + batch_size - 1) // batch_size
    all_rows = []
    columns = None
    
    for batch_idx in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[batch_idx:batch_idx+batch_size]
        batch_codes = "', '".join(batch_stocks)
        current_batch = (batch_idx // batch_size) + 1
        
        # 显示进度
        print(f"  处理批次 {current_batch}/{total_batches} ({len(batch_stocks)} 只股票)...", end=' ', flush=True)
        
        # 根据实际测试结果，使用S_DQ_ADJCLOSE作为复权收盘价（后复权）
        # TRADE_DT是VARCHAR2(8)类型，格式为YYYYMMDD，直接字符串比较
        sql = f"""
        SELECT 
            S_INFO_WINDCODE,
            TRADE_DT,
            S_DQ_CLOSE as CLOSE_PRICE,
            S_DQ_OPEN as OPEN_PRICE,
            S_DQ_HIGH as HIGH_PRICE,
            S_DQ_LOW as LOW_PRICE,
            S_DQ_ADJCLOSE as CLOSE_ADJ,  -- 后复权收盘价（实际可用字段）
            S_DQ_VOLUME as VOLUME,
            S_DQ_AMOUNT as AMOUNT,
            S_DQ_AVGPRICE as VWAP,  -- 均价(VWAP)
            S_DQ_ADJFACTOR as ADJ_FACTOR  -- 复权因子
        FROM FILESYNC.ASHAREEODPRICES
        WHERE S_INFO_WINDCODE IN ('{batch_codes}')
          AND TRADE_DT >= '{start_date}'
          AND TRADE_DT <= '{end_date}'
        ORDER BY S_INFO_WINDCODE, TRADE_DT
        """
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                cursor.execute(sql)
                if columns is None:
                    columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                all_rows.extend(rows)
                print(f"✅ 获取 {len(rows)} 条记录")
                success = True
                
                # 短暂延迟，避免过快请求导致连接问题
                if batch_idx + batch_size < len(stock_list):
                    time.sleep(0.1)
                    
            except oracledb.exceptions.DatabaseError as e:
                retry_count += 1
                error_msg = str(e)
                if "ORA-03113" in error_msg or "connection" in error_msg.lower():
                    print(f"❌ 连接错误（尝试 {retry_count}/{max_retries}）...", end=' ')
                    if connection is not None and retry_count < max_retries:
                        try:
                            # 尝试重新连接
                            time.sleep(2)  # 等待2秒
                            connection.reconnect()
                            cursor = connection.cursor()
                            print("重连成功，重试...", end=' ', flush=True)
                        except Exception as reconnect_err:
                            print(f"重连失败: {reconnect_err}")
                            if retry_count >= max_retries:
                                raise
                    else:
                        raise
                else:
                    raise
    
    if all_rows:
        df = pd.DataFrame(all_rows, columns=columns)
        # TRADE_DT是VARCHAR2(8)格式，转换为datetime
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d', errors='coerce')
        
        # 处理缺失值：CLOSE_ADJ缺失时使用收盘价填充
        if 'CLOSE_ADJ' not in df.columns or df['CLOSE_ADJ'].isna().all():
            df['CLOSE_ADJ'] = df['CLOSE_PRICE']
        else:
            df['CLOSE_ADJ'] = df['CLOSE_ADJ'].fillna(df['CLOSE_PRICE'])
        
        # 处理缺失值：VWAP缺失时使用成交金额/成交量计算，否则用收盘价
        if 'VWAP' in df.columns:
            mask = df['VWAP'].isna()
            if mask.any() and 'AMOUNT' in df.columns and 'VOLUME' in df.columns:
                df.loc[mask, 'VWAP'] = df.loc[mask, 'AMOUNT'] / (df.loc[mask, 'VOLUME'] + 1e-10)
            df['VWAP'] = df['VWAP'].fillna(df['CLOSE_PRICE'])
        elif 'AMOUNT' in df.columns and 'VOLUME' in df.columns:
            df['VWAP'] = df['AMOUNT'] / (df['VOLUME'] + 1e-10)
        else:
            df['VWAP'] = df['CLOSE_PRICE']
        
        print(f"✅ 获取到 {len(df)} 条价格数据，股票数: {df['S_INFO_WINDCODE'].nunique()}")
        return df
    else:
        print("⚠️  未找到价格数据")
        return pd.DataFrame()

def get_market_value_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=None):
    """
    获取市值数据
    
    Parameters:
    -----------
    cursor : Oracle cursor
    stock_list : list, 股票代码列表
    start_date : str, 开始日期 'YYYYMMDD'
    end_date : str, 结束日期 'YYYYMMDD'
    batch_size : int, 批次大小（默认100）
    connection : Oracle connection, 用于重连（可选）
    
    Returns:
    --------
    pd.DataFrame : 市值数据
    """
    if len(stock_list) == 0:
        return pd.DataFrame()
    
    import time
    
    batch_size = 100  # 减小批次大小
    total_batches = (len(stock_list) + batch_size - 1) // batch_size
    all_rows = []
    columns = None
    
    for batch_idx in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[batch_idx:batch_idx+batch_size]
        batch_codes = "', '".join(batch_stocks)
        current_batch = (batch_idx // batch_size) + 1
        
        print(f"  处理批次 {current_batch}/{total_batches} ({len(batch_stocks)} 只股票)...", end=' ', flush=True)
        
        # TRADE_DT字段直接使用字符串比较（已验证可行）
        sql = f"""
        SELECT 
            S_INFO_WINDCODE,
            TRADE_DT,
            S_VAL_MV as TOTAL_MV,  -- 总市值
            S_VAL_MV as FREE_MV    -- 流通市值（如果没有单独字段，用总市值）
        FROM FILESYNC.ASHAREEODDERIVATIVEINDICATOR
        WHERE S_INFO_WINDCODE IN ('{batch_codes}')
          AND TRADE_DT >= '{start_date}'
          AND TRADE_DT <= '{end_date}'
        ORDER BY S_INFO_WINDCODE, TRADE_DT
        """
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                cursor.execute(sql)
                if columns is None:
                    columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                all_rows.extend(rows)
                print(f"✅ 获取 {len(rows)} 条记录")
                success = True
                
                if batch_idx + batch_size < len(stock_list):
                    time.sleep(0.1)
                    
            except oracledb.exceptions.DatabaseError as e:
                retry_count += 1
                error_msg = str(e)
                if "ORA-03113" in error_msg or "connection" in error_msg.lower():
                    print(f"❌ 连接错误（尝试 {retry_count}/{max_retries}）...", end=' ')
                    if connection is not None and retry_count < max_retries:
                        try:
                            time.sleep(2)
                            connection.reconnect()
                            cursor = connection.cursor()
                            print("重连成功，重试...", end=' ', flush=True)
                        except Exception as reconnect_err:
                            print(f"重连失败: {reconnect_err}")
                            if retry_count >= max_retries:
                                raise
                    else:
                        raise
                else:
                    raise
    
    if all_rows:
        df = pd.DataFrame(all_rows, columns=columns)
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d', errors='coerce')
        print(f"✅ 获取到 {len(df)} 条市值数据")
        return df
    else:
        print("⚠️  未找到市值数据")
        return pd.DataFrame()

def get_turnover_rate_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=None):
    """
    获取成交金额数据（用于计算换手率）
    
    根据实际测试，AShareEODPrices表中没有S_DQ_TURNOVER字段。
    换手率需要从成交金额和市值计算：换手率 = 成交金额 / 市值
    
    Parameters:
    -----------
    cursor : Oracle cursor
    stock_list : list, 股票代码列表
    start_date : str, 开始日期 'YYYYMMDD'
    end_date : str, 结束日期 'YYYYMMDD'
    batch_size : int, 批次大小（默认100）
    connection : Oracle connection, 用于重连（可选）
    
    Returns:
    --------
    pd.DataFrame : 成交金额数据，后续结合市值计算换手率
    """
    if len(stock_list) == 0:
        return pd.DataFrame()
    
    import time
    
    batch_size = 100  # 减小批次大小
    total_batches = (len(stock_list) + batch_size - 1) // batch_size
    all_rows = []
    columns = None
    
    for batch_idx in range(0, len(stock_list), batch_size):
        batch_stocks = stock_list[batch_idx:batch_idx+batch_size]
        batch_codes = "', '".join(batch_stocks)
        current_batch = (batch_idx // batch_size) + 1
        
        print(f"  处理批次 {current_batch}/{total_batches} ({len(batch_stocks)} 只股票)...", end=' ', flush=True)
        
        sql = f"""
        SELECT 
            S_INFO_WINDCODE,
            TRADE_DT,
            S_DQ_AMOUNT as AMOUNT
        FROM FILESYNC.ASHAREEODPRICES
        WHERE S_INFO_WINDCODE IN ('{batch_codes}')
          AND TRADE_DT >= '{start_date}'
          AND TRADE_DT <= '{end_date}'
        ORDER BY S_INFO_WINDCODE, TRADE_DT
        """
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                cursor.execute(sql)
                if columns is None:
                    columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                all_rows.extend(rows)
                print(f"✅ 获取 {len(rows)} 条记录")
                success = True
                
                if batch_idx + batch_size < len(stock_list):
                    time.sleep(0.1)
                    
            except oracledb.exceptions.DatabaseError as e:
                retry_count += 1
                error_msg = str(e)
                if "ORA-03113" in error_msg or "connection" in error_msg.lower():
                    print(f"❌ 连接错误（尝试 {retry_count}/{max_retries}）...", end=' ')
                    if connection is not None and retry_count < max_retries:
                        try:
                            time.sleep(2)
                            connection.reconnect()
                            cursor = connection.cursor()
                            print("重连成功，重试...", end=' ', flush=True)
                        except Exception as reconnect_err:
                            print(f"重连失败: {reconnect_err}")
                            if retry_count >= max_retries:
                                raise
                    else:
                        raise
                else:
                    raise
    
    if all_rows:
        df = pd.DataFrame(all_rows, columns=columns)
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d', errors='coerce')
        print(f"✅ 获取到 {len(df)} 条成交金额数据（用于后续计算换手率）")
        return df
    else:
        print("⚠️  未找到成交金额数据")
        return pd.DataFrame()

def get_market_index_data(cursor, index_code='000852.SH', start_date=None, end_date=None):
    """
    获取市场指数数据（用于计算市场收益率）
    
    Parameters:
    -----------
    cursor : Oracle cursor
    index_code : str, 指数代码，默认中证1000
    start_date : str, 开始日期 'YYYYMMDD'
    end_date : str, 结束日期 'YYYYMMDD'
    
    Returns:
    --------
    pd.DataFrame : 指数数据
    """
    # TRADE_DT字段直接使用字符串比较（已验证可行）
    sql = f"""
    SELECT 
        TRADE_DT,
        S_DQ_CLOSE as INDEX_CLOSE
    FROM FILESYNC.AINDEXEODPRICES
    WHERE S_INFO_WINDCODE = '{index_code}'
    """
    
    if start_date:
        sql += f" AND TRADE_DT >= '{start_date}'"
    if end_date:
        sql += f" AND TRADE_DT <= '{end_date}'"
    
    sql += " ORDER BY TRADE_DT"
    
    cursor.execute(sql)
    
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    
    if rows:
        df = pd.DataFrame(rows, columns=columns)
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d', errors='coerce')
        df['MARKET_RETURN'] = df['INDEX_CLOSE'].pct_change()
        print(f"✅ 获取到 {len(df)} 条市场指数数据")
        return df
    else:
        print("⚠️  未找到市场指数数据")
        return pd.DataFrame()

def load_data_from_csv(data_path='./data/'):
    """
    从CSV文件加载数据（如果已经下载过）
    
    Parameters:
    -----------
    data_path : str, 数据文件路径
    
    Returns:
    --------
    dict : 包含所有数据的字典
    """
    import os
    data = {}
    
    try:
        # 加载价格数据
        if os.path.exists(f'{data_path}stock_price_data.csv'):
            data['price_data'] = pd.read_csv(f'{data_path}stock_price_data.csv', encoding='utf-8-sig')
            data['price_data']['TRADE_DT'] = pd.to_datetime(data['price_data']['TRADE_DT'])
            print(f"✅ 加载价格数据: {len(data['price_data'])} 条")
        
        # 加载市值数据
        if os.path.exists(f'{data_path}market_value_data.csv'):
            data['mv_data'] = pd.read_csv(f'{data_path}market_value_data.csv', encoding='utf-8-sig')
            data['mv_data']['TRADE_DT'] = pd.to_datetime(data['mv_data']['TRADE_DT'])
            print(f"✅ 加载市值数据: {len(data['mv_data'])} 条")
        
        # 加载换手率数据
        if os.path.exists(f'{data_path}turnover_rate_data.csv'):
            data['turnover_data'] = pd.read_csv(f'{data_path}turnover_rate_data.csv', encoding='utf-8-sig')
            data['turnover_data']['TRADE_DT'] = pd.to_datetime(data['turnover_data']['TRADE_DT'])
            print(f"✅ 加载换手率数据: {len(data['turnover_data'])} 条")
        
        # 加载市场数据
        if os.path.exists(f'{data_path}market_index_data.csv'):
            data['market_data'] = pd.read_csv(f'{data_path}market_index_data.csv', encoding='utf-8-sig')
            data['market_data']['TRADE_DT'] = pd.to_datetime(data['market_data']['TRADE_DT'])
            print(f"✅ 加载市场数据: {len(data['market_data'])} 条")
        
        # 加载成分股历史变动记录
        if os.path.exists(f'{data_path}csi1000_constituents_history.csv'):
            data['constituents_history'] = pd.read_csv(f'{data_path}csi1000_constituents_history.csv', encoding='utf-8-sig')
            data['constituents_history']['S_CON_INDATE'] = pd.to_datetime(data['constituents_history']['S_CON_INDATE'], errors='coerce')
            data['constituents_history']['S_CON_OUTDATE'] = pd.to_datetime(data['constituents_history']['S_CON_OUTDATE'], errors='coerce')
            print(f"✅ 加载成分股历史变动记录: {len(data['constituents_history'])} 条")
        elif os.path.exists(f'{data_path}csi1000_constituents.csv'):
            # 兼容旧版本（只有单日成分股列表）
            data['constituents'] = pd.read_csv(f'{data_path}csi1000_constituents.csv', encoding='utf-8-sig')
            print(f"✅ 加载成分股列表: {len(data['constituents'])} 只（旧格式，建议重新下载获取历史变动记录）")
        
    except Exception as e:
        print(f"⚠️  加载数据时出错: {e}")
    
    return data

def fetch_all_data(start_date='20220801', end_date=None, save_path='./data/'):
    """
    获取所有需要的数据并保存
    
    Parameters:
    -----------
    start_date : str, 开始日期
    end_date : str, 结束日期，如果为None则使用今天
    save_path : str, 数据保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    
    # 初始化Oracle客户端
    init_oracle_client()
    
    try:
        connection = oracledb.connect(**db_config)
        print("✅ 数据库连接成功")
        cursor = connection.cursor()
        
        # 1. 获取所有历史成分股变动记录（重要：用于构建成分股时间序列）
        print("\n📊 步骤1: 获取中证1000成分股历史变动记录...")
        constituents_history = get_csi1000_constituents_history(cursor)
        if len(constituents_history) == 0:
            print("❌ 无法获取成分股历史数据，程序终止")
            return None
        
        # 保存历史变动记录
        constituents_history.to_csv(f'{save_path}csi1000_constituents_history.csv', index=False, encoding='utf-8-sig')
        
        # 获取所有曾经是成分股的股票列表（用于数据获取）
        # 包括：纳入日期在end_date之前的所有股票
        all_constituent_stocks = constituents_history[
            (constituents_history['S_CON_INDATE'] <= pd.to_datetime(end_date, format='%Y%m%d'))
        ]['S_INFO_WINDCODE'].unique().tolist()
        
        print(f"   共 {len(all_constituent_stocks)} 只股票曾经是成分股（纳入日期 <= {end_date}）")
        
        stock_list = all_constituent_stocks
        
        # 2. 获取价格数据
        print("\n📊 步骤2: 获取股票价格数据...")
        print(f"   共需处理 {len(stock_list)} 只股票，日期范围: {start_date} 至 {end_date}")
        print(f"   将分批处理，每批100只股票（减小批次以降低超时风险）")
        price_data = get_stock_price_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=connection)
        if len(price_data) > 0:
            price_data.to_csv(f'{save_path}stock_price_data.csv', index=False, encoding='utf-8-sig')
            print(f"✅ 价格数据已保存: {len(price_data)} 条记录")
        
        # 3. 获取市值数据
        print("\n📊 步骤3: 获取市值数据...")
        print(f"   将分批处理，每批100只股票")
        mv_data = get_market_value_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=connection)
        if len(mv_data) > 0:
            mv_data.to_csv(f'{save_path}market_value_data.csv', index=False, encoding='utf-8-sig')
            print(f"✅ 市值数据已保存: {len(mv_data)} 条记录")
        
        # 4. 获取换手率数据
        print("\n📊 步骤4: 获取换手率数据...")
        print(f"   将分批处理，每批100只股票")
        turnover_data = get_turnover_rate_data(cursor, stock_list, start_date, end_date, batch_size=100, connection=connection)
        if len(turnover_data) > 0:
            turnover_data.to_csv(f'{save_path}turnover_rate_data.csv', index=False, encoding='utf-8-sig')
            print(f"✅ 换手率数据已保存: {len(turnover_data)} 条记录")
        
        # 5. 获取市场指数数据
        print("\n📊 步骤5: 获取市场指数数据...")
        market_data = get_market_index_data(cursor, index_code='000852.SH', start_date=start_date, end_date=end_date)
        if len(market_data) > 0:
            market_data.to_csv(f'{save_path}market_index_data.csv', index=False, encoding='utf-8-sig')
        
        cursor.close()
        connection.close()
        print("\n✨ 所有数据下载完成！")
        print(f"📁 数据已保存到: {save_path}")
        
        return {
            'constituents_history': constituents_history,  # 历史变动记录
            'price_data': price_data,
            'mv_data': mv_data,
            'turnover_data': turnover_data,
            'market_data': market_data
        }
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # 获取中证1000成分股所有历史数据（2022年8月1日至今）
    data = fetch_all_data(
        start_date='20220801',
        end_date='20251231',
        save_path='d:/programme/vscode_c/courses/Software Enginerring/data/'
    )
