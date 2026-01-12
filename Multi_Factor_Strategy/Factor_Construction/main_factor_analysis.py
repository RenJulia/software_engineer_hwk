# -*- coding: utf-8 -*-
"""
中证1000多因子量化策略 - 主程序
整合数据获取、因子计算、因子评估三大模块
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入自定义模块
from data_collection import fetch_all_data, load_data_from_csv
from factor_calculation import FactorCalculator
from factor_evaluation import FactorEvaluator
from constituent_manager import ConstituentManager

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
            # 兼容旧版本
            data['constituents'] = pd.read_csv(f'{data_path}csi1000_constituents.csv', encoding='utf-8-sig')
            print(f"⚠️  加载旧格式成分股列表: {len(data['constituents'])} 只（建议重新下载获取历史变动记录）")
        
    except Exception as e:
        print(f"⚠️  加载数据时出错: {e}")
    
    return data


def load_factors_from_local(factors_path='./factors/'):
    """
    从本地CSV文件加载因子数据
    
    Parameters:
    -----------
    factors_path : str, 因子文件路径
    
    Returns:
    --------
    dict : 包含所有因子数据的字典
    """
    all_factors = {}
    
    if not os.path.exists(factors_path):
        print(f"❌ 因子目录不存在: {factors_path}")
        return all_factors
    
    print(f"📂 从本地加载因子数据: {factors_path}")
    
    # 预期的因子名称列表
    expected_factors = [
        'SCC', 'TCC', 'APB', 'ARC', 'VRC', 'SRC', 'KRC',
        'BIAS', 'TURNOVER_BIAS', 'NEW_HIGH_RATIO',
        'ID_VOL', 'ID_VOL_DECORR', 'TURN20',
        'CGO', 'RCGO', 'SUE',
        'CANDLE_ABOVE_MEAN', 'CANDLE_ABOVE_STD',
        'CANDLE_BELOW_MEAN', 'CANDLE_BELOW_STD',
        'WILLIAMS_ABOVE_MEAN', 'WILLIAMS_ABOVE_STD',
        'WILLIAMS_BELOW_MEAN', 'WILLIAMS_BELOW_STD',
        'UBL'
    ]
    
    loaded_count = 0
    failed_count = 0
    
    # 遍历因子目录，加载所有CSV文件（排除factor_summary.csv）
    for filename in os.listdir(factors_path):
        if filename.endswith('.csv') and filename != 'factor_summary.csv':
            factor_name = filename[:-4]  # 去掉.csv后缀
            
            try:
                factor_file = os.path.join(factors_path, filename)
                factor_df = pd.read_csv(factor_file, index_col=0, encoding='utf-8-sig')
                
                # 将列名转换为日期格式（如果可能）
                try:
                    factor_df.columns = pd.to_datetime(factor_df.columns)
                except:
                    pass  # 如果转换失败，保持原样
                
                all_factors[factor_name] = factor_df
                loaded_count += 1
                print(f"   ✅ {factor_name}: {factor_df.shape}")
                
            except Exception as e:
                print(f"   ❌ {factor_name}: 加载失败 - {e}")
                failed_count += 1
    
    print(f"\n✅ 因子加载完成: {loaded_count} 个成功，{failed_count} 个失败")
    
    # 检查是否加载了预期的因子
    missing_factors = set(expected_factors) - set(all_factors.keys())
    if missing_factors:
        print(f"⚠️  以下预期因子未找到: {sorted(missing_factors)}")
    
    return all_factors

def main():
    """主函数"""
    print("="*80)
    print("中证1000多因子量化策略 - 完整分析流程")
    print("="*80 + "\n")
    
    # ==================== 第一步：数据获取 ====================
    print("【第一步】数据获取")
    print("-"*80)
    
    data_path = 'd:/programme/vscode_c/courses/Software Enginerring/data/'
    os.makedirs(data_path, exist_ok=True)
    
    # 检查是否已有数据
    if os.path.exists(f'{data_path}stock_price_data.csv'):
        print("检测到已有数据文件，直接加载...")
        data = load_data_from_csv(data_path)
    else:
        print("未检测到数据文件，开始从数据库下载...")
        # 从数据库获取数据
        # 注意：这里需要根据实际情况调整日期范围
        data = fetch_all_data(
            start_date='20200101',  # 开始日期
            end_date='20241231',    # 结束日期（可根据需要调整）
            save_path=data_path
        )
        
        if data is None:
            print("❌ 数据获取失败，程序终止")
            return
    
    # 检查必要数据是否存在
    if 'price_data' not in data or data['price_data'].empty:
        print("❌ 价格数据缺失，程序终止")
        return
    
    # ==================== 第二步：初始化成分股管理器 ====================
    print("\n【第二步】初始化成分股管理器")
    print("-"*80)
    
    constituent_manager = None
    if 'constituents_history' in data and not data['constituents_history'].empty:
        constituent_manager = ConstituentManager(data['constituents_history'])
        print("✅ 成分股管理器初始化成功，将用于回测时的成分股过滤")
    else:
        print("⚠️  未找到成分股历史变动记录，回测将使用所有股票（可能不准确）")
        print("   建议重新运行数据获取，确保包含constituents_history数据")
    
    # ==================== 第三步：加载因子数据 ====================
    print("\n【第三步】加载因子数据")
    print("-"*80)
    
    factors_path = 'd:/programme/vscode_c/courses/Software Enginerring/factors/'
    
    # 从本地加载已计算好的因子
    all_factors = load_factors_from_local(factors_path)
    
    if not all_factors:
        print("❌ 未找到因子数据，程序终止")
        print(f"   请先运行 factor_calculation.py 计算因子，或检查因子路径: {factors_path}")
        return
    
    print(f"✅ 成功加载 {len(all_factors)} 个因子")
    
    # ==================== 第四步：准备收益率数据 ====================
    print("\n【第四步】准备收益率数据")
    print("-"*80)
    
    # 从价格数据计算收益率（用于因子评估）
    # 需要先计算收益率
    price_data = data['price_data'].copy()
    price_data = price_data.sort_values(['S_INFO_WINDCODE', 'TRADE_DT'])
    
    # 确保CLOSE_ADJ存在
    if 'CLOSE_ADJ' not in price_data.columns:
        if 'CLOSE_PRICE' in price_data.columns:
            price_data['CLOSE_ADJ'] = price_data['CLOSE_PRICE']
        else:
            print("❌ 价格数据中缺少CLOSE_PRICE或CLOSE_ADJ字段")
            return
    
    # 计算收益率
    price_data['RETURN'] = price_data.groupby('S_INFO_WINDCODE')['CLOSE_ADJ'].pct_change()
    
    # 转换为宽格式
    price_wide = price_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RETURN']].dropna().pivot(
        index='S_INFO_WINDCODE',
        columns='TRADE_DT',
        values='RETURN'
    )
    
    # 将列名转换为日期格式（如果可能）
    try:
        price_wide.columns = pd.to_datetime(price_wide.columns)
    except:
        pass
    
    print(f"✅ 收益率数据形状: {price_wide.shape}")
    
    # ==================== 第五步：因子评估 ====================
    print("\n【第五步】因子评估")
    print("-"*80)
    
    # 初始化因子评估器（传入成分股管理器）
    evaluator = FactorEvaluator(
        factor_data=all_factors,
        return_data=price_wide,
        price_data=data['price_data'],
        constituent_manager=constituent_manager  # 传入成分股管理器
    )
    
    # 评估所有因子
    results = evaluator.evaluate_all_factors(
        forward_period=1,              # 前瞻期：1日
        layers=5,                      # 分层数：5层
        freq=5,                        # 调仓频率：5日
        correlation_threshold=0.7,     # 相关性阈值：0.7
        save_dir='./results/'
    )
    
    # ==================== 总结 ====================
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    
    print("\n生成的文件:")
    print(f"  数据文件: {data_path}")
    print(f"  因子文件: {factors_path}")
    print(f"  结果文件: ./results/")
    
    print("\n主要结果:")
    if results.get('ic_summary') is not None:
        ic_summary = results['ic_summary']
        print(f"  ✅ 共评估 {len(ic_summary)} 个因子")
        print(f"  ✅ IR最高的5个因子:")
        top5 = ic_summary.head(5)
        for idx, (factor_name, row) in enumerate(top5.iterrows(), 1):
            print(f"     {idx}. {factor_name}: IR={row['IR']:.4f}, IC均值={row['IC_Mean']:.4f}")
    
    if results.get('multi_factor_result') and results['multi_factor_result'].get('factors_to_remove'):
        factors_to_remove = results['multi_factor_result']['factors_to_remove']
        print(f"  ✅ 建议剔除 {len(factors_to_remove)} 个高相关性因子")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
