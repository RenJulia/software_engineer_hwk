import pandas as pd
import numpy as np
import os

# 读取三张表
print("正在读取数据文件...")
basic_info = pd.read_csv('CSI1000_Basic_Info_Real.csv')
futures_eod = pd.read_csv('CSI1000_Futures_EOD_Real.csv')
index_eod = pd.read_csv('H00852_SH_EOD.csv')

print(f"基本信息表行数: {len(basic_info)}")
print(f"期货交易表行数: {len(futures_eod)}")
print(f"指数表行数: {len(index_eod)}")

# 第一步：合并期货交易表和指数表
# 注意：期货表的S_INFO_WINDCODE是合约代码（如IM2208.CFE），
# 指数表的S_INFO_WINDCODE是指数代码（h00852.SH），它们不同
# 所以按照TRADE_DT合并，为每笔期货交易添加对应日期的指数数据
print("\n第一步：合并期货交易表和指数表（按TRADE_DT）...")

merged_step1 = pd.merge(
    futures_eod,
    index_eod,
    on='TRADE_DT',
    how='left',
    suffixes=('_futures', '_index')
)

print(f"第一步合并后行数: {len(merged_step1)}")

# 第二步：与基本信息表合并
# 用户要求按照FS_INFO_LTDLDATE合并
# 但FS_INFO_LTDLDATE是最后交割日，而merged_step1中有TRADE_DT（交易日期）
# 我们需要通过S_INFO_WINDCODE来匹配合约，然后FS_INFO_LTDLDATE会作为一列保留
print("\n第二步：与基本信息表合并...")

# 先通过S_INFO_WINDCODE匹配合约基本信息
# 注意：merged_step1中期货的windcode列名是S_INFO_WINDCODE_futures
final_merged = pd.merge(
    merged_step1,
    basic_info,
    left_on='S_INFO_WINDCODE_futures',
    right_on='S_INFO_WINDCODE',
    how='left',
    suffixes=('', '_basic')
)

print(f"最终合并后行数: {len(final_merged)}")
print(f"最终合并后列数: {len(final_merged.columns)}")

# 显示列名
print("\n最终表的列名:")
print(final_merged.columns.tolist())

# 保存结果
output_file = 'merged_data.csv'
final_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n合并完成！结果已保存到: {output_file}")

# 显示前几行
print("\n前5行数据预览:")
print(final_merged.head())

# 显示数据统计
print("\n数据统计:")
print(f"包含FS_INFO_LTDLDATE的记录数: {final_merged['FS_INFO_LTDLDATE'].notna().sum()}")
print(f"包含指数数据的记录数: {final_merged['S_INFO_WINDCODE_index'].notna().sum()}")

# 第三步：计算三列新数据
print("\n第三步：计算新列...")

# 1. 计算距离到日期的天数（工作日）
print("计算距离到日期的天数（工作日）...")

# 将日期字符串转换为日期格式
def str_to_date(date_str):
    """将YYYYMMDD格式的字符串转换为日期"""
    if pd.isna(date_str):
        return None
    try:
        return pd.to_datetime(str(int(date_str)), format='%Y%m%d')
    except:
        return None

# 转换日期列
final_merged['TRADE_DT_date'] = final_merged['TRADE_DT'].apply(str_to_date)
final_merged['FS_INFO_LTDLDATE_date'] = final_merged['FS_INFO_LTDLDATE'].apply(str_to_date)

# 计算工作日天数差（使用numpy的busday_count，只计算工作日，排除周末）
# busday_count计算的是[start, end)之间的工作日数，不包括end
# 我们需要包括最后交割日，所以需要+1，或者使用end+1
def calc_business_days(start_date, end_date):
    """计算两个日期之间的工作日天数（包括起始日和结束日）"""
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan
    try:
        # numpy的busday_count计算的是[start, end)之间的工作日数
        # 要包括end，我们需要计算[start, end+1)或者使用busday_count(start, end) + 1
        # 但更准确的是：如果end是工作日，则+1；否则不加
        # 简单方法：计算[start, end+1)的工作日数，然后减去1（因为end+1可能不是工作日）
        # 或者更简单：计算[start, end+1)的工作日数，然后检查end是否是工作日
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 使用numpy的busday_count，计算[start, end+1)的工作日数
        # 这样如果end是工作日，会被包括在内
        days = np.busday_count(start_str, (end_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        return days
    except:
        return np.nan

final_merged['距离到日期的天数'] = final_merged.apply(
    lambda row: calc_business_days(row['TRADE_DT_date'], row['FS_INFO_LTDLDATE_date']),
    axis=1
)

# 2. 计算收益率
print("计算收益率...")
# 收益率 = (指数收盘价 - 期货收盘价) / 期货收盘价
final_merged['收益率'] = (
    (final_merged['S_DQ_CLOSE_index'] - final_merged['S_DQ_CLOSE_futures']) 
    / final_merged['S_DQ_CLOSE_futures']
)

# 3. 计算调整后的收益率
print("计算调整后的收益率...")
# 调整后的收益率 = (收益率 + 1) ^ (1/距离到日期的天数)
# 注意：如果距离到日期的天数为0或负数，需要处理
def calc_adjusted_return(return_rate, days):
    """计算调整后的收益率：(收益率+1)的(1/天数)次方"""
    if pd.isna(return_rate) or pd.isna(days) or days <= 0:
        return np.nan
    try:
        # (收益率 + 1) 的 (1/天数) 次方
        return (return_rate + 1) ** (1.0 / days)-1
    except:
        return np.nan

final_merged['调整后的收益率'] = final_merged.apply(
    lambda row: calc_adjusted_return(row['收益率'], row['距离到日期的天数']),
    axis=1
)

# 删除临时日期列
final_merged = final_merged.drop(columns=['TRADE_DT_date', 'FS_INFO_LTDLDATE_date'])

print(f"\n计算完成！")
print(f"距离到日期的天数 - 非空值: {final_merged['距离到日期的天数'].notna().sum()}")
print(f"收益率 - 非空值: {final_merged['收益率'].notna().sum()}")
print(f"调整后的收益率 - 非空值: {final_merged['调整后的收益率'].notna().sum()}")

# 显示统计信息
print("\n新列统计信息:")
print(f"距离到日期的天数 - 平均值: {final_merged['距离到日期的天数'].mean():.2f}")
print(f"距离到日期的天数 - 最小值: {final_merged['距离到日期的天数'].min():.2f}")
print(f"距离到日期的天数 - 最大值: {final_merged['距离到日期的天数'].max():.2f}")
print(f"收益率 - 平均值: {final_merged['收益率'].mean():.6f}")
print(f"调整后的收益率 - 平均值: {final_merged['调整后的收益率'].mean():.6f}")

# 保存更新后的结果
final_merged.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n更新后的数据已保存到: {output_file}")

# 显示包含新列的前几行
print("\n包含新列的前5行数据预览:")
display_cols = ['TRADE_DT', 'FS_INFO_LTDLDATE', 'S_DQ_CLOSE_futures', 'S_DQ_CLOSE_index', 
                '距离到日期的天数', '收益率', '调整后的收益率']
print(final_merged[display_cols].head())

