import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime
import warnings
import os
import platform
warnings.filterwarnings('ignore')

# 设置中文字体 - 针对不同操作系统
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        # macOS 常见中文字体
        font_candidates = [
            'Arial Unicode MS',
            'PingFang SC',
            'STHeiti',
            'Heiti TC',
            'Songti SC',
            'Kaiti SC'
        ]
    elif system == 'Windows':
        # Windows 常见中文字体
        font_candidates = [
            'Microsoft YaHei',
            'SimHei',
            'SimSun',
            'KaiTi'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC'
        ]
    
    # 获取所有可用字体
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果没找到，尝试查找包含中文关键词的字体
    if not selected_font:
        for font_name in available_fonts:
            if any(keyword in font_name for keyword in ['Unicode', 'Heiti', 'PingFang', 'Songti', 'YaHei', 'SimHei']):
                selected_font = font_name
                break
    
    # 设置字体
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        print(f"✓ 已设置中文字体: {selected_font}")
        return True
    else:
        print("⚠ 警告: 未找到中文字体，中文可能显示为方框")
        return False

# 执行字体设置
setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 验证字体设置
try:
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.text(0.5, 0.5, '测试', fontsize=12)
    plt.close(fig)
except Exception as e:
    print(f"字体验证失败: {e}")

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# 读取数据
print("=" * 60)
print("数据探索性分析 (EDA)")
print("=" * 60)
print("\n正在读取数据...")
df = pd.read_csv('merged_data.csv')

print(f"\n数据基本信息:")
print(f"  数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  内存使用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 1. 数据概览
print("\n" + "=" * 60)
print("1. 数据概览")
print("=" * 60)
print("\n前5行数据:")
print(df.head())

print("\n数据类型:")
print(df.dtypes.value_counts())

# 2. 缺失值分析
print("\n" + "=" * 60)
print("2. 缺失值分析")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing,
    '缺失百分比': missing_pct
})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
if len(missing_df) > 0:
    print("\n缺失值统计:")
    print(missing_df.head(10))
else:
    print("\n无缺失值")

# 3. 关键变量的描述性统计
print("\n" + "=" * 60)
print("3. 关键变量描述性统计")
print("=" * 60)

# 选择关键数值变量
key_vars = [
    'S_DQ_CLOSE_futures', 'S_DQ_CLOSE_index', 
    'S_DQ_VOLUME_futures', 'S_DQ_AMOUNT_futures',
    '距离到日期的天数', '收益率', '调整后的收益率'
]

print("\n关键变量统计:")
print(df[key_vars].describe())

# 4. 创建可视化图表
print("\n" + "=" * 60)
print("4. 生成可视化图表")
print("=" * 60)

# 创建图表保存目录
import os
output_dir = 'eda_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 转换日期格式用于时间序列分析
df['TRADE_DT_parsed'] = pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')
df['FS_INFO_LTDLDATE_parsed'] = pd.to_datetime(df['FS_INFO_LTDLDATE'].astype(str), format='%Y%m%d')

# 确保使用中文字体 - 强制设置
# 获取实际可用的中文字体
available_fonts = {f.name for f in fm.fontManager.ttflist}
chinese_font = None
for font in ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC', 'Songti SC']:
    if font in available_fonts:
        chinese_font = font
        break

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font] + [f for f in plt.rcParams['font.sans-serif'] if f != chinese_font]
    print(f"✓ 图表将使用中文字体: {chinese_font}")
else:
    print("⚠ 警告: 未找到中文字体")

# 图1: 距离到日期的天数分布
print("\n生成图表 1: 距离到日期的天数分布...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['距离到日期的天数'].hist(bins=50, ax=axes[0], edgecolor='black', alpha=0.7)
axes[0].set_title('距离到日期的天数 - 直方图', fontsize=14, fontweight='bold')
axes[0].set_xlabel('工作日天数', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].axvline(df['距离到日期的天数'].mean(), color='red', linestyle='--', 
                label=f'均值: {df["距离到日期的天数"].mean():.1f}')
axes[0].legend()

df['距离到日期的天数'].plot(kind='box', ax=axes[1], vert=True)
axes[1].set_title('距离到日期的天数 - 箱线图', fontsize=14, fontweight='bold')
axes[1].set_ylabel('工作日天数', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_距离到日期的天数分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 收益率分布
print("生成图表 2: 收益率分布...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['收益率'].dropna().hist(bins=50, ax=axes[0], edgecolor='black', alpha=0.7, color='skyblue')
axes[0].set_title('收益率分布 - 直方图', fontsize=14, fontweight='bold')
axes[0].set_xlabel('收益率', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].axvline(df['收益率'].mean(), color='red', linestyle='--', 
                label=f'均值: {df["收益率"].mean():.4f}')
axes[0].legend()

df['收益率'].dropna().plot(kind='box', ax=axes[1], vert=True)
axes[1].set_title('收益率分布 - 箱线图', fontsize=14, fontweight='bold')
axes[1].set_ylabel('收益率', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_收益率分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3: 调整后的收益率分布
print("生成图表 3: 调整后的收益率分布...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['调整后的收益率'].dropna().hist(bins=50, ax=axes[0], edgecolor='black', alpha=0.7, color='lightgreen')
axes[0].set_title('调整后的收益率分布 - 直方图', fontsize=14, fontweight='bold')
axes[0].set_xlabel('调整后的收益率', fontsize=12)
axes[0].set_ylabel('频数', fontsize=12)
axes[0].axvline(df['调整后的收益率'].mean(), color='red', linestyle='--', 
                label=f'均值: {df["调整后的收益率"].mean():.6f}')
axes[0].legend()

df['调整后的收益率'].dropna().plot(kind='box', ax=axes[1], vert=True)
axes[1].set_title('调整后的收益率分布 - 箱线图', fontsize=14, fontweight='bold')
axes[1].set_ylabel('调整后的收益率', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/3_调整后的收益率分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4: 期货收盘价与指数收盘价的时间序列
print("生成图表 4: 期货与指数收盘价时间序列...")
fig, ax = plt.subplots(figsize=(15, 6))
df_sorted = df.sort_values('TRADE_DT_parsed')
ax.plot(df_sorted['TRADE_DT_parsed'], df_sorted['S_DQ_CLOSE_futures'], 
        label='期货收盘价', alpha=0.7, linewidth=1.5)
ax.plot(df_sorted['TRADE_DT_parsed'], df_sorted['S_DQ_CLOSE_index'], 
        label='指数收盘价', alpha=0.7, linewidth=1.5)
ax.set_title('期货收盘价 vs 指数收盘价 - 时间序列', fontsize=14, fontweight='bold')
ax.set_xlabel('交易日期', fontsize=12)
ax.set_ylabel('收盘价', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/4_期货指数收盘价时间序列.png', dpi=300, bbox_inches='tight')
plt.close()

# 图5: 收益率与距离到日期的天数关系
print("生成图表 5: 收益率与距离到日期的天数关系...")
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['距离到日期的天数'], df['收益率'], 
                     alpha=0.5, s=20, c=df['距离到日期的天数'], cmap='viridis')
ax.set_title('收益率 vs 距离到日期的天数', fontsize=14, fontweight='bold')
ax.set_xlabel('距离到日期的天数（工作日）', fontsize=12)
ax.set_ylabel('收益率', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='距离到日期的天数')
plt.tight_layout()
plt.savefig(f'{output_dir}/5_收益率与距离天数关系.png', dpi=300, bbox_inches='tight')
plt.close()

# 图6: 调整后的收益率与距离到日期的天数关系
print("生成图表 6: 调整后的收益率与距离到日期的天数关系...")
fig, ax = plt.subplots(figsize=(12, 6))
scatter = ax.scatter(df['距离到日期的天数'], df['调整后的收益率'], 
                     alpha=0.5, s=20, c=df['距离到日期的天数'], cmap='plasma')
ax.set_title('调整后的收益率 vs 距离到日期的天数', fontsize=14, fontweight='bold')
ax.set_xlabel('距离到日期的天数（工作日）', fontsize=12)
ax.set_ylabel('调整后的收益率', fontsize=12)
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='距离到日期的天数')
plt.tight_layout()
plt.savefig(f'{output_dir}/6_调整后收益率与距离天数关系.png', dpi=300, bbox_inches='tight')
plt.close()

# 图7: 相关性热力图
print("生成图表 7: 关键变量相关性热力图...")
corr_vars = ['S_DQ_CLOSE_futures', 'S_DQ_CLOSE_index', 'S_DQ_VOLUME_futures',
             '距离到日期的天数', '收益率', '调整后的收益率']
corr_matrix = df[corr_vars].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('关键变量相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/7_相关性热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# 图8: 不同合约的收益率对比
print("生成图表 8: 不同合约的收益率对比...")
# 提取合约代码（去掉.CFE后缀）
df['合约代码'] = df['S_INFO_WINDCODE_futures'].str.replace('.CFE', '')
# 选择数据量较多的合约
contract_counts = df['合约代码'].value_counts()
top_contracts = contract_counts.head(10).index

fig, ax = plt.subplots(figsize=(14, 6))
df_top = df[df['合约代码'].isin(top_contracts)]
sns.boxplot(data=df_top, x='合约代码', y='收益率', ax=ax)
ax.set_title('不同合约的收益率分布对比（Top 10）', fontsize=14, fontweight='bold')
ax.set_xlabel('合约代码', fontsize=12)
ax.set_ylabel('收益率', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/8_不同合约收益率对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 图9: 收益率时间序列
print("生成图表 9: 收益率时间序列...")
fig, ax = plt.subplots(figsize=(15, 6))
df_sorted = df.sort_values('TRADE_DT_parsed')
ax.plot(df_sorted['TRADE_DT_parsed'], df_sorted['收益率'], 
        alpha=0.7, linewidth=1, color='steelblue')
ax.axhline(y=df['收益率'].mean(), color='red', linestyle='--', 
           label=f'均值: {df["收益率"].mean():.4f}')
ax.set_title('收益率时间序列', fontsize=14, fontweight='bold')
ax.set_xlabel('交易日期', fontsize=12)
ax.set_ylabel('收益率', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/9_收益率时间序列.png', dpi=300, bbox_inches='tight')
plt.close()

# 图10: 调整后的收益率时间序列
print("生成图表 10: 调整后的收益率时间序列...")
fig, ax = plt.subplots(figsize=(15, 6))
df_sorted = df.sort_values('TRADE_DT_parsed')
ax.plot(df_sorted['TRADE_DT_parsed'], df_sorted['调整后的收益率'], 
        alpha=0.7, linewidth=1, color='darkgreen')
ax.axhline(y=df['调整后的收益率'].mean(), color='red', linestyle='--', 
           label=f'均值: {df["调整后的收益率"].mean():.6f}')
ax.set_title('调整后的收益率时间序列', fontsize=14, fontweight='bold')
ax.set_xlabel('交易日期', fontsize=12)
ax.set_ylabel('调整后的收益率', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_调整后收益率时间序列.png', dpi=300, bbox_inches='tight')
plt.close()

# 图11: 距离到日期的天数与收益率的联合分布
print("生成图表 11: 距离天数与收益率联合分布...")
fig, ax = plt.subplots(figsize=(12, 8))
hexbin = ax.hexbin(df['距离到日期的天数'], df['收益率'], 
                   gridsize=30, cmap='YlOrRd', mincnt=1)
ax.set_title('距离到日期的天数 vs 收益率 - 联合分布', fontsize=14, fontweight='bold')
ax.set_xlabel('距离到日期的天数（工作日）', fontsize=12)
ax.set_ylabel('收益率', fontsize=12)
plt.colorbar(hexbin, ax=ax, label='数据点数量')
plt.tight_layout()
plt.savefig(f'{output_dir}/11_距离天数与收益率联合分布.png', dpi=300, bbox_inches='tight')
plt.close()

# 图12: 关键变量分布对比
print("生成图表 12: 关键变量分布对比...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 期货收盘价
df['S_DQ_CLOSE_futures'].hist(bins=50, ax=axes[0, 0], edgecolor='black', alpha=0.7)
axes[0, 0].set_title('期货收盘价分布', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('期货收盘价', fontsize=10)
axes[0, 0].set_ylabel('频数', fontsize=10)

# 指数收盘价
df['S_DQ_CLOSE_index'].dropna().hist(bins=50, ax=axes[0, 1], edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('指数收盘价分布', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('指数收盘价', fontsize=10)
axes[0, 1].set_ylabel('频数', fontsize=10)

# 成交量
df['S_DQ_VOLUME_futures'].hist(bins=50, ax=axes[1, 0], edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_title('期货成交量分布', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('成交量', fontsize=10)
axes[1, 0].set_ylabel('频数', fontsize=10)

# 持仓量
df['S_DQ_OI'].hist(bins=50, ax=axes[1, 1], edgecolor='black', alpha=0.7, color='purple')
axes[1, 1].set_title('持仓量分布', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('持仓量', fontsize=10)
axes[1, 1].set_ylabel('频数', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/12_关键变量分布对比.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 生成分析报告
print("\n" + "=" * 60)
print("5. 生成分析报告")
print("=" * 60)

report = f"""
数据探索性分析报告
{'=' * 60}

1. 数据概览
   - 总记录数: {len(df):,}
   - 总变量数: {len(df.columns)}
   - 时间范围: {df['TRADE_DT_parsed'].min().strftime('%Y-%m-%d')} 至 {df['TRADE_DT_parsed'].max().strftime('%Y-%m-%d')}

2. 关键指标统计
   
   距离到日期的天数:
   - 均值: {df['距离到日期的天数'].mean():.2f} 天
   - 中位数: {df['距离到日期的天数'].median():.2f} 天
   - 最小值: {df['距离到日期的天数'].min():.0f} 天
   - 最大值: {df['距离到日期的天数'].max():.0f} 天
   - 标准差: {df['距离到日期的天数'].std():.2f} 天
   
   收益率:
   - 均值: {df['收益率'].mean():.4f} ({df['收益率'].mean()*100:.2f}%)
   - 中位数: {df['收益率'].median():.4f} ({df['收益率'].median()*100:.2f}%)
   - 最小值: {df['收益率'].min():.4f} ({df['收益率'].min()*100:.2f}%)
   - 最大值: {df['收益率'].max():.4f} ({df['收益率'].max()*100:.2f}%)
   - 标准差: {df['收益率'].std():.4f}
   
   调整后的收益率:
   - 均值: {df['调整后的收益率'].mean():.6f}
   - 中位数: {df['调整后的收益率'].median():.6f}
   - 最小值: {df['调整后的收益率'].min():.6f}
   - 最大值: {df['调整后的收益率'].max():.6f}
   - 标准差: {df['调整后的收益率'].std():.6f}

3. 价格统计
   
   期货收盘价:
   - 均值: {df['S_DQ_CLOSE_futures'].mean():.2f}
   - 中位数: {df['S_DQ_CLOSE_futures'].median():.2f}
   - 范围: {df['S_DQ_CLOSE_futures'].min():.2f} - {df['S_DQ_CLOSE_futures'].max():.2f}
   
   指数收盘价:
   - 均值: {df['S_DQ_CLOSE_index'].mean():.2f}
   - 中位数: {df['S_DQ_CLOSE_index'].median():.2f}
   - 范围: {df['S_DQ_CLOSE_index'].min():.2f} - {df['S_DQ_CLOSE_index'].max():.2f}

4. 相关性分析
   
   收益率与距离到日期的天数相关性: {df['收益率'].corr(df['距离到日期的天数']):.4f}
   调整后的收益率与距离到日期的天数相关性: {df['调整后的收益率'].corr(df['距离到日期的天数']):.4f}
   期货收盘价与指数收盘价相关性: {df['S_DQ_CLOSE_futures'].corr(df['S_DQ_CLOSE_index']):.4f}

5. 数据质量
   - 缺失值记录数: {df.isnull().any(axis=1).sum()}
   - 完整记录数: {df.notna().all(axis=1).sum()}
   - 收益率缺失: {df['收益率'].isna().sum()} 条
   - 调整后的收益率缺失: {df['调整后的收益率'].isna().sum()} 条

6. 合约信息
   - 不同合约数量: {df['S_INFO_WINDCODE_futures'].nunique()}
   - 交易最活跃的合约: {df['S_INFO_WINDCODE_futures'].value_counts().index[0]}
   - 该合约交易次数: {df['S_INFO_WINDCODE_futures'].value_counts().iloc[0]}

{'=' * 60}
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# 保存报告
with open(f'{output_dir}/EDA_分析报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n所有图表和分析报告已保存到 '{output_dir}' 目录")
print(f"共生成 {len([f for f in os.listdir(output_dir) if f.endswith('.png')])} 个图表文件")

