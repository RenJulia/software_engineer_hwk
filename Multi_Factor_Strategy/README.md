# 中证1000截面选股多因子量化投资策略

## 📋 项目概述

本项目实现了一个完整的中证1000截面选股的多因子量化投资策略。策略频率为**日频**，使用**日频因子**对**日度收益率**进行预测，股票池为**中证1000成分股**。

### 核心特点

- ✅ **30个因子**：20个传统因子 + 10个机器学习因子
- ✅ **多模型集成**：因子筛选等权合成、MLP深度神经网络、XGBoost
- ✅ **严格回测**：成分股过滤、防止未来信息泄露、袋外回测
- ✅ **完整评估**：IC分析、分层回测、相关性分析、性能指标

---

## 📁 项目结构

```
.
├── README.md                          # 项目说明文档（本文件）
├── Factor.ipynb                       # Jupyter Notebook主程序（推荐使用）
├── main_factor_analysis.py            # Python主程序（一键运行）
│
├── data_collection.py                 # 数据获取模块
├── factor_calculation.py              # 因子计算模块（20个传统因子）
├── ml_factor_generation_improved.py   # 机器学习因子生成模块（10个ML因子）
├── ml_utils_improved.py               # ML工具函数模块
├── factor_evaluation.py                # 因子评估模块（IC、分层回测、相关性分析）
├── factor_combination.py              # 因子合成模块（多模型集成）
├── strategy_backtest.py              # 策略回测模块（多头选股回测）
├── constituent_manager.py            # 成分股管理器（处理成分股变动）
│
├── data/                              # 数据目录
│   ├── csi1000_constituents_history.csv
│   ├── stock_price_data.csv
│   ├── market_value_data.csv
│   ├── turnover_rate_data.csv
│   └── market_index_data.csv
│
├── factors/                           # 因子数据目录（30个因子CSV文件）
│
├── signal/                            # 合成信号目录
│   ├── model1_factor_selection_prediction.csv
│   ├── model2_mlp_prediction.csv
│   ├── model3_xgboost_prediction.csv
│   └── final_signal.csv
│
├── results/                           # 分析结果目录
│   ├── IC统计结果
│   ├── 分层回测结果
│   ├── 相关性分析结果
│   └── 策略回测结果
│
├── logs/                              # 项目文档目录
│   ├── PROJECT_SUMMARY.md            # 项目完成总结
│   ├── FACTOR_DATA_REQUIREMENTS.md   # 详细数据需求文档
│   ├── ML_FACTOR_README.md           # 机器学习因子说明
│   ├── FACTOR_COMBINATION_README.md  # 因子合成说明
│   ├── OVERFITTING_PREVENTION_GUIDE.md  # 防过拟合指南
│   └── CODE_REVIEW_REPORT.md         # 代码审查报告
│
└── resouces/                          # 参考资料
    ├── Factors.docx                  # 因子定义文档
    └── *.pdf                          # 相关研报
```

---

## 🎯 策略流程

### 1. 数据获取 (`data_collection.py`)

从Oracle数据库获取以下数据：

- **成分股数据**：中证1000成分股历史变动记录
- **价格数据**：OHLC、复权收盘价、成交量、成交额
- **市值数据**：总市值、流通市值
- **换手率数据**：日度换手率
- **市场数据**：中证1000指数数据

**关键特性**：
- ✅ 支持分批处理（避免SQL超时）
- ✅ 支持数据保存和加载（CSV格式）
- ✅ 自动处理成分股历史变动

### 2. 因子计算

#### 2.1 传统因子 (`factor_calculation.py`) - 20个因子

| 编号 | 因子名称 | 类型 | 说明 |
|------|---------|------|------|
| 1 | SCC | 空间中心性 | Spatial Centrality Centrality |
| 2 | TCC | 时间中心性 | Temporal Centrality Centrality |
| 3 | APB | 平均价格偏差 | Average Price Bias |
| 4-7 | ARC/VRC/SRC/KRC | 相对成本各阶矩 | Average/Variance/Skewness/Kurtosis of Relative Cost |
| 8 | BIAS | 价格偏差 | 20日价格偏差 |
| 9 | TURNOVER_BIAS | 换手率偏差 | 20日换手率偏差 |
| 10 | NEW_HIGH_RATIO | 新高日比例 | 20日内新高日比例 |
| 11-12 | ID_VOL / ID_VOL_DECORR | 特质波动率 | 特质波动率（去相关版） |
| 13 | TURN20 | 换手率因子 | 20日平均换手率 |
| 14-15 | CGO / RCGO | 资本收益悬置 | Capital Gain Overhang（残差版） |
| 16 | SUE | 标准化意外收益 | Standardized Unexpected Earnings（需EPS数据） |
| 17-20 | CANDLE_ABOVE/BELOW | K线上下影线 | K线上/下影线（均值/标准差） |
| 21-24 | WILLIAMS_ABOVE/BELOW | Williams上下影线 | Williams上/下影线（均值/标准差） |
| 25 | UBL | 综合上下影线 | Up & Bottom Line |

#### 2.2 机器学习因子 (`ml_factor_generation_improved.py`) - 10个因子

使用5种机器学习模型对日度和月度收益率进行预测：

| 模型 | 日度因子 | 月度因子 |
|------|---------|---------|
| GRU | GRU_DAILY | GRU_MONTHLY |
| Transformer | TRANSFORMER_DAILY | TRANSFORMER_MONTHLY |
| LightGBM | LIGHTGBM_DAILY | LIGHTGBM_MONTHLY |
| SVM | SVM_DAILY | SVM_MONTHLY |
| 随机森林 | RF_DAILY | RF_MONTHLY |

**模型特点**：
- ✅ 使用IC损失函数（神经网络）
- ✅ 防止未来信息泄露（严格时间对齐）
- ✅ 防过拟合机制（早停、Dropout、L2正则化）
- ✅ 序列长度：40天

### 3. 因子评估 (`factor_evaluation.py`)

#### 3.1 单因子分析

- **IC分析**：
  - IC（Information Coefficient）：皮尔逊相关系数
  - RankIC：斯皮尔曼秩相关系数
  - IR（Information Ratio）：IC均值 / IC标准差
  - IC胜率：IC>0的比例

- **分层回测**：
  - 5层分层回测
  - 可调调仓频率（默认5日）
  - 考虑交易成本（默认0.2%）
  - 单调性检测
  - 多空收益计算

#### 3.2 多因子分析

- **相关性分析**：
  - 计算因子间相关性矩阵
  - 绘制相关性热力图
  - 高相关性因子筛选（默认阈值0.7）

- **因子排名**：
  - 按IC均值排名
  - 按RankIC均值排名
  - 按IR排名
  - 按多空收益排名
  - 综合排名（IR+多空收益加权）

### 4. 因子合成 (`factor_combination.py`)

使用三个模型进行因子合成：

#### 模型1：因子筛选等权合成法

- 计算所有因子的IC
- 剔除IC绝对值低于阈值的因子
- 剔除高相关性因子对
- 对筛选后的因子进行截面标准化
- 等权合成得到预测信号

#### 模型2：MLP深度神经网络

- 输入：所有因子（截面标准化）
- 网络结构：128 → 64 → 32 → 1
- 损失函数：CCC（一致性相关系数）
- 数据划分：3:1:1（训练:验证:测试）
- 早停机制：验证集CCC不再提升

#### 模型3：XGBoost

- 输入：所有因子（截面标准化）
- 损失函数：CCC
- 数据划分：4:1（训练:测试）
- 参数：n_estimators=200, max_depth=6, learning_rate=0.1

#### 最终信号

三个模型的预测值等权合成，得到最终交易信号。

**关键特性**：
- ✅ 防止未来信息泄露（t日因子预测t+1日收益）
- ✅ 严格时间序列划分
- ✅ 成分股过滤（使用ConstituentManager）

### 5. 策略回测 (`strategy_backtest.py`)

#### 5.1 单信号分析

对每个信号（模型1、模型2、模型3、最终信号）进行：
- IC/IR分析
- 分层回测
- 单调性检测
- 多空收益计算
- 相关性分析

#### 5.2 多头选股回测

- **选股方式**：根据信号值选择头部股票
- **头部比例**：5%、10%、20%（可配置）
- **调仓频率**：日度调仓（可配置）
- **交易成本**：费率0.2%，滑点0.1%（可配置）
- **成分股过滤**：确保只使用当日成分股

#### 5.3 袋外回测

- **数据划分**：前80%训练，后20%测试
- **最优比例选择**：在训练集上选择最优头部比例（基于夏普比率）
- **测试集回测**：在测试集上使用最优比例进行回测

#### 5.4 性能指标

- 总收益率
- 年化收益率
- 夏普比率
- 最大回撤
- 胜率
- 月度收益统计
- 与基准对比（中证1000指数）

---

## 🚀 快速开始

### 环境要求

- Python 3.7+
- Oracle Instant Client（用于数据库连接）

### 安装依赖

```bash
pip install -r logs/requirements.txt
```

**主要依赖包**：
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- oracledb >= 1.0.0
- torch (PyTorch)
- xgboost
- scikit-learn
- jupyter

### 配置数据库连接

在 `data_collection.py` 中修改数据库配置：

```python
lib_dir = os.path.expanduser("D:\\Software\\Oracle\\instantclient_23_0")  # Oracle客户端路径
db_config = {
    "user": "your_username",
    "password": "your_password",
    "dsn": "host:port/service_name"
}
```

### 运行方式

#### 方式1：使用Jupyter Notebook（推荐）

```bash
jupyter notebook Factor.ipynb
```

按照notebook中的单元格顺序逐步执行：
1. 数据获取
2. 成分股管理器初始化
3. 因子计算（传统因子 + ML因子）
4. 因子评估（单因子 + 多因子分析）
5. 因子合成（三个模型）
6. 策略回测（单信号分析 + 多头选股回测 + 袋外回测）

#### 方式2：使用Python主程序

```bash
python main_factor_analysis.py
```

#### 方式3：分模块使用

```python
# 1. 数据获取
from data_collection import fetch_all_data, load_data_from_csv

data = load_data_from_csv(data_path='./data/')
# 或从数据库获取
# data = fetch_all_data(start_date='20220801', end_date='20251231', save_path='./data/')

# 2. 初始化成分股管理器
from constituent_manager import ConstituentManager
constituent_manager = ConstituentManager(data['constituents_history'])

# 3. 计算传统因子
from factor_calculation import FactorCalculator
calculator = FactorCalculator(
    price_data=data['price_data'],
    mv_data=data.get('mv_data'),
    turnover_data=data.get('turnover_data'),
    market_data=data.get('market_data')
)
all_factors = calculator.calculate_all_factors()

# 4. 生成机器学习因子（如果尚未生成）
from ml_factor_generation_improved import generate_ml_factors
ml_factors = generate_ml_factors(
    data_path='./data/',
    factors_path='./factors/',
    sequence_length=40,
    train_test_split=0.8
)
all_factors.update(ml_factors)

# 5. 因子评估
from factor_evaluation import FactorEvaluator
price_wide = calculator.pivot_to_wide_format(
    calculator.price_data[['S_INFO_WINDCODE', 'TRADE_DT', 'RETURN']].dropna(),
    'RETURN'
)
evaluator = FactorEvaluator(
    factor_data=all_factors,
    return_data=price_wide,
    price_data=data['price_data'],
    constituent_manager=constituent_manager
)
results = evaluator.evaluate_all_factors(
    forward_period=1,
    layers=5,
    freq=5,
    correlation_threshold=0.7,
    save_dir='./results/'
)

# 6. 因子合成
from factor_combination import FactorCombiner
combiner = FactorCombiner(
    factor_data=all_factors,
    return_data=price_wide,
    constituent_manager=constituent_manager
)
model1_pred = combiner.model1_factor_selection_equal_weight(
    min_factors=5,
    max_factors=10,
    ic_threshold=0.02,
    correlation_threshold=0.7
)
model2_pred = combiner.model2_mlp(
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    hidden_dims=[128, 64, 32],
    dropout=0.3,
    epochs=100,
    patience=10
)
model3_pred = combiner.model3_xgboost(
    train_ratio=0.8,
    test_ratio=0.2,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1
)
final_signal = combiner.combine_models()
combiner.save_predictions(save_dir='./signal/')

# 7. 策略回测
from strategy_backtest import StrategyBacktester, load_signals_from_directory
signal_data = load_signals_from_directory('./signal/')
backtester = StrategyBacktester(
    signal_data=signal_data,
    return_data=price_wide,
    benchmark_data=data.get('market_data'),
    price_data=data['price_data'],
    constituent_manager=constituent_manager
)
backtest_results = backtester.backtest_all_signals(
    top_pcts=[0.05, 0.1, 0.2],
    fee=0.002,
    slippage=0.001,
    rebalance_freq=1,
    save_dir='./results/'
)
oos_results = backtester.out_of_sample_backtest(
    all_results=backtest_results,
    train_ratio=0.8,
    fee=0.002,
    slippage=0.001,
    rebalance_freq=1,
    save_dir='./results/'
)
```

---

## 📊 输出结果

### 数据文件 (`./data/`)

- `csi1000_constituents_history.csv` - 成分股历史变动记录
- `stock_price_data.csv` - 价格数据
- `market_value_data.csv` - 市值数据
- `turnover_rate_data.csv` - 换手率数据
- `market_index_data.csv` - 市场指数数据

### 因子文件 (`./factors/`)

30个因子CSV文件，格式：股票×日期

### 信号文件 (`./signal/`)

- `model1_factor_selection_prediction.csv` - 模型1预测
- `model2_mlp_prediction.csv` - 模型2预测
- `model3_xgboost_prediction.csv` - 模型3预测
- `final_signal.csv` - 最终合成信号

### 分析结果 (`./results/`)

#### 单因子分析结果

- `{因子名}_IC_trend.png` - IC趋势图
- `{因子名}_layer_returns.png` - 分层收益曲线
- `{因子名}_IC_stats.csv` - IC统计结果
- `{因子名}_layer_nav.csv` - 分层回测净值

#### 多因子分析结果

- `all_factors_IC_summary.csv` - 所有因子IC统计汇总
- `factor_correlation_matrix.csv` - 因子相关性矩阵
- `factor_correlation_heatmap.png` - 相关性热力图
- `factors_to_remove.csv` - 建议剔除的高相关性因子
- `top5_by_IC_Mean.csv` - 按IC均值排名前5
- `top5_by_RankIC_Mean.csv` - 按RankIC均值排名前5
- `top5_by_IR.csv` - 按IR排名前5
- `top5_by_Long_Short_Return.csv` - 按多空收益排名前5

#### 策略回测结果

- `backtest_summary.csv` - 回测汇总表
- `oos_backtest_summary.csv` - 袋外回测汇总表
- `{信号名}_{比例}_nav.csv` - 各策略净值序列
- `{信号名}_{比例}_oos_nav.csv` - 袋外回测净值序列
- `nav_curves_comparison.png` - 净值曲线对比图
- `oos_nav_curves.png` - 袋外回测净值曲线图
- `{信号名}_{比例}_monthly_returns.png` - 月度收益统计图
- `metrics_comparison_*.png` - 性能指标对比图

---

## 🔧 关键参数配置

### 数据获取参数

```python
start_date = '20220801'  # 开始日期（YYYYMMDD格式）
end_date = '20251231'    # 结束日期
batch_size = 100         # 分批处理大小
```

### 因子计算参数

```python
# 传统因子：使用默认参数
# ML因子：
sequence_length = 40     # 序列长度（天）
train_test_split = 0.8   # 训练集比例
```

### 因子评估参数

```python
forward_period = 1       # 前瞻期（日）
layers = 5              # 分层数
freq = 5                # 调仓频率（交易日）
correlation_threshold = 0.7  # 相关性阈值
```

### 因子合成参数

```python
# 模型1：
min_factors = 5         # 最少保留因子数
max_factors = 10        # 最多保留因子数
ic_threshold = 0.02     # IC阈值（绝对值）

# 模型2（MLP）：
hidden_dims = [128, 64, 32]  # 隐藏层维度
dropout = 0.3          # Dropout比例
epochs = 100           # 训练轮数
patience = 10          # 早停耐心值

# 模型3（XGBoost）：
n_estimators = 200     # 树的数量
max_depth = 6          # 树的最大深度
learning_rate = 0.1    # 学习率
```

### 策略回测参数

```python
top_pcts = [0.05, 0.1, 0.2]  # 头部比例（5%, 10%, 20%）
fee = 0.002                  # 交易费率（0.2%）
slippage = 0.001             # 滑点（0.1%）
rebalance_freq = 1           # 调仓频率（日度）
train_ratio = 0.8            # 训练集比例（袋外回测）
```

---

## ⚠️ 重要注意事项

### 1. 数据完整性

- **必需数据**：复权收盘价、成交量、换手率、OHLC价格、市值
- **重要数据**：市场收益率、成分股列表
- **可选数据**：EPS（仅SUE因子需要）
- **数据周期**：建议至少3-5年历史数据

### 2. 防止未来信息泄露

- ✅ 使用t日因子值预测t+1日收益率
- ✅ 严格时间序列划分（训练/验证/测试）
- ✅ 因子值保存在t日，用于预测t+1日收益

### 3. 成分股过滤

- ✅ 使用`ConstituentManager`确保每个日期只使用当日成分股
- ✅ 回测时自动过滤非成分股
- ✅ 因子计算和评估都考虑成分股变动

### 4. 计算性能

- **因子计算**：可能需要较长时间（取决于数据量和因子复杂度）
- **ML因子训练**：可能需要数小时（取决于硬件和数据量）
- **内存使用**：处理全市场数据时注意内存占用

### 5. 防过拟合机制

- ✅ 早停机制（验证IC不再提升）
- ✅ Dropout正则化（0.3）
- ✅ L2正则化（weight_decay=1e-4）
- ✅ 学习率衰减
- ✅ 梯度裁剪

---

## 📚 详细文档

项目文档位于 `logs/` 目录下：

- `PROJECT_SUMMARY.md` - 项目完成总结
- `FACTOR_DATA_REQUIREMENTS.md` - 详细数据需求文档
- `ML_FACTOR_README.md` - 机器学习因子说明
- `FACTOR_COMBINATION_README.md` - 因子合成说明
- `OVERFITTING_PREVENTION_GUIDE.md` - 防过拟合指南
- `CODE_REVIEW_REPORT.md` - 代码审查报告
- `CONSTITUENT_FIX_SUMMARY.md` - 成分股修复说明

---

## 🐛 常见问题

### Q1: 如何修改调仓频率？

A: 在因子评估和策略回测时修改 `freq` 参数：
```python
# 因子评估
evaluator.layer_backtest_single_factor(factor_name, freq=10)  # 10日调仓

# 策略回测
backtester.backtest_all_signals(rebalance_freq=10)  # 10日调仓
```

### Q2: 如何只评估特定因子？

A: 在评估前筛选因子字典：
```python
selected_factors = {k: v for k, v in all_factors.items() if k in ['BIAS', 'TURN20']}
evaluator = FactorEvaluator(factor_data=selected_factors, ...)
```

### Q3: 因子计算失败怎么办？

A: 检查数据完整性，某些因子需要特定数据字段。查看错误信息，根据提示补充缺失数据。

### Q4: ML因子训练时间过长？

A: 
- 减小批次大小（batch_size）
- 减小序列长度（sequence_length）
- 使用GPU加速（如果可用）
- 减少训练轮数（epochs）

### Q5: 内存不足？

A:
- 分批处理数据
- 减小序列长度
- 使用数据采样
- 关闭不必要的中间结果保存

---

## 📈 策略性能

策略回测结果保存在 `./results/` 目录下，包括：

- **回测汇总表**：各信号在不同头部比例下的性能指标
- **袋外回测结果**：在测试集上的表现
- **净值曲线**：策略净值随时间的变化
- **月度收益统计**：各月的收益分布
- **性能指标对比**：与基准的对比

---

## 🔄 更新日志

### v1.0 (2024)

- ✅ 完成数据获取模块
- ✅ 实现20个传统因子
- ✅ 实现10个机器学习因子
- ✅ 完成因子评估模块（IC分析、分层回测）
- ✅ 完成因子合成模块（三模型集成）
- ✅ 完成策略回测模块（多头选股、袋外回测）
- ✅ 实现成分股管理器
- ✅ 完善文档和注释

---

## 📄 许可证

本项目仅供学习和研究使用。

---

## 👥 贡献者

量化投资研究团队

---

## 📮 联系方式

如有问题或建议，请联系项目维护者。

---

**最后更新**: 2024年12月
