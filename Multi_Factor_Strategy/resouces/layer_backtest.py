# -*- coding: utf-8 -*-
"""
【华泰金工】日频多因子系统

组合回测

"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import datetime


#%% 1.读取数据
alpha_path = './'
alpha = loadmat(f'{alpha_path}alpha.mat')
alpha_daily = loadmat(f'{alpha_path}alpha_daily.mat')
alpha_daily_1 = loadmat(f'{alpha_path}alpha_daily_1.mat')
alpha_daily_3 = loadmat(f'{alpha_path}alpha_daily_3.mat')
# 股票
all_stock_list = alpha['basicinfo']['stock_number_wind'][0][0]
all_stock_list = [code[0][0] for code in all_stock_list]
# 交易日
daily_trading_dates = [datetime.date(1998, 1, 5) + datetime.timedelta(days=int(N - 729760)) for N in alpha_daily['dailyinfo']['dates'][0][0][0]]
daily_trading_dates = [datetime.datetime.strftime(N, '%Y-%m-%d') for N in daily_trading_dates]
# 各类基础数据
data = {}
data['close_adj'] = pd.DataFrame(alpha_daily['dailyinfo']['close_adj'][0][0], index=all_stock_list, columns=daily_trading_dates)
data['amt'] = pd.DataFrame(alpha_daily_1['dailyinfo_1']['amt'][0][0], index=all_stock_list, columns=daily_trading_dates)
data['vwap_adj'] = pd.DataFrame(alpha_daily_1['dailyinfo_1']['vwap'][0][0] * alpha_daily['dailyinfo']['close_adj'][0][0] / alpha_daily['dailyinfo']['close'][0][0], index=all_stock_list, columns=daily_trading_dates)
data['delist_date'] = alpha['basicinfo']['delist_date'][0][0][:, 0]
data['delist_date'] = [datetime.datetime.strftime((datetime.date(1998, 1, 5) + datetime.timedelta(days=int(N - 729760))), '%Y-%m-%d') if not np.isnan(N) else np.nan for N in data['delist_date']]
data['delist_date'] = pd.DataFrame(data['delist_date'], index=all_stock_list)
data['maxupordown'] = pd.DataFrame(alpha_daily_3['dailyinfo_3']['maxupordown'][0][0], index=all_stock_list, columns=daily_trading_dates)


#%% 2.因子分层
def gen_layers(factor, layers=5, freq=5):
    weight = [pd.DataFrame(index=factor.index) for j in range(layers)]
    for i in range(1, factor.shape[1], freq):
        last_date = factor.columns[i - 1]
        date = factor.columns[i]
        factor_date = factor[last_date]
        for j in range(layers):
            threshold1, threshold2 = np.nanquantile(factor_date, j / layers), np.nanquantile(factor_date, (j + 1) / layers)
            weight_date = pd.Series(0.0, index=factor.index)
            weight_date[(factor_date >= threshold1) & (factor_date <= threshold2)] = 1.0
            weight_date = weight_date / weight_date.sum()
            weight[j][date] = weight_date
    return weight

def gen_base(factor, freq=5):
    weight = pd.DataFrame(index=factor.index)
    for i in range(1, factor.shape[1], freq):
        last_date = factor.columns[i - 1]
        date = factor.columns[i]
        factor_date = factor[last_date]
        weight_date = pd.Series(0.0, index=factor.index)
        weight_date[~factor_date.isna()] = 1.0
        weight_date = weight_date / weight_date.sum()
        weight[date] = weight_date
    return weight

# 读取因子数据
factor = pd.read_csv('factor.csv', index_col=0)
factor[factor == 0] = np.nan
# 分层
weight = gen_layers(factor, layers=5)
weight_base = gen_base(factor)


#%% 3.回测
#%% 通用日频回测函数
def backtest_daily(data, 
                   raw_weight,
                   date_bkt_start=None,
                   date_bkt_end=None,
                   fee=0.002, 
                   suspend_correct=1, 
                   type_price='close',):
    
    
    #%% 读取输入
    # 读取数据
    stock_code = pd.Series(data['close_adj'].index)
    daily_dates = pd.to_datetime(data['close_adj'].columns)
    
    close_adj = data['close_adj']
    amt = data['amt']
    if type_price == 'vwap':
        vwap_adj = data['vwap_adj']
    delist_date = data['delist_date']
    delist_date = [pd.to_datetime(x) if type(x)==str else np.nan for x in delist_date.iloc[:,0]]
    delist_date = pd.Series(delist_date,index=stock_code)
    maxupordown = data['maxupordown'].astype(float)
    
    # 读取原始权重和调仓日期
    # raw_weight = dataset.pivot_factor(weight)
    dates_trade = pd.to_datetime(raw_weight.columns)

    # 初始化权重
    stock_weight = pd.DataFrame(0,
                                index=stock_code,
                                columns=dates_trade)
    stock_weight.loc[raw_weight.index,:] = raw_weight.values
    stock_weight = stock_weight.fillna(0)
    
    # 设置回测区间
    if date_bkt_start is None:
        date_bkt_start = dates_trade[0].strftime('%Y-%m-%d')
    if date_bkt_end is None:
        date_bkt_end = dates_trade[-1].strftime('%Y-%m-%d')


    #%% 回测初始化
    # 回测起止日期的序号
    id_date_bkt_start = np.where(daily_dates==pd.to_datetime(date_bkt_start))[0][0]
    id_date_bkt_end = np.where(daily_dates==pd.to_datetime(date_bkt_end))[0][0]
    
    # 换仓日的序号
    id_dates_trade = [np.where(daily_dates==i)[0][0] for i in dates_trade]
    
    # 储存净值
    value_daily_raw = pd.Series(np.nan,index=daily_dates)
    
    # 第一个换仓日及上个交易日的净值置为1
    value_daily_raw.iloc[id_date_bkt_start-1:id_date_bkt_start+1] = 1
    
    # 储存上期权重，用于计算换手率；初始值置为0
    weight_sell = pd.Series(0,index=stock_code)
    
    # 储存换手率
    turnover = pd.Series(0,index=dates_trade)
    
    # 停牌修正得到实际权重
    real_weight = pd.DataFrame(0,index=stock_weight.index,columns=stock_weight.columns)
    
    
    #%% 正式回测
    # 遍历换仓日
    for i_date_trade, id_date_trade in enumerate(id_dates_trade):
        # id_date_trade = id_dates_trade[i_date_trade] # for debug
        #------------------------------------------#
        # 1. 确定每次换仓日对应的净值日区间
        # 换仓日的序号
        if id_date_trade < id_dates_trade[-1]:
            # 若非最后一个换仓日，则净值日为换仓日至下一个换仓日
            id_dates_value = list(range(id_date_trade, id_dates_trade[i_date_trade+1]+1))
        elif id_date_trade < id_date_bkt_end:
            # 若为最后一个换仓日，则净值日为换仓日至回测终止日
            id_dates_value = list(range(id_date_trade, id_date_bkt_end+1))
        else:
            id_dates_value = []
        #------------------------------------------#
        # 2. 针对停牌、退市等特殊情况进行权重修正
        if suspend_correct == 0:
            # 不进行修正
            real_weight.iloc[:,i_date_trade] = stock_weight.iloc[:,i_date_trade].values
        else:
            # 进行修正
            # 是否不可交易，1为不可交易
            flag_non_trade = ~np.logical_and(amt.iloc[:,id_date_trade] > 0, maxupordown.iloc[:,id_date_trade] == 0)
            flag_non_buy = ~np.logical_and(amt.iloc[:,id_date_trade] > 0, maxupordown.iloc[:,id_date_trade] <= 0)
            flag_non_sell = ~np.logical_and(amt.iloc[:,id_date_trade] > 0, maxupordown.iloc[:,id_date_trade] >= 0)
            # 是否退市，1为退市
            flag_delist = (delist_date <= daily_dates[id_date_trade])
            # 理论买入权重
            weight_buy = stock_weight.iloc[:,i_date_trade]
            # 理论买入总权重
            weight_buy_target = sum(weight_buy)
            # 实际卖出权重weight_sell
            # 买卖股票是否可交易
            weight_trade = weight_buy - weight_sell
            if weight_trade.loc[flag_non_trade].sum() == 0:
                # 若买卖均可交易，则实际买入权重为理论买入权重
                real_weight.iloc[:,i_date_trade] = weight_buy
            else:
                # 若存在不可交易的股票
                print('第%d期（%s）存在不可交易股票'%(i_date_trade+1,dates_trade[i_date_trade].strftime('%Y-%m-%d')))
                # ------------------------------------------ #
                # Step1：若存在不可卖出且未退市的股票，则该股票实际买入权重为实际卖出权重（即上期权重）
                # id_stock_non_sell = np.all([weight_sell>0,flag_non_trade==True,flag_delist==False],axis=0)
                # 上期持仓且不可卖出，则当期持仓=上期持仓（问题：加仓情形？）
                # id_stock_non_sell = np.all([weight_sell>0,flag_non_sell==True,flag_delist==False],axis=0)
                # 当期减仓（包含于上期持仓）且不可卖出，则当期持仓=上期持仓
                id_stock_non_sell = np.all([weight_trade<0,flag_non_sell==True,flag_delist==False],axis=0)
                id_stock_non_sell = np.where(id_stock_non_sell==True)[0]
                real_weight.iloc[id_stock_non_sell,i_date_trade] = weight_sell.iloc[id_stock_non_sell]
                # 拟买入权重=0
                weight_buy.iloc[id_stock_non_sell] = 0
                # 输出无法卖出的股票
                if len(id_stock_non_sell) > 0:
                    for i_stock in id_stock_non_sell:
                        print('%s无法卖出'%(stock_code.iloc[i_stock]))
                # ------------------------------------------ #
                # Step2：若存在不可买入的股票，则该股票实际买入权重为0
                #id_stock_non_buy = np.all([weight_buy>0,flag_non_trade==True],axis=0)
                # 当期持仓且不可买入，则拟当期持仓=0（问题：减仓情形？）
                # id_stock_non_buy = np.all([weight_buy>0,flag_non_buy==True],axis=0)
                # weight_buy.loc[id_stock_non_buy] = 0
                # 当期加仓且不可买入，则拟当期持仓=上期持仓
                id_stock_non_buy = np.all([weight_trade>=0,flag_non_buy==True],axis=0)
                id_stock_non_buy = np.where(id_stock_non_buy==True)[0]
                real_weight.iloc[id_stock_non_buy,i_date_trade] = weight_sell.iloc[id_stock_non_buy]
                # 拟买入权重=0
                weight_buy.iloc[id_stock_non_buy] = 0
                # 输出无法买入的股票
                if sum(id_stock_non_buy) > 0:
                    id_stock_non_buy = np.where(id_stock_non_buy==True)[0]
                    for i_stock in id_stock_non_buy:
                        print('%s无法买入'%(stock_code.iloc[i_stock]))
                # ------------------------------------------ #
                # Step3：若存在上期有持仓且已退市的股票，假设通过某种途径已卖出
                # id_stock_delist = np.all([weight_sell>0,flag_non_trade==True,flag_delist==True],axis=0)
                id_stock_delist = np.all([weight_trade<0,flag_non_sell==True,flag_delist==True],axis=0)
                # 输出退市股票
                if sum(id_stock_delist) > 0:
                    id_stock_delist = np.where(id_stock_delist==True)[0]
                    for i_stock in id_stock_delist:
                        print('%s退市卖出'%(stock_code.iloc[i_stock]))
                # ------------------------------------------ #
                # Step4：实际真正买入权重 = 理论买入总权重 - 无法卖出的权重（无法交易，已“内定”）
                weight_buy_real = weight_buy_target - sum(real_weight.iloc[:,i_date_trade])
                # ------------------------------------------ #
                # Step5：修正买入权重，使得总仓位与目标仓位一致
                weight_buy = weight_buy * weight_buy_real / sum(weight_buy) 
                # ------------------------------------------ #
                # Step6：买入（交易）
                # id_stock_buy = np.all([weight_buy>0,flag_non_trade==False],axis=0)
                # id_stock_buy = np.all([weight_buy>0,flag_non_buy==False],axis=0)
                # id_stock_buy = np.where(id_stock_buy==True)[0]
                # real_weight.iloc[id_stock_buy,i_date_trade] = weight_buy.iloc[id_stock_buy]
                id_stock_trade = list(set(range(len(stock_code))) - set(id_stock_non_sell) - set(id_stock_non_buy))
                real_weight.iloc[id_stock_trade,i_date_trade] = weight_buy.iloc[id_stock_trade]
                # 股票划分为3部分：
                # id_stock_non_sell = weight_trade<0 & flag_non_sell==True
                # id_stock_non_buy = weight_trade>=0 & flag_non_buy==True
                # id_stock_trade
                
        #------------------------------------------#
        # 3. 记录权重、买入价
        weight_buy = real_weight.iloc[:,i_date_trade]
        if type_price == 'close':
            price_buy = close_adj.iloc[:,id_date_trade]
        elif type_price == 'vwap':
            # 以均价买入
            #price_buy = vwap.iloc[:,id_date_trade] * close_adj.iloc[:,id_date_trade] / close.iloc[:,id_date_trade]
            price_buy = vwap_adj.iloc[:,id_date_trade]
            # 若vwap为nan或0，则以前一日复权收盘价代替
            id_price_buy_nan = np.where(np.isnan(price_buy)==True)[0]
            id_price_buy_zero = np.where(price_buy==0)[0]
            price_buy.iloc[id_price_buy_nan] = close_adj.iloc[id_price_buy_nan,id_date_trade-1]
            price_buy.iloc[id_price_buy_zero] = close_adj.iloc[id_price_buy_zero,id_date_trade-1]
        # 用于计算净值的基准value = 当期换仓日value
        # 当期换仓日value由上期换仓日的最后一个净值日根据卖出价计算得到
        # 当期换仓日value在随后计算时会根据收盘价更新
        value_buy = value_daily_raw.iloc[id_date_trade]
        turnover.iloc[i_date_trade] = sum(abs(weight_buy.values - weight_sell.values))
        # 根据换手率扣除交易费用
        value_buy = value_buy * (1 - turnover.iloc[i_date_trade] * fee)
        #------------------------------------------#
        # 4. 计算每日净值
        for id_date_value in id_dates_value:
            if id_date_value < id_dates_value[-1]:
                # 非当期最后一个净值日
                price_value = close_adj.iloc[:,id_date_value]
            else:
                # 当期最后一个净值日，卖出
                if type_price == 'close':
                    price_value = close_adj.iloc[:,id_date_value]
                elif type_price == 'vwap':
                    # 以均价卖出
                    #price_value = vwap.iloc[:,id_date_value] * close_adj.iloc[:,id_date_value] / close.iloc[:,id_date_value]
                    price_value = vwap_adj.iloc[:,id_date_value]
                    # 若vwap为nan或0，则以前一日复权收盘价代替
                    id_price_value_nan = np.where(np.isnan(price_value)==True)[0]
                    id_price_value_zero = np.where(price_value==0)[0]
                    price_value.iloc[id_price_value_nan] = close_adj.iloc[id_price_value_nan,id_date_value-1]
                    price_value.iloc[id_price_value_zero] = close_adj.iloc[id_price_value_zero,id_date_value-1]
                # 计算自然增长的权重，用于计算换手率和交易费用
                weight_sell = weight_buy * price_value / price_buy
                weight_sell.loc[np.isnan(weight_sell)] = 0
                # 权重归一化，考虑现金仓位
                weight_sell = weight_sell / (sum(weight_sell) + 1 - sum(weight_buy))
            # 计算收益
            returns = np.nansum(weight_buy * (price_value / price_buy - 1))
            # 计算净值；若为当期换仓日的最后一个净值日，则此净值为下期换仓日计算净值的基准
            value_daily_raw.iloc[id_date_value] = value_buy * (1 + returns)
            
    
    #%% 输出持仓
    real_weight = real_weight.loc[raw_weight.index,:]
    
    
    #%% 输出净值
    value_daily_raw = value_daily_raw.dropna()
    return value_daily_raw

# 回测
value = pd.DataFrame()
for i in range(len(weight)):
    value[f'第{i+1}层'] = backtest_daily(data, 
                                        weight[i],
                                        date_bkt_start=None,
                                        date_bkt_end='2024-07-31',
                                        fee=0, 
                                        suspend_correct=1, 
                                        type_price='close',)
value['等权基准'] = backtest_daily(data, 
                                    weight_base,
                                    date_bkt_start=None,
                                    date_bkt_end='2024-07-31',
                                    fee=0, 
                                    suspend_correct=1, 
                                    type_price='close',)


#%% 4.业绩分析
#%% 计算最大回撤
def compute_mdd(value):
    # value: (T,) numpy.ndarray
    value = np.array(value).reshape((-1))
    dd = []
    for i in range(len(value)):
        dd.append(1 - value[i] / np.max(value[0:i+1]))
    mdd = np.max(dd)
    dd = [-i for i in dd]
    return mdd, dd


#%% 业绩分析
def evaluate(value_portfolio, value_benchmark, save_path='./', turnover_portfolio=None):
    # value_portfolio: (T,) pandas.Series
    # value_benchmark: (T,) pandas.Series
    # save_path: str
    # turnover_portfolio: (T,) pandas.Series
    
    # 每年交易日/周/月数
    DAY = 252
    #WEEK = 52
    MONTH = 12
    
    # Merge组合净值和基准净值
    value = pd.merge(value_portfolio,value_benchmark,'outer',left_index=True,right_index=True)
    value.columns = ['portfolio','benchmark']
    value['benchmark'] = value['benchmark'] / value['benchmark'].iloc[0]
    
    # 计算超额收益净值
    excess_return = value['portfolio'].pct_change().values - value['benchmark'].pct_change().values
    excess_return[0] = 0
    value_excess_compound = pd.Series((1+excess_return).cumprod(),index=value.index)
    value['excess'] = value_excess_compound - 1
    _, value['excess_mdd'] = compute_mdd(value_excess_compound)
    
    # 初始化结果
    col_name = ['annualized_return','annualized_volatility','sharpe_ratio','maximum_drawdown','calmar_ratio',
                'annualized_excess_return','annualized_tracking_error','information_ratio','excess_maximum_drawdown','excess_calmar_ratio',
                'monthly_win_ratio','annualized_turnover']
    evaluation = pd.Series(index=col_name, dtype=float)
    
    # 计算原始收益业绩分析指标
    _value = value['portfolio']
    evaluation.loc['annualized_return'] = np.power(_value.iloc[-1]/_value.iloc[0],DAY/(_value.shape[0]-1)) - 1
    evaluation.loc['annualized_volatility'] = np.nanstd(_value.pct_change()) * np.sqrt(DAY)
    evaluation.loc['sharpe_ratio'] = evaluation.loc['annualized_return'] / evaluation.loc['annualized_volatility']
    evaluation.loc['maximum_drawdown'], _ = compute_mdd(_value)
    evaluation.loc['calmar_ratio'] = evaluation.loc['annualized_return'] / evaluation.loc['maximum_drawdown']
   
    # 计算超额收益业绩分析指标
    _value = value_excess_compound
    evaluation.loc['annualized_excess_return'] = np.power(_value.iloc[-1]/_value.iloc[0],DAY/(_value.shape[0]-1)) - 1
    evaluation.loc['annualized_tracking_error'] = np.nanstd(_value.pct_change()) * np.sqrt(DAY)
    evaluation.loc['information_ratio'] = evaluation.loc['annualized_excess_return'] / evaluation.loc['annualized_tracking_error']
    evaluation.loc['excess_maximum_drawdown'], _ = compute_mdd(_value)
    evaluation.loc['excess_calmar_ratio'] = evaluation.loc['annualized_excess_return'] / evaluation.loc['excess_maximum_drawdown']

    # 计算月度超额收益胜率
    evaluation.loc['monthly_win_ratio'] = np.nan
    _index = value.index
    # 月末日期
    month_list = pd.date_range(
        # start=pd.Timestamp(f"{_index.min()[:4]}0101"),
        # end=pd.Timestamp(f"{_index.max()[:4]}1231"),
        start=pd.Timestamp(f"{_index.min().year}0101"),
        end=pd.Timestamp(f"{_index.max().year}1231"),
        freq="1M")
    # 初始化月度超额收益
    month_return = pd.DataFrame(np.nan,
                                index=range(month_list[0].year,month_list[-1].year+1),
                                columns=range(1,MONTH+1),
                                dtype=float)
    # 遍历月份，计算月度超额收益
    # curr_idx_month_start: 月初索引（上月最后一个交易日）
    # curr_idx_month_end: 月末索引（当月最后一个交易日）
    i, curr_idx_month_start, curr_idx_month_end = 0, None, None
    while i<len(month_list):
        # 若i>0，则当月月初=上月月末
        curr_idx_month_start = curr_idx_month_end if i>0 else 0
        # 若当月月初晚于最后一个净值日，则结束循环
        if curr_idx_month_start >= value.shape[0] - 1:
            break
        # 若当月月末早于第一个净值日，则跳过循环
        if month_list[i] < pd.to_datetime(_index)[0]:
            curr_idx_month_end = 0
            i = i + 1
            continue
        # 获取当月月末索引
        curr_idx_month_end = np.where(pd.to_datetime(_index)<=month_list[i])[0][-1]
        # 计算超额收益
        # 排除当月仅有1个净值日情况
        if curr_idx_month_end > curr_idx_month_start:
            curr_excess_return = value_excess_compound[curr_idx_month_end] / value_excess_compound[curr_idx_month_start] - 1
            month_return.loc[month_list[i].year,month_list[i].month] = curr_excess_return
        i = i+1
    # 计算月度超额收益胜率 
    evaluation.loc['monthly_win_ratio'] = (month_return>0).sum().sum() / month_return.count().sum()
    # 计算年度超额收益
    for i in month_return.index:
        month_return.loc[i,'total'] = (month_return.loc[i,:] + 1).prod() - 1
    
    # 计算年化双边换手率
    if turnover_portfolio is None:
        evaluation.loc['annualized_turnover'] = np.nan
    else:
        evaluation.loc['annualized_turnover'] = turnover_portfolio.sum().values / ((value.shape[0]-1) / DAY)
    
    return value, evaluation

excess_value = pd.DataFrame()
evaluation = pd.DataFrame()
for i in range(len(weight)):
    value_i, evaluation_i = evaluate(value_portfolio=value[f'第{i+1}层'], 
                                    value_benchmark=value['等权基准'], 
                                    save_path='./', 
                                    turnover_portfolio=None)
    excess_value[f'第{i+1}层'] = value_i['excess']
    evaluation[f'第{i+1}层'] = evaluation_i


#%% 5.保存
writer = pd.ExcelWriter('分层回测结果.xlsx')
value.to_excel(writer,sheet_name='净值')
excess_value.to_excel(writer,sheet_name='超额净值')
evaluation.to_excel(writer,sheet_name='业绩分析')
writer.save()
