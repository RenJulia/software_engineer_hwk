import torch
from torch import optim
import torch.nn as nn
import numpy as np
from scipy import stats
import os
import scipy.io as sio
from function import *
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_transformer import MyModel_Transformer, Newdataset
from torch.serialization import add_safe_globals
# from train_model_daily_attention import MyModel, Newdataset

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#gpu
add_safe_globals([MyModel_Transformer])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#加载基本数据
X1 = np.load('data/X1.npy')
Y = np.load('data/Y.npy')
#参数设置
# begid = 3280
begid = 4600
len1_date = 40
period = 120


def load_data(end_date):
    # if end_date >= 4600:
    #     m = 4600
    # else:
    #     m = 3280
    
    model_list = list(range(begid, Y.shape[1], period))
    m = [m for m in model_list if m <= end_date][-1]

    med_x1 = np.load('data/med_x1.npy')
    mad_x1 = np.load('data/mad_x1.npy')

    x1 = X1[:, end_date-len1_date+1:end_date+1, :] #(1000,period,6)
    y = Y[:, end_date-len1_date+1:end_date+1]

    x1_in_sample = np.zeros(
        (int(x1.shape[0]), len1_date, x1.shape[2]))

    y_in_sample = np.zeros((int(y.shape[0]), 1))
    w_in_sample = np.zeros((int(y.shape[0]), 1))
    n_sample = 0
    s_index = n_sample
    # 遍历所有的股票样本
    nonan_index = []
    for i in range(y.shape[0]):
        x1_one = x1[i, :] #200:period
        y_one = y[i, -1]

        if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).any()):
            continue 
        nonan_index.append(i)
        x1_one_last = np.tile(x1_one[-1, :], (x1_one.shape[0], 1))
        x1_one = x1_one / x1_one_last


        x1_in_sample[n_sample, :, :] = x1_one
        y_in_sample[n_sample, :] = y_one
        n_sample += 1

    e_index = n_sample
    y_in_sample[s_index:e_index, 0], w_in_sample[s_index:e_index, 0] = \
        standardize_and_weight(y_in_sample[s_index:e_index, 0])
    for k in range(0, x1_one.shape[0]):
        for s in range(0, x1_one.shape[1]):
            tmpmean = med_x1[k][s]
            tmpstd = mad_x1[k][s]
            x1_in_sample[s_index:e_index, k, s] = (x1_in_sample[s_index:e_index, k, s] - tmpmean) / (tmpstd + 1e-8)


    x1_in_sample = x1_in_sample[:n_sample, :]
    y_in_sample = y_in_sample[:n_sample, :]
    w_in_sample = w_in_sample[:n_sample, ]

    x1_test = x1_in_sample
    y_test = y_in_sample
    w_test = w_in_sample
    return x1_test, y_test, w_test, nonan_index


if __name__ == '__main__':
    for mm in range(1, 2):
        rank_ic_list = []
        dir_alpha_daily = 'data/alpha_daily.mat'
        dailyinfo = sio.loadmat(dir_alpha_daily,variable_names=['dailyinfo'])['dailyinfo']
        date_mat = dailyinfo['dates'][0][0][0]
        fac_1 = pd.DataFrame(np.nan * np.zeros((X1.shape[0],len(date_mat) - begid)))
        trade_date_df = pd.DataFrame(np.nan * np.zeros((2,len(date_mat) - begid)))
        i_panel = 0
        end_list = list(range(begid, Y.shape[1], period))
        for end in end_list:
            #定义模型
            model = MyModel_Transformer()
            # 加载模型
            model_path = f'model_transformer/{mm}/'
            print('--------------------------------')
            print(model_path + str(end) + '.pt')
            print('--------------------------------')
            model = torch.load(model_path + str(end) + '.pt', weights_only=False, map_location=device)
            # for end_date in range(end+1, end+len3_date+1):
            for end_date in range(end+1, end+period+1):
                if end_date < X1.shape[1]:
                    x1_test, y_test, w_test, nonan_index = load_data(end_date)
                    test_ds = Newdataset(x1_test, y_test)
                    test_dl = DataLoader(test_ds, batch_size = len(x1_test))

                    x1, labels = next(iter(test_dl))
                    x1 = x1.to(device)
                    labels = labels.to(device)
                    model.eval()
                    y_pred = model(x1)
                    y_pred = y_pred.cpu().detach().numpy()
                    # 计算ic
                    fac_1.iloc[nonan_index, i_panel] = y_pred[:, -1]
                    trade_date_df.iloc[0, i_panel] = date_mat[end_date]
                    try:
                        trade_date_df.iloc[1, i_panel] = date_mat[end_date+1]
                    except:
                        trade_date_df.iloc[1, i_panel] = date_mat[end_date]+1
                    i_panel += 1
                    try:
                        rank_ic = stats.spearmanr(y_test, y_pred)[0]
                        print(np.mean(rank_ic_list))
                        rank_ic_list.append(rank_ic)
                    except:
                        pass


        fac_1 = fac_1.iloc[:,:i_panel]
        trade_date_df = trade_date_df.iloc[:,:i_panel]
        fac_1.to_csv(model_path + 'fac1.csv', index=None, header=None)
        trade_date_df.to_csv(model_path + 'trade_date.csv',header=False,index=False)

        ic_cumsum = np.cumsum(rank_ic_list)
        df_rank_ic = pd.DataFrame({'rank_ic': rank_ic_list, 'cumsum': ic_cumsum})
        df_rank_ic.to_csv(model_path + 'df_test_ic.csv', index = False)
        print(df_rank_ic.rank_ic.mean())
