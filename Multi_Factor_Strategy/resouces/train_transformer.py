import torch
from torch import optim
import torch.nn as nn
import numpy as np
import os
from function import *
from torch.utils.data.dataset import IterableDataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# 加载基本数据
#gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 一些参数
step = 5
len1_date = 40
begid = 4600
# begid = 6280
# 加载数据

X1 = np.load('data/X1.npy')
Y = np.load('data/Y.npy')
print(X1.shape)
print(Y.shape)

class MyModel_Transformer(nn.Module):
    """
    transformer
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 36,
        nhead: int = 6, 
        num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=4*hidden_dim, batch_first=True)#
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x1):
        hidden = self.embedding(x1) # (batch, seq, hidden_dim)
        hidden = self.transformer_encoder(hidden) # (batch, seq, hidden_dim)
        hidden = hidden[:, -1, :] # (batch, hidden_dim)
        hidden = self.bn(hidden) # (batch, hidden_dim)
        output = self.linear(hidden) # (batch, 1)

        return output



class Mydataset(IterableDataset):
    def __init__(self,data1, label) -> None:
            super().__init__()
            self.data1 = data1.astype(np.float32)
    def __len__(self):
        return len(self.data1)
    def __iter__(self):
        for row in range(len(self.data1)):
            yield self.data1[row], self.label[row]


class Newdataset(Dataset):
    def __init__(self,data1, label) -> None:
            super().__init__()
            self.data1 = data1.astype(np.float32)
            self.label = label.astype(np.float32)
    def __len__(self):
        return len(self.data1)
    def __getitem__(self, index):
        return self.data1[index],  self.label[index]



# 定义数据导入
def load_data(end_date):

    med_x1 = np.load('data/med_x1.npy')
    mad_x1 = np.load('data/mad_x1.npy')

    end_date1 = end_date - 15
    #2410
    start_date = end_date - 2410
    # 取出对应时间跨度的X和Y
    x1 = X1[:, start_date:end_date1, :]
    y = Y[:, start_date:end_date1]

    x1_in_sample = np.zeros(
        (int(x1.shape[0] * x1.shape[1] / step), len1_date, x1.shape[2]))

    y_in_sample = np.zeros((int(y.shape[0] * y.shape[1] / step), 1))
    w_in_sample = np.zeros((int(y.shape[0] * y.shape[1] / step), 1))

    n_sample = 0
    # 沿着时间轴取样本
    for j in range(0, y.shape[1] - len1_date + 1, step):
        s_index = n_sample
        # 遍历所有的股票样本
        for i in range(y.shape[0]):
            x1_one = x1[i, j + len1_date - len1_date:j + len1_date]
            y_one = y[i, j + len1_date - 1]

            if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).any() or np.isnan(
                        y_one).any()):
                continue
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

    split = int(y_in_sample.shape[0] * 0.8)
    x1_train = x1_in_sample[:split, :, :]
    x1_val = x1_in_sample[split:, :, :]
    y_train = y_in_sample[:split, :]
    y_val = y_in_sample[split:, :]
    w_train = w_in_sample[:split, 0]
    w_val = w_in_sample[split:, 0]
    return x1_train, x1_val, y_train, y_val, w_train, w_val




BATCH_SIZE = 5000
MAX_EPOCH = 100
epoch_iter = 10 #早停轮数设置
w = 0
loss_name = 'personr' #mse,personr,ccc
# MAX_EPOCH = 5
# epoch_iter = 3 #早停轮数设置

# 定义模型训练和测试的方法
def train():
    # 模型的训练状态
    ic = 0
    model.train()
    tqdm_ = tqdm(iterable=train_dl)
    for i, batch in enumerate(tqdm_):
        # 获得一个批次的数据和标签
        x1, labels = batch
        x1 = x1.to(device)
        labels = labels.to(device)
        out = model(x1)
        #加入一个惩罚
        # Loss = nn.MSELoss()
        loss = CCC(labels, out)
        if loss_name == 'personr':
            loss = pearson_r_loss(labels, out)
        elif loss_name == 'mse':
            Loss = nn.MSELoss()
            loss = Loss(out, labels)
        else:
            loss = CCC(labels, out)
        # L2_loss = penality(hidden)
        # Loss = pearson_r_loss(labels, out)
        # loss = Loss + w * L2_loss
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 修改权值
        optimizer.step()
        ic += pearson_r(labels, out).item()
        # tqdm_.set_description(
        #     "epoch:{:d} train loss:{:.4f} train -pearson:{:.4f} train penality:{:.4f}".format(epoch, loss.item(), Loss.item(), L2_loss.item()))
        tqdm_.set_description("epoch:{:d} train loss:{:.4f}".format(epoch, loss.item()))
    print("train person:{0}".format(ic / (i+1)))
    return ic / (i+1)


def eval():
    # 模型的测试状态
    model.eval()
    ic = 0  # 测试集准确率
    tqdm_ = tqdm(iterable=val_dl)
    for i, batch in enumerate(tqdm_):
        # 获得一个批次的数据和标签
        x1, labels = batch
        x1 = x1.to(device)
        labels = labels.to(device)
        out = model(x1)
        # corr = penality(hidden)
        # 预测正确的数量
        if loss_name == 'personr':
            ic += pearson_r(labels, out).item()
        elif loss_name == 'mse':
            ic += anti_mse(labels, out).item()
        else:
            ic += -CCC(labels, out).item()
        tqdm_.set_description(
            "epoch:{:d} val {!s}:{:.4f} ".format(epoch, loss_name, ic))
    print("val {0}:{1}".format(loss_name,ic / (i+1)))
    return ic / (i+1)



if __name__ == '__main__':
    for mm in range(1, 2):
        
        os.makedirs(f'model_transformer/{mm}/', exist_ok=True)
        cur_file_path = __file__
        cur_file_name = cur_file_path.split('\\')[-1]
        #cur_file_name = cur_file_path.split('/')[-1]
        shutil.copyfile(cur_file_path, f'model_transformer/{mm}/{cur_file_name}')

        end_list = list(range(begid, Y.shape[1], 120))
        # end_list = list(range(3280, begid, 120))
        for end in end_list:
            x1_train, x1_val, y_train, y_val, w_train, w_val = load_data(end)
            print('data finish')

            train_ds = Newdataset(x1_train, y_train)
            train_dl = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)

            val_ds = Newdataset(x1_val, y_val)
            val_dl = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=True)

            # 定义模型
            model = MyModel_Transformer()
            model.to(device)
            # 定义优化器
            optimizer = optim.Adam(model.parameters(), lr=0.005)# 随机梯度下降
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # 学习率衰减step_size可设置

            max_ic = -10000
            max_epoch = 0

            train_list = []
            val_list = []

            for epoch in range(MAX_EPOCH):
                train_ic = train()
                ic = eval()

                train_list.append(train_ic)
                val_list.append(ic)

                if ic > max_ic:
                    max_ic = ic
                    max_epoch = epoch
                    model_path = f'model_transformer/{mm}/'
                    torch.save(model, model_path + str(end) + '.pt')
                else:
                    if epoch - max_epoch >= epoch_iter:
                        break
                # scheduler.step()

            fig = plt.figure(figsize=[8,6])
            plt.plot(
                np.arange(epoch + 1),
                train_list,
                label='train_scores'
                )
            plt.plot(
                np.arange(epoch + 1),
                val_list,
                label='valid_scores'
                )
            plt.legend()
            plt.title(f"train and valid scores for {end}")
            fig.savefig(f'model_transformer/{mm}/{end}.png')


