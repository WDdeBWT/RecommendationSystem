import time
import math

import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import numpy as np

import FCM
import data
import utils
import evaluator
import model_test


USER_CLUSTERS = 5
ITEM_CLUSTERS = 5
BATCH_SIZE = 2000
LR = 0.005
EPOCHES = 30


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden1(x))      # 激励函数(隐藏层的线性值)
        x = F.relu(self.hidden2(x))      # 激励函数(隐藏层的线性值)
        x = self.predict(x)             # 输出值
        return x


def test():
    data_path = 'data-sub/temp_list.csv'
    rate_list = utils.read_csv(data_path, show_detail=False)[1:]
    train_data, test_data = utils.split_data(rate_list, 1, show_detail=False)
    tdata = data.TrainData(train_data, show_detail=False, only_hot=False)
    tdata.get_rate_mat()

    fcm_u = FCM.FCM(tdata.rate_umat, USER_CLUSTERS, 20, m=3.00, show_detail=True)
    user_membership_mat = fcm_u.get_result()
    user_membership_mat = np.array(user_membership_mat)
    print('user_membership_mat.shape: {}'.format(user_membership_mat.shape))

    fcm_i = FCM.FCM(tdata.rate_imat, ITEM_CLUSTERS, 20, m=3.00, show_detail=True)
    movie_membership_mat = fcm_i.get_result()
    movie_membership_mat = np.array(movie_membership_mat)
    print('movie_membership_mat.shape: {}'.format(movie_membership_mat.shape))

    np_input, np_target = tdata.get_nn_input_mat(user_membership_mat, movie_membership_mat)
    print(np_input.shape)
    print(np_target.shape)

    net = Net(n_feature=USER_CLUSTERS + ITEM_CLUSTERS, n_hidden=10, n_output=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)

    # 先转换成 torch 能识别的 Dataset
    torch_dataset = torchdata.TensorDataset(torch.from_numpy(np_input), torch.from_numpy(np_target))
    # 把 dataset 放入 DataLoader
    loader = torchdata.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )

    for epoch in range(EPOCHES):   # 训练所有!整套!数据 3 次
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            prediction = net(batch_x)     # 喂给 net 训练数据 x, 输出预测值
            loss = loss_func(prediction, batch_y)     # 计算两者的误差
            optimizer.zero_grad()   # 清空上一步的残余更新参数值
            loss.backward()         # 误差反向传播, 计算参数更新值
            optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
            # if step % 20 == 0:
            if step == 0:
                time_str = time.strftime("%H:%M:%S", time.localtime())
                # print('- time: {} - epoch: {} step: {} - loss: {}'.format(time_str, epoch, step, loss))
                print('- time: {} epoch: {} - loss: {}'.format(time_str, epoch, loss))
                print('predict: {}'.format([round(item[0], 2) for item in prediction.tolist()[:5]]))
                print('target : {}'.format([round(item, 2) for item in batch_y.tolist()[:5]]))

if __name__ == "__main__":
    test()
