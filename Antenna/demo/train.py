import time

import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import pandas as pd
from matplotlib import pyplot as plt


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 3)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        # x = self.fc1(x)
        x = torch.relu(self.fc1(x))
        # x = self.bn1(x)
        x = self.dropout1(x)
        # x = torch.relu(self.fc2(x))
        # x = self.bn2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        x = self.fc4(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available')
else:
    device = torch.device('cpu')
    print('CUDA is not available')
data = pd.read_csv("./output1.csv")
# data.sample(frac=1)
loss = nn.MSELoss()
channels = pd.get_dummies(data['ch'])
data.drop(columns='ch', inplace=True)
data[:] = data[:].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = pd.concat([data.iloc[:, :-3], channels], axis=1)
# all_features = data.iloc[:, :-3]
# all_features = data.iloc[:, :-3]
# all_features[:-1] = all_features[:-1].apply(lambda x: (x - x.mean()) / (x.std()))
# channels = pd.get_dummies(all_features['ch'])
# all_features = pd.concat([all_features.iloc[:, :-1], channels], axis=1)
n_train = int(0.9 * data.shape[0])
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(data.iloc[:n_train, -3:].values.reshape(-1, 3), dtype=torch.float32).to(device)
print(train_features.shape)
# print(train_features[0:10, :])
print(test_features.shape)
print(train_labels.shape)
# print(train_labels[:1, :])
in_features = train_features.shape[1]


def log_rmse(net, features, labels):
    rmse = loss(net(features), labels)
    return rmse.item()
# def log_rmse(net, features, labels):
#     # 为了在取对数时进一步稳定该值，将小于1的值设置为1
#     clipped_preds = torch.clamp(net(features), 1, float('inf'))
#     rmse = torch.sqrt(loss(torch.log(clipped_preds),
#                            torch.log(labels)))
#     return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    net.to(device)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            t_s = time.time()
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            t_e = time.time()
            print(t_e-t_s)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = MLP()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx, :]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = MLP()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i >= 0:
        #     d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #              xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
        #              legend=['train', 'valid'], yscale='log')
        #     plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


for lr in range(1, 100):
    lr /= 1e5
    k, num_epochs, weight_decay, batch_size = 5, 200, 0.00001, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')
    if abs(float(train_l) - float(valid_l)) < 0.1:
        print(lr)
# train_and_pred(train_features, test_features, train_labels, test_data,
#                num_epochs, lr, weight_decay, batch_size)
