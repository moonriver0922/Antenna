#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2023 - 2023 Guosheng_W, Inc. All Rights Reserved
#
# @Time    : 2023/5/10 22:31
# @Author  : Guosheng_W
# @File    : boundary.py
# @Email   : 3190102029@zju.edu.cn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from d2l.torch import d2l
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

logger = SummaryWriter('log')

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        print(preds.shape)
        print(labels.shape)
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(96, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dp2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dp3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(32, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dp4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.dp5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dp2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dp3(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dp4(x)
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dp5(x)
        x = self.fc6(x)
        return x


df = pd.read_csv("./data/s2.csv")
df = df.sample(frac=1)
pos = df.iloc[:, :2]
p1 = [3.247750759124756, 9.432153701782227]
p2 = [-2.249511241912842, 9.743700981140137]
p3 = [2.916592836380005, -9.539752960205078]
on = []
i_n = []
out = []
r0 = 1.3
r = 1.6
r1 = 2.0


def is_point_in_circle(point, r):
    return point[0] ** 2 + point[1] ** 2 - r ** 2 < 0


def is_point_in_triangle(point, p1, p2, p3):
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x2, y3 - y2)
    v3 = (x1 - x3, y1 - y3)

    v4 = (x - x1, y - y1)
    v5 = (x - x2, y - y2)
    v6 = (x - x3, y - y3)

    dp1 = v1[0] * v4[1] - v1[1] * v4[0]
    dp2 = v2[0] * v5[1] - v2[1] * v5[0]
    dp3 = v3[0] * v6[1] - v3[1] * v6[0]

    # in
    return (dp1 >= 0 and dp2 >= 0 and dp3 >= 0) or (dp1 <= 0 and dp2 <= 0 and dp3 <= 0)


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available')
else:
    device = torch.device('cpu')
    print('CUDA is not available')
train_data = []
acc_data = []
test_data = []
count = 1
for index, (x, y) in enumerate(zip(pos.iloc[:, 0], pos.iloc[:, 1])):
    if not is_point_in_circle([x, y], r) and is_point_in_circle([x, y], r1):
        on.append([x, y])
        train_data.append(df.iloc[index, :].tolist() + [1, 0])
    elif is_point_in_circle([x, y], r0):
        count += 1
        if count % 10 == 0:
            i_n.append([x, y])
            train_data.append(df.iloc[index, :].tolist() + [0, 1])
        elif count % 10 == 1:
            acc_data.append(df.iloc[index, :].tolist() + [0, 1])
    elif not is_point_in_circle([x, y], r1):
        out.append([x, y])
        test_data.append(df.iloc[index, :].tolist() + [1, 0])
print(len(on))
print(len(i_n))
acc_data = np.array(acc_data)
train_data = np.array(train_data)
np.random.shuffle(train_data)
print(train_data)
test_data = np.array(test_data)
acc_features = torch.tensor(acc_data[:, 3:-2], dtype=torch.float32).to(device)
acc_labels = torch.tensor(acc_data[:, -2:], dtype=torch.float32)
train_features = torch.tensor(train_data[:, 3:-2], dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data[:, -2:], dtype=torch.float32).to(device)
test_features = torch.tensor(test_data[:, 3:-2], dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data[:, -2:], dtype=torch.float32)
train_features = (train_features - train_features.mean()) / train_features.std()
test_features = (test_features - test_features.mean()) / test_features.std()
acc_features = (acc_features - acc_features.mean()) / acc_features.std()
print(train_features)
print(train_features.shape)
print(train_labels.shape)
print(train_labels)
print(test_features.shape)
print(test_labels.shape)
# loss = FocalLoss(alpha=0.25, gamma=2)
# loss = torch.nn.BCEWithLogitsLoss()
loss = torch.nn.CrossEntropyLoss()


# loss = nn.MSELoss()


def train(net, train_features_, train_labels_, test_features_, test_labels_,
          num_epochs_, learning_rate, weight_decay_, batch_size_):
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    net.to(device)
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features_, train_labels_), batch_size_)
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay_,
                                momentum=0.9)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000005)
    log_step_interval = 1000
    num_batches = len(train_iter)
    for epoch in range(num_epochs_):
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs_}") as pbar:
            for step, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                o = net(X)
                l = loss(o, y)
                a = torch.where(o <= 0.5, torch.tensor(0), torch.tensor(1))
                # acc = a.eq(y).sum().item() / 2
                row_equals = torch.all(torch.eq(a, y), dim=1)
                acc = torch.sum(row_equals).item()
                total = a.shape[0]
                l.backward()
                optimizer.step()
                global_iter_num = epoch * num_batches + step + 1
                pbar.update(1)
                pbar.set_postfix_str(f"loss: {l.item():.3f}, acc: {acc/total:.3f}")
                if global_iter_num % log_step_interval == 0:
                    # print("epoch:", epoch)
                    # print("global_step:{}, loss:{:.3}".format(global_iter_num, l.item()))
                    logger.add_scalar("train loss", l.item(), global_step=global_iter_num)
        # scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        # print("当前学习率：", current_lr)
        train_ls.append(loss(net(train_features_), train_labels_).item())
        if test_labels_ is not None:
            test_ls.append(loss(net(test_features_), test_labels_).item())
    return train_ls, test_ls


def get_k_fold_data(k_, i, X, y):
    assert k_ > 1
    fold_size = X.shape[0] // k_
    X_train, y_train = None, None
    for j in range(k_):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k_, X_train, y_train, num_epochs_, learning_rate, weight_decay_,
           batch_size_):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k_):
        data = get_k_fold_data(k_, i, X_train, y_train)
        net = classifier()
        train_ls, valid_ls = train(net, *data, num_epochs_, learning_rate,
                                   weight_decay_, batch_size_)
        train_l_sum += train_ls[-1]

        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs_ + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs_],
                     legend=['train', 'valid'], yscale='log')
            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k_, valid_l_sum / k_


k, num_epochs, lr, weight_decay, batch_size = 5, 5000, 0.000055, 0.00001, 32
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


def train_and_pred(train_features, test_features, train_labels, test_labels,
                   num_epochs, lr, weight_decay, batch_size):
    net = classifier()
    net.to(device)
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = torch.where(net(test_features) <= 0.5, torch.tensor(0), torch.tensor(1)).cpu().detach().numpy()
    row_equals = np.all(np.equal(preds, np.array(test_labels)), axis=1)
    acc = np.sum(row_equals).item()
    print(f"acc: {acc/preds.shape[0]}")


# train_and_pred(train_features, acc_features, train_labels, acc_labels, num_epochs, lr, weight_decay, batch_size)
