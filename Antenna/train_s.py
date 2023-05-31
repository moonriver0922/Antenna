import random
import time

import numpy as np
import torch
import torch.nn as nn
from d2l import torch as d2l
import pandas as pd
from matplotlib import pyplot as plt
from torch import autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dp4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(128, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dp5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp6 = nn.Dropout(0.1)
        self.fc7 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp7 = nn.Dropout(0.1)
        self.fc8_1 = nn.Linear(256, 1)
        self.fc8_2 = nn.Linear(256, 1)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dp1(x)
        x = torch.relu(self.fc2(x))
        x = self.dp2(x)
        x = torch.relu(self.fc3(x))
        x = self.dp3(x)
        x = torch.relu(self.fc4(x))
        x = self.dp4(x)
        x = torch.relu(self.fc5(x))
        x = self.dp5(x)
        x = torch.relu(self.fc6(x))
        x = self.dp6(x)
        x = torch.relu(self.fc7(x))
        x = self.dp7(x)
        feature_map = (x.reshape(-1, 256, 1) * x.reshape(-1, 1, 256)).unsqueeze(1)
        feature_map = self.conv(feature_map)
        feature_map = feature_map.squeeze()
        alpha = self.fc8_1(x)
        beta = self.fc8_2(x)
        return torch.cat((alpha, beta, feature_map.reshape((-1, 256*256))), dim=1)


# def line_distance(p1, v1, p2, v2):
#     v = torch.cross(v1, v2)
#     print(v.shape)
#     p = p2 - p1
#     print(p.dtype)
#     # dot_product = torch.dot(p, v)
#     dot_product = torch.sum(p * v, dim=1)
#     norm = torch.norm(v)
#     dist = torch.abs(dot_product) / norm
#     return dist


def spherical2pointandline(x):
    # print(x.shape)
    alpha, beta = x[:, 0].reshape((-1, 1)), x[:, 1].reshape((-1, 1))
    return torch.cat((torch.sin(beta) * torch.cos(alpha), torch.sin(beta) * torch.sin(alpha), torch.cos(beta)), dim=1)


# x = spherical2pointandline(torch.tensor([[torch.pi / 4, torch.pi / 4]]))
# print(x)


# def spherical2cartesian(x, p: list):
#     alpha, beta = x[:, 0].reshape((-1, 1)), x[:, 1].reshape((-1, 1))
#     px, py, pz = p
#     size = x.shape[0]
#     return torch.cat((torch.ones(size, 1).to(device) - torch.tan(alpha), torch.ones((size, 1)).to(device),
#                       torch.sqrt(torch.tan(beta) ** 2 / (torch.ones(size, 1).to(device) + torch.tan(alpha) ** 2)),
#                       -px * (torch.ones(size, 1).to(device) - torch.tan(alpha)) - py - pz * torch.sqrt(
#                           torch.tan(beta) ** 2 / (torch.ones(size, 1).to(device) + torch.tan(alpha) ** 2))), dim=1)


# def intersect_or_distance(x1, x2):
#     a1, b1, c1, d1 = x1[:, 0], x1[:, 1], x1[:, 2], x1[:, 3]
#     a2, b2, c2, d2 = x2[:, 0], x2[:, 1], x2[:, 2], x2[:, 3]
#     v1 = torch.cat((a1.unsqueeze(1), b1.unsqueeze(1), c1.unsqueeze(1)), dim=1)
#     v2 = torch.cat((a2.unsqueeze(1), b2.unsqueeze(1), c2.unsqueeze(1)), dim=1)
#
#     if torch.all(torch.cross(v1, v2) == 0):
#         dist = abs(d2 - d1) / torch.norm(v1)
#         return 1, dist
#
#     n = torch.cross(v1, v2)
#     n1 = torch.cross(v2, n)
#     n2 = torch.cross(n, v1)
#     t1 = torch.where(torch.sum(n1 * n1, dim=1) != 0, -torch.sum(n1 * v1, dim=1) / torch.sum(n1 * n1, dim=1), -torch.sum(n1 * v1, dim=1) / 0.00001)
#     t2 = torch.where(torch.sum(n2 * n2, dim=1) != 0, torch.sum(n2 * v2, dim=1) / torch.sum(n2 * n2, dim=1), torch.sum(n2 * v2, dim=1) / 0.00001)
#     # t2 = torch.sum(n2 * v2, dim=1) / torch.sum(n2 * n2, dim=1)
#     p_1 = v1 * t1.unsqueeze(1)
#     p_2 = v2 * t2.unsqueeze(1)
#     # print('p_1', p_1.shape)
#     if torch.allclose(p_1, p_2):
#         return 2, p_1
#     else:
#         return 3, torch.where(torch.norm(v1) != 0, torch.norm(torch.cross(p_2 - p_1, v1)) / torch.norm(v1), torch.norm(torch.cross(p_2 - p_1, v1)) / 0.00001)
def intersect_or_distance(p1, v1, p2, v2):
    p1.expand(v1.shape[0], 3)
    p2.expand(v2.shape[0], 3)
    if torch.all(torch.cross(v1, v2) == 0):
        dist = distance_between_lines(p1, v1, p2, v2)
        return 1, dist
    s = compute_intersection(p1, v1, p2, v2)
    if s == 1:
        return 3, distance_between_lines(p1, v1, p2, v2)
    return 2, s


def compute_intersection(p1, v1, p2, v2):
    v1_ = v1.unsqueeze(-1).reshape(-1, 3, 1)
    v2_ = v2.unsqueeze(-1).reshape(-1, 3, 1)
    A = torch.cat((v1_, v2_, torch.ones(v1_.shape)), dim=2)
    b = (p2 - p1).squeeze()
    x = torch.linalg.solve(A, b)
    zero = x[:, 2].unsqueeze(-1)
    if not torch.allclose(zero, torch.zeros_like(zero)):
        return 1
    p = p1 + v1 * x[:, 0].unsqueeze(-1)
    return p


def distance_between_lines(p1, v1, p2, v2):
    # Calculate the direction vector perpendicular to both lines
    cross_product = torch.cross(v1, v2, dim=1)

    # Calculate the distance between the two lines
    distance = torch.abs(torch.einsum('ij, ij->i', (p2 - p1, cross_product))) / torch.norm(cross_product, dim=1)

    return distance.unsqueeze(1)


def area(x1, x2, x3):
    v1 = spherical2pointandline(x1)
    v2 = spherical2pointandline(x2)
    v3 = spherical2pointandline(x3)
    res = [intersect_or_distance(p1, v1, p2, v2), intersect_or_distance(p2, v2, p3, v3),
           intersect_or_distance(p3, v3, p1, v1)]
    s = torch.zeros((x1.shape[0], 1))
    if res[0][0] + res[1][0] + res[2][0] == 6:
        s += tri(res[0][1], res[1][1], res[2][1])
    else:
        for flag, r in res:
            if flag != 2:
                s += r
    return s


class MLP_(nn.Module):
    def __init__(self):
        super(MLP_, self).__init__()
        self.fc1 = nn.Linear(256*256, 32*32)
        self.bn1 = nn.BatchNorm1d(32*32)
        self.fc2 = nn.Linear(32*32, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dp3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(256, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, 16)
        self.fc8 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dp3(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dp4(x)
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x


class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.mlp3 = MLP()
        self.mlp4 = MLP_()

    def forward(self, x):
        x1 = self.mlp1(x[:, :32])
        x2 = self.mlp2(x[:, 32:64])
        x3 = self.mlp3(x[:, 64:])
        s = area(x1[:, :2], x2[:, :2], x3[:, :2])
        p = self.mlp4((x1[:, 2:] + x2[:, 2:] + x3[:, 2:]).reshape(-1, 256*256))
        return torch.cat((s, p), dim=1)


# def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
#     """使用GPU计算模型在数据集上的精度"""
#     if isinstance(net, nn.Module):
#         net.eval()  # 设置为评估模式
#         if not device:
#             device = next(iter(net.parameters())).device
#     # 正确预测的数量，总预测的数量
#     metric = d2l.Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(X, list):
#                 X = [x.to(device) for x in X]
#             else:
#                 X = X.to(device)
#             y = y.to(device)
#             metric.add(d2l.accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]


def tri(a, b, d):
    def angle2point(alpha, beta, d):
        return torch.cat(
            (d * torch.sin(beta) * torch.cos(alpha), d * torch.sin(beta) * torch.sin(alpha), d * torch.cos(beta)),
            dim=1)

    def triangle_area(p_1, p_2, p_3):
        a_ = torch.norm(p_2 - p_3, dim=1)
        b_ = torch.norm(p_1 - p_3, dim=1)
        c = torch.norm(p_1 - p_2, dim=1)
        s_ = (a_ + b_ + c) / 2
        area_ = torch.sqrt(s_ * (s_ - a_) * (s_ - b_) * (s_ - c))
        return area_

    p1_ = angle2point(a[:, 0].reshape(-1, 1), b[:, 0].reshape(-1, 1), d[:, 0].reshape(-1, 1))
    p2_ = angle2point(a[:, 1].reshape(-1, 1), b[:, 1].reshape(-1, 1), d[:, 1].reshape(-1, 1))
    p3_ = angle2point(a[:, 2].reshape(-1, 1), b[:, 2].reshape(-1, 1), d[:, 2].reshape(-1, 1))
    s = triangle_area(p1_, p2_, p3_).reshape((-1, 1))
    print(s)
    return s


# def train(net, train_iter, test_iter, num_epochs, lr, device):
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#
#     net.apply(init_weights)
#     print('training on', device)
#     net.to(device)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     loss = nn.MSELoss()
#     animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
#                             legend=['train loss', 'train acc', 'test acc'])
#     timer, num_batches = d2l.Timer(), len(train_iter)
#     for epoch in range(num_epochs):
#         # 训练损失之和，训练准确率之和，样本数
#         metric = d2l.Accumulator(3)
#         net.train()
#         for i, (X, y) in enumerate(train_iter):
#             timer.start()
#             optimizer.zero_grad()
#             X, y = X.to(device), y.to(device)
#             y_hat = net(X)
#             l = loss(y_hat, y)
#             l.backward()
#             optimizer.step()
#             with torch.no_grad():
#                 metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
#             timer.stop()
#             train_l = metric[0] / metric[2]
#             train_acc = metric[1] / metric[2]
#             if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
#                 animator.add(epoch + (i + 1) / num_batches,
#                              (train_l, train_acc, None))
#         test_acc = evaluate_accuracy_gpu(net, test_iter)
#         animator.add(epoch + 1, (None, None, test_acc))
#     print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
#           f'test acc {test_acc:.3f}')
#     print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
#           f'on {str(device)}')
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            # t_s = time.time()
            for X, y in train_iter:
                # with autograd.detect_anomaly():
                optimizer.zero_grad()
                l = loss(net(X), y)
                assert torch.isnan(l).sum() == 0, print(l)
                # print(l)
                # print(l)
                l.backward()
                # nn.utils.clip_grad_norm(net.parameters, 1, norm_type=2)
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix_str(f"loss: {l.item():.3f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print("当前学习率：", current_lr)
        # t_e = time.time()
        # print(epoch, t_e - t_s)
        train_ls.append(loss(net(train_features), train_labels).item())
        if test_labels is not None:
            test_ls.append(loss(net(test_features), test_labels).item())
    return train_ls, test_ls


def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    if is_train:
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    else:
        data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True)

    return data_iter


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
        net = network()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i >= 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available')
else:
    device = torch.device('cpu')
    print('CUDA is not available')
p1 = torch.tensor([[3.247750759124756, 9.432153701782227, 0.6975647807121277]])
p2 = torch.tensor([[-2.249511241912842, 9.743700981140137, 0.0]])
p3 = torch.tensor([[2.916592836380005, -9.539752960205078, 0.6975647807121277]])
loss = nn.MSELoss()
df = pd.read_csv("./data/s2.csv")
l = len(df)
df = df.sample(frac=1).reset_index(drop=True)
# df[:] = df[:].apply(lambda x: (x - x.mean()) / (x.std()))
train_size = int(0.9 * l)
df.iloc[:, 3:] = df.iloc[:, 3:].apply(lambda x: (x - x.mean()) / (x.std()))
df.insert(0, 's', [0.1 for i in range(l)])
train_data = df.iloc[:train_size, 4:]
# train_data_rx1, train_data_rx2, train_data_rx3 = train_data.iloc[:, :32], train_data.iloc[:, 32:64], \
#     train_data.iloc[:, 64:]
test_data = df.iloc[train_size:, 4:]
test = df.iloc[train_size:, :]
test.to_csv("./test.csv", index=False)
train_features = torch.tensor(train_data.values, dtype=torch.float32)
test_features = torch.tensor(test_data.values, dtype=torch.float32)
train_labels = torch.tensor(df.iloc[:train_size, :4].values, dtype=torch.float32)
test_labels = df.iloc[train_size:, :4].values
print('Training set shape:', train_features.shape)
# print('Training set:', train_features)
print('Test set shape:', test_features.shape)
print('Train labels shape:', train_labels.shape)
# print('Train labels:', train_labels)
in_features = int(train_features.shape[1] / 3)

# train_min = 1e31
# for lr in range(1, 1000):
k, num_epochs, lr, weight_decay, batch_size = 5, 500, 0.002, 0.0002, 16
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)

# # if train_l < train_min:
# #     train_min = train_l
# # print(train_min)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


# print("train_min", train_min)

dis2me = lambda x, y: np.linalg.norm(x - y)
error_l = []


def train_and_pred(train_features, test_features, train_labels, test_labels,
                   num_epochs, lr, weight_decay, batch_size):
    net = network()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    preds = net(test_features).cpu().detach().numpy()
    # l = np.sum((preds - test_labels) ** 2, axis=1)
    for i in range(test_labels.shape[0]):
        error = dis2me(preds[i], test_labels[i])
        error_l.append(error)
        print(f'error {i}:', error)
    print('Median error', np.median(error_l))
    print('Average error', np.average(error_l))
    # test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    df_ = pd.DataFrame(preds.reshape((-1, 4)), columns=['s', 'x', 'y', 'z'])
    df_.to_csv('submission.csv', index=False)


# train_and_pred(train_features, test_features, train_labels, test_labels, num_epochs, lr, weight_decay, batch_size)
plt.show()
