import pandas as pd
import torch
import torch.nn as nn
from d2l.torch import d2l, PositionalEncoding
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dp3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 128)
        self.dp4 = nn.Dropout(0.1)
        self.fc5 = nn.Linear(128, 256)
        self.dp5 = nn.Dropout(0.1)
        self.fc6 = nn.Linear(256, 256)
        self.dp6 = nn.Dropout(0.1)
        self.fc7 = nn.Linear(256, 128)
        self.dp7 = nn.Dropout(0.1)
        self.fc8_1 = nn.Linear(128, 1)
        self.fc8_2 = nn.Linear(128, 128)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        residual = x
        x = self.dp2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dp3(x)
        x = torch.relu(self.fc4(x))
        x = x + residual
        x = self.dp4(x)
        x = torch.relu(self.fc5(x))
        x = self.dp5(x)
        x = torch.relu(self.fc6(x))
        x = self.dp6(x)
        x = torch.relu(self.fc7(x))
        # x = self.dp7(x)
        alpha = self.fc8_1(x)
        # feature_map = self.fc8_2(x)
        feature_map = x
        return torch.cat((alpha, feature_map.reshape(-1, 128)), dim=1)


class MLP_(nn.Module):
    def __init__(self):
        super(MLP_, self).__init__()
        self.fc1 = nn.Linear(128 * 3, 128)
        self.dp1 = nn.Dropout(0.10)
        self.fc2 = nn.Linear(128, 128)
        self.dp2 = nn.Dropout(0.10)
        self.fc3 = nn.Linear(128, 64)
        self.dp3 = nn.Dropout(0.10)
        self.fc4 = nn.Linear(64, 64)
        self.dp4 = nn.Dropout(0.10)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 16)
        self.fc7 = nn.Linear(16, 16)
        self.fc8 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dp1(x)
        x = torch.relu(self.fc2(x))
        x = self.dp2(x)
        x = torch.relu(self.fc3(x))
        # x = self.dp3(x)
        x = torch.relu(self.fc4(x))
        x = self.dp4(x)
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = self.fc8(x)
        return x


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.mlp3 = MLP()
        self.mlp4 = MLP_()
        self.encoder = Encoder()

    def forward(self, x):
        x1 = self.mlp1(x[:, :16])
        x2 = self.mlp2(x[:, 16:32])
        x3 = self.mlp3(x[:, 32:])
        s = area(x1[:, :-128], x2[:, :-128], x3[:, :-128])
        p = self.mlp4(torch.cat((x1[:, 1:], x2[:, 1:], x3[:, 1:]), dim=1).reshape(-1, 128 * 3))
        return torch.cat((s, p), dim=1)


# class PositionalEncoding(nn.Module):
#     def __init__(self, num_hiddens, dropout, max_len=6000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.P = torch.zeros((max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
#             torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#         self.P[:, 0::2] = torch.sin(X)
#         self.P[:, 1::2] = torch.cos(X)
#
#     def forward(self, x):
#         x = x + self.P[:x.shape[0], :].to(device)
#         return self.dropout(x)
class PE(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout, max_len=6000):
        super(PE, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
            10000,
            torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
            num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        # print(X.shape)
        # print(self.P[:, :X.shape[1], :].shape)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class Encoder(nn.Module):
    def __init__(self, max_len=6000):
        super(Encoder, self).__init__()
        self.eb = nn.Embedding(max_len, 16)
        self.p = PE(16, 0)

    def forward(self, x):
        one = torch.randint(1, 2, x.shape).to(device)
        ones = self.eb(one)
        x_ex = x.unsqueeze(-1)
        ex = x_ex.expand(ones.shape)
        x = ex * ones
        x = self.p(x)
        return x.reshape(x.shape[0], -1)


def compute_area(p1, p2, p3):
    v1x = p2[:, 0] - p1[:, 0]
    v1y = p2[:, 1] - p1[:, 1]
    v2x = p3[:, 0] - p1[:, 0]
    v2y = p3[:, 1] - p1[:, 1]

    area = 0.5 * torch.abs(v1x * v2y - v1y * v2x).reshape(-1, 1)

    return area


def are_lines_parallel(v1, v2):
    cross_product = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return torch.all(torch.eq(cross_product, torch.zeros_like(cross_product)))


def compute_intersection(p1, v1, p2, v2):
    p1.expand(v1.shape[0], 2)
    p2.expand(v2.shape[0], 2)
    if are_lines_parallel(v1, v2):
        print('parallel')
        return 0
    v1_ = v1.unsqueeze(-1).reshape(-1, 2, 1)
    v2_ = v2.unsqueeze(-1).reshape(-1, 2, 1)
    A = torch.cat((v1_, -v2_), dim=2)
    b = (p2 - p1).squeeze()
    x = torch.linalg.solve(A, b)
    p = p1 + v1 * x[:, 0].unsqueeze(-1)
    return p


def radian2vector(x):
    alpha = x.reshape((-1, 1))
    return torch.cat((torch.cos(alpha), torch.sin(alpha)), dim=1)


def area(x1, x2, x3):
    v1 = radian2vector(x1)
    v2 = radian2vector(x2)
    v3 = radian2vector(x3)
    s = torch.zeros((x1.shape[0], 1)).to(device)
    res = [compute_intersection(p1, v1, p2, v2), compute_intersection(p2, v2, p3, v3),
           compute_intersection(p3, v3, p1, v1)]
    for r in res:
        if isinstance(r, int):
            return torch.full(s.shape, 1.e1).to(device)
    # s += compute_area(res[0], res[1], res[2])
    s = length(res[0], res[1], res[2])
    # print(s)
    return s


def length(p_1, p_2, p_3):
    # print(p_1.shape)
    l1 = torch.sqrt(((p_1 - p_2)**2).sum(dim=1)).reshape((-1, 1))
    l2 = torch.sqrt(((p_2 - p_3)**2).sum(dim=1)).reshape((-1, 1))
    l3 = torch.sqrt(((p_3 - p_1)**2).sum(dim=1)).reshape((-1, 1))
    # print((l1+l2+l3))
    return l1 + l2 + l3


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)
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
            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available')
else:
    device = torch.device('cpu')
    print('CUDA is not available')
p1 = torch.tensor([[3.247750759124756, 9.432153701782227]]).to(device)
p2 = torch.tensor([[-2.249511241912842, 9.743700981140137]]).to(device)
p3 = torch.tensor([[2.916592836380005, -9.539752960205078]]).to(device)
loss = nn.MSELoss()
df = pd.read_csv("./data/s2.csv")
l = len(df)
df = df.sample(frac=1).reset_index(drop=True)
# df[:] = df[:].apply(lambda x: (x - x.mean()) / (x.std()))
train_size = int(0.9 * l)
df.iloc[:, 3:] = df.iloc[:, 3:].apply(lambda x: (x - x.mean()) / (x.std()))
df.insert(0, 's', [0. for i in range(l)])
train_data = df.iloc[:train_size, 5::2]
# train_data_rx1, train_data_rx2, train_data_rx3 = train_data.iloc[:, :32], train_data.iloc[:, 32:64], \
#     train_data.iloc[:, 64:]
test_data = df.iloc[train_size:, 5::2]
test = df.iloc[train_size:, :]
test.to_csv("./test.csv", index=False)
train_features = torch.tensor(train_data.values, dtype=torch.float32).to(device)
test_features = torch.tensor(test_data.values, dtype=torch.float32).to(device)
train_labels = torch.tensor(df.iloc[:train_size, :3].values, dtype=torch.float32).to(device)
test_labels = df.iloc[train_size:, :3].values
# train_features = torch.cat((p(train_features[:, :16]),
#                             p(train_features[:, 16:32]),
#                             p(train_features[:, 32:48])),
#                            dim=1)
print('Training set shape:', train_features.shape)
# print('Training set:', train_features)
print('Test set shape:', test_features.shape)
print('Train labels shape:', train_labels.shape)
# print('Train labels:', train_labels)
in_features = int(train_features.shape[1] / 3)
k, num_epochs, lr, weight_decay, batch_size = 5, 500, 0.00002, 0.000001, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
