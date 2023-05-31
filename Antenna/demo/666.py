import numpy as np
import torch
import torch.nn as nn
from d2l.torch import d2l, PositionalEncoding
from matplotlib import pyplot as plt
from scipy.io import loadmat
annots = loadmat('../data/gateway-pos.mat')
# print(annots)
# d = {'gateway1': np.array([[3.0153759, 3.1666589, 3.3179417, 3.4692247, 3.0190096,
#                             3.1702926, 3.3215754, 3.4728584, 3.022643, 3.173926,
#                             3.325209, 3.476492, 3.0262768, 3.1775599, 3.3288426,
#                             3.4801257],
#                            [9.494461, 9.44237, 9.390279, 9.338188, 9.505013,
#                             9.452923, 9.400831, 9.348741, 9.515567, 9.463476,
#                             9.411385, 9.359294, 9.526119, 9.474029, 9.421937,
#                             9.369846],
#                            [0.9369801, 0.9369801, 0.9369801, 0.9369801, 0.7773699,
#                             0.7773699, 0.7773699, 0.7773699, 0.61775964, 0.61775964,
#                             0.61775964, 0.61775964, 0.45814943, 0.45814943, 0.45814943,
#                             0.45814943]], dtype=np.float32),
#      'gateway2': np.array([[-2.48336, -2.3274608, -2.1715617, -2.0156624, -2.48336,
#                             -2.3274608, -2.1715617, -2.0156624, -2.48336, -2.3274608,
#                             -2.1715617, -2.0156624, -2.48336, -2.3274608, -2.1715617,
#                             -2.0156624],
#                            [9.689713, 9.725705, 9.761697, 9.797689, 9.689713,
#                             9.725705, 9.761697, 9.797689, 9.689713, 9.725705,
#                             9.761697, 9.797689, 9.689713, 9.725705, 9.761697,
#                             9.797689],
#                            [0.24, 0.24, 0.24, 0.24, 0.08,
#                             0.08, 0.08, 0.08, -0.08, -0.08,
#                             -0.08, -0.08, -0.24, -0.24, -0.24,
#                             -0.24]], dtype=np.float32),
#      'gateway3': np.array([[3.1412113, 2.9882026, 2.8351936, 2.682185, 3.1444745,
#                             2.9914658, 2.8384569, 2.6854482, 3.1477375, 2.9947288,
#                             2.8417199, 2.6887112, 3.1510007, 2.997992, 2.844983,
#                             2.6919744],
#                            [-9.453573, -9.500353, -9.5471325, -9.593912, -9.464247,
#                             -9.511026, -9.557806, -9.604586, -9.47492, -9.5217,
#                             -9.56848, -9.615259, -9.485594, -9.532373, -9.579153,
#                             -9.625933],
#                            [0.9369801, 0.9369801, 0.9369801, 0.9369801, 0.7773699,
#                             0.7773699, 0.7773699, 0.7773699, 0.61775964, 0.61775964,
#                             0.61775964, 0.61775964, 0.45814943, 0.45814943, 0.45814943,
#                             0.45814943]], dtype=np.float32)}
d = {'gateway1': np.array([[2.7933638 , 2.9211454 , 3.0489273 , 3.176709  , 2.8017561 ,
        2.9295378 , 3.0573196 , 3.1851013 , 2.8101485 , 2.93793   ,
        3.065712  , 3.1934936 , 2.8185408 , 2.9463224 , 3.0741043 ,
        3.201886  ],
       [4.1057124 , 4.0094223 , 3.9131317 , 3.8168414 , 4.1168494 ,
        4.020559  , 3.9242685 , 3.8279781 , 4.1279864 , 4.031696  ,
        3.9354055 , 3.8391151 , 4.139123  , 4.042833  , 3.9465423 ,
        3.850252  ],
       [0.6748655 , 0.6748655 , 0.6748655 , 0.6748655 , 0.5154743 ,
        0.5154743 , 0.5154743 , 0.5154743 , 0.35608315, 0.35608315,
        0.35608315, 0.35608315, 0.196692  , 0.196692  , 0.196692  ,
        0.196692  ]], dtype=np.float32), 'gateway2': np.array([[-2.8006117 , -2.6649241 , -2.5292363 , -2.3935487 , -2.8138754 ,
        -2.6781878 , -2.5425    , -2.4068124 , -2.8271391 , -2.6914515 ,
        -2.5557637 , -2.4200761 , -2.8404028 , -2.7047153 , -2.5690274 ,
        -2.4333398 ],
       [ 4.029016  ,  4.113803  ,  4.1985903 ,  4.283377  ,  4.0502424 ,
         4.1350293 ,  4.2198167 ,  4.3046036 ,  4.0714684 ,  4.1562552 ,
         4.2410426 ,  4.3258295 ,  4.0926948 ,  4.1774817 ,  4.262269  ,
         4.347056  ],
       [ 1.0192176 ,  1.0192176 ,  1.0192176 ,  1.0192176 ,  0.86118746,
         0.86118746,  0.86118746,  0.86118746,  0.7031573 ,  0.7031573 ,
         0.7031573 ,  0.7031573 ,  0.54512715,  0.54512715,  0.54512715,
         0.54512715]], dtype=np.float32), 'gateway3': np.array([[-0.01480271, -0.17458344, -0.33436418, -0.49414492, -0.0164005 ,
        -0.17618123, -0.33596197, -0.49574268, -0.01799826, -0.17777899,
        -0.33755973, -0.49734044, -0.01959606, -0.17937678, -0.33915752,
        -0.49893826],
       [-4.8682384 , -4.8598647 , -4.8514915 , -4.8431177 , -4.898726  ,
        -4.8903522 , -4.881979  , -4.8736053 , -4.929214  , -4.9208403 ,
        -4.912467  , -4.9040933 , -4.9597015 , -4.951328  , -4.9429545 ,
        -4.934581  ],
       [ 1.1896355 ,  1.1896355 ,  1.1896355 ,  1.1896355 ,  1.0325751 ,
         1.0325751 ,  1.0325751 ,  1.0325751 ,  0.8755148 ,  0.8755148 ,
         0.8755148 ,  0.8755148 ,  0.7184545 ,  0.7184545 ,  0.7184545 ,
         0.7184545 ]], dtype=np.float32)}
x1 = np.sum(d['gateway1'][0]) / 16
x2 = np.sum(d['gateway2'][0]) / 16
x3 = np.sum(d['gateway3'][0]) / 16
y1 = np.sum(d['gateway1'][1]) / 16
y2 = np.sum(d['gateway2'][1]) / 16
y3 = np.sum(d['gateway3'][1]) / 16
z1 = np.sum(d['gateway1'][2]) / 16
z2 = np.sum(d['gateway2'][2]) / 16
z3 = np.sum(d['gateway3'][2]) / 16


print([x1, y1, z1])
print([x2, y2, z2])
print([x3, y3, z3])
# x = torch.ones((4, 3))
# y = 2 * torch.ones((4, 3))
# x = y * x
# print(x)
# p1 = [1, 1, 1]
# p2 = [2, 2, 2]
# pp = dict(zip(p2, p1))
# p = [i - j for i, j in zip(p2, p1)]
# print(p)


# 定义系数矩阵 A 和常数向量 b
# A = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
# b = torch.tensor([-1.0, 0.0, 1.0])
# print(A.shape)
# # 求解最小二乘问题 Ax=b
# x = torch.linalg.solve(A, b)
#
# print("最小二乘解 x:")
# print(x)
# A = torch.randn(3, 3)
# print(A)
# b = torch.randn(3)
# print(b)
# x = torch.linalg.solve(A, b)
# torch.allclose(A @ x, b)
# A = torch.randn(2, 3, 3)
# B = torch.randn(2, 3, 4)
# X = torch.linalg.solve(A, B)
# print(X)
# torch.allclose(A @ X, B)
#
# A = torch.randn(2, 3, 3)
# b = torch.randn(3, 1)
# x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3, 1)
# print(x.shape)
# torch.allclose(A @ x, b)
# b = torch.randn(3)
# x = torch.linalg.solve(A, b) # b is broadcasted to size (2, 3)
# print(x.shape)
# Ax = A @ x.unsqueeze(-1)
# torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
# a = torch.randn(2,3)
# print(a)
# b = torch.ones(2,3)
# print(b[:,0].unsqueeze(-1).shape)
# a = a * b[:,0].unsqueeze(-1)
# print(a)
# def distance_between_lines(p1, v1, p2, v2):
#     # Calculate the direction vector perpendicular to both lines
#     cross_product = torch.cross(v1, v2, dim=1)
#
#     # Calculate the distance between the two lines
#     distance = torch.abs(torch.einsum('ij, ij->i', (p2 - p1, cross_product))) / torch.norm(cross_product, dim=1)
#
#     return distance.unsqueeze(1)
#
#
# print(distance_between_lines(torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32),
#                              torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.float32),
#                              torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.float32),
#                              torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32)))
# a = torch.tensor([[1, 1, 1], [1, 2, 3], [2, 2, 2]])
# b = torch.tensor([[1, 1, 1], [3, 2, 1], [2, 3, 2]])
#
# # 按行比较两个张量是否相同，结果为布尔型张量
# row_equals = torch.all(torch.eq(a, b), dim=1)
#
# # 计算相同行的数量
# num_same_rows = torch.sum(row_equals).item()
#
# print(num_same_rows)
# print(a.eq(b).sum().item())
# dis2me = lambda x, y: np.linalg.norm(x - y)
# print(dis2me(a, b))
# a = np.array([[1, 1, 1], [1, 2, 1]])
# b = np.array([[1, 2, 1], [1, 2, 2]])
#
# # 按行比较两个数组是否相等，结果为布尔型数组
# row_equals = np.all(np.equal(a, b), axis=1)
# print(row_equals.sum().item())
def compute_intersection(p1, v1, p2, v2):
    p1.expand(v1.shape[0], 2)
    p2.expand(v2.shape[0], 2)
    v1_ = v1.unsqueeze(-1).reshape(-1, 2, 1)
    v2_ = v2.unsqueeze(-1).reshape(-1, 2, 1)
    A = torch.cat((v1_, -v2_), dim=2)
    print('A', A.shape)
    b = (p2 - p1).squeeze()
    x = torch.linalg.solve(A, b)
    p = p1 + v1 * x[:, 0].unsqueeze(-1)
    return p


# p1 = torch.zeros((1, 2))
# p2 = torch.ones((1, 2))
# v1 = torch.tensor([[1., 0.]])
# v2 = torch.tensor([[0., -1.]])
# print(compute_intersection(p1, v1, p2, v2))


# @save
# class PositionalEncoding(nn.Module):
#     """位置编码"""
#
#     def __init__(self, num_hiddens, dropout, max_len=6000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         # 创建一个足够长的P
#         self.P = torch.zeros((1, max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#         self.P[:, :, 0::2] = torch.sin(X)
#         self.P[:, :, 1::2] = torch.cos(X)
#
#     def forward(self, X):
#         print(X)
#         X = X + self.P[:, :X.shape[1], :].to(X.device)
#         print(X[0,16])
#         return self.dropout(X)
#
#
# encoding_dim, num_steps = 32, 60
# pos_encoding = PositionalEncoding(encoding_dim, 0)
# pos_encoding.eval()
# X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
# P = pos_encoding.P[:, :X.shape[1], :]
# d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
#          figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])
# plt.show()
import torch
import torch.nn as nn


# class MyModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MyModel, self).__init__()
#         self.eb = nn.Embedding(30, 3)
#         self.positional_encoding = PositionalEncoding(input_dim, 0)  # Positional Encoding 层
#         self.fc = nn.Linear(input_dim, hidden_dim)  # 全连接层
#         self.relu = nn.ReLU()
#         self.output_layer = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = self.positional_encoding(x)  # 应用 Positional Encoding
#         x = self.fc(x)
#         x = self.relu(x)
#         x = self.output_layer(x)
#         return x
#
#
# # 示例数据
# N = 10
# seq_len = 8
# d_model = 16
# input_dim = seq_len * d_model
# hidden_dim = 32
# output_dim = 2
#
# x = torch.randn(N, seq_len, d_model)
#
# # 创建模型并进行前向传播
# model = MyModel(input_dim, hidden_dim, output_dim)
# output = model(x)
#
# print(output.shape)  # 输出: torch.Size([10, 2])
# class PE(nn.Module):
#     """Positional encoding."""
#
#     def __init__(self, num_hiddens, dropout, max_len=6000):
#         super(PE, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         # Create a long enough `P`
#         self.P = d2l.zeros((1, max_len, num_hiddens))
#         X = d2l.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(
#             10000,
#             torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
#             num_hiddens)
#         self.P[:, :, 0::2] = torch.sin(X)
#         self.P[:, :, 1::2] = torch.cos(X)
#
#     def forward(self, X):
#         X = X + self.P[:, :X.shape[1], :].to(X.device)
#         return self.dropout(X)
#
#
# class Encoder(nn.Module):
#     def __init__(self, max_len=6000):
#         super(Encoder, self).__init__()
#         self.eb = nn.Embedding(max_len, 16)
#         self.p = PE(16, 0)
#
#     def forward(self, x):
#         one = torch.randint(1, 2, x.shape)
#         ones = self.eb(one)
#         x_ex = x.unsqueeze(-1)
#         ex = x_ex.expand(ones.shape)
#         x = ex * ones
#         x = self.p(x)
#         return x.reshape(x.shape[0], -1)
#
#
# a = torch.rand((64, 16))
# En = Encoder()
# print(En(a).shape)
def length(p_1, p_2, p_3):
    # print(p_1.shape)
    l1 = torch.sqrt(((p_1 - p_2)**2).sum(dim=1)).reshape((-1, 1))
    l2 = torch.sqrt(((p_2 - p_3)**2).sum(dim=1)).reshape((-1, 1))
    l3 = torch.sqrt(((p_3 - p_1)**2).sum(dim=1)).reshape((-1, 1))
    # print((l1+l2+l3))
    return l1 + l2 + l3


# l1 = torch.tensor([[1, 0]])
# l2 = torch.tensor([[0, 0]])
# l3 = torch.tensor([[1, 1]])
# print(length(l1, l2, l3))
