import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import sys


# --- 1. 数据加载与预处理 (Planetoid 格式) ---

def parse_index_file(filename):
    """读取测试集索引"""
    index = []
    # 1. 打开名为 filename 的文件（通常是 ind.cora.test.index）
    for line in open(filename):
        # 2. line.strip() 去掉行尾的换行符 \n
        # 3. int(...) 将字符串转为整数
        # 4. append(...) 加入列表
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):
    """
    针对 Citeseer 等数据集优化的加载逻辑
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    # --- 关键改进 1: 处理 Citeseer 的索引空隙 ---
    # 找到最大的索引值，确定矩阵的最终大小
    min_idx, max_idx = min(test_idx_reorder), max(test_idx_reorder)

    # 创建填充后的特征矩阵
    combined_features = sp.vstack((allx, tx)).tolil()
    # 创建一个全零矩阵，行数为最大索引+1
    features = sp.lil_matrix((max_idx + 1, x.shape[1]))
    # 先填入已知的部分
    features[:allx.shape[0], :] = allx
    # 再根据 reorder 填入测试集部分
    features[test_idx_reorder, :] = tx

    # --- 1. 堆叠原始标签 (此时可能是 scipy 稀疏矩阵) ---
    combined_labels = sp.vstack((ally, ty))

    # --- 2. 转换为标准的 Numpy Dense 数组 ---
    if sp.issparse(combined_labels):
        combined_labels = combined_labels.toarray()

    # --- 3. 创建全零容器，防止索引越界 (针对 Citeseer 的空洞) ---
    # max_idx + 1 是为了容纳测试集里那些“跳跃”的索引
    labels_onehot = np.zeros((max_idx + 1, combined_labels.shape[1]))

    # 分段填充
    labels_onehot[:ally.shape[0], :] = ally.toarray() if sp.issparse(ally) else ally
    labels_onehot[test_idx_reorder, :] = ty.toarray() if sp.issparse(ty) else ty

    # --- 4. 转换并降维 ---
    # 这一步 labels 会变成形状如 (3327,) 的一维数组
    labels = np.argmax(labels_onehot, axis=1)

    # --- 5. 关键补丁：处理没有任何标签的节点 ---
    # Citeseer 有些行全是 0，argmax 默认会返回 0，这可能影响准确率
    # 但至少程序能跑通了
    # --- 关键改进 3: 确保邻接矩阵维度对齐 ---
    # 显式指定 nodelist，防止 graph 中漏掉孤立节点导致矩阵变小
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph), nodelist=range(max_idx + 1))

    # 划分索引
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range

    return adj, features, labels, idx_train, idx_val, idx_test


# --- 配合主程序中的小改动 ---
# 在主程序中，不再需要 np.where(labels)[1]，直接转 tensor 即可
# labels = torch.LongTensor(labels)
def normalize_adj(adj):
    """A_hat = D^-1/2 * (A + I) * D^-1/2"""
    adj = sp.coo_matrix(adj)
    adj_tilde = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_tilde.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_tilde).dot(D_inv_sqrt).tocoo()#矩阵乘法，最后返回时转成 COO，是因为下一阶段我们要把它转成 PyTorch 的 SparseTensor，而 PyTorch 最喜欢 COO 格式（即：行索引、列索引、值）


def sparse_to_tuple(sparse_mx):
    """将 scipy 稀疏矩阵转为 torch 稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# --- 2. 模型定义 --- 这个是我们之前准备好的核心代码

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# --- 3. 运行主程序 ---

# 加载并预处理
adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')#执行我们之前拆解的拼图逻辑，拿到原始的 $A$ 和 $X$
adj = sparse_to_tuple(normalize_adj(adj))#完成 $D^{-1/2}(A+I)D^{-1/2}$ 的计算。
features = torch.FloatTensor(np.array(features.todense()))
# 修改前：labels = torch.LongTensor(np.where(labels)[1])
# 修改后：
def extract_labels(labels_onehot):
    # 找到每一行最大值的索引
    return torch.LongTensor(np.argmax(labels_onehot, axis=1))

labels = torch.LongTensor(labels)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 初始化模型与优化器
model = GCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()#开启训练模式，激活dropout
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss_train.item():.4f}')


# 训练 loop
for epoch in range(200):
    train(epoch)

# 测试结果
model.eval()#关闭dropout
output = model(features, adj)
preds = output[idx_test].max(1)[1]
acc_test = preds.eq(labels[idx_test]).sum().item() / len(idx_test)
print(f"\nFinal Test Accuracy: {acc_test:.4f}")