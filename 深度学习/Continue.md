# 基于兴趣的GCN学习记录
##  two weeks

### 一、起源
- 在课上接触到深度学习的算法
- 实验室面试提到深度学习有关论文复现的问题
- 试图找一篇论文复现深度学习相关算法

### 二、论文
- Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
- 问题：英语水平不够，只能读懂大致意思，不得要领
- 解决：用Gemini等AI大模型进行辅助学习和翻译，下面是我对此论文的理解：

1.  FAST APPROXIMATE CONVOLUTIONS ON GRAPHS （快速卷积在图上应用）
- 公式：$$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$
- 初始的图卷积定义为：$$g_{\theta} \star x = U g_{\theta}(\Lambda) U^T x$$
 $U$：$L$ 的特征向量矩阵（傅里叶基）。
 $g_{\theta}(\Lambda)$：卷积核，可以看作是对特征值（频率）的函数。
- 采用ChebNet不等式来近似卷积，并进行一阶近似，他假设 $K=1$（只看 1 阶邻居），并令最大特征值 $\lambda_{max} \approx 2$
 公式简化为$$g \star x \approx \theta_0 x + \theta_1 (D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) x$$
- 重正化 $$\tilde{A} = A + I_N, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$
可得：$$\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$
- PS:A为邻接矩阵，也就是每个节点收到邻居的影响，而不是最初单独的存在,D为度矩阵，每个节点的度为列之和
- 下面为代码复现
``` python
 import numpy as np
  # 1. 定义原始邻接矩阵 A (假设 3 个节点，0-1, 1-2 相连)
  A = np.array([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0]
  ], dtype=float)
    
  # 2. Renormalization Trick: 加上自环 (Self-loop)
  # A_tilde = A + I#如果不加单位阵，模型在聚合邻居特征时会丢掉节点自身的特征
  I = np.eye(3) #生成单位矩阵 np.eye 3为维度
  A_tilde = A + I
    
  # 3. 计算度矩阵 D_tilde (每个节点的度是行/列之和)
  # D_tilde 是一个对角矩阵
  d = np.sum(A_tilde, axis=1)#1为列和 0为行和
  D_tilde = np.diag(d)#以d这个向量作为对角线构建矩阵
    
  # 4. 计算对称归一化项: D_tilde^{-1/2}
  # 即对角线元素的 -0.5 次方
  D_inv_sqrt = np.diag(np.power(d, -0.5))
    
  # 5. 最终得到 GCN 的传播矩阵: \hat{A} = D^{-1/2} * A_tilde * D^{-1/2}
  # 这里的乘法是矩阵乘法
  A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
  print("原始 A:\n", A)
  print("\n加上自环后的 A_tilde:\n", A_tilde)
  print("\n最终归一化后的 A_hat:\n", np.round(A_hat, 3))
```

2. 两层的 GCN 模型用于半监督分类
- 论文中公式：$$Z = f(X, A) = \text{softmax} \left( \hat{A} \cdot \text{ReLU} \left( \hat{A} X W^{(0)} \right) W^{(1)} \right)$$
- 第一层（Hidden Layer）：线性变换：$X W^{(0)}$。将输入特征从 $F$ 维降到隐藏层维度 $H$（论文中通常取 16）。图卷积（邻居聚合）：$\hat{A} \times (X W^{(0)})$。每个节点吸纳邻居的信息。非线性激活：通过 $\text{ReLU}$ 函数。
- 第二层（Output Layer）：重复卷积：再次乘以 $\hat{A}$，这意味着信息传播到了 2-hop（两步以外）的邻居。维度映射：通过 $W^{(1)}$ 将特征映射到类别数 $C$。
- 分类层（Softmax）：对每一行（每个节点）做 Softmax，得到属于每个类别的概率。
- 模型每层只使用单个权重矩阵，并通过对邻接矩阵的适当归一化处理不同节点度

3. GCN算法的实现：使用pyTorch
```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GraphConvolution(nn.Module):#在 PyTorch 中，所有的神经网络层都要继承 nn.Module继承它之后，系统会自动帮你处理反向传播（BP）、参数更新和模型保存等底层繁琐工作
        def __init__(self, in_features, out_features):#W的特征维度
            super(GraphConvolution, self).__init__()#找到GC的父类转换对象
            # 对应公式中的 W
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            nn.init.xavier_uniform_(self.weight)#论文中提到的Glorot 初始化，它通过数学手段让每一层输出的方差保持一致，防止训练初期梯度爆炸
        #前向传播
        def forward(self, x, adj):
            # 对应公式: A_hat * X * W
            support = torch.mm(x, self.weight)
            output = torch.spmm(adj, support) # 稀疏矩阵乘法？
            return output
        #x：节点特征矩阵 $X$（形状：$N \times \text{in\_features}$）。
        # adj：预处理好的归一化邻接矩阵 $\hat{A}$（形状：$N \times N$）。
        # torch.mm：标准的稠密矩阵乘法（Matrix Multiplication）。
        # torch.spmm：稀疏矩阵乘法（Sparse Matrix Multiplication）。
        # 重点：因为 $\hat{A}$ 绝大多数位置都是 0，直接用普通矩阵乘法会极其浪费内存。spmm 只计算非零项，这是复现大图数据的关键。
        
```
- 对于 self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
- torch.FloatTensor(in_features, out_features)创建空张量Tensor
- nn.Parameter(...)把矩阵包装到模型中
```python
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        nfeat: 输入特征维度 (Cora 是 1433)
        nhid:  隐藏层维度 (论文推荐 16)
        nclass: 类别数 (Cora 是 7)
        dropout: 失活率 (论文推荐 0.5)
        """
        super(GCN, self).__init__()

        # 定义两个图卷积层
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        
        # 定义 Dropout 率
        self.dropout = dropout

    def forward(self, x, adj):
        # 第一层卷积
        # 计算: A_hat * X * W1
        x = self.gc1(x, adj)
        
        # 激活函数: 增加非线性表达能力
        x = F.relu(x)
        
        # Dropout: 随机丢弃 50% 的神经元防止过拟合
        # 注意: training=self.training 是关键，测试时会自动关闭 Dropout
        x = F.dropout(x, self.dropout, training=self.training)
        
        # 第二层卷积
        # 计算: A_hat * H * W2 (此时输入是隐藏层特征 H)
        x = self.gc2(x, adj)
        
        # 输出层: 使用 log_softmax
        # 配合 NLLLoss (Negative Log Likelihood Loss) 使用，效果等同于 CrossEntropy
        return F.log_softmax(x, dim=1)
```
- Relu进行非线性，扩充模型学习复杂的能力
- https://github.com/tkipf/gcn/tree/master/gcn/data 在这里找到了相关的数据集

### 三、复现论文代码
基于我们刚刚已经完成了GCN算法的实现，即该公式$$Z = f(X, A) = \text{softmax} \left( \hat{A} \cdot \text{ReLU} \left( \hat{A} X W^{(0)} \right) W^{(1)} \right)$$
将结果又应用dropout防止过拟合，下一步将官方的数据集提供的数据进行图的组合并结合论文中给出的数据进行代码复现
组合起来代码如下：
```python
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
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):
    """从原始文件加载拼图"""
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

    # 核心拼图逻辑
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # 标签处理: One-hot 转为类别索引
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # 划分数据集索引
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(adj):
    """A_hat = D^-1/2 * (A + I) * D^-1/2"""
    adj = sp.coo_matrix(adj)
    adj_tilde = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_tilde.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj_tilde).dot(D_inv_sqrt).tocoo()


def sparse_to_tuple(sparse_mx):
    """将 scipy 稀疏矩阵转为 torch 稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# --- 2. 模型定义 ---

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
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
adj = sparse_to_tuple(normalize_adj(adj))
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])  # 转为类别索引

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

# 初始化模型与优化器
model = GCN(nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train(epoch):
    model.train()
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
model.eval()
output = model(features, adj)
preds = output[idx_test].max(1)[1]
acc_test = preds.eq(labels[idx_test]).sum().item() / len(idx_test)
print(f"\nFinal Test Accuracy: {acc_test:.4f}") 
```
- 第一次运行结果如图（cora）数据集
- ![result 1](GCN%20FIRST.png)
- 对应论文结果如图
- ![paper_result](PAPER%20RESULTS.png)
- 由图可知该运行结果与论文结果相近
- citeseer数据集运行结果如图
- 运行时由于一些编码格式（类别索引（1D））和（0ne-hot）不同而维度出错，在AI帮助下修改代码运行成功
- ![result_2](result%202.png)
- 由图可知该运行结果与论文结果相近
PS：一些值得关注的点:
- optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
- 优化器用来优化W权重
