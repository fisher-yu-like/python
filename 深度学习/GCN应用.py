import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 1. 数据准备
def generate_data(num_nodes=200):
    adj = torch.zeros((num_nodes, num_nodes))
    adj[:100, :100] = (torch.rand(100, 100) > 0.9).float()
    adj[100:, 100:] = (torch.rand(100, 100) > 0.9).float()
    adj += (torch.rand(num_nodes, num_nodes) > 0.998).float()
    adj += torch.eye(num_nodes)  # 自环

    x = torch.randn(num_nodes, 16)  # 初始随机特征
    labels = torch.tensor([0] * 100 + [1] * 100)  # 真实标签

    # 创建训练掩码：每组只选 5 个点作为已知标签进行训练
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:10] = True
    train_mask[100:110] = True

    return x, adj, labels, train_mask


# 2. 定义 GCN 模型
class DeepGCN(nn.Module):
    def __init__(self, in_feat, h_feat, out_feat, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        # 堆叠很多层
        self.layers.append(nn.Linear(in_feat, h_feat))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(h_feat, h_feat))
        self.layers.append(nn.Linear(h_feat, out_feat))

    def forward(self, x, adj):
        for layer in self.layers:
            # 每一层都在做 A * X * W
            x = torch.relu(layer(torch.mm(adj, x)))
        return x


# 3. 训练流程
x, adj, labels, train_mask = generate_data()
model = DeepGCN(16, 8, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("开始训练...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(x, adj)
    loss = criterion(out[train_mask], labels[train_mask])  # 只根据已知标签算误差
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred = out.argmax(dim=1)
        acc = (pred == labels).float().mean()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | 全局准确率: {acc:.2%}")

# 4. 可视化最终学习到的表征
model.eval()
with torch.no_grad():
    final_emb = model(x, adj)
    # 使用 t-SNE 降维
    z = TSNE(n_components=2).fit_transform(final_emb.numpy())

plt.scatter(z[:100, 0], z[:100, 1], c='red', label='Class 0', alpha=0.6)
plt.scatter(z[100:, 0], z[100:, 1], c='blue', label='Class 1', alpha=0.6)
plt.title("GCN Semi-supervised Classification Result")
plt.legend()
plt.show()