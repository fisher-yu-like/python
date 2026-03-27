import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# 记录开始时间
start_time = time.time()

# --- 这里放置你的程序代码 ---
# 模拟程序运行

# -------------------------


# 1. 定义残差块 (Basic Block)
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 快捷连接 (Shortcut)：如果输入输出维度不同，用 1x1 卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 核心：将输入直接加到输出上
        out = torch.relu(out)
        return out

# 2. 定义微型 ResNet (针对 CIFAR-10)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        # 1. 生产第一个块（处理尺寸缩减和通道增加）
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes  # 更新状态：之后的输入通道数变成了当前输出的通道数

        # 2. 生产后续的块（保持尺寸和通道不变）
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = F.avg_pool2d(out, 32) # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- 2. 封装训练过程 ---
def train_model(num_blocks_list, epochs=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    history = {} # 用于存储每个模型的 loss 记录

    for n in num_blocks_list:
        print(f"\n开始训练 num_blocks = {n} 的模型...")
        net = ResNet(BasicBlock, [n]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # 降低点学习率让曲线更平滑

        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(trainloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        history[f'num_blocks={n}'] = epoch_losses

    return history

# --- 3. 执行实验并绘图 ---
num_blocks_to_test = [1,2,3,4,5]
results = train_model(num_blocks_to_test, epochs=5) # 建议跑10轮看趋势
# 记录结束时间
end_time = time.time()

# 计算差值
duration = end_time - start_time
print(f"程序运行耗时: {duration:.4f} 秒")
plt.figure(figsize=(10, 6))
for label, losses in results.items():
    plt.plot(range(1, len(losses)+1), losses, marker='o', label=label)

plt.title('ResNet Convergence: Impact of num_blocks')
plt.xlabel('Epoch')
plt.ylabel('Average CrossEntropy Loss')
plt.legend()
plt.grid(True)
plt.show()

