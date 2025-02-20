# import torch
# import torch_directml
#
# # 确认 DirectML 是否可用
# print("DirectML 是否可用:", torch_directml.is_available())
#
# # 创建 DirectML 设备
# device = torch_directml.device()
#
# # 创建 PyTorch 张量并移动到 DirectML 设备
# x = torch.tensor([1.0, 2.0, 3.0], device=device)
# y = torch.tensor([4.0, 5.0, 6.0], device=device)
#
# # 进行计算
# z = x + y
#
# # 如果需要同步，可以调用 `.cpu()` 来避免与设备直接交互时的问题
# z_numpy = z.cpu().numpy()
#
# # 打印结果（已同步到 CPU 上）
# print("计算结果:", z_numpy)


# import torch
# import torch_directml
# import torch.nn as nn
# import torch.optim as optim
#
# # 确保 DirectML 是否可用
# print("DirectML 是否可用:", torch_directml.is_available())
#
# # 创建 DirectML 设备
# device = torch_directml.device()
#
# # 创建一个简单的神经网络（MLP）
# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
#         self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  # ReLU 激活函数
#         x = self.fc2(x)  # 输出层
#         return x
#
# # 网络的输入、隐藏和输出层大小
# input_size = 10
# hidden_size = 20
# output_size = 2
#
# # 创建神经网络模型并移动到 DirectML 设备
# model = SimpleMLP(input_size, hidden_size, output_size).to(device)
#
# # 创建随机的输入和目标数据
# input_data = torch.randn(64, input_size, device=device)  # 64 个样本
# target_data = torch.randint(0, 2, (64, output_size), dtype=torch.float32, device=device)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
#
# # 训练过程
# num_epochs = 5
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(input_data)
#     loss = criterion(outputs, target_data)
#
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # 打印损失值
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
#
# # 使用模型进行预测
# with torch.no_grad():
#     test_input = torch.randn(10, input_size, device=device)  # 10 个测试样本
#     test_output = model(test_input)
#     print("预测结果:", test_output.cpu().numpy())  # 同步到 CPU 打印结果
#
#


# import torch
# import torch_directml
# import torch.nn as nn
# import torch.optim as optim
#
# # 检查 DirectML 是否可用
# device = torch_directml.device()
# print("DirectML 是否可用:", torch_directml.is_available())
#
#
# # 定义一个简单的多层感知器（MLP）
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# # 模型超参数
# input_size = 784  # 输入维度
# hidden_size = 512  # 隐藏层大小
# output_size = 10  # 输出类别数
# batch_size = 128  # 批量大小
# epochs = 5  # 训练轮数
#
# # 初始化模型、损失函数和优化器
# model = MLP(input_size, hidden_size, output_size).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练过程
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0.0
#
#     # 使用随机生成的数据
#     for _ in range(100):  # 每个epoch进行100次迭代
#         inputs = torch.randn(batch_size, input_size).to(device)  # 随机生成输入数据
#         targets = torch.randint(0, output_size, (batch_size,)).to(device)  # 随机生成目标标签
#
#         optimizer.zero_grad()  # 清除之前的梯度
#         outputs = model(inputs)  # 前向传播
#         loss = criterion(outputs, targets)  # 计算损失
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#
#         total_loss += loss.item()
#
#     # 打印每个epoch的平均损失
#     print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / 100:.4f}")
#
# # 模拟模型训练后的输出预测
# inputs = torch.randn(10, input_size).to(device)
# predictions = model(inputs)
# print("预测结果:", predictions)


import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置设备为 DirectML
device = torch_directml.device()

# 模拟一些数据
input_size = 784  # 输入特征的大小（例如28x28的图像）
hidden_size = 128  # 增加隐藏层的神经元数量
output_size = 10  # 输出类别数量

# 构造数据集
X = torch.randn(1000, input_size, device=device)  # 1000个样本
y = torch.randint(0, output_size, (1000,), device=device)  # 1000个标签

# 使用 DataLoader 加载数据
batch_size = 64
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 构建一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建模型实例
model = SimpleNN(input_size, hidden_size, output_size).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # 反向传播
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 优化步骤
        optimizer.step()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# 预测
model.eval()  # 切换为评估模式
with torch.no_grad():
    test_input = torch.randn(10, input_size, device=device)  # 10个测试样本
    predictions = model(test_input)
    _, predicted_labels = torch.max(predictions, 1)
    print("预测结果:", predicted_labels)


