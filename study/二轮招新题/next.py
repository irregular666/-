


import numpy as np
import torch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块




# 重新初始化数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 定义线性回归模型
class LinearRegressionModel(nn.Module): #nn.module是pytorch中所有类型神经网络的集合，这里初始化以调用其中的线性回归功能
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1) #线性回归输出一个标量，所以维数为1

    def forward(self, x): #定义前向传播
        return self.linear(x) #这里调用了前面已经定义的线性回归函数

# 实例化模型、损失函数和优化器
model = LinearRegressionModel(num_inputs)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) #lr:学习率，控制梯度下降的步长

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad() #清除每一轮得出的梯度
    output = model(features) #前面定义的线性回归模型，进行前向传播
    l = loss(output.view(-1), labels) #调整为一维
    l.backward()
    optimizer.step() #更新一次参数

# 获取训练后的权重和偏置（weight是权重，bias是偏置）
trained_w = model.linear.weight.detach().numpy()
trained_b = model.linear.bias.detach().numpy()

print(trained_w, trained_b)
