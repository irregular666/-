
#(2,)，表示这是一个一维数组，且该维度的大小为2。元组中的逗号表示这是一个单元素元组，
# 用于明确指出这是一个一维数组，而不是一个简单的数字。

#x_train.shape[0] 这一行代码的意思是：从 x_train.shape 返回的元组中取出第一个元素（即数组的长度）。
import numpy as np

import matplotlib.pyplot as plt


x_train=np.array([1.0,2.0])

print(x_train.shape)
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
