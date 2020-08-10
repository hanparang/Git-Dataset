# 넘파이로 텐서 만들기(벡터와 행렬 만들기)

import numpy as np
import torch

# 1D with Numpy
test1 = np.array([0., 1., 2., 3., 4., 5., 6.])
print(test1)

print("Rank of test1 : ",test1.ndim)
print("Shape of test1 : ",test1.shape)

# index slicing
print("test1[2:5] = ", test1[2:5])

# 2D with Numpy
test2 = np.array([[1., 2., 3.], 
                  [4., 5., 6.], 
                  [7., 8., 9.],
                  [10., 11., 12.]])
print('Rank of test2: ', test2.ndim)
print('Shape of test2: ', test2.shape)

# 1D with PyTorch
test3 = torch.FloatTensor([0, 1, 2, 3, 4, 5, 6])
print(test3.dim)
print(test3.shape)
print(test3.size())

# 2D with PyTorch
test4 = torch.FloatTensor([[1., 2., 3.], 
                           [4., 5., 6.], 
                           [7., 8., 9.],
                           [10., 11., 12.]])
print(test4.dim)
print(test4.shape)
print(test4.size())

# Broadcasting
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1+m2)

m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2]])
print(m1+m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

"""
브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되었는가?
[1, 2]
==> [[1, 2],
     [1, 2]]
[3]
[4]
==> [[3, 3],
     [4, 4]]
     자동으로 시행하기 때문에 주의해서 사용해야한다.
"""

# Matrix Multiplication Vs. Multiplication

# 행렬곱셈(Matrix Multiplication)
m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print(m1.matmul(m2))

# element-wise 곱셈
print(m1.mul(m2)) # or m1*m2

# 평균(mean)
test = torch.FloatTensor([1, 2])
print(test.mean())