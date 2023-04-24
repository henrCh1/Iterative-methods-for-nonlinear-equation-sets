# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:45:38 2023

@author: 86319
"""

import numpy as np

# 定义非线性方程组
def F(x):
    u = x[0]**2 + x[1]**2 - 1
    v = x[0]**3 - x[1]
    return np.array([u, v])

# 定义Jacobi矩阵
def J(x):
    J11 = 2*x[0]
    J12 = 2*x[1]
    J21 = 3*x[0]**2
    J22 = -1
    return np.array([[J11, J12], [J21, J22]])

# 定义牛顿迭代法
def newton(F, J, x0, tol, maxiter):
    # x0：初始点
    # tol：精度要求
    # maxiter：最大迭代次数
    for i in range(maxiter):
        # 计算Jacobi矩阵的逆和非线性函数的值，并求解线性方程组
        dx = np.linalg.solve(J(x0), -F(x0))
        # 求出新的迭代点
        x1 = x0 + dx
        # 判断是否满足精度要求
        if np.linalg.norm(dx) < tol:
            break
        # 更新迭代点
        x0 = x1
    # 返回最终的迭代点
    return x1

# 调用函数求解
x0 = np.array([0.8, 0.6])
tol = 1e-6
maxiter = 3

x = newton(F, J, x0, tol, maxiter)
print("方程组的解为 ", x)
