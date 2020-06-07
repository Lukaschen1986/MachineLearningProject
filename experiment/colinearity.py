# -*- coding: utf-8 -*-
"""
实验	                      线性回归	                  逻辑回归
插入一列完全一样的特征	      原系数被平均分配给两个特征	   两个特征系数相等
插入一列强相关特征（共线性）	强相关特征的系数为0	        两个特征系数相似
"""
import numpy as np
from sklearn.datasets import (make_regression, make_classification)
from sklearn.linear_model import (LinearRegression, LogisticRegression)

# 线性回归
x, y = make_regression(n_samples=1000, n_features=2, n_targets=1, random_state=1)
estimator = LinearRegression(fit_intercept=True)
estimator.fit(x, y)
estimator.coef_ # array([87.91985605, 16.88001901])

# 插入一列完全一样的特征，系数被平分
x = np.concatenate([x, x[:,0].reshape(-1, 1)], axis=1)
estimator.fit(x, y)
estimator.coef_ # array([43.95992803, 16.88001901, 43.95992803])

# 插入一列强相关特征，强相关特征的系数为0
eps = np.random.randn(len(x)) * 0.01
x = np.concatenate([x, (x[:,0]+eps).reshape(-1, 1)], axis=1)
np.corrcoef(x.T) # 0.99995359
estimator.fit(x, y)
estimator.coef_ # array([8.79198561e+01, 1.68800190e+01, 1.05705021e-13])

# 逻辑回归
x, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=1)
estimator = LogisticRegression(fit_intercept=True, max_iter=1000)
estimator.fit(x, y)
estimator.coef_ # array([[ 0.65983474, -0.26241595,  0.54780573,  1.47517278]])

# 插入一列完全一样的特征，系数相等
x = np.concatenate([x, x[:,0].reshape(-1, 1)], axis=1)
estimator.fit(x, y)
estimator.coef_ # array([[ 0.43907773, -0.35922036,  0.66567224,  1.31840149,  0.43907773]])

# 插入一列强相关特征，
eps = np.random.randn(len(x)) * 0.01
x = np.concatenate([x, (x[:,0]+eps).reshape(-1, 1)], axis=1)
np.corrcoef(x.T) # 0.99998098
estimator.fit(x, y)
estimator.coef_ # array([[ 0.42921955, -0.36351032,  0.67088193,  1.31134053,  0.45860676]])
