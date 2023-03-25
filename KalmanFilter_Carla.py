# https://medium.com/analytics-vidhya/kalman-filters-a-step-by-step-implementation-guide-in-python-91e7e123b968
# https://machinelearningspace.com/object-tracking-python/

# -*- coding=utf-8 -*-
# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
import cv2
import numpy
import numpy as np
import pylab

# 这里是假设A=1，H=1的情况

# 参数初始化
n_iter = 50
sz = (n_iter,)  # size of array
x = -0.37727  # 真实值
z = numpy.random.normal(x, 0.1, size=sz)  # 观测值 ,观测时存在噪声

Q = 1e-5  # process variance

# 分配数组空间
xhat = numpy.zeros(sz)  # x 滤波估计值
P = numpy.zeros(sz)  # 滤波估计协方差矩阵
xhatminus = numpy.zeros(sz)  # x 估计值
Pminus = numpy.zeros(sz)  # 估计协方差矩阵
K = numpy.zeros(sz)  # 卡尔曼增益

R = 0.1 ** 2  # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

kalman = cv2.KalmanFilter(1, 1)
kalman.transitionMatrix = np.array([[1]], np.float32)  # 转移矩阵 A
kalman.measurementMatrix = np.array([[1]], np.float32)  # 测量矩阵    H
kalman.measurementNoiseCov = np.array([[1]], np.float32) * 0.01  # 测量噪声 R
kalman.processNoiseCov = np.array([[1]], np.float32) * 1e-5  # 过程噪声 Q
kalman.errorCovPost = np.array([[1.0]], np.float32)  # 最小均方误差 P

xhat = np.zeros(sz)  # x 滤波估计值
kalman.statePost = np.array([xhat[0]], np.float32)
for k in range(1, n_iter):
    # print(np.array([z[k]], np.float32))
    mes = np.reshape(np.array([z[k]], np.float32), (1, 1))
    # print(mes.shape)
    xhat[k] = kalman.predict()
    kalman.correct(np.array(mes, np.float32))

pylab.figure()
pylab.plot(z, 'k+', label='noisy measurements')  # 观测值
pylab.plot(xhat, 'b-', label='a posteri estimate')  # 滤波估计值
pylab.axhline(x, color='g', label='truth value')  # 真实值
pylab.legend()
pylab.xlabel('Iteration')
pylab.ylabel('Voltage')
pylab.show()
