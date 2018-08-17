import numpy as np
import matplotlib.pyplot as plt

def input_data(txt_filepath):
    data = np.loadtxt(txt_filepath)
    return data

def my_scatter(dataMat):
    '''
    绘制原始数据的散点图
    :param dataMat:
    :return:
    '''
    x = np.array(dataMat[:, 1])
    y = np.array(dataMat[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)

    plt.savefig('original.png')

    plt.show()

def my_scatter_end(dataMat,theta_result):
    '''
    绘制散点图及回归图像
    :param dataMat:
    :return:
    '''
    x = np.array(dataMat[:, 1])
    y = np.array(dataMat[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y)

    m = np.arange(0,1,0.05)
    n = theta_result[0]+theta_result[1] * m
    plt.plot(m,n)

    plt.savefig('LinearRegression.png')

    plt.show()


def computeCost(X, y, theta):
    inner = np.power(((X*np.transpose(theta))-y), 2)
    return np.sum(inner)/(2*(len(y)))

def compute_theta(theta_init, step, X, y):
    temp = ((X*np.transpose(np.mat(theta_init))) - y)
    theta_init[0] = theta_init[0] - (step * np.sum(temp)/(len(y)))
    theta_init[1] = theta_init[1] - np.sum(step * (np.transpose(X[:,1])*temp )/(len(y)))
    return theta_init

def result(theta_init, n, X, y, step):
    '''
    输出每次更新的theta值和Cost损失
    :param theta_init: theta的初始化值
    :param n:训练次数
    :param X:训练数据
    :param y:对应的标签
    :param step:学习效率
    :return:
    '''
    i = 0
    while i < n:
        Cost = computeCost(X, y, np.mat(theta_init))
        print("参数theta为：", theta_init, "损失为：", Cost)
        theta_init = compute_theta(theta_init, step, X, y)
        i = i + 1

    return theta_init



