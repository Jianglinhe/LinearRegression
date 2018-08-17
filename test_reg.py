import reg

# 1.读数据文件,并对数据预处理(txt的数据的第一列都是1)
data = reg.input_data('ex0.txt')
X = data[:,0:2]
y = data[:,2:3]

print(X.shape)
print(y.shape)

# 2.绘制数据的散点图，并进行观察，确定目标函数为线性回归目标函数y = theta0 + theta1 * x
reg.my_scatter(data)

# 3.初始化参数（theta值，学习效率step,以及使用梯度下降法的训练次数）
theta_init = [1,3]
step = 0.5
n = 100

# 4.得到训练的结果
theta_result = reg.result(theta_init, n, X, y, step)

# 5、绘制图像
reg.my_scatter_end(data,theta_result)



