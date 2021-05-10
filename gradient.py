import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 线性回归
# def jj(x):
# 	return (x-1)**2+4
# def djj(x):
# 	return (x-1)*2
# def grad_des(x0,eta,erro=1e-8,n=1e4):
# 	x = x0
# 	i = 0
# 	while i < n:
# 		grad = djj(x)
# 		last_x = x
# 		x = x - grad*eta
# 		if(abs(jj(x)-jj(last_x)))<erro:
# 			break
# 		i+=1
# 	print("循环次数",i,"学习率",eta)
# 	return x
# x = grad_des(0,0.05)

#多元线性回归MSE
def J(theta,x_b,y):
    return np.sum((y-x_b.dot(theta))**2)/len(y)
def dJ(theta,x_b,y):
    return x_b.T.dot(x_b.dot(theta)-y) * 2 /len(x_b)

def gradient_descent1(x_b, y, eta, theta_initial, erro=1e-8, n=1e4):
    theta = theta_initial
    i = 0
    while i < n:
        gradient = dJ(theta, x_b, y)
        last_theta = theta
        theta = theta - gradient * eta
        if (abs(J(theta, x_b, y) - J(last_theta, x_b, y))) < erro:
            break
        i += 1
    return theta
def predict(x_t,theta1):
    X_b=np.hstack([np.ones((len(x_t),1)),x_t])
    return X_b.dot(theta1)
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.array(range(2009,2016))
x = (x-x.min())/(x.max()-x.min())
x = x.reshape(-1,1)
X = x**2
y = np.array([0.5,9.36,33.6,191,350.19,571,912])
# X = np.hstack([x, x2])

x_b = np.hstack([np.ones((len(X),1)),X])
theta0 = np.zeros(x_b.shape[1])
print(theta0,x_b)
eta = 0.01
theta1 = gradient_descent1(x_b, y, eta,  theta0)
y_pre=[]
for i in x_b[:,1]:
    y_pre.append(predict([[i]], theta1))
# print (theta1,'\n',[j for i in x_b[:,1] for j in i])
print(y_pre)

# plt.plot(range(2009,2016),y)
plt.plot(range(2009,2016),y_pre,color='red')
plt.scatter(range(2009,2016),y)
plt.show()