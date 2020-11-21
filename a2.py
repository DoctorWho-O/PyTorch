import numpy as np

x = np.array([1,2,3,4,5,6])   #输入数据
y = np.array([150])           #输出数据
w = np.array([1.0,1.0,1.0,1.0,1.0,1.0]).reshape(6,1)  #权重，6条数据，每条数据1个

learning_rate = 1e-6  
for t in range(100000):
	pred = x.dot(w)  #输入点乘权重得出的结果
	loss = np.square(pred - y).sum() 
	print(t,loss)  

        #更新权重的思路是，使用得出结果与输出数据的两倍差值是1x1，点乘权重6x1，做T处理变为1x6.当被乘数的数据条数与乘数每条多少个一致可点乘。乘以learning_rate以后得出一个1x6列的小的差距，用权重减去这个小的差距慢慢调整
	grad = 2.0 * (pred-y) 
	grad_w = x.reshape(6,1).dot(grad).reshape(6,1) #只有一行的矩阵.T不能变为纵列，不然只需要x.T.dot(grad)
	w -= learning_rate * grad_w

print(w)
print(x.dot(w))