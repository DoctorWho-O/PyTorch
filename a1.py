import numpy as np

x = np.random.randn(10,100)   #x输入有10条数据,每条数据100个
y = np.random.randn(10,1)     #y输出有10条数据，每条数据1个 最后比较x和y修正中间的权重，使x接近输出y

w1 = np.random.randn(100,10)  #权重w1，有100条数据，每条数据有10个
w2 = np.random.randn(10,1)    #权重w2，有10条数据，每条数据1个

learning_rate = 1e-6 #乘以次数以后会变为一个0.00x的小数
for t in range(50000):    #经过500次相似迭代
	h = x.dot(w1)   #x和w1矩阵相乘相加后会变成10条数据，每条10个
	h_relu = np.maximum(h,0)   #正数保留,负数变为0
	y_pred = h_relu.dot(w2)    #10条数据每条10个与10条数据每条1个相乘变为10个数据

	loss = np.square(y_pred - y).sum() #每条数据都先平方然后相加
	print(t,loss)
 
	#总体思想应该是把相差的值乘回去跟w1比较
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h_relu.T.dot(grad_y_pred)  #.T改变矩阵方向，横着的一列数据变为竖着的。
        
	grad_h_relu = grad_y_pred.dot(w2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0
	grad_w1 = x.T.dot(grad_h)

	#更新w1，w2
	w1 -= learning_rate * grad_w1
	w2 -= learning_rate * grad_w2


#打印看下效果
h = x.dot(w1)
h_relu = np.maximum(h,0)
y_pred = h_relu.dot(w2)

print(y)
print(y_pred)
  