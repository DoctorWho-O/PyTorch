# -*- coding: utf-8 -*-
import torch

dtype = torch.float
device = torch.device("cpu")

x = torch.FloatTensor([[2, 9, 12, 17, 28, 32, 5]])
y = torch.FloatTensor([[1, 4, 12, 20, 25, 32, 2]])
w = torch.randn(7,7,device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(1000):
   
    y_pred = x.mm(w)
    loss = (y_pred - y).pow(2).sum()
    
    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        
        w.grad.zero_()
        
print(y.mm(w))
