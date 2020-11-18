import torch
x = torch.ones(1,1,5,5)
conv = torch.nn.Conv2d(1,1,2)
out = conv(x)
print(out)
print(list(conv.parameters()))