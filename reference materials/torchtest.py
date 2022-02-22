import torch
#x = torch.rand(5, 3)
#print(x)
print(torch.cuda.is_available())

t = torch.tensor([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
print(t)
print(t[:,0])