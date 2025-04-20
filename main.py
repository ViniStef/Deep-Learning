import torch
import numpy as np

list = [[1,2,3],
        [4,5,6]]

tns = torch.Tensor(list)
print(tns)

tns = torch.FloatTensor(list)
print(tns.dtype)
print(tns)

tns = torch.LongTensor(list)
print(tns.dtype)
print(tns)

arr = np.random.rand(3, 4)
tns = torch.from_numpy(arr) # Preserves the original array's type, (float64) in this case
print(arr)
print(tns)

tns1 = torch.ones(2, 3)
tns0 = torch.zeros(4, 5)
tnsr = torch.randn(3, 3)

print(tns1)
print(tns0)
print(tnsr)

#Tensor to numpy
print(type (arr))
arr = tnsr.data.numpy()
print(type (arr))

print(tnsr)
tnsr[0, 2] = -10
print(" ")
print(tnsr)

print(" ")
print(tnsr[:, 2])

print(tnsr[0, 2].size())

tns = tnsr[0:2, :]
print(tns.shape)
print(tns1.shape)

print(tns+tns1)
print(torch.mm(tns1, tns.T))

tns = torch.randn(2,2,3)
print(tns)

print(tns.size())
print(tns.view(tns.size(), -1))

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

tns = tns.to(device)
print(tns)