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
