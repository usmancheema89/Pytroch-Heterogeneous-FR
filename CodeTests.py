
from operator import sub
import numpy as np
import torch

indexes = np.zeros(((5**2)*2), dtype=np.int32).reshape((5**2,2))
indexes = []
no_images = 5
lst_index = 0
for i in range(0,no_images):
    for j in range(i,no_images):        
        # indexes[lst_index,0] = i
        # indexes[lst_index,1] = j
        indexes.append([i,j])
        lst_index += 1
indexes = np.array(indexes,np.int16)
np.random.shuffle(indexes)

print(len(indexes))
# for i in range(indexe.shape[0]):
#     print(i, indexe[i])