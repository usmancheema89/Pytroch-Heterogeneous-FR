
from operator import sub
import numpy as np
import torch

def CrossInputNDiff(x, dim):

    op_1 = torch.nn.Upsample(scale_factor=dim,mode='nearest')
    x = torch.cat((torch.sub(op_1(x),x.tile(1,1,dim,dim)), 
                torch.sub(x.tile(1,1,dim,dim),op_1(x))), dim=1)
    

    return x

dim = 7
batch = 2
ch = 1024

values = dim * dim * ch * batch
x = torch.tensor( np.arange(0, values).reshape((batch,ch,dim,dim)) ).type(torch.FloatTensor)

y_pred = CrossInputNDiff(x, dim)

print(y_pred)


## input x
    # u_sample = torch.nn.ConvTranspose2d(1, 1, 3, stride=3, padding=0,output_padding=0,padding_mode='zeros', bias=False)
    # w = torch.tensor([[1,1,1],[0,0,0],[0,0,0]]).type(torch.FloatTensor)
    # w = w.view(1, 1, 3, 3).repeat(1, x.size()[1], 1, 1)
    # u_sample.requires_grad_(False)
    # u_sample.weight = torch.nn.Parameter(w)
    # x_3  = u_sample(x)
    # print(x[0])
    # print(x_3[0])