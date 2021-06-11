from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
import numpy as np

class CMD_BLOCK(nn.Module):
    def __init__(self,emb_s=256):
        super(CMD_BLOCK,self).__init__()
        dim = emb_s*2
        # self.emb_Layer = EmbfixLayer()
        self.L1 = nn.Linear(dim,dim)
        self.activation = nn.ReLU()
        self.L2 = nn.Linear(dim,int(dim/2))
        self.L3 = nn.Linear(int(dim/2),int(dim/4))
        self.L4 = nn.Linear(int(dim/4),1)
        self.dropout = nn.Dropout(0.2)
        self.sig_class = nn.Sigmoid()
    
    def forward(self,x):
        x = torch.cat((x[0],x[1]),dim=1)
        x = self.L1(x) #.type(torch.FloatTensor)
        x = self.activation(x)
        x = self.L2(x)
        x = self.activation(x)
        x = self.L3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.L4(x)
        # x = self.sig_class(x)
        return x

class CIND_Block(nn.Module):
    def __init__(self, dim = 7):
        super(CIND_Block,self).__init__()
        in_ch = 2048 
        n_ch = 256
        self.dim = dim
        self.op_1 = CIN_diff
        self.cnv1 = nn.Conv2d(n_ch, n_ch, 5, 5)
        self.cnv2 = nn.Conv2d(n_ch, n_ch, 3)
        self.cnv3 = nn.Conv2d(n_ch, n_ch, 3)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_ch)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.L1 = nn.Linear(n_ch, 1)
        
    def forward(self, x):

        x = self.op_1(x)
        x = self.cnv1(x)
        x = self.relu(x)
        x = self.cnv2(x)
        x = self.relu(x)
        x = self.cnv3(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.avgpool(x)
        #dropout Flatten, Dense, Lambda
        x = x.view(x.size(0), -1)
        x = self.L1(x)

        return x

class CIND(nn.Module):
    def __init__(self):
        super(CIND, self).__init__()
        cind_block = CIND_Block
        self.f_block = self._make_layers(cind_block)

    def _make_layers(self,block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.f_block(x)
        return x

class CMD(nn.Module):
    def __init__(self, emb_s = 256):
        super(CMD, self).__init__()
        cmd_block = CMD_BLOCK
        self.f_block = self._make_layers(cmd_block)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _make_layers(self,block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.f_block(x)
        return x

class EmbfixLayer(nn.Module):
    def __init__(self):
        super(EmbfixLayer,self).__init__()
        # self.no_img, self.emb_size = x.size()[0], x.size()[1]

    def forward(self, x):
        x = torch.cat((x[0],x[1]),dim=1)
        return x

def CMDLoss(input, target):
    # edit targets
    target_mat = torch.eq(target[0], target[1]).type(torch.float)#.to(torch.device('cuda'))
    loss = nn.BCEWithLogitsLoss()
    return loss(input.view(-1), target_mat)

def CMD_Lbl_maker(target): #(1744, 1)
    if not torch.is_tensor(target):
            target = torch.tensor(target)
    no_img = target.size()[0]
    t = target.view((-1))
    t_v = torch.transpose(t.tile(1,no_img),0,1)
    t_h = t.view(no_img,1).tile(1,no_img).view(no_img*no_img,1)
    t = torch.eq(t_v, t_h).type(torch.FloatTensor).to(torch.device('cuda'))
    return t

def CIN_diff(x): #(32 128 7 7)
    scale = 5
    upsam_op = torch.nn.Upsample(scale_factor=scale,mode='nearest') ## fixit
    pading = torch.nn.ZeroPad2d(2)
    
    # A to B tiling the batch dimension
    x_r = x[0] # 123,123,123
    x_l = x[1] #111,222,333
    
    # tiling in the spacial dimension
    x_l = upsam_op(x_l)
    
    x_r = pading(x_r)
    slice_x = []
    slice_y = []
    for x_i in range(2, x_r.size()[2] -2):
        for x_j in range(2, x_r.size()[3]-2):
            slice_y.append(x_r[:,:,x_i-2:x_i+3,x_j-2:x_j+3])
            # print(x_i, x_j)
            # print(x_r[0,0,x_i-2:x_i+3,x_j-2:x_j+3])        
        slice_x.append(torch.cat(slice_y,dim=3))
        slice_y= []

    x_r = torch.cat(slice_x,dim=2)


    diff = torch.sub(x_l,x_r)

    # # B to A tiling the batch dimension
    # x_l = y #123,123,123
    # x_r = x #111,222,333
    
    # # tiling in the spacial dimension
    # x_l = upsam_op(x_l)
    
    # x_r = pading(x_r)
    # slice_x = []
    # slice_y = []
    # for x_i in range(2, x_r.size()[2] -2):
    #     for x_j in range(2, x_r.size()[3]-2):
    #         slice_y.append(x_r[:,:,x_i-2:x_i+3,x_j-2:x_j+3])     
    #     slice_x.append(torch.cat(slice_y,dim=3))
    #     slice_y= []
    # x_r = torch.cat(slice_x,dim=2)

    # diff = torch.cat((torch.sub(x_l,x_r),diff),dim=1)




    # for x_i in range(2, x_r.size()[2] -2):
    #     for x_j in range(2, x_r.size()[3]-2):
    #         slice_y.append(x_r[:,:,x_i-2:x_i+3,x_j-2:x_j+3])
    #         print(x_i, x_j)
    #         print(x_r[0,0,x_i-2:x_i+3,x_j-2:x_j+3])
    #         print(x_l[0,0,x_i-2:x_i+3,x_j-2:x_j+3])

    return diff

def CINDLoss(input, target):
    target_mat = torch.eq(target[0], target[1]).type(torch.float)#.to(torch.device('cuda'))
    loss = nn.BCEWithLogitsLoss()
    return loss(input.view(-1), target_mat)

# dim = 7
# ch = 128
# emb = 400
# x = torch.arange(0,emb * ch* dim * dim).view([emb, ch, dim, dim]).type(torch.float)
# y = CIND_Block()
# x =  y(x)

