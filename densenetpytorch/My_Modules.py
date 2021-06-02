from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

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
        # x = self.emb_Layer(x)
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
        x = torch.cat((
            x.view(1,x.size()[0],x.size()[1]).tile((x.size()[0],1,1)),
            x.tile(1,1,x.size()[0]).reshape(x.size()[0], x.size()[0],x.size()[1])
            ),2).reshape(-1,x.size()[1]*2)
        return x

def CMDLoss(input, target):
    # edit targets
    no_img = target.size()[0]
    t_v = torch.transpose(target.tile(1,no_img),0,1)
    t_h = target.view(no_img,1).tile(1,no_img).view(no_img*no_img,1)
    target_mat = torch.eq(t_v, t_h).type(torch.FloatTensor).to(torch.device('cuda'))
    loss = nn.BCEWithLogitsLoss()
    return loss(input, target_mat)

# cmd = CMD(4)

# x = torch.tensor([[1, 1, 1, 1], [2, -100, -100, 2],[3, 1, 1, 3], [4, -100, -100, 4]])
# y_true = torch.tensor([1, 2, 1, 2]).type(torch.FloatTensor)
# y_pred = cmd(x)

# print(x)
# print(y_true)
# print(y_pred)

# loss = CMDLoss(y_pred,y_true)
# print(loss)
