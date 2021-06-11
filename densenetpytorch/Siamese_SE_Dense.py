from __future__ import division
# from Siamese_Modules import CIND, CIN_diff, CINDLoss, CMD, CMDLoss

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch, numpy

class DenseBottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=32, dropRate=0): # 64, 32 0
        super(DenseBottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1) 
            # SE_BLOCK
            # se = self.global_avg(out)
            # se = se.view(se.size(0), -1)
            # se = self.fc1(se)
            # se = self.relu(se)
            # se = self.fc2(se)
            # se = self.sigmoid(se)
            # se = se.view(se.size(0), se.size(1), 1, 1)
            # out = out * se.expand_as(out)
        return out

class SEBottleneck(nn.Module):
    def __init__(self, inplanes, growthRate=32): # 64, 32 0
        super(SEBottleneck, self).__init__()
        outplanes = inplanes
        self.inp = inplanes
        self.outp = inplanes + growthRate
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(outplanes, outplanes // 16)
        self.fc2 = nn.Linear(outplanes // 16, outplanes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        se = self.global_avg(x)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)
        x = x * se.expand_as(x)
        return x

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,
                               bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.avgpool(out)
        return out

class Siamese_Dense(nn.Module):
    def __init__(self, growthRate=32, head7x7=True, dropOut=.4,
                 increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16), num_classes=256, cind = False):
        """ Constructor
        Args:
            layers: config of layers, e.g., (6, 12, 24, 16)
            num_classes: number of classes
        """
        super(Siamese_Dense, self).__init__()
        self.cind = cind
        block = DenseBottleneck
        se_block = SEBottleneck
        self.growthRate = growthRate
        self.increasingRate = increasingRate
        headplanes = growthRate * pow(increasingRate, 2)
        self.inplanes = headplanes * 2  # default 64
        self.dropRate = 0
        self.head7x7 = head7x7
        if self.head7x7:
            self.conv1 = nn.Conv2d(3, headplanes * 2, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(headplanes * 2)
        else:
            self.conv1 = nn.Conv2d(3, headplanes, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(headplanes)
            self.conv2 = nn.Conv2d(headplanes, headplanes, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(headplanes)
            self.conv3 = nn.Conv2d(headplanes, headplanes * 2, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(headplanes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.red_conv = nn.Conv2d(1024, 256, 1, 1)

        # Dense-Block 1 and transition (56x56)
        self.dense1 = self._make_layer(block, layers[0])
        self.se1 = self._make_se(se_block)
        self.trans1 = self._make_transition(compressionRate)
        self.se_1 = self._make_se(se_block,True)
        
        # Dense-Block 2 and transition (28x28)
        self.dense2 = self._make_layer(block, layers[1])
        self.se2 = self._make_se(se_block)
        self.trans2 = self._make_transition(compressionRate)
        self.se_2 = self._make_se(se_block,True)
        
        # Dense-Block 3 and transition (14x14)
        self.dense3 = self._make_layer(block, layers[2])
        self.se3 = self._make_se(se_block)
        self.trans3 = self._make_transition(compressionRate)
        self.se_3 = self._make_se(se_block,True)
        
        # Dense-Block 4 (14x14)
        self.dense4 = self._make_layer(block, layers[3])
        self.se4 = self._make_se(se_block)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropOut)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct DenseNet
            blocks: number of blocks to be built
        Returns: a Module consisting of n sequential bottlenecks.
        """
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate)) #64, 32, 0
            self.inplanes += self.growthRate
            
        return nn.Sequential(*layers)

    def _make_se(self,block, transi = False):
        layers = []
        layers.append(block(self.inplanes, self.growthRate))
        # if transi == False:
        #     self.inplanes += self.growthRate
        
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        self.growthRate *= self.increasingRate
        return Transition(inplanes, outplanes)

    def _forward_one(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.dense1(x)
        x = self.se1(x)
        x = self.trans1(x)
        x = self.se_1(x)

        x = self.dense2(x)
        x = self.se2(x)
        x = self.trans2(x)
        x = self.se_2(x)

        x = self.dense3(x)
        x = self.se3(x)
        x = self.trans3(x)
        x = self.se_3(x)

        x = self.dense4(x)
        x = self.se4(x)
        x = self.bn(x)
        x = self.relu(x)
        cind = self.red_conv(x)
        #add Activation
        x = self.avgpool(x)
        #dropout Flatten, Dense, Lambda
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x,p=2,dim=1)
        if not self.cind:
            return x
        return x, cind

    def forward(self, x):
        y, cind_y = self._forward_one(x[0])
        z, cind_z = self._forward_one(x[1])

        return [y, z], [cind_y, cind_z]
     

def get_siam_dense(classes, cind= False):
    model = Siamese_Dense(growthRate=32, head7x7=True, dropOut=0,
                        increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16),
                        num_classes=classes,cind = cind)
                # (self, growthRate=32, head7x7=True, dropRate=0,
                # increasingRate=1, compressionRate=2, layers=(6, 12, 24, 16), num_classes=1000)
    
    return model

# model = get_siam_dense(256, cind = True)

# x = torch.rand(64, 3, 224, 224)
# y = torch.rand(64, 3, 224, 224)
# lbl = torch.rand(64, 1)
# x, y = model([x,y])

# cind_x = CIN_diff(x[1], y[1])
# cind_net = CIND()
# cind_y = cind_net(cind_x)
# cind_loss = CINDLoss(cind_y, [lbl, lbl])
# print(cind_loss)

# cmd_net = CMD()
# cmd_y = cmd_net([x[0], y[0]])
# cmd_loss = CMDLoss(cmd_y, [lbl,lbl]) 