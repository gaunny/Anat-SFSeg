import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch import sigmoid

class ResPointNetfeat(nn.Module):
    def __init__(self,use_conv=True):
        super(PointNetfeat, self).__init__()
        # 3-layer MLP (via 1D-CNN) : encoder points individually
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128,1024, 1)
        if use_conv:
            self.conv4 = torch.nn.Conv1d(3,128,1)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))#torch.Size([batch_size, 64, 15]) 
        x = self.bn2(self.conv2(x))
        if self.conv4:
            identity = self.conv4(identity)
        x += identity
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x
class ResPointNetCls(nn.Module):
    def __init__(self, k=2):  #log264=6   log2161 = 8  log2572 = 10  log2 16 = 4 log2 58 = 6
        super(ResPointNetCls, self).__init__()
        #self.emb_layer1 = nn.Embedding(8,120)  #emb_dim=8  swm 
        self.emb_layer1 = nn.Embedding(34,120)   #emb_dim=34 dwm
        self.emb_layer2 = nn.Embedding(105,7)
        self.feat1 = ResPointNetfeat()
        # 3 fully connected layers for classification
        # self.fc1 = nn.Linear(1024, 1024)
        self.fc1 = nn.Linear(1024+120+7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, input1,input2,input3):  #Input1 is point，input2 is cluster-level anatomical features input3 is individual-level anatomical features
        input2 = input2.int()
        input2 = input2.view(input2.size()[0],1) #(batch_size,)  region
        input3 = torch.transpose(input3, 2,1)  #(batch_size,105,1)  ind_region
        input3_ = torch.squeeze(input3)
        emb1 = self.emb_layer1(input2)
        emb1 = emb1.mean(dim=1)
        input_tensor = input3_.long()
        emb2 = self.emb_layer2(input_tensor)
        emb2 = emb2.mean(dim=1) 
        
        z = torch.cat([self.feat1(input1),emb1],dim=1)
        z = torch.cat([z,emb2],dim=1)
        z = F.relu(self.bn1(self.fc1(z)))
        z= F.relu(self.bn2(self.fc2(z)))
        z = F.relu(self.bn3(self.dropout(self.fc3(z))))
        z = self.fc4(z)
       
        return F.log_softmax(z, dim=1)  
