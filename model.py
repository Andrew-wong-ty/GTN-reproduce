import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):
    
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        self.linear1 = nn.Linear(self.w_out*self.num_channels, self.w_out)
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def gcn_conv(self,X,H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            temp = (torch.eye(H.shape[0])==0).type(torch.FloatTensor)
            temp = temp.cuda()
            H = H*temp
            # H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            temp1 = (torch.eye(H.shape[0])==0).type(torch.FloatTensor)
            temp1 = temp1.cuda()
            temp2 = torch.eye(H.shape[0]).type(torch.FloatTensor)
            temp2 = temp2.cuda()
            H = H*temp1 + temp2

            #H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        temp = torch.eye(H.shape[0]).type(torch.FloatTensor)
        temp = temp.cuda()
        deg_inv = deg_inv*temp
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        A = A.unsqueeze(0).permute(0,3,1,2) # [1,n_edges,n_nodes, n_nodes]
        Ws = []
        for i in range(self.num_layers):  # k*GTLayer
            if i == 0:
                H, W = self.layers[i](A)  # H即为 A(1)
            else:
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)   # 此时H就是A^l,  shape=[n_channel, n_node, n_node]
        for i in range(self.num_channels):  # 产生的新的meta-path, 用gcn 卷积
            if i==0:
                X_ = F.relu(self.gcn_conv(X,H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X,H[i]))
                X_ = torch.cat((X_,X_tmp), dim=1)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_=None):
        if self.first == True:
            a = self.conv1(A) # Q1(1)
            b = self.conv2(A) # Q2(1)
            H = torch.bmm(a,b) # 图卷积结果相乘, 得到A(1)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(),(F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            a = self.conv1(A)  # 得到最后一个conv的结果, 
            H = torch.bmm(H_,a) #与前2个layer 的output相乘
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H,W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
