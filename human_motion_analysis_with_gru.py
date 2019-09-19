import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

class SelfAttentiveEncoder(nn.Module):

    def __init__(self):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(256, 20, bias=False)
        self.ws2 = nn.Linear(20, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
#        self.init_weights()
        self.attention_hops = 1

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp):
        size = outp.size()  # [bsz, len, nhid] 50,25,128
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp) #bsz,hop,nhid

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class A_LSTM(nn.Module):
    def __init__(self):
        super(A_LSTM, self).__init__()
        # input size?
#         self.lnlstm = LNLSTM(30,64,2)#input,output,layer num_layers=1
        self.gru = nn.GRU(input_size=34, hidden_size=128,num_layers=2,bidirectional=True)
        self.bn1 = nn.BatchNorm1d(50)
        #self.selfattention = SelfAttention(128)
        # fix attention output size to 25
        self.selfattention = SelfAttentiveEncoder()
        self.fc1 = nn.Linear(256, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(320, 320)
        self.bn3 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(320, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(34)
        
    def forward(self, x):#input should be length,bsz,30
        length,bsz,feature = x.shape
        x = x.contiguous().view(-1,feature)
        x = self.bn5(x)
        x = x.contiguous().view(length,bsz,feature)
        x,hidden = self.gru(x)
        #output 25,bsz,128
        x = x.permute(0,2,1)#25,128,bsz
        #  x = self.maxpooling(x)#25,128,bsz/2
        #print("pooling",x.shape)
        x = x.permute(2,0,1)#bsz/2,25,128
        #print("pooling",x.shape)
        x = self.bn1(x)
        #print("bn1",x)
        # x = self.selfattention(x).squeeze()
        x = self.selfattention(x).squeeze(1)
        #print("attention",x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn4(x)
        l2_norm = torch.norm(x,p=2,dim=1)#bsz
        return l2_norm 

class MMD_NCA_Net(nn.Module):
    def __init__(self):
        super(MMD_NCA_Net, self).__init__()
        self.A_LSTM = A_LSTM()

    def forward(self, x):
        return self.A_LSTM(x)


