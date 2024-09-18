import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sparse
from torch.nn.parameter import Parameter

class DNN(nn.Module):
    
    def __init__(self, input_dim, hidden_dim,device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim,hidden_dim)            # fully connected layer: maps last hidden vector to model prediction
        self.activation1 = nn.ReLU()                # coz binary classification
        self.drop=nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim,1)
        self.activation2 = nn.Sigmoid()  
        self.device=device



    def forward(self, x):
        e = self.drop(self.activation1(self.fc1(x)))
        out=self.activation2(self.fc2(e))
        return out
    



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(),nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(),nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
  
        channel_att_x = self.mlp_x(x)
        channel_att_g = self.mlp_g(g)
        # print(channel_att_g.shape)

        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        
        scale = torch.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out
    

class COVID_Pred(nn.Module):
    def __init__(self,input_dim,latent,graph_conv,device):
        super().__init__()
        self.cca = CCA(input_dim,latent)
        self.model = graph_conv
        self.HISTORY=Parameter(torch.fmod(torch.randn(1,latent).to(device), 2)) 
        
    def forward(self, x):
        H=self.HISTORY.repeat(len(x),1)
        h=self.cca(x,H)
        # print(h.shape)
        out=self.model(h)
        return out

    def get_emb(self, x):
        H=self.HISTORY.repeat(len(x),1)
        h=self.cca(x,H)
        # print(h.shape)
        return h,h


