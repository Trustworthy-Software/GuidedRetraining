"""
Adapted from: https://github.com/HobbitLong/SupContrast
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
    


class encoder(nn.Module):
    """encoder"""
    def __init__(self, dim_in):
        super(encoder, self).__init__()
        
        self.lin1 = nn.Linear(dim_in, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()      
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.lin1(x)))
        out = self.relu2(self.bn2(self.lin2(out)))
        out = self.relu3(self.bn3(self.lin3(out)))
        out = self.relu4(self.bn4(self.lin4(out)))
        out = self.relu5(self.bn5(self.lin5(out)))
        return out 
    
    
class Model_supcon(nn.Module):
    """encoder + Projection"""
    def __init__(self, dim_in=65):
        super(Model_supcon, self).__init__()
        
        self.encoder = encoder(dim_in)

        self.head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                #nn.Dropout(p=0.2),
                nn.Linear(64, 32)
            )

    def forward(self, x):
        out = self.encoder(x)
        out = self.head(F.normalize(out))
        return out 
    
    
    
class Model_linear(nn.Module):
    """classifier"""
    def __init__(self, dim_in=128, num_classes=2):
        super(Model_linear, self).__init__()
        self.lin1 = nn.Linear(dim_in, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.relu3 = nn.ReLU()
        self.lin4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.relu4 = nn.ReLU()
        self.lin5 = nn.Linear(8, 2)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        out = self.relu1(self.bn1(self.lin1(x)))
        out = self.relu2(self.bn2(self.lin2(out)))
        out = self.relu3(self.bn3(self.lin3(out)))
        out = self.relu4(self.bn4(self.lin4(out)))
        out = self.sig(self.lin5(out))
        return out

    
