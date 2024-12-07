# @inproceedings{HM-TDF,  
#   title={Hard Sample Mining-Based Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HM-TDF}  
# }  

import torch
import torch.nn as nn
import sys
import os
script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)
from KANLinear import KANLinear as KAN_Linear
from CNNKAN import CNNKan as CNN_Kan

class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []

        self.IE = CNN_Kan()
        self.RR = nn.Sequential(nn.BatchNorm1d(512),
                                nn.Dropout(drop_rate),
                                nn.Linear(512, 32),
                                nn.BatchNorm1d(32),
                                nn.ReLU(),
                                nn.Dropout(drop_rate),
                                nn.Linear(32, 1))


    def forward(self, X, f_p):
        f_i = self.IE(X)
        out = self.RR(f_i)
        return out
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )

# define net structure
def get_net(num_features, num_labels, drop_rate):
    net = MffKan(num_labels, num_features, drop_rate)
    return net
