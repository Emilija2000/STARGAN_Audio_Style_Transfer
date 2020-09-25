import torch
import torch.nn as nn
import torch.nn.functional as F

#MODEL - DISKRIMINATOR

class Discriminator(nn.Module):
    
    def __init__(self, num_classes):
        super(Discriminator,self).__init__()
        modules = []

        modules.append(nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(8))
        modules.append(nn.ReLU(inplace = True))

        modules.append(nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.ReLU(inplace = True))

        modules.append(nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU(inplace = True))

        modules.append(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.ReLU(inplace = True))

        modules.append(nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(32))
        modules.append(nn.ReLU(inplace = True))
        
        
        modules.append(nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 2, stride = 2))
        modules.append(nn.BatchNorm2d(16))
        modules.append(nn.ReLU(inplace = True))
        
        modules.append(nn.AdaptiveAvgPool2d(output_size = 1))

        self.fc=nn.Linear(in_features=16,out_features=1,bias=False)
        
        self.fc_classif = nn.Linear(in_features=16, out_features = num_classes,bias=False)
        
        self.sequence = nn.Sequential(*modules)
        
    def forward(self,x):
        y = self.sequence(x)
        y = torch.flatten(y,start_dim=1)
        predict_fake = nn.Sigmoid()(self.fc(y))
        predict_genre = nn.Softmax(dim = 0)(self.fc_classif(y))
        return predict_genre, predict_fake
