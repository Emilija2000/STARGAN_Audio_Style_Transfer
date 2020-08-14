import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    '''
    Convolutional neural network with fully connected layer as its output layer.
    4 "up" (more output channels) and 2 "down" layers.
    Softmax is applied on the output -> probabilities.
    Classifing output_num classes of input data.
    '''
    def __init__(self, output_num):
        super(Classifier,self).__init__()
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

        self.fc = nn.Linear(in_features=16,out_features=output_num,bias=True)
        
        self.sequence = nn.Sequential(*modules)


    def forward(self,x):
        y = self.sequence(x)
        y = torch.flatten(y,start_dim=1)
        y = self.fc(y)
        return nn.Softmax(dim = 1)(y)
        