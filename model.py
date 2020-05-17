import torch.nn as nn
from torchvision import models
from cell import *

class ConvLSTM_Model(nn.Module):

    def initialize_conv_model(self, feature_extract):
        conv_model = self.convNet
        for param in conv_model.parameters():
            param.requires_grad = not feature_extract
    
    def setConvModel(self):
        if self.conv_name == 'alexnet':
            alexnet = models.alexnet(pretrained=True)
            self.convNet = nn.Sequential(*list(alexnet.features.children()))
            self.lin1 = nn.Linear(3*3*self.mem_size, 1000)
        elif self.conv_name == 'resnet':
            resnet18 = models.resnet18(pretrained=True)
            self.convNet = nn.Sequential(*list(resnet18.children())[0:-3])
            self.lin1 = nn.Linear(7*7*self.mem_size, 1000)
        else:
            resnet50 = models.resnet50(pretrained=True)
            self.convNet = nn.Sequential(*list(resnet50.children())[0:-2])

    def __init__(self, mem_size, feature_extract, model_name):
        super(ConvLSTM_Model, self).__init__()
        self.mem_size = mem_size
        self.conv_name = model_name
        self.setConvModel()
        self.initialize_conv_model(feature_extract)
        self.conv_lstm = ConvLSTMCell(self.mem_size, self.mem_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.lin2 = nn.Linear(1000, 256)
        self.lin3 = nn.Linear(256, 10)
        self.lin4 = nn.Linear(10, 2)
        self.BN1 = nn.BatchNorm1d(1000)
        self.BN2 = nn.BatchNorm1d(256)
        self.BN3 = nn.BatchNorm1d(10)
        self.drop = nn.Dropout()
        self.classifier = nn.Sequential(self.drop, self.lin1, self.BN1, self.relu, self.drop, self.lin2, self.BN2, self.relu, self.drop, self.lin3, self.BN3, self.relu, self.lin4)

    def forward(self, x):
        state = None
        seqLen = x.size(0) - 1
        for t in range(0, seqLen):
            x1 = x[t] - x[t+1]
            #x1 = x[t]
            x1 = self.convNet(x1)
            state = self.conv_lstm(x1, state)
        x = self.maxpool(state[0])
        print(x.type())
        x = self.classifier(x.view(x.size(0), -1))
        return x
