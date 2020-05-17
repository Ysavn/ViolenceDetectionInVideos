import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import models, transforms, datasets
import random
from model import * 
from cell import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class My_Dataset(data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, transform, path):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load(self.path + ID)
        y = self.labels[ID]
        X_ = torch.tensor((), dtype=torch.float32)
        X_ = X_.new_empty((X.shape[0], X.shape[1], 224, 224))

        if self.transform:
            for i in range(X.shape[0]):
                X_[i] = self.transform(X[i])
        return X_, y

def make_dataset(batch_size):
    dir_in = "/home/csci5980/saluj012/TMP/"
    dir_in2 = "/home/csci5980/saluj012/TRAIN/"
    spatial_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: crops[torch.randint(high=10, size=(1, 1)).item()]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    spatial_transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: crops[torch.randint(high=10, size=(1, 1)).item()]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print("Batch size: ", batch_size)

    partition = {'train': [], 'val': []}
    labels = {}
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 5}

    for file in os.listdir(dir_in):
        file_split = file.split("_")
        id = file
        label = file_split[1]
        partition[file_split[0]].append(id)
        labels[id] = torch.tensor([int(label)])

    for file in os.listdir(dir_in2):
        file_split = file.split("_")
        id = file
        label = file_split[1]
        partition[file_split[0]].append(id)
        labels[id] = torch.tensor([int(label)])

    dataloaders_dict = {'train': [], 'test': []}

    
    train_set = My_Dataset(partition['train'], labels,spatial_transform2, dir_in2)
    test_set = My_Dataset(partition['val'], labels,spatial_transform, dir_in)
    dataloaders_dict['train'].append(data.DataLoader(train_set, **params))
    dataloaders_dict['test'].append(data.DataLoader(test_set, **params))

    return dataloaders_dict

def evaluate(model, dataloader_test):
    model.eval()
    cnt_v = 0
    cnt_nv = 0
    running_v = 0
    running_nv = 0
    with torch.no_grad():
        running_corrects = 0
        for inputs, labels in dataloader_test[0]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_model = Variable(inputs.permute(1, 0, 2, 3, 4)).to(device)
            output_label = model(input_model)
            labels = torch.squeeze(labels)
            _, preds = torch.max(output_label, 1)
            print(preds.shape)
            print(labels.data.shape)
            running_corrects += torch.sum(preds == labels.data)
            running_v += torch.sum((preds == labels.data) * (labels.data == 1))
            running_nv += torch.sum((preds == labels.data) * (labels.data == 0))
            cnt_v += torch.sum(labels.data == 1)
            cnt_nv += torch.sum(labels.data == 0)
        accuracy = running_corrects.double() / len(dataloader_test[0].dataset)
        running_v = running_v.item() / cnt_v.item()
        running_nv = running_nv.item() / cnt_nv.item()
    return running_v*100, running_nv*100, accuracy*100

if __name__ == '__main__':
    #model = torch.load("/home/csci5980/saluj012/ConvLSTM/MODEL_75").to(device)
    model = ConvLSTM_Model(mem_size = 256, feature_extract=True, model_name = 'resnet')
    path = torch.load("/home/csci5980/saluj012/ConvLSTM/TrainedModels/Model_140_20:11:13.pt")
    model.load_state_dict(path)
    model.to(device)
    dataloaders_dict = make_dataset(16)
    #acc_v, acc_nv, acc_t = evaluate(model, dataloaders_dict['train'])
    #print("For TRAIN")
    #print("Violence: {}, Non-Violence: {}, Overall: {}".format(acc_v, acc_nv, acc_t))
    #print("")
    acc_v, acc_nv, acc_t = evaluate(model, dataloaders_dict['test'])
    print("For TEST")
    print("Violence: {}, Non-Violence: {}, Overall: {}".format(acc_v, acc_nv, acc_t))

