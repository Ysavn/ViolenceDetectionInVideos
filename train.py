import os
import argparse
import numpy as np
import random
from torch import optim
import time, copy, datetime
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset
from model import *
from torch.utils import data
from ConvLSTM_Dataset import *


def make_dataset(num_frames, batch_size):
    dir_in = "/home/csci5980/saluj012/ConvLSTM/RWF-2000-ConvLSTM_RGB_IN" + "_" + str(num_frames) + "/"
    spatial_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.CenterCrop(224),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: crops[torch.randint(high=10, size=(1, 1)).item()]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print("Batch size: ", batch_size)
    print("Train Transform: ", spatial_transform)
    spatial_transform2 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    print("Test Transform: ", spatial_transform2)

    partition = {'train': [], 'val': []}
    labels = {}
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 1}


    for file in os.listdir(dir_in):
        file_split = file.split("_")
        id = file
        label = file_split[1]
        partition[file_split[0]].append(id)
        labels[id] = torch.tensor([int(label)])

    k_fold = 1
    N_train = len(partition['train'])
    sz_val = 400
    print("Validation Set Size: ", sz_val)
    random.shuffle(partition['train'])
    dataloaders_dict = {'train': [], 'val': [], 'test': []}
    for i in range(k_fold):
        curr_val = partition['train'][i * sz_val:(i + 1) * sz_val]
        curr_train = partition['train'][0:i * sz_val]
        curr_train.extend(partition['train'][(i + 1) * sz_val:])
        curr_train_set = ConvLSTM_Dataset(curr_train, labels, num_frames, spatial_transform)
        curr_val_set = ConvLSTM_Dataset(curr_val, labels, num_frames, spatial_transform)
        dataloaders_dict['train'].append(data.DataLoader(curr_train_set, **params))
        dataloaders_dict['val'].append(data.DataLoader(curr_val_set, **params))

    test_set = ConvLSTM_Dataset(partition['val'], labels, num_frames, spatial_transform2)
    dataloaders_dict['test'].append(data.DataLoader(test_set, **params))

    return dataloaders_dict


def evaluate(model, dataloader_test):
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        for inputs, labels in dataloader_test[0]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            input_model = Variable(inputs.permute(1, 0, 2, 3, 4)).to(device)
            output_label = model(input_model)
            labels = torch.squeeze(labels)
            _, preds = torch.max(output_label, 1)
            running_corrects += torch.sum(preds == labels.data)
            #print(running_corrects)
        accuracy = running_corrects.double() / len(dataloader_test[0].dataset)
    return accuracy*100


def train_model(model, dataloader_dict, criterion, optimizer, optimScheduler, num_epochs):
    since = time.time()
    best_acc = 0
    losses = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloader_dict[phase][0]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                input_model = Variable(inputs.permute(1, 0, 2, 3, 4)).to(device)
                output_label = model(input_model)
                labels = torch.squeeze(labels)

                loss = criterion(output_label, labels)

                prob, preds = torch.max(output_label, 1)
                #print("prob: ", prob)
                #print("preds:", preds)
                #print("output_label", output_label)
                #print("actual_label", labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader_dict[phase][0].dataset)
            epoch_acc = running_corrects.double() / len(dataloader_dict[phase][0].dataset)
            losses[phase].append(epoch_loss)

            print('{}: {} Loss: {:.4f} Acc: {:.2f}'.format(epoch, phase, epoch_loss, epoch_acc*100))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'train':
                optimScheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:2f}'.format(best_acc*100))

    # load best model weights
    model.load_state_dict(best_model_wts)

    print("Test Accuracy: {:2f}".format(evaluate(model, dataloader_dict['test'])))
    return model, losses


if __name__ == '__main__':
    num_epochs = 100
    num_frames = 64
    feature_extract = True
    memSize = 256
    stepSize = 50
    decayRate = 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_size", help="set memory size")
    parser.add_argument("--epochs", help="set number of epochs")
    parser.add_argument("--model_name", help="name of conv model")
    parser.add_argument("--batch_size", help="batch size of input")
    parser.add_argument("--lr", help="learning rate for optimizer")
    parser.add_argument("--weight_decay", help="l2 regularization")
    args = parser.parse_args()
    if args.mem_size:
        memSize = int(args.mem_size)
    if args.epochs:
        num_epochs = int(args.epochs)
    if args.model_name:
        model_name = str(args.model_name).lower()
    if args.batch_size:
        batch_size = int(args.batch_size)
    if args.lr:
        lr = float(args.lr)
    if args.weight_decay:
        weight_decay = float(args.weight_decay)
    dataloader_dict = make_dataset(num_frames, batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM_Model(mem_size=memSize, feature_extract=feature_extract, model_name=model_name)
    #parallelize across the 2 GPUs
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    model.to(device)
    #Testing saved model
    #model.load_state_dict(torch.load('/home/csci5980/saluj012/ConvLSTM/Model_resnet_1_2020-04-29_05:01:21.pt'))
    #acc = evaluate(model, dataloader_dict['test'])
    #print("Test Acc: ",acc)
 
    criterion = nn.CrossEntropyLoss()
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    # optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
    optimizer = optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    print(optimizer)
    optimScheduler = optim.lr_scheduler.StepLR(optimizer, stepSize, decayRate)
    optimizer_name = 'Adam'
    print("Model Name: {}\nEpochs: {}\nFeatureExtract: {}\nOptimizer: {}\n".format(model_name, num_epochs,
                                                                                   feature_extract, optimizer_name))
    best_model, losses = train_model(model, dataloader_dict, criterion, optimizer, optimScheduler, num_epochs)
    date_time = str(datetime.datetime.now())
    date_time = date_time.split(" ")
    date = date_time[0]
    time = date_time[1][0:date_time[1].rfind(":")+3]
    np.save("/home/csci5980/saluj012/ConvLSTM/Accs_" + model_name + "_" + str(num_epochs) + "_" + str(date) + "_" + str(time) + ".npy", losses)
    torch.save(best_model.state_dict(), "/home/csci5980/saluj012/ConvLSTM/Model_" + model_name + "_" + str(num_epochs) + "_" + str(date) + "_" + str(time) + ".pt")
