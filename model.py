import numpy as np
import os
import torch
import random

from torch import optim
from torchvision import models, transforms, datasets
import torch.nn as nn
import time, copy
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, target, optical_flow, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.optical_flow = torch.from_numpy(optical_flow).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        z = self.optical_flow[index]

        if self.transform:
            x = self.transform(x)

        return x, y, z

    def __len__(self):
        return len(self.data)


def get_dataloaders(num_of_frames):
    # training data path
    src_train_fight_path = 'RWF-2000-CroppedOpticalFlow/train/Fight/'
    src_train_non_fight_path = 'RWF-2000-CroppedOpticalFlow/train/NonFight/'

    # validation data path
    src_val_fight_path = 'RWF-2000-CroppedOpticalFlow/val/Fight/'
    src_val_non_fight_path = 'RWF-2000-CroppedOpticalFlow/val/NonFight/'

    num_train_data = len(os.listdir(src_train_fight_path)) + len(os.listdir(src_train_non_fight_path))
    num_val_data = len(os.listdir(src_val_fight_path)) + len(os.listdir(src_val_non_fight_path))

    X_tmp_train = np.empty(shape=[num_train_data, num_of_frames, 4, 224, 224])
    y_tmp_train = np.empty(shape=[num_train_data])

    X_tmp_val = np.empty(shape=[num_val_data, num_of_frames, 4, 224, 224])
    y_tmp_val = np.empty(shape=[num_val_data])

    # input for training data
    idx = 0
    for fight_data in os.listdir(src_train_fight_path):
        X_tmp_train[idx] = np.load(src_train_fight_path + fight_data)
        y_tmp_train[idx] = 1
        idx += 1

    for non_fight_data in os.listdir(src_train_non_fight_path):
        X_tmp_train[idx] = np.load(src_train_non_fight_path + non_fight_data)
        y_tmp_train[idx] = 0
        idx += 1

    # input for validation data
    idx = 0
    for fight_data in os.listdir(src_val_fight_path):
        X_tmp_val[idx] = np.load(src_val_fight_path + fight_data)
        y_tmp_val[idx] = 1
        idx += 1

    for non_fight_data in os.listdir(src_val_non_fight_path):
        X_tmp_val[idx] = np.load(src_val_non_fight_path + non_fight_data)
        y_tmp_val[idx] = 0
        idx += 1

    # final training set
    X_train = np.empty(shape=[num_train_data * num_of_frames, 3, 224, 224])
    y_train = np.empty(shape=[num_train_data * num_of_frames])
    optical_flow_train = np.empty(shape=[num_train_data * num_of_frames, 1, 7, 7])
    idx = 0
    for i in range(num_train_data):
        for j in range(num_of_frames):
            X_train[idx] = X_tmp_train[i][j][0:3]
            y_train[idx] = y_tmp_train[i]
            optical_flow_train[idx] = X_tmp_train[i][j][3][108:115, 108:115] + 1
            idx += 1

    # final validation set
    idx = 0
    X_val = np.empty(shape=[num_val_data * num_of_frames, 3, 224, 224])
    y_val = np.empty(shape=[num_val_data * num_of_frames])
    optical_flow_val = np.empty(shape=[num_val_data * num_of_frames, 1, 7, 7])
    for i in range(num_val_data):
        for j in range(num_of_frames):
            X_val[idx] = X_tmp_val[i][j][0:3]
            y_val[idx] = y_tmp_val[i]
            optical_flow_val[idx] = X_tmp_val[i][j][3][108:115, 108:115] + 1
            idx += 1

    # shuffle train dataset
    temp = list(zip(list(X_train), list(y_train)))
    random.shuffle(temp)
    r1, r2 = zip(*temp)
    X_train = np.asarray(r1)
    y_train = np.asarray(r2)

    # shuffle val dataset
    temp = list(zip(list(X_val), list(y_val)))
    random.shuffle(temp)
    r1, r2 = zip(*temp)
    X_val = np.asarray(r1)
    y_val = np.asarray(r2)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    train_dataset = MyDataset(X_train, y_train, optical_flow_train, data_transforms['train'])
    test_dataset = MyDataset(X_val, y_val, optical_flow_val, data_transforms['val'])

    # Create training and validation dataloaders
    dataloaders_dict = {}
    batch_size = 10
    dataloaders_dict['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                            num_workers=4)
    dataloaders_dict['val'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                          num_workers=4)
    return dataloaders_dict


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    # resize fcc layer
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, optical_flow in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optical_flow = optical_flow.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        model_conv = nn.Sequential(*list(model.children())[0:-2])
                        # conv output
                        output_interim = model_conv(inputs)
                        optical_flow = optical_flow.repeat(1, 512, 1, 1)
                        # feature amplification
                        output_interim = output_interim * optical_flow
                        # passing features through avg pool layer
                        model_avg_pool = nn.Sequential(*list(model.children())[-2:-1])
                        output_interim2 = model_avg_pool(output_interim)
                        output_interim2 = output_interim2.reshape(output_interim2.shape[0], -1)
                        model_fc = nn.Sequential(*list(model.children())[-1:])
                        # passing features through fcc layer
                        outputs = model_fc(output_interim2)
                        print(outputs.shape)
                        loss = criterion(
                            outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':

    num_of_classes = 2
    num_of_frames = 4
    feature_extract = True
    dataloaders_dict = get_dataloaders(num_of_frames)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft, input_size = initialize_model(num_of_classes, feature_extract, True)
    #torch.cuda.set_device(device)
    #model_ft.cuda()

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    model_name = "resnet"

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))
