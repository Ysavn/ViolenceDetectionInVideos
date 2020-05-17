import numpy as np
import torch
from torchvision import transforms
from torch.utils import data
import cv2 as cv
from model import *
from cell import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
import time
#from google.colab.patches import cv2_imshow

class Test_Dataset(data.Dataset):
    def __init__(self, X, transform):
        self.X = X
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X_ = torch.tensor((), dtype=torch.float32)
        X_ = X_.new_empty((self.X[index].shape[0], self.X[index].shape[1], 224, 224))
        if self.transform:
            for i in range(self.X[index].shape[0]):
                X_[i] = self.transform(self.X[index][i])
        return X_

#path = torch.load("/content/drive/My Drive/MODEL_75.75")
#model = ConvLSTM_Model(mem_size = 256, feature_extract=True, model_name = 'resnet')
#model.load_state_dict(path)
video = './video_v_21.avi'
list_frames = []
cap = cv.VideoCapture(video)
cap2 = cap
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.resize(frame, (256, 256))
    list_frames.append(frame)
list_frames = np.asarray(list_frames)
N = list_frames.shape[0]
spatial_transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

partition = {'train': [], 'val': []}
labels = {}
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader, ith):
    #model.eval()
    #model.eval()
    model.eval()
    with torch.no_grad():
        running_corrects = 0
        for inputs in test_loader:
            #print(inputs.shape)
            inputs = inputs.to(device)
            input_model = Variable(inputs.permute(1, 0, 2, 3, 4)).to(device)
            output_label = model(input_model)
            output_prob = F.softmax(output_label)
            #print(output_prob)
            pred_prob, pred_label = torch.max(output_label, 1)
            #print(ith, pred_prob.item(), pred_label)
    return output_prob[0, 1].item()

def show_video(cap, model):
    list_frames = []
    list_preds = []
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        frame = frame2
        frame = cv.resize(frame, (256, 256))
        list_frames.append(frame)
        if(len(list_frames)==64):
            test_data = np.asarray(list_frames)
            test_data = test_data.reshape(1, 64, 3, 256, 256)
            test_set = Test_Dataset(torch.from_numpy(test_data), spatial_transform2)
            test_loader = data.DataLoader(test_set, **params, pin_memory=False)
            prediction = evaluate(model, test_loader, 0)
            labels = ['Violence', 'Non-Violence']
            x = np.arange(len(labels))
            y = []
            if prediction < 0.5:
                prediction = 1 - prediction
                y = [0, prediction]
            else:
                y = [prediction, 0]
            list_preds.append(prediction)
            plt.bar(x , y, width=0.35, color='r')
            ax = plt.gca()
            #ax.set_xlim(0, 0.1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            #plt.savefig('plot.png')
            #img = cv.imread('./plot.png')
            frame2 = cv.resize(frame2, (360, 360))
            #img = cv.resize(img, (300, 360), interpolation = cv.INTER_AREA)
            list_frames.pop(0)
            cv2_imshow(frame2)
            print(prediction)
            #cv2_imshow(img)
    list_preds = np.asarray(list_preds)
    preds = torch.from_numpy(list_preds)
    torch.save(preds, './preds.pt')

def show_video_hack(cap):
    list_frames = []
    list_preds = torch.load('./preds_v_21.pt')
    print(list_preds.shape)
    stride = 1
    id = 0
    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        frame = frame2
        frame = cv.resize(frame, (256, 256))
        list_frames.append(frame)
        if (len(list_frames) == 64):
            test_data = np.asarray(list_frames)
            test_data = test_data.reshape(1, 64, 3, 256, 256)
            test_set = Test_Dataset(torch.from_numpy(test_data), spatial_transform2)
            test_loader = data.DataLoader(test_set, **params, pin_memory=False)
            #prediction = evaluate(model, test_loader, 0)
            labels = ['Violence', 'Non-Violence']
            x = np.arange(len(labels))
            y = []
            prediction = list_preds[id].item()
            print(prediction)
            if prediction < 0.5:
                prediction = 1 - prediction
                y = [0, prediction]
            else:
                y = [prediction, 0]
            plt.bar(x , y, width=0.35, color='r')
            ax = plt.gca()
            #ax.set_xlim(0, 0.1)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            plt.savefig('plot' + str(id) + '.png')
            img = cv.imread('plot' + str(id) + '.png')
            frame2 = cv.resize(frame2, (360, 360))
            img = cv.resize(img, (300, 360), interpolation=cv.INTER_AREA)
            for j in range(stride):
                list_frames.pop(0)
            im_h = cv.hconcat([frame2, img])
            cv.imshow("window", im_h)
            cv.waitKey(1)
            id+=1
            if id==1:
                time.sleep(3)
            plt.clf()
            plt.close()
            #plt.pause(0.02)
            #cv.imshow("window", img)
    list_preds = np.asarray(list_preds)
    preds = torch.from_numpy(list_preds)
    #torch.save(preds, './preds.pt')

#model.to(device)
cap = cv.VideoCapture(video)
show_video_hack(cap)
#evaluate(model, test_loader, i)