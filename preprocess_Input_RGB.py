import cv2 as cv
import torch
import numpy as np
import os

def generate_diffImg_nd_OF(cap, num_frames):
    list_frames = []
    list_diffImg_OF = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (224, 224))
        list_frames.append(frame)
    list_frames = np.asarray(list_frames)
    N = list_frames.shape[0]
    if (N-1)%(num_frames-1)==0:
        d = (N-1)//(num_frames-1) - 1
    else:
        d = (N-1)//(num_frames-1)
    for i in range(0, N, d):
        if i//d == num_frames:
            break
        frame = list_frames[i]
        nxt_frame = list_frames[i+1]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        nxt_gray = cv.cvtColor(nxt_frame, cv.COLOR_BGR2GRAY)
        diff = nxt_gray - gray
        flow = cv.calcOpticalFlowFarneback(gray, nxt_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        ip = np.asarray([diff, flow_x, flow_y])
        list_diffImg_OF.append(ip)
    list_diffImg_OF = np.asarray(list_diffImg_OF)
    return list_diffImg_OF

def gen_RGB(cap, num_frames):
    list_frames = []
    list_Img = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (256, 256))
        list_frames.append(frame)
    list_frames = np.asarray(list_frames)
    N = list_frames.shape[0]
    if (N-1)%(num_frames-1)==0:
        d = (N-1)//(num_frames-1) - 1
    else:
        d = (N-1)//(num_frames-1)
    for i in range(0, N, d):
        if i//d == num_frames:
            break
        frame = list_frames[i]
        ip = np.asarray([frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]])
        list_Img.append(ip)
    list_Img = np.asarray(list_Img)
    return list_Img



if __name__ == '__main__':
    dir_in = "/home/csci5980/saluj012/RWF-2000-Original-Data/RWF-2000/"
    dir_out = "/home/csci5980/saluj012/ConvLSTM/RWF-2000-ConvLSTM_RGB_IN"

    i = 1
    num_frames = 64
    dir_out = dir_out + "_" + str(num_frames) + "/"
    
    for root, dirs, files in os.walk(dir_in):
        paths = root.split("/")
        for video in files:
            if (video == ".DS_Store"):
                continue
            video = root + '/' + video
            cap = cv.VideoCapture(video)
            ip = gen_RGB(cap, num_frames)
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)
            label = "0"
            if paths[-1] == 'Fight':
                label = "1"
            new_file_name = dir_out + paths[-2] + "_" + label + "_" + str(i) + ".pt"
            i+=1
            torch.save(torch.from_numpy(ip), new_file_name)
            print("Saved: " + new_file_name)

