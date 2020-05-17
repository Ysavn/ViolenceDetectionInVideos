import torch
from torch import Tensor
from torch.utils import data


class ConvLSTM_Dataset(data.Dataset):
    # 'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, num_frames, transform):
        self.labels = labels
        self.num_frames = num_frames
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        X = torch.load('/home/csci5980/saluj012/ConvLSTM/RWF-2000-ConvLSTM_RGB_IN_' + str(self.num_frames) + "/" + ID)
        y = self.labels[ID]
        X_ = torch.tensor((), dtype=torch.float32)
        X_ = X_.new_empty((X.shape[0], X.shape[1], 224, 224))
        idx = torch.randint(high=10, size = (1,1)).item()
        if self.transform:
            for i in range(X.shape[0]):
                X_[i] = self.transform(X[i])
        return X_, y
