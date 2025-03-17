from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from torchvision import transforms

img_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456, 0.406), (0.229, 0.224, 0.225))
])


class preProc(Dataset):
    def __init__(self, img_size=(220, 220), data_dir=None, model='CA-SeqNet'):
        self.img_size = img_size
        self.small_batch_test = False
        self.data_dir = data_dir
        self.label = []
        self.npy_list = []
        self.png_list = []
        self.get_list()
        self.model = model

    def get_list(self):
        listdir = os.listdir(self.data_dir)
        for i in range(len(listdir)):
            tem = os.listdir(self.data_dir + '/' + listdir[i])
            for j in range(len(tem)):

                if tem[j][-4:] == '.npy':
                    self.npy_list.append(self.data_dir + '/' + listdir[i] + '/' + tem[j])
                    if listdir[i] == 'mcg':
                        self.label.append(0)
                    if listdir[i] == 'mog_fast':
                        self.label.append(1)
                    if listdir[i] == 'mog_normal':
                        self.label.append(2)
                    if listdir[i] == 'non-artifact':
                        self.label.append(3)
                if tem[j][-4:] == '.png':
                    self.png_list.append(self.data_dir + '/' + listdir[i] + '/' + tem[j])

    def __len__(self):
        return len(self.npy_list)

    def __getitem__(self, index):
        x1 = np.load(self.npy_list[index])
        # Zero mean normalization
        min_x = min(x1)
        max_x = max(x1)
        x = (x1 - min_x) / (max_x - min_x)
        if self.model == 'SeqNet':
            return x, self.label[index]
        if self.model == 'SpaceNet':
            img = cv2.imread(self.png_list[index], cv2.IMREAD_COLOR)
            img = img[:, :, ::-1].copy()
            resized_img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            resized_img = img_transf(resized_img)
            channel = [2, 0, 1]
            resized_img = resized_img[channel]
            return resized_img, self.label[index]

