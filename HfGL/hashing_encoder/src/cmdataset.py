
from logging import getLogger

from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import h5py

import scipy.io as sio
import cv2

logger = getLogger()

class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)

def text_transform(text):
    return text

class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        data_name,
        return_index=False,
        partition='train'
    ):
        self.data_name = data_name
        self.partition = partition
        training = 'train' in partition.lower()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = []
        if training:
            trans.extend([transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ])
        else:
            trans.extend([transforms.Compose([
                    # transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ])
        self.trans = trans
        self.return_index = return_index
        self.open_data()

    def open_data(self):
        if self.data_name.lower() == 'cvact_fea':
            data = CVACT_fea(self.partition)
        elif self.data_name.lower() == 'cvusa_fea':
            data = CVUSA_fea(self.partition)

        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
            self.imgs = self.imgs
        else:
            (self.imgs, self.texts, self.labels, root) = data
            self.imgs = Sampler(root, self.imgs)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]

    def __getitem__(self, index):
        image = self.imgs[index]
        text = self.texts[index]
        if isinstance(self.imgs, Sampler):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            text = list(map(lambda trans: trans(text), [text_transform] * len(self.trans)))
        else:
            multi_crops = [image]
            text = [text]

        label = self.labels[index]

        if self.return_index:
            return index, multi_crops, text, label
        return multi_crops, text, label
        # return multi_crops, text, index

    def __len__(self):
        return self.length

def CVACT_fea(partition):
    root = './data/CVACT/'
    data_img_train = sio.loadmat(os.path.join(root, 'ACTdesTrain.mat'))['sat_des'][0:-1]
    data_txt_train = sio.loadmat(os.path.join(root, 'ACTdesTrain.mat'))['grd_des'][0:-1]

    data_img_test = sio.loadmat(os.path.join(root, 'ACTdesTest.mat'))['sat_des']
    data_txt_test = sio.loadmat(os.path.join(root, 'ACTdesTest.mat'))['grd_des']

    labels_train = np.random.randint(0, 2, size=(data_img_train.shape[0], 24))
    labels_test = np.random.randint(0, 2, size=(data_img_test.shape[0], 24))

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img_test.astype(np. float32), data_txt_test.astype(np.float32), labels_test.astype(np.uint8)
    else:
        data_img, data_txt, labels = data_img_train.astype(np.float32), data_txt_train.astype(np.float32), labels_train.astype(np.uint8)
    return data_img, data_txt, labels

def CVUSA_fea(partition):
    root = './data/CVUSA/'
    data_img_train = sio.loadmat(os.path.join(root, 'USAdesTrain.mat'))['sat_des'][0:-1]
    data_txt_train = sio.loadmat(os.path.join(root, 'USAdesTrain.mat'))['grd_des'][0:-1]

    data_img_test = sio.loadmat(os.path.join(root, 'USAdesTest.mat'))['sat_des']
    data_txt_test = sio.loadmat(os.path.join(root, 'USAdesTest.mat'))['grd_des']

    # 随机，无效值
    labels_train = np.random.randint(0, 2, size=(data_img_train.shape[0], 24))
    labels_test = np.random.randint(0, 2, size=(data_img_test.shape[0], 24))

    if 'test' in partition.lower():
        data_img, data_txt, labels = data_img_test.astype(np. float32), data_txt_test.astype(np.float32), labels_test.astype(np.uint8)
    else:
        data_img, data_txt, labels = data_img_train.astype(np.float32), data_txt_train.astype(np.float32), labels_train.astype(np.uint8)
    return data_img, data_txt, labels