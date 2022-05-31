import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from cv2 import transform
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class ImageTransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


image_file_path = 'data/goldenretriever-3724972_640.jpg'
img = Image.open(image_file_path)

resize = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = ImageTransform(resize, mean, std)
img_transformed = transform(img, phase='train')

img_transformed = img_transformed.numpy().transpose((1,2,0))
img_transformed = np.clip(img_transformed, 0,1)
# plt.imshow(img_transformed)
# plt.show()


def make_datapath_list(phase="train"):

    rootpath = "./data/hymenoptera_data/"
    target_path = rootpath+phase+'/**/*.jpg'
    
    

    path_list = []  # ここに格納する

    # globを利用してサブディレクトリまでファイルパスを取得する
    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


class HymenopterDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]
        
        if label == 'ants':
            label = 0
        else: 
            label = 1

        return img_transformed, label

train_dataset = HymenopterDataset(
    file_list = make_datapath_list('train'),
    transform = transform,
    phase='train',
)

val_dataset = HymenopterDataset(
    file_list = make_datapath_list('val'),
    transform = transform,
    phase = 'val',
)

batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

net = models.vgg16(pretrained=True)

net.train()

for param in net.parameters():
    param.requires_grad = False

net.classifier[6] = nn.Linear(net.classifier[6].in_features, 1)

net.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.classifier[6].parameters())

num_epochs = 2

for epoch in range(num_epochs):
    net.train()

    train_loss = 0.0
    train_corrects = 0
    train_num = 0

    optimizer.zero_grad()

    if (epoch != 0): # no train at epoch 0 to check the performance with no train.
        for inputs, labels in tqdm(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            train_num += inputs.size(0)
            outputs = net(inputs).squeeze()
            loss = criterion(outputs, labels.float())

            preds = torch.where(outputs > 0, 1, 0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels)

    net.eval()

    eval_loss = 0.0
    eval_corrects = 0
    eval_num = 0

    for inputs, labels in tqdm(val_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        eval_num += inputs.size(0)
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, labels.float())

        preds = torch.where(outputs > 0, 1, 0)

        eval_loss += loss.item() * inputs.size(0)
        eval_corrects += torch.sum(preds == labels)

    if epoch:
        print(f"epoch: {epoch + 1}  train loss: {train_loss/train_num}  train acc: {train_corrects.double()/train_num}  eval loss: {eval_loss/eval_num}  eval acc: {eval_corrects.double()/eval_num}")

    else:
        print(f"epoch: {epoch + 1} eval loss: {eval_loss/eval_num}  eval acc: {eval_corrects.double()/eval_num}")
