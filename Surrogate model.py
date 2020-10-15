import numpy as np
from numpy import random as r
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as opt
import torch.autograd as autog
import torchvision as tv

from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader

### temporary ###
import os   # instead conda install nomkl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### temporary ###

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # add more layers

        self.deconv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=9, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.deconv2 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, h, thresholds=[]):

        #s = h.shape
        #print(s)
        #print(thresholds[0])
        #h = torch.cat((h, thresholds[0].cuda().unsqueeze(dim=1)), dim=1)   # concatenate threshold tensor to current layer
        #h = h.reshape([s[0], s[1], s[2], s[3]])

        h = self.conv1(h)
        h = self.conv2(h)
        # residual blocks or conv layers
        h = self.deconv1(h)     # starting deconvolution
        h = self.deconv2(h)
        return h

    def train(self, epoch):
        tr_loss = 0
        x_train, y_train = autog.Variable(train_x), autog.Variable(train_y)
        #x_val, y_val = autog.Variable(val_x), autog.Variable(val_y)

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda()
        #x_val = x_val.cuda()
        #y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        #x_train = x_train.unsqueeze(0)
        #y_train = y_train.unsqueeze(0)
        output_train = net(x_train, [t1, t2])
        #output_val = net(x_val)

        # computing the training and validation loss
        y_train = y_train
        loss_train = criterion(output_train, y_train)
        #loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        #val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss += loss_train.item()
        return output_train


class CannyDataset(Dataset):

    def __init__(self, data, train=True, transform=None, target_transform=None,
                 download=False):
        self.data, self.targets = data.data, data.targets


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]



        return img, target

# declare variables
batchsize = 32
n_epochs = 1
train_losses = []
val_losses = []
labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# get dataset
fmnist_data = tv.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, transform=tf.Compose([tf.ToTensor()]),
                                       target_transform=None, download=True)

# create contour images (y) and store thresholds as dimensions in X
train = [(np.reshape(a.numpy(), (28, 28))*255).astype(np.uint8) for a in fmnist_data.data]
trainX = []
trainY = []
for a in train:
    t2, t1 = r.randint(1, 256), r.randint(1, 256)
    trainY.append(cv.Canny(a, t1, t2))
    trainX.append([a, np.full((28, 28), t1), np.full((28, 28), t2)])

# update dataset with new X and y
fmnist_data.targets = trainY
fmnist_data.data = torch.tensor(trainX)
dataset = CannyDataset(fmnist_data)
data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

# create network
net = Net()
optimizer = opt.Adam(net.parameters(), lr=0.07)
criterion = nn.MSELoss()    # BCELoss()
net = net.cuda()
criterion = criterion.cuda()

# training process, loops through epoch (and batch or data-entries)
for epoch in range(n_epochs):
    for i, batch in enumerate(data_loader):
        train_x, train_y = batch
        output = net.train(epoch)
    if epoch % 2 == 0:
        print('Epoch : ', epoch + 1, '\t')  # , 'loss :', loss_val)

# plot loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()

# visualize sample of X and y
for i, batch in enumerate(data_loader):
    if i == 1:
        x, y = batch
        x0 = (np.reshape(x[0][0].numpy(), (28, 28))*255).astype(np.uint8)
        y0 = (np.reshape(y[0].numpy(), (28, 28))*255).astype(np.uint8)
        plt.imshow(x0, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(y0, cmap=plt.cm.gray)
        plt.show()

# visualize last output of network
axs = plt.subplots(4, 8)[1].ravel()
for i, ax in enumerate(axs):
    im = output[i].cpu().detach().numpy() * 255
    ax.imshow(im, cmap=plt.cm.gray)
plt.show()
print(im, ' with shape: ', im.shape)

