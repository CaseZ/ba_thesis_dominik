import cv2 as cv
import numpy as np
from numpy import random as r
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as opt
import torch.autograd as autog
import torchvision as tv
import gc
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from functools import partial

### temporary ###
import os  # instead conda install nomkl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### temporary ###


class Conv2dAutoPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


convAuto = partial(Conv2dAutoPad, kernel_size=3, bias=False)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({  'conv': conv(in_channels, out_channels, *args, **kwargs),
                                        'bn': nn.BatchNorm2d(out_channels)
                                      }))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, downsampling=1, conv=convAuto, *args, **kwargs):
        super().__init__()

        self.in_channels, self.out_channels, self.expansion, self.downsampling, self.conv \
            = in_channels, out_channels, expansion, downsampling, conv

        self.blocks = self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             nn.ReLU(inplace=True),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             nn.ReLU(inplace=True),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

        self.shortcut = nn.Sequential(OrderedDict(
            {   'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)
            })) if self.should_apply_shortcut else None

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResidualBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        return self.blocks(x)


print(conv_bn(3, 3, nn.Conv2d, kernel_size=3))
print(ResidualBlock(32, 64))
print(ResNetLayer(64, 128, n=3))


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
            nn.ConvTranspose2d(64, 32, kernel_size=9, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(16, 16))
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(28, 28))
        )

    def forward(self, h, thresholds=[]):
        # s = h.shape
        # print(s)
        # print(thresholds[0])
        # h = torch.cat((h, thresholds[0].cuda().unsqueeze(dim=1)), dim=1)   # concatenate threshold tensor to current layer
        # h = h.reshape([s[0], s[1], s[2], s[3]])

        h = self.conv1(h)
        h = self.conv2(h)
        # residual blocks or conv layers
        h = self.deconv1(h)  # starting deconvolution
        h = self.deconv2(h)
        return h

    def train(self, epoch):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        tr_loss = 0
        x_train, y_train = autog.Variable(x_train), autog.Variable(y_train)
        #x_val, y_val = autog.Variable(x_val), autog.Variable(y_val)

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.cuda()
        y_val = y_val.cuda().unsqueeze(1)
        x_val.requires_grad = False
        y_val.requires_grad = False

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = net(x_train, [t1, t2])
        with torch.no_grad():
            output_val = net(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        with torch.no_grad():
            loss_val = criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

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

gc.collect()
torch.cuda.empty_cache()

# declare variables
batchsize = 32
n_epochs = 25
train_losses = []
val_losses = []
labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# get dataset
fmnist_data = tv.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, transform=tf.Compose([tf.ToTensor()]),
                                       target_transform=None, download=True)

# create contour images (y) and store thresholds as dimensions in X
data = [(np.reshape(a.numpy(), (28, 28)) * 255).astype(np.uint8) for a in fmnist_data.data]
X = []
Y = []
for a in data:
    t2, t1 = r.randint(1, 256), r.randint(1, 256)
    Y.append(cv.Canny(a, t1, t2))
    X.append([a, np.full((28, 28), t1), np.full((28, 28), t2)])

# update dataset with new X and y
fmnist_data.targets = Y
fmnist_data.data = torch.FloatTensor(X)
dataset = CannyDataset(fmnist_data)
data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

# create network
net = Net()
optimizer = opt.Adam(net.parameters(), lr=0.07)
criterion = nn.MSELoss()  # BCELoss()
net = net.cuda()
criterion = criterion.cuda()

# training process, loops through epoch (and batch or data-entries)
for epoch in range(n_epochs):
    for i, batch in enumerate(data_loader):
        #if i % 100 == 0:
        #    print('Batch : ', i + 1, '\t')
        X, y = batch
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
        x0 = (np.reshape(x[0][0].numpy(), (28, 28)) * 255).astype(np.uint8)
        y0 = (np.reshape(y[0].numpy(), (28, 28)) * 255).astype(np.uint8)
        plt.imshow(x0, cmap=plt.cm.gray)
        plt.show()
        plt.imshow(y0, cmap=plt.cm.gray)
        plt.show()

# visualize last output of network
axs = plt.subplots(2, 8)[1].ravel()
for i, ax in enumerate(axs):
    im = output[i][0].cpu().detach().numpy()
    ax.imshow(im, cmap=plt.cm.gray)
plt.show()
