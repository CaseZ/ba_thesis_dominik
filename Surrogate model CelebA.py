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
from torchsummary import summary
import PIL

### temporary ###
import os  # instead conda install nomkl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### temporary ###


class Conv2dAutoPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # auto padding, here kernelsize / 2
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

# create convolution layer with auto padding
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

        b1 = block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling)
        self.blocks = nn.Sequential(
            b1,
            *[block(b1.expanded_channels,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        return self.blocks(x)


#print(conv_bn(3, 3, nn.Conv2d, kernel_size=3))
#print(ResidualBlock(32, 64))


#dummy = torch.ones((1, 64, 48, 48))
#blo = ResNetLayer(64, 64, n=3, expansion=1)
#print(blo(dummy).shape)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        )

        layers = [ResNetLayer(128, 128, n=3, expansion=1) for _ in range(9)]
        self.ResLayer1 = layers[0]
        self.ResLayer2 = layers[0]
        self.ResLayer3 = layers[0]
        self.ResLayer4 = layers[0]
        self.ResLayer5 = layers[0]
        self.ResLayer6 = layers[0]
        self.ResLayer7 = layers[0]
        self.ResLayer8 = layers[0]
        self.ResLayer9 = layers[0]

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        # convolution

    def forward(self, h, thresholds=[]):
        # s = h.shape
        # print(s)
        # print(thresholds[0])
        # h = torch.cat((h, thresholds[0].cuda().unsqueeze(dim=1)), dim=1)   # concatenate threshold tensor to current layer
        # h = h.reshape([s[0], s[1], s[2], s[3]])

        h = self.conv1(h)
        h = self.conv2(h)
        # Residual Layers to prevent vanishing gradient
        h = self.ResLayer1(h)
        h = self.ResLayer2(h)
        h = self.ResLayer3(h)
        h = self.ResLayer4(h)
        h = self.ResLayer5(h)
        h = self.ResLayer6(h)
        h = self.ResLayer7(h)
        h = self.ResLayer8(h)
        h = self.ResLayer9(h)
        # starting deconvolution
        h = self.deconv1(h)
        h = self.deconv2(h)
        return h

    def train(self, epoch):
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33) #, random_state=42
        tr_loss = 0
        x_train, y_train = autog.Variable(x_train), autog.Variable(y_train)
        #x_val, y_val = autog.Variable(x_val), autog.Variable(y_val)

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.float().cuda()
        y_val = y_val.float().cuda().unsqueeze(1)
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
batchsize = 128
n_epochs = 5
train_losses = []
val_losses = []
labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

# LOADING DATASET
#celebA_data = tv.datasets.CelebA(root='./data/CelebA', split='train',
#                                 transform=tf.Compose([tf.transforms.Grayscale(1), tf.ToTensor()]),
#                                 target_transform=None, download=False
#                                 )

celebA_data = tv.datasets.ImageFolder(root='./data/CelebA',
                                 transform=tf.Compose([tf.transforms.Grayscale(1), tf.ToTensor()]),
                                 target_transform=None
                                 )

shapeX, shapeY = 218, 178
data = []

# create tensor of data, convert to UTF-8
#data = torch.tensor([(a[0].numpy() * 255).astype(np.uint8) for (a, b) in celebA_data])
'''
print("done creating tensor")
torch.save(data.clone(), 'data.pt')
print("saved!")
'''

data = torch.load('data.pt')
print('loaded!')
celebA_data.data = data
print(len(celebA_data.data))

# create contour images (y) and store thresholds as dimensions in X
x = []
Y = []
for a in data:
    #a = cv.blur(a, (3, 3))
    t2, t1 = r.randint(1, 256), r.randint(1, 256)
    Y.append([cv.Canny(a.numpy(), t1, t2)])
    x.append([a, torch.tensor(np.full((shapeX, shapeY), t1)), torch.tensor(np.full((shapeX, shapeY), t2))])


print('shapes: ')
print(len(Y), len(Y[0]), len(Y[0][0]), len(Y[0][0][0]))
print(len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))

# update dataset with new X and y
celebA_data.targets = Y
celebA_data.data = x
dataset = CannyDataset(celebA_data)
data_loader = DataLoader(celebA_data, batch_size=batchsize, shuffle=False, drop_last=True)

# create network
net = Net()
optimizer = opt.Adam(net.parameters(), lr=0.07)
criterion = nn.MSELoss()  # BCELoss()
net = net.cuda()
criterion = criterion.cuda()
print(summary(net, (3, 218, 178)))
# training process, loops through epoch (and batch or data-entries)
for epoch in range(n_epochs):
    for i, batch in enumerate(data_loader):
        # this fucks up!
        # fucking dataloader loses my 3 dimensions, batch already 1D
        # bruh
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
    if i == (len(data_loader)-1):
        x, y = batch
        axs = plt.subplots(8, 3)[1]

        for a, ax in enumerate(axs):
            im = output[a][0].cpu().detach().numpy()
            x0 = (np.reshape(x[a][0].numpy(), (28, 28))).astype(np.uint8)
            y0 = (np.reshape(y[a].numpy(), (28, 28))).astype(np.uint8)
            t1, t2 = int(x[a][1][0][0].numpy()), int(x[a][2][0][0].numpy())

            ax[0].imshow(x0, cmap=plt.cm.gray)
            ax[0].set_title('input with Thresholds: ' + str(t1) + 'and ' + str(t2))
            ax[1].imshow(y0, cmap=plt.cm.gray)
            ax[1].set_title('target')
            ax[2].imshow(im, cmap=plt.cm.gray)
            ax[2].set_title('output')
        plt.show()

# visualize last output of network
axs = plt.subplots(3, 16)[1].ravel()
for i, ax in enumerate(axs):
    im = output[i][0].cpu().detach().numpy()
    ax.imshow(im, cmap=plt.cm.gray)
plt.show()

