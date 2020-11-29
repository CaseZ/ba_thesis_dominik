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
import piq
from typing import Union, Tuple
import hiddenlayer as hl
import winsound
import os

### temporary ###

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    #instead conda install nomkl

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"       #debug cuda errors

### temporary ###


class Conv2dAutoPad(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # auto padding, here kernelsize / 2
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


# create convolution layer with auto padding
convAuto = partial(Conv2dAutoPad, kernel_size=3, bias=False)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                                      'bn': nn.BatchNorm2d(out_channels)
                                      }))


def render(net, path):
    transforms = [
        hl.transforms.Fold("Conv > BatchNorm > Relu > MaxPool", "ConvBnReluMaxP", "Convolution with Pooling"),
        hl.transforms.Fold("ConvTranspose > BatchNorm > Relu", "ConvTransBnRelu", "Transposed Convolution"),
        hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu", "Convolution"),
        hl.transforms.Fold("Conv > BatchNorm", "ConvBn", "Convolution without ReLU"),
        hl.transforms.Fold("""(ConvBnRelu > ConvBnRelu > ConvBn) > Add""", "BottleneckBlock", "Bottleneck Block"),
        hl.transforms.Fold("""BottleneckBlock > BottleneckBlock > BottleneckBlock""", "ResLayer", "Residual Layer"),
        hl.transforms.FoldDuplicates(),
    ]
    graph = hl.build_graph(net, torch.zeros([1, 3, 128, 128]).cuda(), transforms=transforms)
    graph.theme = hl.graph.THEMES["blue"].copy()
    Viz = graph.build_dot()
    Viz.attr(rankdir="TB")
    directory, file_name = os.path.split(path)
    Viz.render(file_name, directory=directory, cleanup=True, format='png')

def cuda_np(tensor):
    return tensor.cpu().detach().numpy()


def createClassDict(class_folder, printingClasses=True):
    # create dictionary for classes
    path, folder = os.path.split(class_folder)
    classes = {}
    i = 0

    for subdir, dirs, files in os.walk(path):
        for dir in dirs:
            classes[i] = dir
            i += 1

    if printingClasses:
        print('ID Class')
        for id, name in classes.items():
            print('{} {}'.format(id, name))

    return classes


class ResidualBottleNeckBlock(nn.Module):
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
            {'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
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
    def __init__(self, in_channels, out_channels, block=ResidualBottleNeckBlock, n=1, *args, **kwargs):
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


# visualizing blocks

# print(conv_bn(3, 3, nn.Conv2d, kernel_size=3))
# print(ResidualBlock(32, 64))

# dummy = torch.ones((1, 64, 48, 48))
# blo = ResNetLayer(64, 64, n=3, expansion=1)
# print(blo(dummy).shape)


class Net(nn.Module):
    def __init__(self, BCEL=False):
        super(Net, self).__init__()
        self.BCEL = BCEL    # option to enable sigmoid layer for Binary Cross Entropy Loss

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
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        )
        # more pooling

        self.ResLayer1 = ResNetLayer(128, 128, n=9, expansion=1)
        self.ResLayer2 = ResNetLayer(128, 256, n=9, expansion=1)
        self.ResLayer3 = ResNetLayer(256, 256, n=9, expansion=1)
        self.ResLayer4 = ResNetLayer(256, 256, n=9, expansion=1)
        self.ResLayer5 = ResNetLayer(256, 256, n=9, expansion=1)
        self.ResLayer6 = ResNetLayer(256, 256, n=9, expansion=1)

        # transposed convolution
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        '''
        self.deconv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(109, 89), mode='bilinear')
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(218, 178), mode='bilinear')
        )
        '''
        # same-convolution after upsampling / deconvoluting
        self.sConv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.sConv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, h):

        h = self.conv1(h)
        h = self.conv2(h)

        # Residual Layers
        h = self.ResLayer1(h)
        h = self.ResLayer2(h)
        h = self.ResLayer3(h)
        h = self.ResLayer4(h)
        h = self.ResLayer5(h)
        h = self.ResLayer6(h)

        # starting deconvolution
        h = self.deconv3(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.sConv1(h)
        h = self.sConv2(h)

        #if self.BCEL:
            #h = self.sigmoid(h)
        return h

    def train(self, epoch):
        if epoch == False:
            return []
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        tr_loss = 0
        x_train, y_train = autog.Variable(x_train), autog.Variable(y_train)
        # x_val, y_val = autog.Variable(x_val), autog.Variable(y_val)

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.float().cuda()
        y_val = y_val.float().cuda().unsqueeze(1)
        x_val.requires_grad = False
        y_val.requires_grad = False

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = net(x_train)
        with torch.no_grad():
            output_val = net(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)
        with torch.no_grad():
            loss_val = criterion(output_val, y_val)
        train_losses.append(cuda_np(loss_train))
        val_losses.append(cuda_np(loss_val))

        # SSIM loss
        ssim_train = piq.ssim(output_train, y_train, data_range=1.)
        ssim_val = piq.ssim(output_val, y_val, data_range=1.)
        SSIM_train.append(cuda_np(ssim_train))
        SSIM_val.append(cuda_np(ssim_val))

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss += loss_train.item()
        return output_train


class CannyDataset(Dataset):

    def __init__(self, data, topMargin=0, bottomMargin=0, normalize=False, norm=None, tnorm=None,
                 download=False):
        self.data = data
        self.topMargin = topMargin
        self.bottomMargin = bottomMargin
        self.norm = norm
        self.tnorm = tnorm
        self.normalize = normalize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        t1 = r.randint(1 + self.bottomMargin, 900 - self.topMargin)
        t2 = r.randint(t1, 900 - self.topMargin)
        img = self.data[index][0]

        # convert image to UTF-8 numpy array for cv module
        cvimg = (img[0].numpy() * 255).astype(np.uint8)

        # create contour image (y)
        target = torch.tensor(cv.Canny(cvimg, t1, t2).astype(float))

        # blurring input image
        blurimg = torch.tensor(cv.GaussianBlur(cvimg, (3, 3), 0)).unsqueeze(0)

        # create input with image and thresholds as dimension
        img = torch.cat([blurimg
                            , torch.full(img.shape, t1, dtype=torch.float)
                            , torch.full(img.shape, t2, dtype=torch.float)
                         ])
        # normalize input imgae, threshold dimensions and target contour-image
        if normalize:
            img = norm(img)
            target = tnorm(target.unsqueeze(0)).squeeze(0)

        return img, target


gc.collect()
torch.cuda.empty_cache()

##########################################
############# USER INTERFACE #############
##########################################

duplicates = None   # optional
batchsize = 14
topMargin = 400
bottomMargin = 150
n_epochs = 7
lr = 0.005
trained = 0
train_valid = False
continueTraining = 1
printingClasses = True
normalize = True
BCEL = False

##########################################
############# USER INTERFACE #############
##########################################

#### additional declarations ####
# path to save model
PATH = "state_dict_model_latest.pt"
# path of image folders (folder name = class name)
class_folder = r'C:\Users\dschm\PycharmProjects\ba_thesis\data\ImageNet\imagenet_images'
# lists to store loss
train_losses, val_losses = [], []
SSIM_train, SSIM_val = [], []
maxT = 900  # DO NOT CHANGE (maximum Canny Threshold value, depends on function and dataset)

# target normalization
tnorm = tf.Normalize(mean=0., std=255.0)
inv_norm = tf.Normalize(mean=-0., std=1/255.0)


std = float(maxT - topMargin - bottomMargin)
# image + thresholds normalization
norm = tf.Normalize(mean=[0., bottomMargin, bottomMargin], std=[255.0, std, std])
inv_input_norm = tf.Normalize(mean=np.negative([0., 0., 0.]), std=np.reciprocal([255.0, std, std]))


                                    ##### LOADING DATASET #####

ImageNet_data = \
    tv.datasets.ImageFolder(root='./data/ImageNet/imagenet_images'
                            , transform=tf.Compose([tf.transforms.Grayscale(1)
                                                       , tf.Resize(256)     # resize to smallest dimension first
                                                       , tf.CenterCrop(218)
                                                       , tf.ToTensor()
                                                    ])
                            , target_transform=None)

if duplicates:
    for _ in range(1, duplicates):
        oldLength = len(dataset)
        print(f"adding {duplicates} duplicates to the dataset")
        dataset2 = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin)
        dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
        print(f"new length of dataset: {len(dataset)}  ##  (old length: {oldLength})")

classes = createClassDict(class_folder, printingClasses)
dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                       , normalize=normalize, norm=norm, tnorm=tnorm)
# set shuffle true
data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)



# create network

if BCEL:
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = nn.MSELoss().cuda()  # nn.SmoothL1Loss().cuda()

net = Net()
optimizer = opt.AdamW(net.parameters(), lr=lr)
net = net.cuda()

# visualizing architecture
# print(summary(net, (3, 218, 178)))
# render(net, path='data/graph_minimal')




if trained or continueTraining:
    # Load model
    print("loading model")
    net.load_state_dict(torch.load(PATH))
    net.eval()

# training process, loops through epoch (and batch or data-entries)
if not trained:
    print("training model")
    for epoch in range(n_epochs):
        dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin)
        data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)  # set shuffle true

        # training with minibatch
        for i, batch in enumerate(data_loader):
            X, y = batch
            output = net.train(epoch)
        if epoch % 2 == 0:
            print('Epoch : ', epoch + 1, '\t')  # , 'loss :', loss_val)
    # Save model
    print("saving model")
    torch.save(net.state_dict(), PATH)
    print("saved model")
    gc.collect()
    torch.cuda.empty_cache()
    # winsound.PlaySound('RAN-D & VILLAIN - CORONA GO FCK YOURSELF.wav', winsound.SND_FILENAME)
    winsound.Beep(500, 1000)

    axs = plt.subplots(2, 1)[1].ravel()
    # plot loss
    axs[0].plot(train_losses, label='Training loss', alpha=0.3)
    axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[0].set_xlabel('batches')
    axs[0].legend()

    # plot SSIM
    axs[1].plot(SSIM_train, label='Training SSIM', alpha=0.3)
    axs[1].plot(SSIM_val, label='Validation SSIM', alpha=0.6)
    axs[1].set_xlabel('batches')
    axs[1].legend()
    plt.show()

# visualize sample of X and y
print("visualizing  output")
for i, batch in enumerate(data_loader):
    if i == 0:  # (len(data_loader)-1)
        X, y = batch[0:20]
        x_show = X[0:20].float().cuda()
        y_show = y[0:20].float().cuda()
        output = net(x_show)

        #print(output[0][0])
        # invert normalization
        x_show, y_show = [inv_input_norm(x).int() for x in x_show], [inv_norm(y.unsqueeze(0)).squeeze(0).int() for y in y_show]
        output = [inv_norm(out).int() for out in output]
        #print(output[0][0])

        axs = plt.subplots(6, 3)[1]
        # image comparison plot
        for a, ax in enumerate(axs):
            im = cuda_np(output[a][0])
            x0 = (cuda_np(x_show[a][0])).astype(np.uint8)
            y0 = (cuda_np(y_show[a])).astype(np.uint8)
            t1, t2 = int(cuda_np(x_show[a][1][0][0])), int(cuda_np(x_show[a][2][0][0]))

            ax[0].axis('off'), ax[1].axis('off'), ax[2].axis('off')
            ax[0].imshow(x0, cmap=plt.cm.gray, interpolation='nearest')
            ax[0].set_title('Thresholds: ' + str(t1) + ' and ' + str(t2))
            ax[1].imshow(y0, cmap=plt.cm.gray, interpolation='nearest')
            # ax[1].set_title('target')
            ax[2].imshow(im, cmap=plt.cm.gray, interpolation='nearest')
            # ax[2].set_title('output')
        plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
        plt.show()

        # threshold comparison
        axs = plt.subplots(10, 3)[1]
        a = np.random.randint(0, len(X))
        step = (((900 - topMargin) - bottomMargin) / len(axs))
        listT = np.arange(bottomMargin, 900 - topMargin, step)

        for index, ax in enumerate(axs):
            x0 = (cuda_np(x_show[a][0])).astype(np.uint8)
            y0 = (cuda_np(y_show[a])).astype(np.uint8)
            t1, t2 = int(X[a][1][0][0]), int(X[a][2][0][0].numpy())
            t1 = listT[index]
            t2 = listT[index] + bottomMargin
            # random thresholds
            # t1 = r.randint(bottomMargin, 900 - topMargin)
            # t2 = r.randint((t1-topMargin if t1 > (maxT-topMargin) else t1), maxT - topMargin)
            # show extremes
            # t1 = (maxT-topMargin) if index == 0 else maxT if index == 1 else 0 if index == 2 else bottomMargin if index == 3 else t1
            # t2 = (maxT-topMargin) if index == 0 else maxT if index == 1 else 0 if index == 2 else bottomMargin if index == 3 else t2
            y0 = cv.Canny(x0, t1, t2)

            x1 = x_show[a][0].unsqueeze(0)
            x1 = torch.cat([x1, torch.full(x1.shape, t1, dtype=torch.float, device="cuda"),
                            torch.full(x1.shape, t2, dtype=torch.float, device="cuda")]).unsqueeze(0).cuda()
            x1 = net(x1)
            x1 = inv_norm(x1.squeeze(0))
            x1 = cuda_np(x1[0])

            ax[0].axis('off'), ax[1].axis('off'), ax[2].axis('off')
            ax[0].imshow(x0, cmap=plt.cm.gray, interpolation='nearest')
            ax[0].set_title(str(t1) + '   and   ' + str(t2))
            ax[1].imshow(y0, cmap=plt.cm.gray, interpolation='nearest')
            # ax[1].set_title('target')
            ax[2].imshow(x1, cmap=plt.cm.gray, interpolation='nearest')
            # ax[2].set_title('output')
        plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
        plt.show()

# visualize last output of network
axs = plt.subplots(2, 7)[1].ravel()
for i, ax in enumerate(axs):
    ax.axis('off')
    im = output[i][0].cpu().detach().numpy()
    ax.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
plt.show()