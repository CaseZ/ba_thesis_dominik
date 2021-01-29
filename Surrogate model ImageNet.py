import cv2 as cv
import numpy as np
from numpy import random as r
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as opt
import torchvision as tv
import gc

from torch.optim import lr_scheduler
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torchsummary import summary
import piq
import hiddenlayer as hl
import winsound
import os
import sys
import time
import traceback

# ## temporary ###
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    # instead conda install nomkl
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"       # debug cuda errors
# ## temporary ###


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


class Interpolate(nn.Module):
    def __init__(self, mode, scale=None, size=None):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.scale = scale
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale, mode=self.mode, size=self.size)
        return x


def get_time(seconds=True):
    if seconds:
        return time.mktime(time.localtime())
    else:
        return time.strftime('%T', time.localtime())


def time_elapsed(t0):
    t_delta = get_time(True) - t0
    return time.strftime("%T", time.gmtime(t_delta))


def list_shape(x):
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0]
    return shape


def minmax(input):
    if isinstance(input, torch.Tensor):
        return f"max : {torch.max(input).item():.5f} \t min : {torch.min(input[input.nonzero(as_tuple=True)]).item()}"
    else:
        return f"max : {np.max([torch.max(x) for x in input]).item():.5f} \t " \
               f"min : {np.min([torch.min(x[x.nonzero(as_tuple=True)]) for x in input]).item()}"


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze model


def render(net, path, input=[14, 3, 128, 128]):
    transforms = [
        hl.transforms.Fold("Conv > BatchNorm > Relu > MaxPool", "ConvBnReluMaxP", "Convolution with Pooling"),
        hl.transforms.Fold("ConvTranspose > BatchNorm > Relu", "ConvTransBnRelu", "Transposed Convolution"),
        hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu", "Convolution"),
        hl.transforms.Fold("Conv > BatchNorm", "ConvBn", "Convolution without ReLU"),
        hl.transforms.Fold("""(ConvBnRelu > ConvBnRelu > ConvBn) > Add""", "BottleneckBlock", "Bottleneck Block"),
        hl.transforms.Fold("""BottleneckBlock > BottleneckBlock > BottleneckBlock
        > BottleneckBlock > BottleneckBlock > BottleneckBlock
        > BottleneckBlock > BottleneckBlock > BottleneckBlock""", "ResLayer", "Residual Layer"),
        hl.transforms.FoldDuplicates(),
    ]
    graph = hl.build_graph(net, torch.zeros(input).cuda(), transforms=transforms)
    graph.theme = hl.graph.THEMES["blue"].copy()
    Viz = graph.build_dot()
    Viz.attr(rankdir="TB")
    directory, file_name = os.path.split(path)
    Viz.render(file_name, directory=directory, cleanup=True, format='png')


def saveimg(name, show):
    if show:
        plt.show()
    else:
        plt.savefig((img_folder + name + parameters + ".png"), dpi=250, bbox_inches='tight')


def cuda_np(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    return tensor.detach().numpy()


def weights_init(m, customgain=True):
    if isinstance(m, nn.Conv2d): #or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data, gain=(nn.init.calculate_gain('relu') if customgain else 1))
        #nn.init.xavier_normal_(m.bias.data)


def create_threshImage(img, t1, t2, blur, preprocess=True):
    img = img[0]
    if isinstance(img, torch.Tensor):
        img = cuda_np(img)
    if preprocess:
        img = (img * 255).astype(np.uint8)

    blurimg = torch.tensor(cv.GaussianBlur(img, (blur, blur), 0), device="cuda").unsqueeze(0)

    buffer = torch.cat([blurimg
                        , torch.full(img.shape, t1, dtype=torch.float, device="cuda").unsqueeze(0)
                        , torch.full(img.shape, t2, dtype=torch.float, device="cuda").unsqueeze(0)
                     ], dim=0)
    return buffer


def create_surrogate_input(h, img):
    new_h = []
    for i, thresh in enumerate(cuda_np(h)):
        new_h.append(create_threshImage(np.float32([img[i]]), thresh[0], thresh[1], blur, preprocess=False).unsqueeze(0))
    return torch.cat(new_h).squeeze(2)


def deconstruct_input(input, index=None, img=True):
    image = (cuda_np(input[index][0]))
    if input[0].size() == (3, 218, 218):
        t1, t2 = int((input[index][1][0][0]).item()), int((input[index][2][0][0]).item())
    else:
        t1, t2 = 0, 0

    if img:
        return image, t1, t2
    return t1, t2


def compare_images(x_show, output, showTarget, name, threshs=None, y_show=None, nr=6):

    dim = 3 if showTarget else 2
    fig_i, axs = plt.subplots(nr, dim)

    for a, ax in enumerate(axs):
        im = cuda_np(output[a][0])
        x0, t1, t2 = deconstruct_input(x_show, a)
        if threshs is not None:
            t1, t2 = threshs[a][0], threshs[a][1]

        ax[0].axis('off'), ax[1].axis('off')
        ax[0].imshow(x0, cmap=plt.get_cmap('gray'), interpolation='nearest')
        ax[0].set_title(f'({str(t1)}, {str(t2)})')
        ax[1].imshow(im, cmap=plt.get_cmap('gray'), interpolation='nearest')

        # set column header
        #if a == 0:
        #    ax[1].set_title('output')

        if showTarget:
            if y_show is None:
                if a == 0:
                    print("creating canny image")
                y0 = torch.tensor(cv.Canny(im.astype(np.uint8), t1, t2).astype(float))
            else:
                y0 = (cuda_np(y_show[a])).astype(np.uint8)

            ax[2].axis('off')
            #err = metrics.mean_squared_error(im, y0)/255
            #IQ1 = piq.ssim((output[a][0]).type(torch.FloatTensor), y_show[a].type(torch.FloatTensor), data_range=1.)
            #IQ2 = piq.gmsd((output[a][0]).type(torch.FloatTensor), y_show[a].type(torch.FloatTensor), data_range=1.)
            ax[2].imshow(y0, cmap=plt.get_cmap('gray'), interpolation='nearest')
            #ax[2].set_title(f'mse: {err:.2f}, ssim: {IQ1:.5f}, GMSD: {IQ2:.5f}')

            # set column header
            if a == 0:
                ax[2].set_title('target')

    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
    saveimg(f"comparison_{name}_", show)    # move function here?
    plt.close(fig_i)


def compare_thresholds(x_show, name, nr=8, round=False):
    fig_c, axs = plt.subplots(nr, 3)

    a = np.random.randint(0, len(X))
    step = (((900 - topMargin) - bottomMargin) / len(axs))
    listT = np.arange(bottomMargin, 900 - topMargin, step)

    for index, ax in enumerate(axs):
        x0 = (cuda_np(x_show[a][0])).astype(np.uint8)
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
        x1 = create_threshImage(x1, t1, t2, blur, preprocess=True)

        with torch.no_grad():
            x1 = surrogate_net(x1.unsqueeze(0), rounding=round).squeeze(0)
        x1 = inv_norm(x1)
        x1 = cuda_np(x1[0])

        ax[0].axis('off'), ax[1].axis('off'), ax[2].axis('off')
        ax[0].imshow(x0, cmap=plt.get_cmap('gray'), interpolation='nearest')
        # ax[0].set_title(f'({str(t1)}, {str(t2)})')

        ax[1].imshow(x1, cmap=plt.get_cmap('gray'), interpolation='nearest')
        ax[2].imshow(y0, cmap=plt.get_cmap('gray'), interpolation='nearest')

        # set column header
        if index == 0:
            ax[1].set_title('output')
            ax[2].set_title('target')

    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
    saveimg(f"compare_thresholds_{name}_", show)
    plt.close(fig_c)


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
#RBNB = ResidualBottleNeckBlock(128, 64).cuda()
#print(summary(RBNB, (128, 1, 1), batch_size=14, device='cuda'))

# dummy = torch.ones((1, 64, 48, 48))
#blo = ResNetLayer(128, 128, n=9, expansion=1).cuda()
#print(summary(blo, (128, 1, 1), batch_size=14, device='cuda'))
#render(blo, path='data/reslayer_minimal', input=[128, 128, 1, 1])
# print(blo(dummy).shape)

class AlexCustom(nn.Module):
    def __init__(self, og_model, layers):
        super(AlexCustom, self).__init__()
        layers = 13-layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            *list(og_model.features.children())[1:-layers]
        )

    def forward(self, x):
        x = self.features(x)
        return x


class SurrogateNet(nn.Module):
    def __init__(self):
        super(SurrogateNet, self).__init__()

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
        # add upsample to ensure always same output
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

    def forward(self, h, rounding=False):

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

        if rounding:
            h = torch.round(h)  # rounding output for high contrast picture

        return h

    def train(self, optimizer_internal, epoch=False):
        if epoch == False:
            print(" --- WARNING : not training because epoch is False or 0 --- ")
            return []
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        x_train.requires_grad = True
        y_train.requires_grad = True

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.float().cuda()
        y_val = y_val.float().cuda().unsqueeze(1)
        x_val.requires_grad = False
        y_val.requires_grad = False

        # clearing the Gradients of the model parameters
        optimizer_internal.zero_grad()

        # prediction for training and validation set
        output_train = self.forward(x_train)
        with torch.no_grad():
            output_val = self.forward(x_val)

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
        del ssim_train, ssim_val, output_train

        # computing the updated weights of all the model parameters
        loss_train.backward()
        loss = loss_train.item()
        del loss_train
        optimizer_internal.step()
        return loss


class PredictNet(nn.Module):
    def __init__(self, surrogate=None, validate=None):
        super(PredictNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
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

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=256, out_features=2, bias=True)
        self.sig = nn.Sigmoid()
        self.sig.requires_grad=False

        if surrogate is not None:
            freeze(surrogate)
            self.surrogate = surrogate

        if validate is not None:
            #freeze(validate)
            self.validate = validate

    def forward(self, h, outputs=False):
        og_im = h           # save original input image

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        h = self.avgpool(h)
        thresholds = self.fc(h.flatten(start_dim=1))
        thresholds = self.sig(thresholds)

        # inv normalize thresholds for showing / control purpose
        threshs = cuda_np(thresholds)
        threshs = [(int(a*std)+bottomMargin, int(b*std)+bottomMargin) for (a, b) in threshs]

        h_3 = create_surrogate_input(thresholds, cuda_np(og_im))
        contour_im = self.surrogate(h_3, rounding=False)        # surrogate pass

        classes = self.validate(contour_im)     # validation pass

        if outputs:
            return classes, threshs, contour_im, h_3
        return classes


class Decoder(nn.Module):
    def __init__(self, surrogate=None, validate=None):
        super(Decoder, self).__init__()

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

        #self.ResLayer1 = ResNetLayer(128, 128, n=9, expansion=1)
        #self.ResLayer2 = ResNetLayer(128, 256, n=9, expansion=1)
        #self.ResLayer3 = ResNetLayer(256, 256, n=9, expansion=1)
        self.sConv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.sConv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # transposed convolution
        self.deconv3 = nn.Sequential(
            Interpolate(mode='nearest', scale=2),
            nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2),
            #nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.deconv1 = nn.Sequential(
            Interpolate(mode='nearest', scale=2),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            #nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            Interpolate(mode='nearest', size=dims),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            #nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # add upsample to ensure always same output

        # same-convolution after upsampling / deconvoluting
        self.sConv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.sConv2 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, h):

        h = self.conv1(h)
        h = self.conv2(h)

        # Residual Layers
        #h = self.ResLayer1(h)
        #h = self.ResLayer2(h)
        #h = self.ResLayer3(h)

        # starting deconvolution
        #h = self.deconv3(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.sConv1(h)
        h = self.sConv2(h)

        return h

    def train(self, optimizer_internal, epoch=False, val=False, alex=None):
        if epoch == False:
            print(" --- WARNING : not training because epoch is False or 0 --- ")
            return []
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        #print(x_train)
        #x_train.requires_grad = True
        #y_train.requires_grad = True

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.float().cuda()
        y_val = y_val.float().cuda().unsqueeze(1)
        #x_val.requires_grad = False
        #y_val.requires_grad = False

        # clearing the Gradients of the model parameters
        optimizer_internal.zero_grad()
        # prediction for training and validation set
        output_train = self.forward(x_train)
        with torch.no_grad():
            output_val = self.forward(x_val)

        if val:
            output_train2 = alex(output_train)
            y_train2 = alex(y_train)
            output_val2 = alex(output_val)
            y_val2 = alex(y_val)

            # computing the training and validation loss
            loss_train = criterion(output_train2, y_train2)
            with torch.no_grad():
                loss_val = criterion(output_val2, y_val2)
            train_losses.append(cuda_np(loss_train))
            val_losses.append(cuda_np(loss_val))
        else:
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
        del ssim_train, ssim_val

        # computing the updated weights of all the model parameters
        loss_train.backward()
        loss = loss_train.item()
        del loss_train
        optimizer_internal.step()
        return output_train, loss



class CannyDataset(Dataset):

    def __init__(self, data, topMargin=0, bottomMargin=0, normalize=False, norm=None, tnorm=None, blur=0,
                 download=False, noThresholds=False):
        self.data = data
        self.topMargin = topMargin
        self.bottomMargin = bottomMargin
        self.norm = norm
        self.tnorm = tnorm
        self.normalize = normalize
        self.blur = blur
        self.addThresholds = not noThresholds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        t1 = r.randint(1 + self.bottomMargin, 900 - self.topMargin)
        t2 = r.randint(t1, 900 - self.topMargin)

        img = self.data[index][0]
        id = self.data[index][1]

        # convert image to UTF-8 numpy array for cv module
        cvimg = (img[0].numpy() * 255).astype(np.uint8)

        # create contour image (y)
        target = torch.tensor(cv.Canny(cvimg, t1, t2).astype(float))
        img = create_threshImage(img, t1, t2, blur, preprocess=True)
        if not self.addThresholds:
            # create input with image without thresholds as dimension
            img = img[0]

        # normalize input image, threshold dimensions and target contour-image
        if normalize and self.addThresholds:
            img = norm(img)
            target = tnorm(target.unsqueeze(0)).squeeze(0)
        elif normalize:     # use tnorm for img if no threshold dimensions
            img = tnorm(img.unsqueeze(0))
            target = tnorm(target.unsqueeze(0)).squeeze(0)
        return img, target, id


gc.collect()
torch.cuda.empty_cache()

# -----------------------------------------
# ------------ USER INTERFACE -------------
# -----------------------------------------

duplicates = None   # optional
topMargin = 400     # threshold value top margin cutoff
bottomMargin = 150  # threshold value bottom margin cutoff
blur = 5            # kernel size for cv.GaussianBlur preprocessing when passing to surrogate
n_epochs = 50       # epochs for training session
total_eps = 50      # total epochs for saving

lrs = 0.05          # (starting) learning rate surrogate model
lrv = 0.1           # (starting) learning rate validation model
lrp = 0.1           # (starting) learning rate prediction model

# -- surrogate model control--
batchsize_surrogate = 15#23     # batchsize for surrogate training
train_surrogate = False          # if true loading model otherwise train from scratch
load_surrogate = True          # continue train when model loaded
schedule_surrogate = True       # use learning rate scheduler
viz_surrogate = True            # visualize output of surrogate network

# -- validation model control--
batchsize_validation = 60       # batchsize for validation training
train_valid = False             # if true loading model otherwise train from scratch
load_valid = False              # continue train when model loaded
schedule_valid = True           # use learning rate scheduler

# -- prediction model control--
batchsize_predict = 14          # batchsize for predict training
train_predict = False          # if true loading model otherwise train from scratch
load_predict = True            # continue train when model loaded
schedule_predict = True         # use learning rate scheduler
use_alex = True                 # use alexNet conv layer as validation
viz_predict = True            # visualize output of predict network

# -- misc --
shutdown_txt = False            # write stdout to txt and shutdown after training
saving = False                  # saving model with parameters as name
printingClasses = False
normalize = True                # normalizing input to [0,1]
show = False                    # show or save plots


# -----------------------------------------
# ------------ USER INTERFACE -------------
# -----------------------------------------

if shutdown_txt:
    print(" ## WARNING ## \n ---- saving console to file -- NO CONSOLE ---- \n ## WARNING ##")
    sys.stdout = open(r'C:\Users\dschm\Uni\Uni\BA Thesis\normalized\Decoder\console.txt', 'w')

# #### additional declarations ####

PATH = "state_dict_model_latest.pt"
class_folder = r'C:\Users\dschm\PycharmProjects\ba_thesis\data\ImageNet\imagenet_images'
img_folder = r'C:\Users\dschm\Uni\Uni\BA Thesis\normalized\Decoder\_'

train_losses, val_losses, vloss = [], [], []
SSIM_train, SSIM_val = [], []
accuracy_train, accuracy_val = [], []

maxT = 900  # DO NOT CHANGE (maximum Canny Threshold value, depends on function and dataset)
predict_net = None      # variable has to be initialized
dims = (218, 218)

parameters = f"{total_eps}eps_lrs{lrs}{'_normalized' if normalize else ''}_{blur}blur_topM{topMargin}_lowM{bottomMargin}"
print("parameters: ", parameters)


# ----------- NORMALIZATION -----------
tnorm = tf.Normalize(mean=0., std=255.0)    # target normalization
inv_norm = tf.Normalize(mean=-0., std=1/255.0)  # revert target normalization

std = float(maxT - topMargin - bottomMargin)
norm = tf.Normalize(mean=[0., bottomMargin, bottomMargin], std=[255.0, std, std])
inv_input_norm = tf.Compose([tf.Normalize(mean=[0., 0., 0.], std=np.reciprocal([255.0, std, std])),
                             tf.Normalize(mean=np.negative([0., bottomMargin, bottomMargin]), std=[1., 1., 1.])])


# ----------- DATASET -----------
ImageNet_data = \
    tv.datasets.ImageFolder(root='./data/ImageNet/imagenet_images'
                            , transform=tf.Compose([     tf.ColorJitter(0, (0, 2.5), 0, 0)
                                                       , tf.transforms.Grayscale(1)
                                                       , tf.Resize(256)     # resize to smallest dimension first
                                                       , tf.CenterCrop(218)
                                                       , tf.ToTensor()
                                                    ])
                            , target_transform=None)
dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                                   , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur)
if duplicates:
    for _ in range(1, duplicates):
        oldLength = len(dataset)
        print(f"adding {duplicates} duplicates to the dataset")
        dataset2 = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin)
        dataset = torch.utils.data.ConcatDataset([dataset, dataset2])
        print(f"new length of dataset: {len(dataset)}  ##  (old length: {oldLength})")

classes = createClassDict(class_folder, printingClasses)


# ################################# SURROGATE MODEL ##################################

criterion = nn.MSELoss().cuda()  # nn.SmoothL1Loss().cuda()
surrogate_net = SurrogateNet().cuda()
optimizer = opt.SGD(surrogate_net.parameters(), lr=lrs, momentum=0.9, weight_decay=0.005, nesterov=True)
#opt.AdamW(surrogate_net.parameters(), lr=lrs)
surrogate_net.apply(weights_init)  # xavier init for conv2d layer weights

# visualizing architecture
#print(summary(surrogate_net, (3, 218, 178), batch_size=14, device='cuda'))
#render(surrogate_net, path='data/surrogate_full')
'''
try:
        return input(prompt)
    except KeyboardInterrupt:
        print("press control-c again to quit")
    return input(prompt)

except KeyboardInterrupt:
  print("...")
  traceback.print_exc()
'''
# ----------- TRAINING -----------

if load_surrogate:
    print("loading surrogate model")
    surrogate_net.load_state_dict(torch.load(PATH))
    surrogate_net.eval()

if train_surrogate:
    print("training surrogate model")
    t_start = get_time()
    print(f"starting time : {get_time(False)}")
    if schedule_surrogate:
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(n_epochs*0.85), eta_min=0.0001, verbose=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, threshold=0.001, cooldown=1, verbose=True)

    for epoch in range(1, n_epochs):
        try:
            data_loader = DataLoader(dataset, batch_size=batchsize_surrogate, shuffle=True, drop_last=True)
            for i, batch in enumerate(data_loader):
                X, y, _ = batch
                loss = surrogate_net.train(optimizer, epoch)
                if i % 25 == 0:
                    print(f"batch {i} \t loss : {loss:.4f}")

            if epoch % 1 == 0:
                print(f'Epoch : { epoch + 1} \t Lr : {optimizer.param_groups[0]["lr"]} \t Loss : {np.mean(val_losses[-i:]):.4f} \t Time :  {time_elapsed(t_start)}')
            if epoch in [0,1,2,3,4,5,6,7,8,9] or (epoch % 5 == 0):
                x_show, y_show, _ = batch
                with torch.no_grad():
                    output = surrogate_net.forward(x_show)
                x_show = [inv_input_norm(x).int() for x in x_show]
                y_show = [inv_norm(y.unsqueeze(0)).squeeze(0).int() for y in y_show]
                output2 = [inv_norm(out).int() for out in output]

                compare_images(x_show, output2, name=f"surrogate_{epoch}", showTarget=True, y_show=y_show, nr=7)
                compare_thresholds(x_show, name=f"surrogate_{epoch}", nr=7)

            if schedule_surrogate:
                scheduler.step(np.mean(val_losses[-i:]))
        except KeyboardInterrupt:
            print(f"model froze in batch : {i}")
            traceback.print_exc()
            exit()

    print(f"end time : {get_time(False)}")
    # Save model
    print("saving model")
    if saving:
        PATH = parameters + str(time_elapsed(t_start)) + PATH
    torch.save(surrogate_net.state_dict(), PATH)
    print("saved model")
    gc.collect()
    torch.cuda.empty_cache()
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
    plt.subplots_adjust(top=1.0)
    saveimg("loss_", show)


# ----------- VISUALIZING -----------
if viz_surrogate:
    data_loader = DataLoader(dataset, batch_size=batchsize_surrogate, shuffle=True, drop_last=True)
    print("visualizing  output")
    batch = next(iter(data_loader))
    X, y, _ = batch
    x_show = X.cuda()
    y_show = y.cuda()

    with torch.no_grad():
        output = surrogate_net(x_show, rounding=True)
    # invert normalization
    x_show, y_show = [inv_input_norm(x).int() for x in x_show], [inv_norm(y.unsqueeze(0)).squeeze(0).int() for y in y_show]
    output = [inv_norm(out).int() for out in output]

    compare_images(x_show, output, name="surrogate", showTarget=True, y_show=y_show, nr=int(batchsize_surrogate/2))

    compare_thresholds(x_show, name="surrogate_rounded", round=True)
    compare_thresholds(x_show, name="surrogate", round=False)

    # visualize last output of network
    '''
    axs = plt.subplots(2, 7)[1].ravel()
    for i, ax in enumerate(axs):
        ax.axis('off')
        im = output[i][0].cpu().detach().numpy()
        ax.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    '''


'''

resnet152 = tv.models.resnet152()
# change first and last layer for 1d input and output class length
resnet152.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet152.fc = nn.Linear(in_features=2048, out_features=len(classes), bias=True)
resnet152 = resnet152.cuda()
resnet152.train()
resnet152.apply(weights_init)   # xavier init for conv2d layer weights

val_net = resnet152
'''
# ################################ VALIDATION MODEL #################################
val_net = Decoder().cuda()
val_net.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        ).cuda()
val_net.train(None)
val_net.apply(weights_init)   # xavier init for conv2d layer weights
criterion = nn.MSELoss().cuda()     #piq.SSIMLoss()
optimizer2 = opt.AdamW(surrogate_net.parameters(), lr=lrv)
# opt.SGD(val_net.parameters(), lr=lrv, momentum=0.9, weight_decay=0.005, nesterov=True)

alex_og = tv.models.alexnet(pretrained=True)
alex = AlexCustom(alex_og, layers=5).cuda()     # 2,5,8,10,13
alex.eval()
freeze(alex)
print(alex)
'''
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
'''

freeze(surrogate_net)   # freeze surrogate
# visualizing architecture
#print(summary(val_net, (1, 218, 178), batch_size=14, device='cuda'))
#render(val_net, path='data/val_minimal', input=[1, 1, 128, 128])
# ----------- TRAINING -----------
if load_valid:
    # Load model
    print("loading validation model")
    val_net.load_state_dict(torch.load("validation_" + PATH))

if train_valid:
    print("training validation model")
    t_start = get_time()
    print(f"starting time : {get_time(False)}")
    if schedule_valid:
        scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=(n_epochs*0.85), eta_min=0.0001, verbose=True)
        # scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=5, T_mult=2, eta_min=0, last_epoch=-1, verbose=True)

    for epoch in range(1, n_epochs):
        data_loader = DataLoader(dataset, batch_size=batchsize_validation, shuffle=True, drop_last=True)
        for i, batch in enumerate(data_loader):
            X, _, _ = batch
            # surrogate is input, normal image is target
            y = X.narrow(1, 0, 1).squeeze()
            X = surrogate_net(X.cuda(), rounding=False)
            output, loss = val_net.train(optimizer2, epoch, val=True, alex=alex)
            if i % 50 == 0:
                print(f"batch : {i}")
                print(minmax(output))

        if epoch%1 == 0:    # epoch % 5 == 0
            X, _, _ = batch
            x_t = [inv_input_norm(x).int() for x in X]
            threshs = [deconstruct_input(x_t, index=i, img=False) for i, x in enumerate(x_t)]
            y = X.narrow(1, 0, 1).squeeze()
            with torch.no_grad():
                X = surrogate_net(X.cuda())
            x_show = X.cuda()
            y_show = y.cuda()
            with torch.no_grad():
                output = val_net(x_show)

            # invert normalization
            x_show, y_show = [inv_norm(x.unsqueeze(0)).squeeze(0).int() for x in x_show]\
                , [inv_norm(y.unsqueeze(0)).squeeze(0).int() for y in y_show]
            output2 = [inv_norm(out).int() for out in output]

            compare_images(x_show, output2, name=f"validate_{epoch}", showTarget=True, y_show=y_show, threshs=threshs, nr=int(batchsize_validation/10))
            #compare_thresholds(x_show, name=f"validate_{epoch}")

        if epoch % 1 == 0:
            print(f'Epoch : {epoch + 1} \t Loss : {np.mean(val_losses[-i:]):.4f} \t Time :  {time_elapsed(t_start)}')

        scheduler2.step()
        print(f"output : {cuda_np(output[0][0][125]).astype(np.int)}")
        #print(f"target : {cuda_np(y)}")
        print("--------------------------------------")


    # Save model
    print("saving validation model")
    print(f"end time : {get_time(False)}")
    torch.save(val_net.state_dict(), "validation_" + PATH)
    print("saved validation model")
    t_end = time_elapsed(t_start)   # training time
    gc.collect()
    torch.cuda.empty_cache()
    winsound.Beep(500, 1000)

    axs = plt.subplots(2, 1)[1].ravel()
    # plot accuracy score
    axs[0].plot(train_losses, label='Training loss', alpha=0.3)
    axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[0].set_xlabel('batches')
    axs[0].legend()

    # plot loss
    axs[1].plot(SSIM_train, label='Training SSIM', alpha=0.3)
    axs[1].plot(SSIM_val, label='Validation SSIM', alpha=0.6)
    axs[1].set_xlabel('batches')
    axs[1].legend()
    plt.subplots_adjust(top=1.0)
    saveimg("valmodel_loss_", show)


# ################################ PREDICTION MODEL #################################

predict_net = PredictNet().cuda()
# criterion same as validateNet

optimizer3 = opt.AdamW(predict_net.parameters(), lr=lrp)
predict_net.apply(weights_init)  # xavier init for conv2d layer weights
freeze(surrogate_net), #freeze(val_net)
predict_net.surrogate = surrogate_net
predict_net.validate = val_net


#print(predict_net)
#print(summary(predict_net, (1, 218, 218), batch_size=14, device='cuda'))
#render(predict_net, path='data/predict_full', input=[14, 1, 128, 128])

if load_predict:
    # Load model
    print("loading prediction model")
    predict_net.load_state_dict(torch.load("predict_" + PATH))

if train_predict:
    print("training prediction model")
    t_start = get_time()
    print(f"starting time : {get_time(False)}")
    if schedule_predict:
        scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer3, T_max=(n_epochs*0.85), eta_min=0.001, verbose=True)

    for epoch in range(n_epochs):
        dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                               , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur, noThresholds=True)
        data_loader = DataLoader(dataset, batch_size=batchsize_predict, shuffle=True, drop_last=True)

        for i, batch in enumerate(data_loader):
            X, _, _ = batch
            # surrogate is input, normal image is target
            y = X.narrow(1, 0, 1).cuda() #.squeeze()
            X = X.cuda()
            optimizer3.zero_grad()
            output, thresholds, contour_imgs, input_im = predict_net.forward(X, outputs=True)

            if use_alex:
                output_train2 = alex(output)
                y_train2 = alex(y)

                # computing the training and validation loss
                train_loss = criterion(output_train2, y_train2)
                loss = train_loss.item()
                train_loss.backward()
                del train_loss
            else:
                train_loss = criterion(output, y)
                loss = train_loss.item()
                train_loss.backward()
                del train_loss


            #output = torch.tensor([torch.topk(out, 1)[1] for out in output]).float().cuda()  # extract class labels
            #acc = metrics.accuracy_score(cuda_np(output), cuda_np(y))
            acc = piq.ssim(output.detach(), y.detach(), data_range=1.)
            accuracy_train.append(acc)
            vloss.append(loss)

            if schedule_predict:
                optimizer3.step()

        if epoch % 1 == 0:
            print(f'Epoch : {epoch + 1} \t Loss : {loss:.4f} \t Time :  {time_elapsed(t_start)} \t sample thresholds : {thresholds}')

        if epoch % 1 == 0:
            contour_imgs = [inv_norm(im).int() for im in contour_imgs]
            input_im = [inv_input_norm(og).int() for og in input_im]
            compare_images(input_im, contour_imgs, name=f"predict{epoch+1}_", threshs=thresholds, showTarget=True, nr=4)

        scheduler3.step()
        #print(f"output : {cuda_np(output).astype(np.int)}")
        #print(f"target : {cuda_np(y)}")
        print("--------------------------------------")

    # Save model
    print("saving prediction model")
    print(f"end time : {get_time(False)}")
    torch.save(predict_net.state_dict(), "predict_" + PATH)
    print("saved prediction model")
    t_end = time_elapsed(t_start)   # training time
    gc.collect()
    torch.cuda.empty_cache()
    winsound.Beep(500, 1000)

    axs = plt.subplots(2, 1)[1].ravel()
    # plot accuracy score
    axs[0].plot(accuracy_train, label='Training Accuracy', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[0].set_xlabel('batches')
    axs[0].legend()

    # plot loss
    axs[1].plot(vloss, label='Training loss', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[1].set_xlabel('batches')
    axs[1].legend()
    saveimg("predictmodel_loss_", show)


# ----------- VISUALIZING -----------
if viz_predict:
    dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                           , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur, noThresholds=True)
    data_loader = DataLoader(dataset, batch_size=batchsize_predict, shuffle=True, drop_last=True)
    print("visualizing predict output")
    batch = next(iter(data_loader))
    X, _, _ = batch
    # surrogate is input, normal image is target
    y = X.narrow(1, 0, 1).cuda()  # .squeeze()
    X = X.cuda()

    with torch.no_grad():
        output, thresholds, contour_imgs, input_im = predict_net.forward(X, outputs=True)

    # invert normalization
    contour_imgs = [inv_norm(im).int() for im in contour_imgs]
    input_im = [inv_input_norm(og).int() for og in input_im]
    compare_images(input_im, contour_imgs, name=f"predict_", threshs=thresholds, showTarget=True, nr=4)


# save console to txt and shutdown
if shutdown_txt:
    print(" ## WARNING ## \n ---- shutting down in 5 minutes ---- \n ## WARNING ##")
    os.system("shutdown /s /t 300")
    time.sleep(180)
    print(" ## WARNING ## \n ---- shutting down in 2 minutes ---- \n ## WARNING ##")
    sys.stdout.close()
