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
import torch.autograd as autog
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
import time
import sys

# ## temporary ###
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    # instead conda install nomkl
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"       # debug cuda errors
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


def saveimg(name, show):
    if show:
        plt.show()
    else:
        plt.savefig((img_folder + name + parameters + ".png"), dpi=250, bbox_inches='tight')


def cuda_np(tensor):
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    return tensor.detach().numpy()


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal_(m.bias.data)


def create_threshImage(img, t1, t2, blur, preprocess=True):
    img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if preprocess:
        img = (img * 255).astype(np.uint8)

    blurimg = torch.tensor(cv.GaussianBlur(img, (blur, blur), 0)).unsqueeze(0)
    buffer = torch.cat([blurimg
                        , torch.full(img.shape, t1, dtype=torch.float).unsqueeze(0)
                        , torch.full(img.shape, t2, dtype=torch.float).unsqueeze(0)
                     ], dim=0)
    return buffer


def create_surrogate_input(h, img):
    new_h = []
    for i, thresh in enumerate(cuda_np(h)):
        new_h.append(create_threshImage(np.float32([img[i]]), thresh[0], thresh[1], blur, preprocess=False).cuda().unsqueeze(0))
    return torch.cat(new_h).squeeze(2)


def deconstruct_input(input, index):
    image = (cuda_np(input[index][0]))
    t1, t2 = int(cuda_np(input[index][1][0][0])), int(cuda_np(input[index][2][0][0]))
    return image, t1, t2


def compare_images(x_show, output, showTarget, name, y_show=None):

    dim = 3 if showTarget else 2
    axs = plt.subplots(6, dim)[1]

    for a, ax in enumerate(axs):
        im = cuda_np(output[a][0])
        x0, t1, t2 = deconstruct_input(x_show, a)

        ax[0].axis('off'), ax[1].axis('off')
        ax[0].imshow(x0, cmap=plt.get_cmap('gray'), interpolation='nearest')
        ax[0].set_title('Thresholds: ' + str(t1) + ' and ' + str(t2))
        ax[1].imshow(im, cmap=plt.get_cmap('gray'), interpolation='nearest')

        # set column header
        if a == 0:
            ax[1].set_title('output')

        if showTarget:
            ax[2].axis('off')
            y0 = (cuda_np(y_show[a])).astype(np.uint8)
            ax[2].imshow(y0, cmap=plt.get_cmap('gray'), interpolation='nearest')

            # set column header
            if a == 0:
                ax[2].set_title('target')

    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
    saveimg(f"comparison_{name}_", show)    # move function here?


def compare_thresholds(x_show):
    axs = plt.subplots(10, 3)[1]

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
        x1 = torch.cat([x1, torch.full(x1.shape, t1, dtype=torch.float, device="cuda"),
                        torch.full(x1.shape, t2, dtype=torch.float, device="cuda")]).unsqueeze(0).cuda()
        x1 = surrogate_net(x1)
        x1 = inv_norm(x1.squeeze(0))
        x1 = cuda_np(x1[0])

        ax[0].axis('off'), ax[1].axis('off'), ax[2].axis('off')
        ax[0].imshow(x0, cmap=plt.get_cmap('gray'), interpolation='nearest')
        # ax[0].set_title(str(t1) + '   and   ' + str(t2))

        ax[1].imshow(x1, cmap=plt.get_cmap('gray'), interpolation='nearest')
        ax[2].imshow(y0, cmap=plt.get_cmap('gray'), interpolation='nearest')

        # set column header
        if index == 0:
            ax[1].set_title('output')
            ax[2].set_title('target')

    plt.subplots_adjust(top=1.0, bottom=0.0, left=0.25, right=0.5, hspace=0.01, wspace=0.05)
    saveimg("compare_thresholds_", show)


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

        return h

    def train(self, epoch):
        if epoch == False:
            print(" --- WARNING : not training because epoch is False or 0 --- ")
            return []
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        x_train, y_train = x_train.requires_grad(), y_train.requires_grad()

        x_train = x_train.float().cuda()
        y_train = y_train.float().cuda().unsqueeze(1)
        x_val = x_val.float().cuda()
        y_val = y_val.float().cuda().unsqueeze(1)
        x_val.requires_grad = False
        y_val.requires_grad = False

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = surrogate_net(x_train)
        with torch.no_grad():
            output_val = surrogate_net(x_val)

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
        return output_train, loss_train.item()


class PredictNet(nn.Module):
    def __init__(self, surrogate, validate):
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

        for param in surrogate.parameters():
            param.requires_grad = False     # freeze model
        self.surrogate = surrogate

        for param in validate.parameters():
            param.requires_grad = False     # freeze model
        self.validate = validate


    def forward(self, h):
        og_im = h           # save original input image

        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        h = self.avgpool(h)
        thresholds = self.fc(h.flatten(start_dim=1))
        thresholds = self.sig(thresholds)

        h_3 = create_surrogate_input(thresholds, cuda_np(og_im))
        #print("img: ", minmax(h_3[0]))
        #print("t1: ", minmax(h_3[0][1]))
        #print("t2: ", minmax(h_3[0][2]))
        contour_im = self.surrogate(h_3)
        #print("contour: ", minmax(contour_im))
        classes = self.validate(contour_im)
        classes.requires_grad = True

        return classes, thresholds, contour_im, h_3


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

        img = create_threshImage(img, t1, t2, blur)

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
batchsize = 14
topMargin = 400     # threshold value top margin cutoff
bottomMargin = 150  # threshold value bottom margin cutoff
blur = 5            # kernel size for cv.GaussianBlur preprocessing when passing to surrogate
n_epochs = 20       # epochs for training session
total_eps = 20      # total epochs for saving

lrs = 0.0025        # learning rate surrogate model
lrv = 0.005         # learning rate validation model
lrp = 0.1          # learning rate prediction model

# -- surrogate model control--
trained_surrogate = True    # if true loading model otherwise train from scratch
continueTraining = False    # continue train when model loaded
viz_surrogate = True       # visualize output of surrogate network

# -- validation model control--
trained_valid = True        # if true loading model otherwise train from scratch
continueVal = False         # continue train when model loaded

# -- prediction model control--
trained_predict = False        # if true loading model otherwise train from scratch
continuePredict = False         # continue train when model loaded

# -- misc --
shutdown_txt = False        # write stdout to txt and shutdown after training
saving = False              # saving model with parameters as name
printingClasses = False
normalize = True            # normalizing input to [0,1]
show = False                # show or save plots


# -----------------------------------------
# ------------ USER INTERFACE -------------
# -----------------------------------------

if shutdown_txt:
    sys.stdout = open(r'C:\Users\dschm\Uni\Uni\BA Thesis\normalized\ImageNet\console.txt', 'w')

# #### additional declarations ####

PATH = "state_dict_model_latest.pt"
class_folder = r'C:\Users\dschm\PycharmProjects\ba_thesis\data\ImageNet\imagenet_images'
img_folder = r'C:\Users\dschm\Uni\Uni\BA Thesis\normalized\ImageNet\_'

train_losses, val_losses, vloss = [], [], []
SSIM_train, SSIM_val = [], []
AUCS_train, AUCS_val = [], []

maxT = 900  # DO NOT CHANGE (maximum Canny Threshold value, depends on function and dataset)

parameters = f"{total_eps}eps_lrs{lrs}{'_normalized' if normalize else ''}_{blur}blur_topM{topMargin}_lowM{bottomMargin}"
print("parameters: ", parameters)


# ----------- NORMALIZATION -----------
tnorm = tf.Normalize(mean=0., std=255.0)    # target normalization
inv_norm = tf.Normalize(mean=-0., std=1/255.0)  # revert target normalization

std = float(maxT - topMargin - bottomMargin)
norm = tf.Normalize(mean=[0., bottomMargin, bottomMargin], std=[255.0, std, std])
inv_input_norm = tf.Normalize(mean=np.negative([0., bottomMargin, bottomMargin]), std=np.reciprocal([255.0, std, std]))


# ----------- DATASET -----------
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
                       , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur)
data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)



# ################################# SURROGATE MODEL ##################################

criterion = nn.MSELoss().cuda()  # nn.SmoothL1Loss().cuda()
surrogate_net = SurrogateNet().cuda()
optimizer = opt.AdamW(surrogate_net.parameters(), lr=lrs)

# visualizing architecture
#print(summary(surrogate_net, (3, 218, 178)))
#render(surrogate_net, path='data/graph_minimal')


# ----------- TRAINING -----------
if trained_surrogate or continueTraining:
    print("loading model")
    surrogate_net.load_state_dict(torch.load(PATH))
    surrogate_net.eval()

if not trained_surrogate:
    print("training model")
    surrogate_net.apply(weights_init)  # xavier init for conv2d layer weights
    t_start = time.time()
    optimizer = opt.SGD(surrogate_net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.005, nesterov=True)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0, verbose=True)

    for epoch in range(1, n_epochs):
        dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                               , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur)
        data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
        for i, batch in enumerate(data_loader):
            X, y, _ = batch
            output, loss = surrogate_net.train(epoch)
        if epoch % 2 == 0:
            t_end = time.time() - t_start  # training time
            t_string = "" + str(int(t_end / 60)) + "m" + str(int(t_end % 60)) + "s"
            print(f'Epoch : { epoch + 1} \t Loss : {loss:.4f} \t Time :  {t_string}')
        scheduler.step()

    # Save model
    print("saving model")
    if saving:
        PATH = parameters + t_string + PATH
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
    print("visualizing  output")
    for i, batch in enumerate(data_loader):
        if i == 0:  # (len(data_loader)-1)
            X, y, _ = batch[0:20]
            x_show = X[0:20].cuda()
            y_show = y[0:20].cuda()

            #print("before surrogate img : ", minmax(x_show[0]))
            #print("before surrogate t1 : ", minmax(x_show[1]))
            #print("before surrogate t2 : ", minmax(x_show[2]))

            output = surrogate_net(x_show)

            #print("after surrogate: ", minmax(output))

        # invert normalization
            x_show, y_show = [inv_input_norm(x).int() for x in x_show], [inv_norm(y.unsqueeze(0)).squeeze(0).int() for y in y_show]
            output = [inv_norm(out).int() for out in output]

            #print("after denorm: ", minmax(output))

            compare_images(x_show, output, name="surrogate", showTarget=True, y_show=y_show)

            compare_thresholds(x_show)

    # visualize last output of network
    '''
    axs = plt.subplots(2, 7)[1].ravel()
    for i, ax in enumerate(axs):
        ax.axis('off')
        im = output[i][0].cpu().detach().numpy()
        ax.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    '''


# ################################ VALIDATION MODEL #################################

resnet152 = tv.models.resnet152()
# change first and last layer for 1d input and output class length
resnet152.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet152.fc = nn.Linear(in_features=2048, out_features=len(classes), bias=True)
resnet152 = resnet152.cuda()
resnet152.train()
resnet152.apply(weights_init)   # xavier init for conv2d layer weights

criterion = nn.CrossEntropyLoss().cuda()
optimizer2 = opt.SGD(resnet152.parameters(), lr=lrv, momentum=0.9, weight_decay=0.005, nesterov=True)
#scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=5, T_mult=2, eta_min=0, last_epoch=-1, verbose=True)
scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=n_epochs, eta_min=0, verbose=True)

# visualizing architecture
#print(summary(resnet152, (1, 218, 178)))
#render(surrogate_net, path='data/graph_minimal')


# ----------- TRAINING -----------
if continueVal:
    # Load model
    print("loading validation model")
    resnet152.load_state_dict(torch.load("validation_" + PATH))

if not trained_valid:
    print("training validation model")
    t = time.time()
    for epoch in range(n_epochs):
        dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                               , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur)
        data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

        for i, batch in enumerate(data_loader):
            X, _, z = batch
            X, y = surrogate_net(X.cuda()), z.cuda()
            X, y = autog.Variable(X), autog.Variable(y)

            optimizer2.zero_grad()
            output = resnet152(X)
            loss = criterion(output, y)

            output = torch.tensor([torch.topk(out, 1)[1] for out in output]).float().cuda()  # extract class labels
            acc = metrics.accuracy_score(cuda_np(output), cuda_np(y))
            AUCS_train.append(acc)
            vloss.append(loss.item())
            loss.backward()
            optimizer2.step()

        if epoch % 2 == 0:
            t_end = time.time() - t  # training time
            t_string = "" + str(int(t_end / 60)) + "m" + str(int(t_end % 60)) + "s"
            print(f'Epoch : {epoch + 1} \t Loss : {loss:.4f} \t Accuracy :  {acc:.4f} \t Time :  {t_string}')

        scheduler2.step()
        print(f"output : {cuda_np(output).astype(np.int)}")
        print(f"target : {cuda_np(y)}")
        print("--------------------------------------")

    # Save model
    print("saving validation model")
    torch.save(resnet152.state_dict(), "validation_" + PATH)
    print("saved validation model")
    t_end = time.time() - t   # training time
    t_string = "_" + str(int(t_end / 60)) + "m" + str(int(t_end % 60)) + "s_"
    gc.collect()
    torch.cuda.empty_cache()
    winsound.Beep(500, 1000)

    axs = plt.subplots(2, 1)[1].ravel()
    # plot auc score
    axs[0].plot(AUCS_train, label='Training Accuracy', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[0].set_xlabel('batches')
    axs[0].legend()

    # plot loss
    axs[1].plot(vloss, label='Training loss', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[1].set_xlabel('batches')
    axs[1].legend()
    saveimg("valmodel_loss_", show)


# ################################ PREDICTION MODEL #################################

predict_net = PredictNet(surrogate_net, resnet152).cuda()
# criterion same as validateNet
optimizer3 = opt.AdamW(predict_net.parameters(), lr=lrp)
scheduler3 = lr_scheduler.CosineAnnealingLR(optimizer3, T_max=n_epochs, eta_min=0.001, verbose=True)

#print(predict_net)
#print(summary(predict_net, (1, 218, 218)))

if not trained_predict:
    print("training prediction model")
    t = time.time()
    for epoch in range(n_epochs):
        dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                               , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur, noThresholds=True)
        data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)

        for i, batch in enumerate(data_loader):
            X, _, z = batch
            X, y = X.cuda(), z.cuda()    #long needed?
            X.requires_grad = True
            optimizer3.zero_grad()
            output, thresholds, contour_imgs, input_im = predict_net(X)
            loss = criterion(output, y)

            output = torch.tensor([torch.topk(out, 1)[1] for out in output]).float().cuda()  # extract class labels
            acc = metrics.accuracy_score(cuda_np(output), cuda_np(y))
            AUCS_train.append(acc)
            vloss.append(loss.item())
            loss.backward()
            optimizer3.step()

        if epoch % 2 == 0:
            t_end = time.time() - t  # training time
            t_string = "" + str(int(t_end / 60)) + "m" + str(int(t_end % 60)) + "s"
            print(f'Epoch : {epoch + 1} \t Loss : {loss:.4f} \t Accuracy :  {acc:.4f} \t Time :  {t_string} \t sample thresholds : {thresholds}')

        if epoch % 1 == 0:
            #contour_imgs = [inv_norm(im).int() for im in contour_imgs]
            #input_im = [inv_input_norm(og).int() for og in input_im]


            compare_images(input_im, contour_imgs, name=f"predict{epoch+1}_", showTarget=False)

        scheduler3.step()
        print(f"output : {cuda_np(output).astype(np.int)}")
        print(f"target : {cuda_np(y)}")
        print("--------------------------------------")

    # Save model
    print("saving prediction model")
    torch.save(resnet152.state_dict(), "predict_" + PATH)
    print("saved prediction model")
    t_end = time.time() - t   # training time
    t_string = "_" + str(int(t_end / 60)) + "m" + str(int(t_end % 60)) + "s_"
    gc.collect()
    torch.cuda.empty_cache()
    winsound.Beep(500, 1000)

    axs = plt.subplots(2, 1)[1].ravel()
    # plot auc score
    axs[0].plot(AUCS_train, label='Training Accuracy', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[0].set_xlabel('batches')
    axs[0].legend()

    # plot loss
    axs[1].plot(vloss, label='Training loss', alpha=0.3)
    #axs[0].plot(val_losses, label='Validation loss', alpha=0.6)
    axs[1].set_xlabel('batches')
    axs[1].legend()
    saveimg("predictmodel_loss_", show)



# save console to txt and shutdown
if shutdown_txt:
    print(" ## WARNING ## \n ---- shutting down in 5 minutes ---- \n ## WARNING ##")
    os.system("shutdown /s /t 300")
    time.sleep(180)
    print(" ## WARNING ## \n ---- shutting down in 2 minutes ---- \n ## WARNING ##")
    sys.stdout.close()
