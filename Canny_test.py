import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
import torch
import torchvision as tv
from torchvision import transforms as tf
from skimage import feature
from numpy import random as r
from torch.utils.data import Dataset, DataLoader

### temporary ###
import os   # instead conda install nomkl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### temporary ###

# ----------- NORMALIZATION -----------
topMargin = 400     # threshold value top margin cutoff
bottomMargin = 150
maxT = 900
blur = 5
normalize = True

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

def create_threshImage(img, t1, t2, blur, preprocess=True):
    img = img[0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if preprocess:
        img = (img * 255).astype(np.uint8)

    blurimg = torch.tensor(cv.GaussianBlur(img, (blur, blur), 0), device="cuda").unsqueeze(0)

    buffer = torch.cat([blurimg
                        , torch.full(img.shape, t1, dtype=torch.float, device="cuda").unsqueeze(0)
                        , torch.full(img.shape, t2, dtype=torch.float, device="cuda").unsqueeze(0)
                     ], dim=0)
    return buffer


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


#mnist_data = tv.datasets.MNIST(root='./data/MNIST', train=True, transform=tf.Compose([tf.ToTensor()]),
#                                       target_transform=None, download=True)
#data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=10, shuffle=False, drop_last=True)

dataset = CannyDataset(ImageNet_data, topMargin=topMargin, bottomMargin=bottomMargin
                                   , normalize=normalize, norm=norm, tnorm=tnorm, blur=blur)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)

labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

sigl, sigh = 100, 200
#trainX = [(np.reshape(a.numpy(), (28, 28))*255).astype(np.uint8) for a in fmnist_data.data]
#testX = [(np.reshape(a, (28, 28))*255).astype(np.uint8) for a in x_valid]
#trainY = [cv.Canny(a, sigl, sigh) for a in trainX]
#testY = [cv.Canny(a, sigl, sigh) for a in trainX]
# sigl, sigh = 100, 200
# trainX = list(zip(trainX, trainT))
# testX = list(zip(testX, testT))
#fmnist_data.targets = trainY


# plt.imshow(fmnist_data.targets[1], cmap=plt.cm.gray)
# plt.show()
# print batch
#'''
for i, batch in enumerate(data_loader):
    if i == 0:
        x, y, _ = batch
        print(len(x[0]))
        x0 = (x[0][0].detach().cpu().numpy()*255).astype(np.uint8)
        y0 = y[0]

        v = np.random.randint(1, 256)
        t1 = int(max(0, (1.0 - 0.23) * v))
        t2 = int(min(255, (1.0 + 0.23) * v))
        #t1, t2 = 80, 50
        print(v, t1, t2)
        x1 = cv.Canny(x0, t1, t2)
        x2 = feature.canny(x0, 10, 20)
        plt.imshow(x0, cmap=plt.cm.gray)
        #plt.title(labels[y0])
        plt.show()
        plt.imshow(x1, cmap=plt.cm.gray)
        #plt.title(labels[y0])
        plt.show()
#'''

# Load image openCV
#im2 = cv.imread(r'F:\Users\Dominik\Pictures\Orbception Warframe (bug).png')

# try different threshold values

'''
fig, axs = plt.subplots(nrows=1, ncols=10, figsize=(16, 6), sharex=True, sharey=True)

for i, batch in enumerate(data_loader):
    if i < 10:
        x, y = batch
        print(i)
        x = (np.reshape(x[0].numpy(), (28, 28))*255).astype(np.uint8)
        axs[i].imshow(x, cmap=plt.cm.gray)
        axs[i].axis('off')
fig.tight_layout()
plt.show()
'''


'''       
# Compute the Canny filter for two values of sigma
sigl, sigh = 100, 200
edges1 = cv.Canny(x0, 10, 20)
edges2 = cv.Canny(x0, 50, 80)
cvcanny = cv.Canny(x0, sigl, sigh)
# Display results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(16, 6), sharex=True, sharey=True)

ax1.imshow(x0, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title(r'Scipy Canny filter, $\sigma=$'+str(sigl), fontsize=20)

ax4.imshow(cvcanny, cmap=plt.cm.gray)
ax4.axis('off')
ax4.set_title(r'CV Canny filter, $\sigma_{low}=$'+str(sigl) +"$\sigma_{high}=$"+str(sigh), fontsize=20)

fig.tight_layout()
plt.show()
'''