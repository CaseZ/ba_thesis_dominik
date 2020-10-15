import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2 as cv
import torch
import torchvision as tv
from torchvision import transforms as tf

### temporary ###
import os   # instead conda install nomkl
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
### temporary ###

fmnist_data = tv.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, transform=tf.Compose([tf.ToTensor()]),
                                       target_transform=None, download=True)
data_loader = torch.utils.data.DataLoader(fmnist_data, batch_size=4, shuffle=True, drop_last=True)

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
    if i == 1:
        x, y = batch
        print(len(x))
        x0 = (np.reshape(x[0].numpy(), (28, 28))*255).astype(np.uint8)
        y0 = y[0].item()

        v = np.random.randint(1, 256)
        t1 = int(max(0, (1.0 - 0.23) * v))
        t2 = int(min(255, (1.0 + 0.23) * v))
        t1, t2 = 80, 50
        print(v, t1, t2)
        x0 = cv.Canny(x0, t1, t2)
        plt.imshow(x0, cmap=plt.cm.gray)
        plt.title(labels[y0])
        plt.show()
#'''


# Load image openCV
#im2 = cv.imread(r'F:\Users\Dominik\Pictures\Orbception Warframe (bug).png')

# try different threshold values
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