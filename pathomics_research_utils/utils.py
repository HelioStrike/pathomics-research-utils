import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

#Read numpy array of image file
def read_image(path):
    return np.array(Image.open(path))

#Divides image into patches of a certain size
def get_image_subpatches(im, size=(32,32)):
    patches = []
    i = 0
    while(i < im.shape[0]-size[0]):
        j = 0
        while(j < im.shape[1]-size[1]):
            patches.append(im[i:i+size[0],j:j+size[1]])
            j += size[1]
        i += size[0]
    return np.array(patches)

#Display images side-by-side
def displayImagesSideBySide(imgs, size=(20,40)):
    f, ax = plt.subplots(1,len(imgs), figsize=size)
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i])
    plt.show()