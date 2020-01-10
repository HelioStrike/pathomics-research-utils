import os
import random
import numpy as np
from copy import deepcopy
from pathomics_research_utils import utils
from tensorflow.keras.utils import Sequence

#Send a list of corresponding image paths
#Divides images into patches and sends them
class SubPatchingSegmentationDataGenerator(Sequence):
    def __init__(self, paired_images_list=None, patch_height=32, patch_width=32,
                 batch_size=32, shuffle=True, augmentation=None,
                 magnify=None, num_channels=3, output_channels=1):
        self.batch_size = batch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_channels = num_channels
        self.output_channels = output_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.paired_images_list = paired_images_list
        self.len = int(len(self.paired_images_list) * (len(utils.get_image_subpatches(utils.read_image(paired_images_list[0][0])))**(0.5)-1)**2 // self.batch_size)
        self.shuffle = shuffle
        self.magnify = magnify
        self.image_ptr = 0
        self.cur = 0
        self.org_patches = []
        self.mask_patches = []

    def __len__(self):
        return self.len-1
    
    def on_epoch_start(self):
        self.image_ptr = 0
        self.cur = 0
        if self.shuffle:
            random.shuffle(self.paired_images_list)

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, self.patch_height, self.patch_width, self.num_channels))
        y = np.empty((self.batch_size, self.patch_height, self.patch_width, self.output_channels))
        for i in range(self.batch_size):
            if self.cur == len(self.org_patches):
                self.cur = 0
                if(self.image_ptr == len(self.paired_images_list)):
                    self.image_ptr = 0
                self.org_patches = utils.get_image_subpatches(utils.read_image(self.paired_images_list[self.image_ptr][0]))
                self.mask_patches = utils.get_image_subpatches(utils.read_image(self.paired_images_list[self.image_ptr][1]))
                self.image_ptr += 1

            dims = self.org_patches[self.cur].shape
            if(len(dims) == 2):
                X[i] = self.org_patches[self.cur].reshape(*dims, 1)
            else:
                X[i] = self.org_patches[self.cur]

            dims = self.mask_patches[self.cur].shape
            if(len(dims) == 2):
                y[i] = self.mask_patches[self.cur].reshape(*dims, 1)
            else:
                y[i] = self.mask_patches[self.cur]

            if self.augmentation:
                X[i] = self.augmentation(X[i])
                y[i] = self.augmentation(y[i])
            self.cur += 1
        return X, y

#Use the path to a directory containing nested directories of images (each corresponding to a class)
#Divides images into patches
#WARNING: This loads the whole dataset into memory, to effectively shuffle the data
class SubPatchingClassificationDataGenerator(Sequence):
    def __init__(self, images_dir=None, patch_height=32, patch_width=32,
                 batch_size=32, shuffle=True, augmentation=None,
                 magnify=None, num_channels=3, num_classes=3):
        self.batch_size = batch_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.shuffle = shuffle
        self.images = np.array([]).reshape(0,patch_height,patch_width,num_channels)
        self.image_labels = []
        self.dir_names = os.listdir(images_dir)
        for i in range(len(self.dir_names)):
            dir_path = os.path.join(images_dir, self.dir_names[i])
            fnames = os.listdir(dir_path)
            cur=0
            for fname in fnames:
                patches = utils.get_image_subpatches(utils.read_image(os.path.join(dir_path, fname)))
                self.images = np.concatenate([self.images, patches], axis=0)
                self.image_labels += [i]*len(patches)
                cur+=1
        self.magnify = magnify
        self.len = len(self.images) // self.batch_size

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if shuffle:
            self.images, self.image_labels = zip(*random.shuffle(list(zip(self.images, self.image_labels))))

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, self.patch_height, self.patch_width, self.num_channels))
        y = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            X[i] = self.images[idx*self.batch_size+i]
            y[i][self.image_labels[idx*self.batch_size+i]] = 1
            if self.augmentation:
                X[i] = self.augmentation(X[i])
        return X, y