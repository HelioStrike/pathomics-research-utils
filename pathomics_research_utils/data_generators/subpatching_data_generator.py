import os
import random
import numpy as np
from copy import deepcopy
from pathomics_research_utils import utils
from tensorflow.keras.utils import Sequence

class SubPatchingSegmentationDataGenerator(Sequence):
    def __init__(self, paired_images_list=None, patch_height=32, patch_width=32,
                 batch_size=32, shuffle=True, augmentation=None,
                 magnify=None, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.paired_images_list = paired_images_list
        self.len = len(self.paired_images_list) * len(utils.get_image_subpatches(utils.read_image(paired_images_list[0][0]))) // self.batch_size
        self.shuffle = shuffle
        self.magnify = magnify
        self.image_ptr = 0
        self.cur = 0
        self.org_patches = []
        self.mask_patches = []

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if self.shuffle:
            random.shuffle(self.paired_images_list)

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, self.patch_height, self.patch_width, self.num_channels))
        y = np.empty((self.batch_size, self.patch_height, self.patch_width, self.num_channels))
        for i in range(self.batch_size):
            if self.cur == len(self.org_patches):
                image_ptr += 1
                cur = 0
                self.org_patches = utils.get_image_subpatches(utils.read_image(self.paired_images_list[image_ptr][0]))
                self.mask_patches = utils.get_image_subpatches(utils.read_image(self.paired_images_list[image_ptr][1]))
            X[i] = self.org_patches[cur]
            y[i] = self.mask_patches[cur]
            if self.augmentation:
                X[i] = self.augmentation(X[i])
                y[i] = self.augmentation(y[i])
            cur += 1
        return X, y

class SubPatchingClassificationDataGenerator(Sequence):
    def __init__(self, images_dir=None, patch_height=32, patch_width=32,
                 batch_size=32, shuffle=True, augmentation=None,
                 magnify=None, num_channels=3):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.image_paths = []
        self.image_labels = []
        self.dir_names = os.listdir(images_dir)
        for i in len(self.dir_names):
            dir_path = os.path.join(images_dir, dir_names[i])
            fnames = os.listdir(dir_path)
            self.image_paths += map(lambda name: os.path.join(dir_path, name), fnames)
            self.image_labels += [i]*len(fnames)
        self.shuffle = shuffle
        self.len = len(self.image_paths) * len(utils.get_image_subpatches(utils.read_image(self.image_paths[0]))) // self.batch_size
        self.magnify = magnify
        self.image_ptr = 0
        self.cur = 0
        self.org_patches = []

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if shuffle:
            self.image_names, self.image_labels = zip(*random.shuffle(list(zip(self.image_names, self.image_labels))))

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, self.resized_height, self.resized_width, self.num_channels))
        y = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            if self.cur == len(self.org_patches):
                image_ptr += 1
                cur = 0
                self.org_patches = utils.get_image_subpatches(utils.read_image(self.image_paths[image_ptr]))
            X[i] = self.org_patches[cur]
            y[i][self.image_labels[image_ptr]] = 1
            if self.augmentation:
                X[i] = self.augmentation(X[i])
            cur += 1
        return X, y