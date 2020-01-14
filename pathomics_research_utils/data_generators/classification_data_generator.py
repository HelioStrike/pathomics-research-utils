import os
import random
import numpy as np
import cv2
from copy import deepcopy
from pathomics_research_utils import utils
from tensorflow.keras.utils import Sequence

#Use the path to a directory containing nested directories of images (each corresponding to a class)
class ClassificationDataGenerator(Sequence):
    def __init__(self, images_dir=None, height=32, width=32, resize=False,
                 batch_size=32, shuffle=True, augmentation=None,
                 magnify=None, num_channels=3, num_classes=3):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation = augmentation

        self.shuffle = shuffle
        self.images = []
        self.image_labels = []
        self.dir_names = os.listdir(images_dir)
        for i in range(len(self.dir_names)):
            dir_path = os.path.join(images_dir, self.dir_names[i])
            fnames = os.listdir(dir_path)
            cur=0
            for fname in fnames:
                try:
                    img = os.path.join(dir_path, fname)
                    self.images.append(img)
                    self.image_labels += [i]
                    cur+=1
                except:
                    pass
        self.height = height
        self.width = width
        self.resize = resize
        self.magnify = magnify
        self.len = len(self.images) // self.batch_size
        if self.shuffle:
            self.shuffle_data()

    def __len__(self):
        return self.len

    def shuffle_data(self):
        a = list(zip(self.images, self.image_labels))
        random.shuffle(a)
        self.images, self.image_labels = zip(*a)

    def on_epoch_start(self):
        if self.shuffle:
            self.shuffle_data()

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, self.height, self.width, self.num_channels))
        y = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            img = utils.read_image(self.images[idx*self.batch_size+i])
            if self.resize:
                img = cv2.resize(img, (self.height, self.width))
            X[i] = img
            y[i][self.image_labels[idx*self.batch_size+i]] = 1
            if self.augmentation:
                X[i] = self.augmentation(X[i])
        return X, y