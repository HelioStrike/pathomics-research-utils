import os
import random
import numpy as np
from copy import deepcopy
from pathomics_research_utils import utils
from tensorflow.keras.utils import Sequence

class ClassificationDataGenerator(Sequence):
    def __init__(self, images_dir=None,
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
                    img = utils.read_image(os.path.join(dir_path, fname))
                    self.image_height = img.shape[0]
                    self.image_width = img.shape[1]
                    self.images.append(img)
                    self.image_labels += [i]
                    cur+=1
                except:
                    pass
        self.magnify = magnify
        self.len = len(self.images) // self.batch_size

    def __len__(self):
        return self.len
    
    def on_epoch_start(self):
        if shuffle:
            self.images, self.image_labels = zip(*random.shuffle(list(zip(self.images, self.image_labels))))

    def __getitem__(self, idx):
        print(idx)
        X = np.empty((self.batch_size, self.image_height, self.image_width, self.num_channels))
        y = np.zeros((self.batch_size, self.num_classes))
        for i in range(self.batch_size):
            X[i] = self.images[idx*self.batch_size+i]
            y[self.image_labels[idx*self.batch_size+i]] = 1
            if self.augmentation:
                X[i] = self.augmentation(X[i])
        return X, y