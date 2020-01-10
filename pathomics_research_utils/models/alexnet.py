import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Sequential

def AlexNet(input_shape=(32,32,3), num_classes=3, dropout=0):
    model = Sequential()
    model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3,3), 2))
    model.add(layers.Conv2D(32, (5,5), input_shape=(32,32,3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model