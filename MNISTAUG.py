from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing import image
from sys import exit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.set_printoptions(threshold = np.nan)
import h5py

nn = models.Sequential()
nn.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
nn.add(layers.Dropout(1))
nn.add(layers.MaxPooling2D((2, 2)))                  

nn.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
nn.add(layers.Dropout(1))
nn.add(layers.MaxPooling2D((2, 2)))                  

nn.add(layers.Conv2D(64, (3, 3), activation = 'relu')) 
nn.add(layers.Dropout(1))
nn.add(layers.Flatten())                            

nn.add(layers.Dense(64, activation = 'relu'))
nn.add(layers.Dropout(1))

nn.add(layers.Dense(10, activation = 'softmax')) 

nn.compile(
    optimizer = "rmsprop",            
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']            
)

(train_data, train_labels), (test_data, test_labels) \
    = mnist.load_data()

train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data.astype('float32') / 255  
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

datagen = image.ImageDataGenerator(
    rotation_range = 1,
    width_shift_range = 0.01,
    height_shift_range = 0.01,
    shear_range = 0.03,
    zoom_range = 0.08,
)

aug_iter = datagen.flow(train_data, train_labels, \
    batch_size = len(train_data))

aug_images = [next(aug_iter) for i in range(4)]

aug_data = np.asarray([img[0] for img in aug_images]) \
    .reshape((240000, 28, 28, 1))

aug_labels = np.asarray([img[1] for img in aug_images]) \
    .reshape((240000, 10))

test_datagen = image.ImageDataGenerator()

aug_test = test_datagen.flow(test_data, test_labels, 
    batch_size = len(test_data))

test_images = next(aug_test)
test_d = np.asarray(test_images[0]).reshape(10000, 28, 28, 1)
test_l = np.asarray(test_images[1]).reshape(10000, 10)

hst = nn.fit(aug_data, aug_labels, epochs = 4, 
    batch_size = 64, validation_data = (test_d, test_l))

nn.save('MNIST.h5')