from keras import models, layers
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.preprocessing import image
from sys import exit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
np.set_printoptions(threshold=np.nan)
import h5py

nn = models.load_model('MNIST.h5')

(train_data, train_labels), (test_data, test_labels) \
    = mnist.load_data()

train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

train_data = train_data.astype('float32') / 255  
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

predictions = nn.predict(test_data, 
    batch_size = len(test_data), steps = None, verbose = 0)

for p in predictions:
    print(np.argmax(p, axis = None, out = None))