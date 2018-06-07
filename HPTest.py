import numpy as np
from sys import stdin
from keras import models, layers, regularizers
from keras.datasets import mnist
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LyrParams(object):

    def __init__(self, channels, dropRate = 0, regType = None):
        self.channels = channels
        self.dropRate = dropRate
        self.regType = regType

    def printP(self):
        print(self.channels, 'Channels\t', self.dropRate, 'Drop Rate\t',
            self.regType, 'Reg Type')


def main():
    myParams = genParams()
    layersD = []

    f = open('HPTest.out', 'w')

    nn = models.Sequential()

    for i in range(0, len(myParams)):
        printableParams = str(myParams[i].channels)
            + ' Channels    ' + str(myParams[i].dropRate)
            + ' Drop Rate    ' + str(myParams[i].regType)
            + ' Reg Type\n'
        f.write(printableParams)

        if(i == 0):
            if(myParams[i].regType == None):
                nn.add(layers.Conv2D(
                    myParams[i].channels, 
                    (3, 3), 
                    activation = 'relu', 
                    input_shape = (28, 28, 1)
                    )
                )
            else:
                if(myParams[i].regType == 'kernel'):
                    nn.add(layers.Conv2D(
                        myParams[0].channels, 
                        (3, 3),
                        activation = 'relu', 
                        input_shape = (28, 28, 1),
                        kernel_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'activity'):
                    nn.add(layers.Conv2D(
                        myParams[i].channels, 
                        (3, 3),
                        activation = 'relu', 
                        input_shape = (28, 28, 1),
                        activity_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'both'):
                    nn.add(layers.Conv2D(
                        myParams[i].channels, 
                        (3, 3),
                        activation = 'relu', 
                        input_shape = (28, 28, 1),
                        kernel_regularizer = regularizers.l2(0.01),
                        activity_regularizer = regularizers.l1(0.01)
                        )
                    )
                else:
                    print("Bad regType...Exiting\n")
                    exit()

            if(myParams[i].dropRate != 0):
                nn.add(layers.Dropout(myParams[i].dropRate))

            nn.add(layers.MaxPooling2D((2, 2)))                  

        elif(i < (len(myParams)-1)):
            if(myParams[i].regType == None):
                nn.add(layers.Conv2D(
                    myParams[i].channels, 
                    (3, 3), 
                    activation = 'relu'
                    )
                )
            else:
                if(myParams[i].regType == 'kernel'):
                    nn.add(layers.Conv2D(
                        myParams[i].channels, 
                        (3, 3),
                        activation = 'relu', 
                        kernel_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'activity'):
                    nn.add(layers.Conv2D(
                        myParams[i].channels, 
                        (3, 3),
                        activation = 'relu', 
                        activity_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'both'):
                    nn.add(layers.Conv2D(
                        myParams[i].channels, 
                        (3, 3),
                        activation = 'relu', 
                        kernel_regularizer = regularizers.l2(0.01),
                        activity_regularizer = regularizers.l1(0.01)
                        )
                    )
                else:
                    print("Bad regType...Exiting\n")
                    exit()

            if(myParams[i].dropRate != 0):
                nn.add(layers.Dropout(myParams[0].dropRate))

            if(i < (len(myParams) - 2)):
                nn.add(layers.MaxPooling2D((2, 2)))                  
            else:
                nn.add(layers.Flatten())

        else:
            if(myParams[i].regType == None):
                nn.add(layers.Dense(
                    myParams[i].channels, 
                    activation = 'relu'
                    )
                )
            else:
                if(myParams[i].regType == 'kernel'):
                    nn.add(layers.Dense(
                        myParams[i].channels, 
                        activation = 'relu' ,
                        kernel_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'activity'):
                   nn.add(layers.Dense(
                        myParams[i].channels, 
                        activation = 'relu', 
                        activity_regularizer = regularizers.l2(0.01)
                        )
                    )
                elif(myParams[i].regType == 'both'):
                    nn.add(layers.Dense(
                        myParams[i].channels, 
                        activation = 'relu', 
                        kernel_regularizer = regularizers.l2(0.01),
                        activity_regularizer = regularizers.l1(0.01)
                        )
                    )
                else:
                    print("Bad regType...Exiting\n")
                    exit()

            if(myParams[i].dropRate != 0):
                nn.add(layers.Dropout(myParams[i].dropRate))
    

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

    hst = nn.fit(train_data, train_labels, epochs = 4, batch_size = 64,
        validation_data = (test_data, test_labels), verbose = 0)

    hst = hst.history

    for i in range(0, len(hst['val_loss'])):
        string = '\t' + str(hst['acc'][i]) + '/' 
            + str(hst['loss'][i]) + '    ' 
            + str(hst['val_acc'][i]) + '/' 
            + str(hst['val_loss'][i]) + '\n' 

        f.write(string)
    f.close()


def genParams():
    listParams = [
        [32, 1.0, None],
        [64, 1.0, None],
        [64, 1.0, None],
        [64, 1.0, None]
    ]

    listLyrParams = []
    for params in listParams:
        listLyrParams.append(LyrParams(params[0], params[1], params[2]))
    
    return listLyrParams


if __name__ == "__main__":
    main()

