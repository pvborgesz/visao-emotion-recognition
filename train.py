import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from math import ceil

# Loads csv files and appends pixels to X and labels to y

def preProcessData():
    data = pd.read_csv('data/fer2013.csv')
    labels = pd.read_csv('data/fer2013new.csv')

    classNames = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    nSamples = len(data)
    w = 48
    h = 48

    y = np.array(labels[classNames])
    X = np.zeros((nSamples, w, h, 1))
    for i in range(nSamples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def cleanDataAndNormalize(X, y): 
    classNames = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Usando máscara para remover imagens desconhecidas ou NF
    y_mask = y.argmax(axis=-1)
    mask = y_mask < classNames.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y



def splitData(X, y):
    test_size = ceil(len(X) * 0.1)

    # Dividindo o dataset em Train e Test
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size)

    return x_train, y_train, x_val, y_val, x_test, y_test


def dataAugmentation(x_train): # Data augmentation para balancear o dataset
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen


def showAugmentedImages(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
        plt.xlabel(class_names[y_train[i]])
    plt.show()


def define_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64

    model = Sequential() # Camada de entrada

    #i wanna build a model with yolo architecture
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))


    model.add(Dense(classes, activation='softmax'))

    # return model

    # # i wanna build a model with SSD architecture
    # # 1st block
    # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    # model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))

    # # 2nd block
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))
    
    # # 3rd block
    # model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu'))
    # model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))


    # # 1st stage 
    # model.add(Conv2D(num_features, kernel_size=(3, 3), input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Conv2D(num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Dropout(0.5)) #Dropout is a simple and powerful regularization technique for neural networks and deep learning models.

    # # 2nd stage
    # model.add(Conv2D(num_features, (3, 3), activation='relu'))
    # model.add(Conv2D(num_features, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # # 3rd stage
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Conv2D(2 * num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))

    # # 4th stage
    # model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    # model.add(Conv2D(2 * num_features, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # # 5th stage
    # model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))
    # model.add(Conv2D(4 * num_features, kernel_size=(3, 3)))
    # model.add(BatchNormalization())
    # model.add(Activation(activation='relu'))

    # model.add(Flatten()) # achatando o modelo

    # # Fully connected neural networks
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))

    # model.add(Dense(classes, activation='softmax'))

    return model