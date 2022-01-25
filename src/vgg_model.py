import numpy as np
import keras
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Normalization

from src.pretprocesing import load_train_data


def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(30, 21, 2), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))

    model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model


def fit_model(model, train_data, train_labels, validation_data, validation_labels, name):
    model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs=100,
              batch_size=256, verbose=2)
    model.summary()
    model.save_weights(name)


if __name__ == '__main__':
    train_data, train_label = load_train_data("../data/train.csv", "../data/train/")
    validation_data, validation_labels = load_train_data("../data/val.csv", "../data/val/")
    fit_model(create_model(), train_data, train_label, validation_data, validation_labels, "../model/weights-vgg1.h5")
