import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from pretprocesing import load_train_data


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(30, 21, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model


def fit_model(model, train_data, train_labels, validation_data, validation_labels):
    model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs=100,
              batch_size=256, verbose=2)
    model.summary()
    model.save_weights("../model/weights3.h5")


if __name__ == '__main__':
    train_data, train_label = load_train_data("../data/train.csv", "../data/train/")
    validation_data, validation_labels = load_train_data("../data/val.csv", "../data/val/")
    fit_model(create_model(), train_data, train_label, validation_data, validation_labels)
