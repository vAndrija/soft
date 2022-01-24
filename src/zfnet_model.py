from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf

from src.pretprocesing import load_train_data


def create_model():
    model = Sequential()
    model.add(Conv2D(96, (7, 7), strides=(2, 2), input_shape=(30, 21, 2), padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.8))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()

    return model


def fit_model(model, train_data, train_labels, val_data, val_labels, name):
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=150, batch_size=256, verbose=2)
    model.summary()
    model.save_weights(name)


if __name__ == '__main__':
    train_data, train_label = load_train_data("../data/train.csv", "../data/train/")
    val_data, val_labels = load_train_data("../data/val.csv", "../data/val/")
    fit_model(create_model(), train_data, train_label, val_data, val_labels, "../model/weights-zfnet2.h5")
