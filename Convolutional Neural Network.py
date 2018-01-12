##  Lynn Regressor 2017.11.22


import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam


# data load
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1, 28, 28)/ 255
X_test = X_test.reshape(-1, 1, 28, 28)/ 255

Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)


# build your convolutional Neural Network
model = Sequential()
# first layer (32,28,28)
model.add(Convolution2D(
    input_shape=(1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
    activation='relu'
))
# second layer 输入 (32,28,28) 输出 (32,14,14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))
# third layer输入 (32,14,14) 输出(64,14,14)
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
    activation='relu'
))
# forth layer 输入(64,14,14) 输出(64,7,7)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first'
))

# fifth layer 输入(64,7,7) 输出(64*7*7,1)
model.add(Flatten())

# fully connection
model.add(Dense(output_dim=1000, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))

# complier
adam = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


# train
model.fit(X_train, Y_train, epochs=1, batch_size=64)

# test
loss, accuracy = model.evaluate(X_test, Y_test)
print('loss', loss)
print('accuracy', accuracy)