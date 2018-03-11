import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
import numpy
from keras.datasets import mnist

### DATA SETUP

width, height, depth = 28, 28, 1
num_classes = 10

# Splitting the data for training and testing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping the input data to be compatible with the convolutional net
# Format is: reshape(# of entries, width of each entry, height of each entry, depth of each entry)
X_train = X_train.reshape(60000, width, height, 1).astype('float32')
X_test = X_test.reshape(10000, width, height, 1).astype('float32')

# Normalizing the input data between 0 and 1
X_train /= 255
X_test /= 255

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

######################

### Creating the Convolutional Model

# Architecture: INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> -> DROPOUT->FC

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape = (width, height, depth)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

### Train the Model
model.fit(X_train, y_train, epochs=5)

########################
### Evaluating the model
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy: {}".format(acc))
print("Loss: {}".format(loss))
