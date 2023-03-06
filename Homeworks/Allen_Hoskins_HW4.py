from __future__ import print_function
import datetime
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras import backend as K
import os
from keras.utils import img_to_array, load_img
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
import numpy as np

now = datetime.datetime.now
batch_size = 128
epochs = 10
num_classes = 10
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3
if K.image_data_format() == "channels_first":
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )
    t = now()
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
    )
    print("Training time: %s" % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])


(x_train, y_train), (x_test, y_test) = mnist.load_data()

feature_layers = [
    Conv2D(filters, kernel_size, padding="valid", input_shape=input_shape),
    Activation("relu"),
    Conv2D(filters, kernel_size),
    Activation("relu"),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation("relu"),
    Dropout(0.5),
    Dense(num_classes),
    Activation("softmax"),
]

# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model, (x_train, y_train), (x_test, y_test), num_classes=10)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False


folder = "../letters HW4/"
x = np.array([])
y = np.array([])
for f in os.listdir(folder):
    if f.endswith(".png"):
        img = load_img(
            os.path.join(folder, f), color_mode="grayscale", target_size=(28, 28)
        )
        img = img_to_array(img)
        label = ord(f[0]) - ord("A")
        y = np.append(y, label)
        if len(x) == 0:
            x = np.array([img])
        else:
            x = np.vstack((x, [img]))

x = x.reshape(-1, 28, 28, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


model.pop()

classification_layers2 = [
    Dense(128),
    Activation("relu"),
    Dropout(0.5),
    Dense(5),
    Activation("softmax"),
]

model2 = Sequential(feature_layers + classification_layers2)


# Freeze feature layers
for l in feature_layers:
    l.trainable = False

# Train model2 on letter dataset
train_model(model2, (x_train, y_train), (x_test, y_test), num_classes=5)
