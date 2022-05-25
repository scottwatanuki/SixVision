# Importing the libraries
# numpy is for calculations
# tensorflow is a library for deep learning
# matplotlib displyas data
# keras is a library based on tensorflow specifically for neural networks
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import os
# from uti.addreg import *
import pickle
from keras.preprocessing import image
from keras.layers import Dropout

# Part 1 - Data Preprocessing

# Preprocessing the Training set
# prevents overfitting; this is data augmentation
# applies feature scaling to all pixels.
image_size = 224
input_shape = (image_size, image_size, 3) 
print("success")

epochs = 150
batch_size = 16

#altering data to allow for better training
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
training_set = train_datagen.flow_from_directory(r"C:\Users\Esports_Player\Downloads\CNN\dataset1\training_set",
                                                target_size=(image_size, image_size),
                                                batch_size=batch_size,
                                                class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(r"C:\Users\Esports_Player\Downloads\CNN\dataset1\test_set",
                                           target_size=(image_size, image_size),
                                           batch_size=batch_size,
                                           class_mode='binary')

# Part 2 - Building the CNN

# Initialising the CNN
# create CNN as sequence of layers
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# filters - number of output filters in the convolution
# kernel_size - height and width of the 2d convolution layer
# input_shape - 3 bc color image. 1 for black & white
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[image_size, image_size, 3]))

# Step 2 - Pooling
# pool_size - height and width of the pool
# strides - shifting pool size
# recommended for max pooling is 2 and 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening - make it into 1d to input for the final layer
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# units - number of fully connected neurons
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(Dropout(0.5))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
# adam - for stochastic gradient descent and minimum loss
opt = RMSprop(learning_rate=0.01)
cnn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
# epoch - increase it
e01 = cnn.fit(x=training_set, validation_data=test_set, epochs=epochs)

# history plot for accuracy
plt.plot(e01.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# history plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# Training the CNN on the Training set and evaluating it on the Test set
# epoch - increase it
e1 = model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy'])
e1history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)
e2 = model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
e2history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)
e3 = model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
e3history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)
e4 = model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
e4history = model.fit(x=training_set, validation_data=test_set, epochs=epochs)

plt.plot(e1.e1history["val_accuracy"], label='1e-2')
plt.plot(e2.e2history["val_accuracy"], label='1e-3')
plt.plot(e3.e3history["val_accuracy"], label='1e-4')
plt.plot(e4.e4history["val_accuracy"], label='1e-5')

# # history plot for accuracy
# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy of Different Learning Rates on Adam Optimizer")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['1e-2', '1e-3', '1e-4', '1e-5'])
plt.show()

plt.plot(e1.e1history["val_loss"], label='1e-2')
plt.plot(e2.e2history["val_loss"], label='1e-3')
plt.plot(e3.e3history["val_loss"], label='1e-4')
plt.plot(e4.e4history["val_loss"], label='1e-5')

plt.title("Model Loss of Different Learning Rates on Adam Optimizer")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['1e-2', '1e-3', '1e-4', '1e-5'])
plt.show()

# Part 4 - Making a single prediction
test_image = image.load_img(
   r"C:\Users\Esports_Player\Downloads\CNN\dataset1\single_prediction\normal.png",
   target_size=(image_size, image_size))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
   prediction = 'normal'
else:
   prediction = 'cataract'
print(prediction)