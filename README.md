# SixVision

SixVision is a deep learning model designed to diagnose cataracts for the purpose of early treatment. It achieves a high accuracy rate of 93.41% using a dataset of only 900 images for training. This model was developed as part of the John & Violet Kay Summer Research Fellowship. For more detailed information about the research behind this model, please refer to the [research paper](https://docs.google.com/document/d/11tB722xq19NcC6rRLnfAbXxbYVvLGwq3LYyqD7YDi3c/edit?usp=sharing).

## Table of Contents
- [Introduction](#sixvision)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Building the CNN](#building-the-cnn)
- [Training the CNN](#training-the-cnn)
- [Evaluation](#evaluation)
- [Making Predictions](#making-predictions)
- [License](#license)

## Installation
To use SixVision, you need to have the following dependencies installed:

- numpy
- tensorflow
- matplotlib
- keras

You can install these dependencies using the following command:
    pip install numpy tensorflow matplotlib keras

## Data Preprocessing
Before training the model, it is necessary to preprocess the data. The following steps are performed:

1. Import the required libraries, including numpy, tensorflow, matplotlib, and keras.
2. Set the image size and input shape for the model.
3. Define the number of epochs and batch size.
4. Create an image data generator for the training set, which applies data augmentation techniques such as rotation, shifting, shear, zooming, and flipping.
5. Load the training and test datasets using the image data generators, specifying the target size and batch size.

## Building the CNN
The Convolutional Neural Network (CNN) architecture is constructed in the following steps:

1. Initialize the CNN as a sequential model.
2. Add a convolutional layer to the model with 32 filters, a kernel size of 3, and ReLU activation function.
3. Add a max-pooling layer to the model with a pool size of 2 and strides of 2.
4. Add a second convolutional layer and another max-pooling layer.
5. Flatten the output of the previous layer to prepare for the fully connected layers.
6. Add a dense layer with 128 neurons and ReLU activation.
7. Apply dropout regularization to reduce overfitting.
8. Add the output layer with a sigmoid activation function to classify the images.

## Training the CNN
To train the CNN on the training set and evaluate its performance on the test set, follow these steps:

1. Compile the CNN using the RMSprop optimizer with a learning rate of 0.01 and binary cross-entropy loss.
2. Fit the model to the training set using the specified number of epochs.
3. Plot the training and validation accuracy for each epoch.
4. Plot the training and validation loss for each epoch.
5. Repeat the training process with different learning rates (1e-2, 1e-3, 1e-4, 1e-5) and plot the validation accuracy and loss for each learning rate.

## Evaluation
The performance of the model can be assessed using the plotted accuracy and loss curves. Higher accuracy and lower loss indicate better performance. The model achieved an accuracy rate of 93.41% on the test set.

## Making Predictions
To make a prediction using the trained model, follow these steps:

1. Load the test image and preprocess it to match the input size of the model.
2. Use the model to predict the class of the image.
3. Map the prediction result to the corresponding class label.
4. Display the predicted class (either "normal" or "cataract").

## License
This project is licensed under the [MIT License](LICENSE).
