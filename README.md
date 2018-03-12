# MNIST Convolutional Neural Network

## Overview
This code trains a convolutional neural network on the MNIST dataset. Uses the [Keras](https://keras.io/) and [NumPy](http://www.numpy.org/) libraries.

## Architecture
INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> DROPOUT -> FC

## Accuracy
This model has achieved 99.19% test accuracy on the MNIST dataset with batch size of 128 and 12 epochs.

## Dependencies

```sudo pip3 install -r requirements.txt```

## Usage
```python3 mnist_conv.py```
