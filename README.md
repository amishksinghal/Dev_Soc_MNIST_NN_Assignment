# Dev_Soc_NN_Assignment

# Neural Network Framework

This repository contains a simple neural network framework built from scratch using NumPy. The framework provides the essential components to build, train, and evaluate neural networks. It is designed to be easy to understand and extend.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Loading Model Weights](#loading-model-weights)
- [Example](#example)
- [Google Collab Notebook](#google-collab-notebook)

## Features

- **Linear Layers**: Fully connected layers for building deep networks.
- **Activation Functions**: Includes ReLU, Sigmoid, Tanh, and Softmax.
- **Loss Functions**: Cross-Entropy Loss and Mean Squared Error (MSE).
- **Optimizer**: Stochastic Gradient Descent (SGD) for updating model parameters.
- **Model Class**: Encapsulates the entire neural network model, including methods for training, predicting, evaluating, saving, and loading.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/amishksinghal/Dev_Soc_NN_Assignment.git
   cd Dev_Soc_NN_Assignment
   ```

2. Install the required dependencies:

   ```bash
   pip install numpy
   ```

## Usage

### Building a Model

To create a model, import the necessary components from the framework and add layers to the model.

```python
from neural_network_framework import Model, Linear, ReLU, Softmax, CrossEntropyLoss, SGD

# Initialize the model
model = Model()

# Add layers
model.add_layer(Linear(784, 128))
model.add_layer(ReLU())
model.add_layer(Linear(128, 10))
model.add_layer(Softmax())

# Compile the model with a loss function and an optimizer
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)
```

### Training the Model

Train your model using the `train` method.

```python
model.train(x_train, y_train, epochs=20, batch_size=64)
```

### Evaluating the Model

Evaluate the model’s performance on a test dataset.

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
```

## Loading Model Weights

You can save and load the model's weights using the `save` and `load` methods.

### Saving the Model Weights

```python
model.save('model_weights.npy')
```

### Loading the Model Weights

To load the saved weights into your model:

```python
model.load('model_weights.npy')
```

Make sure that the architecture of the model when loading the weights matches the one used during saving.

## Example

Below is a complete example of using the framework to train a neural network on the MNIST dataset.

```python
from neural_network_framework import Model, Linear, ReLU, Softmax, CrossEntropyLoss, SGD

# Initialize the model
model = Model()

# Add layers
model.add_layer(Linear(784, 128))
model.add_layer(ReLU())
model.add_layer(Linear(128, 10))
model.add_layer(Softmax())

# Compile the model with a loss function and an optimizer
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)
model.compile(loss, optimizer)

# Train the model
model.train(x_train, y_train, epochs=20, batch_size=64)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the model weights
model.save('model_weights.npy')

# Load the model weights
model.load('model_weights.npy')
```

## Google Collab Notebook

You can find the notebook where this framework was used to train a model on the MNIST dataset [here](https://colab.research.google.com/drive/1V3ahHYvdxdbUrMJlb324hddI_44wqHSN?usp=sharing).
