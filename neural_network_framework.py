import numpy as np
import pandas as pd

class Linear:
    def __init__(self, input_dim, output_dim):
        """
        Initializes a fully connected (linear) layer.

        Parameters:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        """
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))

    def forward(self, x):
        """
        Performs the forward pass through the linear layer.

        Parameters:
        x (ndarray): Input data of shape (n_samples, input_dim).

        Returns:
        ndarray: The output of the linear layer of shape (n_samples, output_dim).
        """
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dout):
        """
        Performs the backward pass through the linear layer.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        ndarray: Gradient of the loss with respect to the input of this layer.
        """
        self.dweights = np.dot(self.input.T, dout)
        self.dbiases = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.weights.T)

    def update_params(self, lr):
        """
        Updates the parameters of the layer using the calculated gradients.

        Parameters:
        lr (float): Learning rate for the update.
        """
        self.weights -= lr * self.dweights
        self.biases -= lr * self.dbiases

class ReLU:
    def forward(self, x):
        """
        Applies the ReLU activation function.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Output after applying ReLU, with the same shape as input.
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the loss with respect to the input of the ReLU function.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        ndarray: Gradient of the loss with respect to the input of this layer.
        """
        return dout * (self.input > 0)

class Sigmoid:
    def forward(self, x):
        """
        Applies the Sigmoid activation function.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Output after applying Sigmoid, with the same shape as input.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dout):
        """
        Computes the gradient of the loss with respect to the input of the Sigmoid function.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        ndarray: Gradient of the loss with respect to the input of this layer.
        """
        return dout * self.output * (1 - self.output)

class Tanh:
    def forward(self, x):
        """
        Applies the Tanh activation function.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Output after applying Tanh, with the same shape as input.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, dout):
        """
        Computes the gradient of the loss with respect to the input of the Tanh function.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        ndarray: Gradient of the loss with respect to the input of this layer.
        """
        return dout * (1 - self.output ** 2)

class Softmax:
    def forward(self, x):
        """
        Applies the Softmax activation function.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Output after applying Softmax, with the same shape as input.
        """
        exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward(self, dout):
        """
        Computes the gradient of the loss with respect to the input of the Softmax function.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
        ndarray: Gradient of the loss with respect to the input of this layer.
        """
        return dout  # Gradient computation depends on the specific loss function used

class CrossEntropyLoss:

    def forward(self, y_pred, y_true):
        """
        Computes the cross-entropy loss.

        Parameters:
        y_pred (ndarray): Predicted probabilities (output from softmax).
        y_true (ndarray): True labels, one-hot encoded.

        Returns:
        float: The cross-entropy loss.
        """
        samples = len(y_true)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.log(correct_confidences)

    def backward(self, y_pred, y_true):
        """
        Computes the cross-entropy loss.

        Parameters:
        y_pred (ndarray): Predicted probabilities (output from softmax).
        y_true (ndarray): True labels, one-hot encoded.

        Returns:
        float: The cross-entropy loss.
        """
        samples = len(y_true)
        grad = y_pred.copy()
        grad[range(samples), y_true] -= 1
        grad = grad / samples
        return grad

class MSELoss:
    def forward(self, y_pred, y_true):
        """
        Computes the Mean Squared Error (MSE) loss.

        Parameters:
        y_pred (ndarray): Predicted values.
        y_true (ndarray): True values.

        Returns:
        float: The MSE loss.
        """
        self.loss = np.mean((y_pred - y_true) ** 2)
        return self.loss

    def backward(self, dout):
        """
        Computes the gradient of the MSE loss.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output (usually 1).

        Returns:
        ndarray: Gradient of the loss with respect to the input.
        """
        return dout * 2 * (y_pred - y_true) / y_true.size

class SGD:
    def __init__(self, learning_rate=0.01):
        """
        Initializes the Stochastic Gradient Descent (SGD) optimizer.

        Parameters:
        learning_rate (float): The learning rate for the optimizer.
        """
        self.lr = learning_rate

    def step(self, layers):
        """
        Updates the parameters of the model's layers using the computed gradients.

        Parameters:
        layers (list): A list of layers in the model.
        """
        for layer in layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(self.lr)

class Model:
    def __init__(self):
        """
        Initializes the model, creating an empty list of layers.
        """
        self.layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the model.

        Parameters:
        layer: The layer to add to the model.
        """
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Compiles the model by setting the loss function and optimizer.

        Parameters:
        loss: The loss function to use.
        optimizer: The optimizer to use.
        """
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x):
        """
        Performs a forward pass through all layers of the model.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Output of the model.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        """
        Performs a backward pass through all layers of the model.

        Parameters:
        dout (ndarray): Gradient of the loss with respect to the output of the model.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def train(self, x_train, y_train, epochs, batch_size):
        """
        Trains the model on the training data.

        Parameters:
        x_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        epochs (int): Number of epochs to train for.
        batch_size (int): Number of samples per batch.
        """
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                y_pred = self.forward(x_batch)

                loss = self.loss.forward(y_pred, y_batch)

                dout = self.loss.backward(y_pred, y_batch)
                self.backward(dout)

                self.optimizer.step(self.layers)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    def predict(self, x):
        """
        Generates predictions for the input data.

        Parameters:
        x (ndarray): Input data.

        Returns:
        ndarray: Predicted values.
        """
        return self.forward(x)


    def evaluate(self, x_test, y_test):
        """
        Evaluates the model on the test data.

        Parameters:
        x_test (ndarray): Test data.
        y_test (ndarray): Test labels.

        Returns:
        tuple: A tuple containing the loss, accuracy and predicted values of the test set.
        """
        y_pred = self.predict(x_test)
        loss = self.loss.forward(y_pred, y_test)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_test) * 100
        y_out = np.argmax(y_pred, axis=1)
        return loss, accuracy, y_out

    def save(self, filepath):
        """
        Saves the model's weights and biases to a file.

        Parameters:
        filepath (str): The file path where the model should be saved.
        """
        np.savez(filepath, *[layer.weights for layer in self.layers if hasattr(layer, 'weights')],
                 *[layer.biases for layer in self.layers if hasattr(layer, 'biases')])

    def load(self, filepath):
        """
        Loads the model's weights and biases from a file.

        Parameters:
        filepath (str): The file path from where the model should be loaded.
        """
        data = np.load(filepath)
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights = data['arr_%d' % idx]
                idx += 1
            if hasattr(layer, 'biases'):
                layer.biases = data['arr_%d' % idx]
                idx += 1
