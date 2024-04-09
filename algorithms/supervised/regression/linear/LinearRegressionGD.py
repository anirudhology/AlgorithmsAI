import numpy as np

"""
Linear Regression Algorithm using Gradient Descent
"""


class LinearRegressionGD:
    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.training_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Predict values
            y_predicted = self.predict(X)
            # Reshape y to match y_predicted
            y_predicted_reshaped = y_predicted.reshape(-1, 1)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted_reshaped - y))
            db = (1 / n_samples) * np.sum(y_predicted_reshaped - y)

            # Reshape dw to match self.weights
            dw_flat = dw.flatten()

            # Update parameters
            self.weights -= self.learning_rate * dw_flat
            self.bias -= self.learning_rate * db

            # Calculate cost (MSE)
            cost = self.calculate_cost(X, y)
            self.training_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def calculate_cost(self, X, y):
        n_samples = len(X)
        prediction = self.predict(X)
        error = prediction - y
        cost = np.sum(error ** 2) / (2 * n_samples)
        return cost
