import numpy as np

"""
Linear Regression Algorithm using Gradient Descent
"""


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-4) -> None:
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.coefficients = None
        self.intercept = None
        self.cost_history = []

    def fit(self, x, y):
        """
        Fit the linear regression model to the given data.

        :param x: features of the given data; array-like shape. Training data.
        :param y: observed values; array-like shape. Observed output
        """
        # Add a column of ones to x for intercept term
        x = np.c_[np.ones(x.shape[0]), x]

        # Initialize coefficients and intercepts to a random value
        self.coefficients = np.random.randn(x.shape[1])
        self.intercept = np.random.randn()

        # Perform Gradient Descent for max_iterations
        for i in range(self.max_iterations):
            # Compute predictions
            y_predicted = np.dot(x, self.coefficients) + self.intercept

            # Compute error
            error = y_predicted - y

            # Compute gradients
            gradient_coefficients = 2 * np.dot(x.T, error) / len(y)
            gradient_intercept = 2 * np.sum(error) / len(y)

            # Update coefficients and intercepts for this iteration
            self.coefficients -= self.learning_rate * gradient_coefficients
            self.intercept -= self.learning_rate * gradient_intercept

            # Compute cost for current iteration (MSE)
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)

            # Check for convergence
            if len(self.cost_history) > 1 and abs(cost - self.cost_history[-2]) < self.tolerance:
                break

    def predict(self, x):
        """
        Predict target value based on the given features and previously calculated
        values of intercept and coefficients
        :param x: Input features
        :return: y_predicted - predicted value of target
        """
        # Add a column of ones for intercept term
        x = np.c_[np.ones(x.shape[0]), x]

        return np.dot(x, self.coefficients) + self.intercept
