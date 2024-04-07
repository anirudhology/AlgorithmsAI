import numpy as np

"""
Linear Regression Algorithm implementation using Ordinary Least Squares method.
This method is also called "closed-form" or "normal-form" solution.
"""


class LinearRegressionOLS:
    def __init__(self):
        # Coefficients/weights of features
        self.coefficients = None
        # Intercept term
        self.intercept = None

    def fit(self, x, y):
        """
        Fit the linear regression model to the given data.

        :param x: features of the given data; array-like shape. Training data.
        :param y: observed values; array-like shape. Observed output
        """
        # Since in the given features, there is no provision for adding the intercept,
        # we will add one more column of ones for the intercept.
        x = np.c_[np.ones(x.shape[0]), x]

        # Perform normal equation to solve for intercept and coefficients
        solution = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

        # Intercept
        self.intercept = solution[0]
        # Coefficients
        self.coefficients = solution[1:]

    def predict(self, x):
        """
        Predict target value based on the given features and previously calculated
        values of intercept and coefficients
        :param x: Input features
        :return: y_predicted - predicted value of target
        """
        if self.coefficients is None:
            raise Exception("Model has not been trained yet. Please call fit() first")

        # Add intercept term
        x = np.c_[np.ones(x.shape[0]), x]

        # Calculate predictions
        y_predicted = x.dot(np.concatenate([[self.intercept], self.coefficients]))
        return y_predicted
