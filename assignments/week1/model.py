import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = []
        self.b = 0

    """Fit the model to the data"""

    def fit(self, X, y):
        # Closed form
        X = np.array(X)
        y = np.array(y)
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.b = self.w[0]
        self.w = self.w[1:]

    """Make predictions with the model"""

    def predict(self, X):
        return self.w * X + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    """Fit the model to the data"""

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        for _ in range(epochs):
            Y_pred = self.predict(X)
            Dw = (-2 / len(X)) * sum(X * (y - Y_pred))
            Db = (-2 / len(X)) * sum(y - Y_pred)
            self.w = self.w - lr * Dw
            self.b = self.b - lr * Db

    """Make predictions with the model"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return self.w * X + self.b
