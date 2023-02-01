import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = []
        self.b = 0

    def fit(self, X, y):
        # Closed form 
        X = np.array(X)
        y = np.array(y)
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.b = self.w[0]
        self.w = self.w[1:]


    def predict(self, X):
        X = np.array(X)
        X = np.c_[np.ones((len(X), 1)), X]
        return X.dot(np.r_[self.b, self.w])


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        raise NotImplementedError()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        raise NotImplementedError()
