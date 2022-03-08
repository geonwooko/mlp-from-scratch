import numpy as np

class linear_regression:

    def __init__(self, feature_dim):
        self.d = feature_dim

        # Initalize parameters
        self.w = np.random.uniform(low=-1, high=1, size=feature_dim)
        self.b = np.random.uniform(low=-1, high=1, size=1)

    def forward(self, X):
        return X.dot(self.w) + self.b
