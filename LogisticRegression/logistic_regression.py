import numpy as np

class logistic_regression:
    def __init__(self, feature_dim, nclasses):
        self.d = feature_dim
        self.k = nclasses
        # Initalize parameters
        self.w = np.random.uniform(low=-1, high=1, size=(self.d, self.k))
        self.b = np.random.uniform(low=-1, high=1, size=self.k)

    def softmax(self, x):
        score = np.exp(x)
        score_sum = np.sum(score, axis = 1, keepdims=True)
        probability = score / score_sum

        return probability

    def forward(self, X):
        o = X.dot(self.w) + self.b
        return self.softmax(o)
