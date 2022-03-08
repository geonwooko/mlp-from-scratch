import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Make dataloader as iterator for SGD
class dataloader:
    def __init__(self, X, t, bs=64):
        self.N = X.shape[0]
        self.bs = bs
        self.batch_list = []

        # Split X, t by batch size
        idx = 0
        self.n_batch = 0

        while (True):
            if (idx + self.bs < self.N):
                self.batch_list.append((X[idx:idx + self.bs], t[idx:idx + self.bs]))
                idx += self.bs
                self.n_batch += 1
            else:
                self.batch_list.append((X[idx:self.N], t[idx:self.N]))
                self.n_batch += 1
                break

    def __iter__(self):
        self.index = 0
        return self

    # Get next items
    def __next__(self):
        if self.index < self.n_batch:
            batch_list = self.batch_list[self.index]

            self.index += 1
            return batch_list[0], batch_list[1]
        # If all batches are read
        else:
            np.random.shuffle(self.batch_list)  # Shuffle the mini-batch list
            self.index = 0
            raise StopIteration

def get_MNIST_data():
    # Load the diabetes dataset
    MNIST_X, MNIST_t = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

    # preprocess the feature
    MNIST_X = MNIST_X.to_numpy().astype(np.float32)
    MNIST_X /= 255.
    MNIST_X -= MNIST_X.mean(axis=0)

    # Make target to one-hot vector
    encoder = OneHotEncoder(sparse=False)
    MNIST_t = encoder.fit_transform(MNIST_t.to_numpy().reshape(-1, 1))

    # Split train / valid / test data with straify sampling
    X_not_test, X_test, t_not_test, t_test = train_test_split(
        MNIST_X.astype(np.float32),
        MNIST_t.astype(np.float32),
        test_size=0.1,
        stratify = MNIST_t
    )

    X_train, X_valid, t_train, t_valid = train_test_split(
        X_not_test.astype(np.float32),
        t_not_test.astype(np.float32),
        test_size=(0.05 / 0.9),
        stratify = t_not_test
    )

    MNIST_dataset = {}
    MNIST_dataset["X"] = {}
    MNIST_dataset["t"] = {}

    MNIST_dataset['X']['train'] = X_train
    MNIST_dataset['X']['valid'] = X_valid
    MNIST_dataset['X']['test'] = X_test

    MNIST_dataset['t']['train'] = t_train
    MNIST_dataset['t']['valid'] = t_valid
    MNIST_dataset['t']['test'] = t_test

    return MNIST_dataset