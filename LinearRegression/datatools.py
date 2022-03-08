import numpy as np
import pickle
from sklearn import datasets

# Make dataloader as iterator for SGD
class dataloader:
    def __init__(self, X, t, bs=100):
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

# Get dataset and get parameters if it has it
def get_dataset_with_parameters(data_name):
    assert data_name in ["random_dataset", "diabetes_dataset"]

    if data_name == "random_dataset":
        # Load the random generated data
        with open(f"datasets/myrandomdataset.pkl", "rb") as f:
            random_dataset = pickle.load(f)
        with open(f"datasets/parameters_myrandomdataset.pkl", "rb") as f:
            random_dataset_parameters = pickle.load(f)
        return random_dataset, random_dataset_parameters

    else:
        diabetes_dataset = get_diabetes_data()
        return diabetes_dataset, None

def get_diabetes_data():
    # Load the diabetes dataset
    diabetes_X, diabetes_t = datasets.load_diabetes(return_X_y=True)
    n_total = diabetes_X.shape[0]
    train_valid_test_rate = [0.85, 0.05, 0.1]

    # Cacluate each dataset's number
    n_train = round(n_total * train_valid_test_rate[0])
    n_valid = round(n_total * train_valid_test_rate[1])
    n_test = round(n_total * train_valid_test_rate[2])
    if (n_train + n_valid + n_test) == n_total:  # 1개가 남을 경우
        n_valid += 1

    # Sample each dataset's index
    total_idx = list(range(n_total))
    train_idx = np.random.choice(total_idx, size=n_train, replace=False)
    remain_idx = list(set(total_idx) - set(train_idx))
    valid_idx = np.random.choice(remain_idx, size=n_valid, replace=False)
    test_idx = list(set(remain_idx) - set(valid_idx))

    # Split dataset
    X_train, t_train = diabetes_X[total_idx], diabetes_t[total_idx]
    X_valid, t_valid = diabetes_X[valid_idx], diabetes_t[valid_idx]
    X_test, t_test = diabetes_X[test_idx], diabetes_t[test_idx]

    # Save all data in dictionary
    diabetes_dataset = {}
    diabetes_dataset["X"] = {}
    diabetes_dataset["t"] = {}

    diabetes_dataset['X']['train'] = X_train
    diabetes_dataset['X']['valid'] = X_valid
    diabetes_dataset['X']['test'] = X_test

    diabetes_dataset['t']['train'] = t_train
    diabetes_dataset['t']['valid'] = t_valid
    diabetes_dataset['t']['test'] = t_test

    return diabetes_dataset