import numpy as np
import pickle
import fire

def gen_random_dataset(n_total,
                       feature_dim,
                       uniform_range=10,
                       alpha=0.1,
                       train_valid_test_rate=[0.85, 0.05, 0.1],
                       file_name="myrandomdataset.pkl"):
    # Parameter Initialization
    w = np.random.uniform(-uniform_range, uniform_range, size=feature_dim)
    b = np.random.uniform(-uniform_range, uniform_range, size=1)

    # Data Initialization
    X = np.random.uniform(-uniform_range, uniform_range, size=(n_total, feature_dim))
    mu = X.dot(w) + b;
    sigma = alpha * uniform_range
    t = np.array([np.random.normal(mu[i], sigma) for i in range(n_total)])

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
    X_train, t_train = X[train_idx], t[train_idx]
    X_valid, t_valid = X[valid_idx], t[valid_idx]
    X_test, t_test = X[test_idx], t[test_idx]

    # Save all data in dictionary
    dataset = {}
    dataset["X"] = {}
    dataset["t"] = {}

    dataset['X']['train'] = X_train
    dataset['X']['valid'] = X_valid
    dataset['X']['test'] = X_test

    dataset['t']['train'] = t_train
    dataset['t']['valid'] = t_valid
    dataset['t']['test'] = t_test

    parameters = {}
    parameters['w'] = w
    parameters['b'] = b

    # Save datasets as file
    with open(f"datasets/{file_name}", "wb") as f:
        pickle.dump(dataset, f)
    with open(f"datasets/parameters_{file_name}", "wb") as f:
        pickle.dump(parameters, f)

    print("랜덤 데이터 생성이 완료되었습니다.")
    print(f"X = {X.shape}, t = {t.shape}, w = {w.shape}, b = {b.shape}")

if __name__ == '__main__':
    fire.Fire(gen_random_dataset)