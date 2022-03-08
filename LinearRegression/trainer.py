import numpy as np
from copy import deepcopy
from datatools import dataloader, get_dataset_with_parameters

class linear_regression_trainer:
    def __init__(self,
                 model,
                 data_name,
                 batch_size=64,
                 learning_rate=1e-2,
                 weight_decay=1e-2,
                 early_stopping=False,
                 n_iter_no_change=5,
                 tol=0.001):

        self.model = model
        self.lr = learning_rate
        self.reg = weight_decay  # Weight decay(L2 Regularization) hyperparamater
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        # Get dataset and make train data as dataloader
        dataset, self.parameters = get_dataset_with_parameters(data_name)
        self.train_dataloader = dataloader(dataset['X']['train'], dataset['t']['train'], batch_size)
        self.X_valid, self.t_valid = dataset['X']['valid'], dataset['t']['valid']
        self.X_test, self.t_test = dataset['X']['test'], dataset['t']['test']

    # Train model for n epochs
    def train(self, n_epoch):
        losses = {}
        losses["train"] = []
        losses['valid'] = []
        losses['test'] = []

        # If use early stopping
        if self.early_stopping:
            early_count = 0
            best_model = None
            best_loss = 1e+20
            best_epoch = 0

        for epoch in range(1, n_epoch + 1):

            batch_losses = []
            for X, t in self.train_dataloader:
                # Predict target with train data
                t_pred = self.model.forward(X)

                # Get MSE Loss and gradient
                train_loss, w_grad, b_grad = self.get_mse_loss(X, t, t_pred, return_grad=True)
                batch_losses.append(train_loss)

                # Apply weight decay(L2 Regularization)
                train_loss += self.reg * np.sum(np.power(self.model.w, 2))
                w_grad += self.reg * self.model.w

                # Update parameters
                self.model.w -= self.lr * w_grad
                self.model.b -= self.lr * b_grad

            # Evaluate with validation / test data
            train_loss = np.mean(batch_losses)
            valid_loss = self.evaluate(self.X_valid, self.t_valid)
            test_loss = self.evaluate(self.X_test, self.t_test)

            # Save the losses
            losses['train'].append(train_loss)
            losses['valid'].append(valid_loss)
            losses['test'].append(test_loss)

            # Print train /valid loss and parameters' squred error
            print(
                f"epoch : {epoch}   train_loss : {train_loss:.3f}   valid_loss : {valid_loss:.3f}   test_loss : {test_loss:.3f}")
            if self.parameters is not None:
                w_SE = np.power((self.parameters['w'] - self.model.w), 2).sum()
                b_SE = np.power((self.parameters['b'] - self.model.b), 2).sum()
                print(f"            w_SE : {w_SE:.3f}   b_SE : {b_SE:.3f}")

            # Check early stopping option
            if self.early_stopping:
                diff = best_loss - valid_loss

                # If current loss is lower than best loss
                if diff > self.tol:
                    best_model = deepcopy(self.model)
                    best_loss = valid_loss
                    best_epoch = epoch
                    early_count = 0
                else:
                    early_count += 1

                # If no change loss during n_iter_no_change epoch
                if early_count == self.n_iter_no_change:
                    self.model = best_model
                    break
        if self.early_stopping:
            print(f"Train finished. Best model's test loss : {losses['test'][best_epoch - 1]:.3f}")
        else:
            print(f"Train finished. Final model's test loss : {losses['test'][-1]:.3f}")

    # Get mse loss and gradient
    def get_mse_loss(self, X, t, t_pred, return_grad=False):
        m = t.shape[0]
        mse_loss = np.power(t - t_pred, 2).sum() / m
        w_grad = X.T.dot(t_pred - t) / m  # Calculate mean of each observation's gradient
        b_grad = (t_pred - t).sum() / m # Calculate mean of each observation's gradient

        if return_grad:
            return mse_loss, w_grad, b_grad

        else:
            return mse_loss

    # Evaluate about the eval data
    def evaluate(self, X_eval, t_eval):
        t_pred = self.model.forward(X_eval)
        mse_loss = self.get_mse_loss(X_eval, t_eval, t_pred, return_grad=False)
        return mse_loss
