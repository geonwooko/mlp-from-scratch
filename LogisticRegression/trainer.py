import numpy as np
from copy import deepcopy
from datatools import dataloader, get_MNIST_data

class logistic_regression_trainer:
    def __init__(self,
                 model,
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
        dataset = get_MNIST_data()
        self.train_dataloader = dataloader(dataset['X']['train'], dataset['t']['train'], batch_size)
        self.X_valid, self.t_valid = dataset['X']['valid'], dataset['t']['valid']
        self.X_test, self.t_test = dataset['X']['test'], dataset['t']['test']

    # Train model for n epochs
    def train(self, n_epoch):
        losses = {}
        losses["train"] = []
        losses['valid'] = []
        losses['test'] = []

        accuracies = {}
        accuracies['train'] = []
        accuracies['valid'] = []
        accuracies['test'] = []

        # If use early stopping
        if self.early_stopping:
            early_count = 0
            best_model = None
            best_accuracy = 0
            best_epoch = 0

        for epoch in range(1, n_epoch + 1):

            batch_losses = []
            batch_accuracies = []
            for X, t in self.train_dataloader:
                # Predict target with train data
                t_pred = self.model.forward(X)

                # Get MSE Loss and gradient
                train_loss, w_grad, b_grad, train_accuracy = self.evaluate(X, t, return_grad=True)
                batch_losses.append(train_loss)
                batch_accuracies.append(train_accuracy)

                # Apply weight decay(L2 Regularization)
                train_loss += self.reg * np.sum(np.power(self.model.w, 2))
                w_grad += self.reg * self.model.w

                # Update parameters
                self.model.w -= self.lr * w_grad
                self.model.b -= self.lr * b_grad

            # Evaluate with validation / test data
            train_loss = np.mean(batch_losses)
            valid_loss, valid_accuracy = self.evaluate(self.X_valid, self.t_valid, return_grad=False)
            test_loss, test_accuracy = self.evaluate(self.X_test, self.t_test, return_grad=False)

            # Save the losses and accuracies
            losses['train'].append(train_loss)
            losses['valid'].append(valid_loss)
            losses['test'].append(test_loss)
            accuracies['train'].append(train_accuracy)
            accuracies['valid'].append(valid_accuracy)
            accuracies['test'].append(test_accuracy)

            # Print train /valid / test loss and accuracy
            print(
                f"epoch : {epoch}   train_loss : {train_loss:.3f}   valid_loss : {valid_loss:.3f}   test_loss : {test_loss:.3f}")
            print(
                f"              train_acc : {train_accuracy:.3f}   valid_acc : {valid_accuracy:.3f}   test_acc : {test_accuracy:.3f}")

            # Check early stopping option
            if self.early_stopping:
                diff = valid_accuracy - best_accuracy

                # If current loss is lower than best loss
                if diff > self.tol:
                    best_model = deepcopy(self.model)
                    best_accuracy = valid_accuracy
                    best_epoch = epoch
                    early_count = 0
                else:
                    early_count += 1

                # If no change loss during n_iter_no_change epoch
                if early_count == self.n_iter_no_change:
                    self.model = best_model
                    break
        if self.early_stopping:
            print(f"Train finished. Best model's test accuracy : {accuracies['test'][best_epoch - 1]:.3f}")
        else:
            print(f"Train finished. Final model's test accuracy : {accuracies['test'][-1]:.3f}")

    # Get mse loss and gradient
    def get_cross_entropy_loss(self, X, t, t_prob, return_grad=False):
        m = t.shape[0]
        cross_entropy_loss = (-t * np.log(t_prob)).sum() / m
        w_grad = X.T.dot(t_prob - t) / m  # Calculate mean of each observation's gradient
        b_grad = (t_prob - t).sum(axis = 0) / m # Calculate mean of each observation's gradient
        if return_grad:
            return cross_entropy_loss, w_grad, b_grad

        else:
            return cross_entropy_loss

    # Evaluate data and return loss & accuracy
    def evaluate(self, X_eval, t_eval, return_grad = True):
        t_prob = self.model.forward(X_eval)
        t_pred_label = np.argmax(t_prob, axis=1)
        t_eval_label = np.argmax(t_eval, axis=1)
        accuracy = np.sum(t_pred_label == t_eval_label) / t_eval.shape[0]

        if return_grad:
            cross_entropy_loss, w_grad, b_grad = self.get_cross_entropy_loss(X_eval, t_eval, t_prob, return_grad)
            return cross_entropy_loss, w_grad, b_grad, accuracy
        else:
            cross_entropy_loss = self.get_cross_entropy_loss(X_eval, t_eval, t_prob, return_grad)
            return cross_entropy_loss, accuracy
