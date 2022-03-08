import numpy as np
from copy import deepcopy
from datatools import dataloader, get_MNIST_data
from MLP_mnist import FC


class MLP_trainer:
    def __init__(self,
                 model,
                 batch_size=64,
                 learning_rate=1e-2,
                 weight_decay=0,
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
                t_prob = self.model.forward(X)
                train_accuracy = self.get_accuracy(t, t_prob)
                train_loss = self.get_cross_entropy_loss(X, t, t_prob)

                # Backward the loss and get parameters' gradient
                weight_grad_list, bias_grad_list = self.model.backward(t, t_prob)

                # Apply weight decay(L2 Regularization) and update parameters
                for layer in self.model.hidden_layers: # hidden layer
                    if isinstance(layer, FC):
                        W_grad = weight_grad_list.pop()
                        b_grad = bias_grad_list.pop()
                        W_grad += self.reg * layer.W
                        b_grad += self.reg * layer.b
                        train_loss += self.reg * (np.sum(layer.W**2) + np.sum(layer.b**2))
                        layer.W -= self.lr * W_grad
                        layer.b -= self.lr * b_grad

                for layer in self.model.output_layers: # output layer
                    if isinstance(layer, FC):
                        W_grad = weight_grad_list.pop()
                        b_grad = bias_grad_list.pop()
                        W_grad += self.reg * layer.W
                        b_grad += self.reg * layer.b
                        train_loss += self.reg * (np.sum(layer.W**2) + np.sum(layer.b**2))
                        layer.W -= self.lr * W_grad
                        layer.b -= self.lr * b_grad

                batch_losses.append(train_loss)
                batch_accuracies.append(train_accuracy)

            # Evaluate with validation / test data
            train_loss = np.mean(batch_losses)
            valid_loss, valid_accuracy = self.evaluate(self.X_valid, self.t_valid)
            test_loss, test_accuracy = self.evaluate(self.X_test, self.t_test)

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
            return self.model, losses, accuracies
        else:
            print(f"Train finished. Final model's test accuracy : {accuracies['test'][-1]:.3f}")
            return self.model

    # Get mse loss 
    def get_cross_entropy_loss(self, X, t, t_prob):
        m = t.shape[0]
        cross_entropy_loss = (-t * np.log(t_prob + 1e-10)).sum() / m
        w_grad = X.T.dot(t_prob - t) / m  # Calculate mean of each observation's gradient
        b_grad = (t_prob - t).sum(axis = 0) / m # Calculate mean of each observation's gradient

        return cross_entropy_loss
    
    # Get the accuracy
    def get_accuracy(self, t, t_prob):
        t_pred_label = np.argmax(t_prob, axis=1)
        t_eval_label = np.argmax(t, axis=1)
        accuracy = np.sum(t_pred_label == t_eval_label) / t.shape[0]
        
        return accuracy
    
    # Evaluate data and return loss & accuracy
    def evaluate(self, X_eval, t_eval):
        t_prob = self.model.forward(X_eval)

        accuracy = self.get_accuracy(t_eval, t_prob)        
        cross_entropy_loss = self.get_cross_entropy_loss(X_eval, t_eval, t_prob)
        return cross_entropy_loss, accuracy
