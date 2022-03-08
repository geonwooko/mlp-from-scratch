import fire
from trainer import logistic_regression_trainer
from logistic_regression import logistic_regression

def main(n_epoch,
         feature_dim = 784,
         nclasses = 10,
         batch_size=64,
         learning_rate=1e-2,
         weight_decay=1e-2,
         early_stopping=False,
         n_iter_no_change=5,
         tol=0.001):

    model = logistic_regression(feature_dim, nclasses)
    trainer = logistic_regression_trainer(model,
                                        batch_size,
                                        learning_rate,
                                        weight_decay,
                                        early_stopping,
                                        n_iter_no_change,
                                        tol)

    losses = trainer.train(n_epoch)

if __name__ == '__main__':
    fire.Fire(main)