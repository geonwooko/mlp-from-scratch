import fire
from trainer import linear_regression_trainer
from linear_regression import linear_regression
def main(data_name,
         feature_dim,
         n_epoch,
         batch_size=64,
         learning_rate=1e-2,
         weight_decay=1e-2,
         early_stopping=False,
         n_iter_no_change=5,
         tol=0.001):

    model = linear_regression(feature_dim)
    trainer = linear_regression_trainer(model,
                                        data_name,
                                        batch_size,
                                        learning_rate,
                                        weight_decay,
                                        early_stopping,
                                        n_iter_no_change,
                                        tol)

    losses = trainer.train(n_epoch)

if __name__ == '__main__':
    fire.Fire(main)