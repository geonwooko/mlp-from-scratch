import fire, pickle
from trainer import MLP_trainer
from MLP_mnist import MLP
import matplotlib.pyplot as plt

def main(n_epoch,
         input_dim = 784,
         nhidden = 1,
         hidden_dim = [30],
         nclasses = 10,
         batch_size=64,
         learning_rate=1e-2,
         weight_decay=1e-2,
         early_stopping=False,
         n_iter_no_change=5,
         tol=0.001):

    # Make multi layer perceptron model
    model = MLP(input_dim,
                nhidden,
                hidden_dim,
                nclasses)
    # Make trainer class instance
    trainer = MLP_trainer(model,
                          batch_size,
                          learning_rate,
                          weight_decay,
                          early_stopping,
                          n_iter_no_change,
                          tol)

    # train model n_epoch and get best model, loss, accuracy
    model, losses, accuracies = trainer.train(n_epoch)

    # Save best model as a file
    with open("result/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Plot and save the loss & accuracy
    save_plot(nhidden, losses, accuracies)

# Save result as a plot
def save_plot(nhidden, losses, accuracies):
    x = range(len(losses['train']))
    col = ['black', 'blue', 'red']

    for i, (key, loss) in enumerate(losses.items()):
        plt.plot(x, loss, color = col[i], label = key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"result/loss_{nhidden}.png")
    plt.close()

    for i, (key, accuracy) in enumerate(accuracies.items()):
        plt.plot(x, accuracy, color = col[i], label = key)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(f"result/accuracy_{nhidden}.png")
    plt.close()


if __name__ == '__main__':
    fire.Fire(main)