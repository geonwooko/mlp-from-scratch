import numpy as np

class Relu:
    def __init__(self):
        pass

    def __call__(self, X):
        self.X = X
        return self.forward(X)

    def forward(self, X):
        mask = X > 0
        return X * mask

    def backward(self, delta_h):
        mask = self.X > 0
        return delta_h * mask

class FC:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = np.zeros((in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.reset_parameters()

    # Use xavier initialization
    def reset_parameters(self):
        std = np.sqrt(6 / (self.in_dim + self.out_dim))
        self.W = np.random.uniform(-std, std, self.W.shape)

    def  __call__(self, X):
        self.X = X
        return self.forward(X)

    def forward(self, X):
        return X.dot(self.W) + self.b

    def backward(self, delta_z):
        delta_X = delta_z.dot(self.W.T)
        delta_W = self.X.T.dot(delta_z) / len(self.X)
        delta_b = delta_z.sum(axis=0) / len(self.X)
        return delta_X, delta_W, delta_b

class Softmax:
    def  __init__(self):
        pass

    def __call__(self, X):
        self.X = X
        return self.forward(X)

    def forward(self, X):
        score = np.exp(X)
        score_sum = np.sum(score, axis=1, keepdims=True)
        self.probability = score / score_sum
        return self.probability

    def backward(self, delta_L, t):
        delta_Z = delta_L * (self.probability - t)
        return delta_Z

# Main class (Wrapper)
class MLP:
    def __init__(self,
                 input_dim,
                 nhidden,
                 hidden_dim,
                 nclasses):

        self.d = input_dim
        self.h = hidden_dim
        self.L = nhidden
        self.k = nclasses
        self.mode = 'train' # If train, save the each forward outputs to storage

        self.hidden_layers = []
        in_dim = self.d

        for out_dim in self.h:
            hidden_fc = FC(in_dim, out_dim)
            relu = Relu()
            self.hidden_layers.append(hidden_fc)
            self.hidden_layers.append(relu)
            in_dim = out_dim

        # Initialize output layer parameters
        self.output_layers = []
        out_fc = FC(in_dim, nclasses)
        softmax = Softmax()
        self.output_layers.append(out_fc)
        self.output_layers.append(softmax)

    # Forward X to y and store the outputs
    def forward(self, X):

        # Forward the hidden layer
        for hidden_layer in self.hidden_layers:
            X = hidden_layer(X)

        # Forward the output layer
        X = self.output_layers[0](X)
        prob = self.output_layers[1](X)

        return prob

    # Backpropagation from class probability to all weights
    def backward(self, t, t_prob):
        weight_grad_list, bias_grad_list = [], []
        m = t.shape[0]

        # Logistic regreesion (fully connected and softmax) backward
        delta_L = 1
        delta_Z = self.output_layers[1].backward(delta_L, t)
        delta_X, delta_W, delta_b = self.output_layers[0].backward(delta_Z)
        weight_grad_list.append(delta_W)
        bias_grad_list.append(delta_b)

        # hidden layer backward
        for l in range(1, (self.L * 2)+1):
            layer = self.hidden_layers[-l]
            delta_h = delta_X

            if isinstance(layer, Relu):
                delta_z = layer.backward(delta_h)

            elif isinstance(layer, FC):
                delta_X, delta_W, delta_b = layer.backward(delta_z)
                weight_grad_list.append(delta_W)
                bias_grad_list.append(delta_b)

        return weight_grad_list, bias_grad_list


