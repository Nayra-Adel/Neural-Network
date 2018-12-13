import numpy as np


def calculate_mse(delta):
    return np.sum(delta ** 2, axis=0) / delta.shape[0]


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


class NeuralNetwork:
    def __init__(self, m, l, n, input_vector):
        self.hidden_weights = np.random.rand(m, l) * 10 + (-5)  # m*L
        self.output_weights = np.random.rand(l, n) * 10 + (-5)  # L*n
        self.learning_rate = 0.001
        self.input_layers = input_vector
        self.hidden_layers = None

    # biases to be 0 [Wx+b] => (b = 0)
    def feed_forward(self):
        self.hidden_layers = sigmoid(np.dot(self.input_layers, self.hidden_weights))  # (k*m).(m*L)
        output_layers = sigmoid(np.dot(self.hidden_layers, self.output_weights))  # (k*L).(L*n)
        return output_layers  # return the normalized output


def take_input():
    inp = [int(x) for x in input().split()]
    m = inp[0]  # input nodes = features
    l = inp[1]  # hidden neurons
    n = inp[2]  # output
    k = int(input())  # num of training examples
    x_input  = []  # k * m
    y_actual = []  # k * n (output value)

    for i in range(k):
        ex = [float(x) for x in input().split()]
        x_input.append(np.array(ex[0:m]))
        y_actual.append(np.array(ex[m:m + n]))

    # scale the x_input array (dividing by the max value)
    x_input = np.array(x_input) / np.amax(x_input, axis=0)
    y_actual = np.array(y_actual)

    return m, l, n, x_input, y_actual


def __main__():
    m, l, n, x_input, y_actual = take_input()

    NN = NeuralNetwork(m, l, n, x_input)

    y_predict = NN.feed_forward()
    print("Before back propagation", calculate_mse(y_actual - y_predict))


__main__()