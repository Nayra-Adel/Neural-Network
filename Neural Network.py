import numpy as np
import pickle


def calculate_mse(delta):
    return np.sum(delta ** 2, axis=0) / delta.shape[0]


def sigmoid(net):
    return 1 / (1 + np.exp(-net))


def sigmoid_prime(s):
    # derivative of sigmoid
    return s * (1 - s)


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

    def backward(self, x_train, y_actual, y_predict, num_of_iterations, threshold):
        for x in range(num_of_iterations):

            error = calculate_mse(y_actual - y_predict)

            if any(error > threshold):

                o_error = y_actual - y_predict  # error in output

                # how much our hidden layer weights contributed to output error
                # applying derivative of sigmoid to hidden_layer_output_error
                # update hidden weights (input --> hidden) weights
                hidden_layer_output_error = o_error.dot(self.output_weights.T)
                apply_derivative_sigmoid = sigmoid_prime(self.hidden_layers) * hidden_layer_output_error
                self.hidden_weights += (self.learning_rate * x_train.T.dot(apply_derivative_sigmoid))

                # update output weights (hidden --> output) weights
                self.output_weights += (self.learning_rate * self.hidden_layers.T.dot(o_error))

            else:
                break

    def train(self, x_train, y_actual, num_of_iterations, threshold):
        y_predict = self.feed_forward()
        self.backward(x_train, y_actual, y_predict, num_of_iterations, threshold)


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

    NN.train(x_input, y_actual, 500, 0.1)

    y_predict = NN.feed_forward()
    print("After back propagation", calculate_mse(y_actual - y_predict))

    # save the weights
    pickle.dump([NN.hidden_weights, NN.output_weights], open("weights", 'wb'))


__main__()