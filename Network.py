import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(vector):
    one = np.array([1]*len(vector))
    one = one.reshape(len(one), 1)
    Y = vector * np.subtract(one, vector)
    return Y


def max_value(X):
    champ = (-1, float('-inf'))
    for i in range(0, len(X)):
        if X[i] > champ[1]:
            champ = (i, X[i])

def cost_derivative(output_vector, expected):
    return np.subtract(output_vector, expected)


class Network:
    def __init__(self, topology):
        self.layers = len(topology)
        self.topology = topology
        self.biases = [np.zeros((i, 1)) for i in topology[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(topology[:-1], topology[1:])]

    def predict(self, input_vector):
        x = np.array(input_vector).reshape(len(input_vector), 1)
        activations = self.feed_forward(x)
        return activations[-1]
    
    def train(self, input_vector, expected):
        x = np.array(input_vector).reshape(len(input_vector), 1)
        y = np.array(expected).reshape(len(expected), 1)
        (w, b) = self.backprop(x, y)
        for i in range(0, len(self.weights)):
            self.weights[i] = np.subtract(self.weights[i], w[i])
        for i in range(0, len(self.biases)):
            self.biases[i] = np.subtract(self.biases[i], b[i])
        
    def feed_forward(self, input_vector):
        activations = [[] for _ in range(0, self.layers)]
        activations[0] = input_vector
        for i in range(0, self.layers-1):
            a = sigmoid(np.dot(self.weights[i], activations[i]) + self.biases[i])
            activations[i+1] = a
        return activations
    
    def backprop(self, input_vector, expected):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations = self.feed_forward(input_vector)
        delta = cost_derivative(activations[-1], expected)
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, np.transpose(activations[-2]))
        for l in range(2, self.layers):
            sp = sigmoid_prime(activations[-l])
            delta = np.matmul(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        return(nabla_w, nabla_b)





            
