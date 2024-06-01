from Network import Network

LENGTH = 28
DIMS = LENGTH*LENGTH

def load_data_from_file(filename):
    data = []
    with(open(filename) as file):
        for line in file:
            line_data = [0]*DIMS
            contents = line.split(',')
            label = contents[0]
            for i in range(1, len(contents)):
                line_data[i-1] = float(contents[i])/255
            data.append((int(label), line_data))
    return data

def max_value(vector):
    champ = (-1, float('-inf'))
    for i in range(0, len(vector)):
        if vector[i] > champ[1]:
            champ = (i, vector[i])
    return champ[0]

def train_network(network, data):
    for datum in data:
        (label, vector) = datum
        expected = [0]*10
        expected[label] = 1
        network.train(vector, expected)

def test_network(network, data):
    correct = 0
    total = len(data)
    for datum in data:
        (label, vector) = datum
        values = network.predict(vector)
        result = max_value(values)
        if label == result:
            correct += 1
    print("Correct: " + str(correct))
    print("total: " + str(total))
    print("Percentage: " + str((correct/total)*100))


def main():
    network = Network([DIMS, 24, 10])
    training_data = load_data_from_file("C:/Users/goald/VSC/Neural_Network/datasets/mnist_train.csv")
    train_network(network, training_data)
    test_data = load_data_from_file("C:/Users/goald/VSC/Neural_Network/datasets/mnist_test.csv")
    test_network(network, test_data)


if __name__ == "__main__":
    main()