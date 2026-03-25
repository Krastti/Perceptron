from neural_network import NeuronNetwork

nn = NeuronNetwork(2, 1)

dataset_and = [[[0, 0], 0], [[0, 1], 0], [[1, 0], 0], [[1, 1], 1]]
dataset_or = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 1]]
dataset_implication = [[[0, 0], 1], [[0, 1], 1], [[1, 0], 0], [[1, 1], 1]]
dataset_xor = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]

nn.train(dataset_xor, 10000)
test_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
nn.test(test_data, 'XOR')