import numpy as np
from math import ceil
from random import randint

class Layer:
    def __init__(self, layer_size, prev_layer, parent_network):
        self.prev_layer = prev_layer
        self.network = parent_network
        self.neurons = [Neuron(self, prev_layer) for _ in range(layer_size)]

    def set_input_data(self, val_list):
        for i in range(len(val_list)):
            self.neurons[i].set_value(val_list[i])

class Neuron:
    def __init__(self, layer: Layer, previous_layer: Layer):
        self.value = 0
        self._layer = layer
        self.inputs = [Input(prev_neuron, randint(0, 10) / 10) for prev_neuron in previous_layer.neurons] if previous_layer else []

    def set_value(self, val):
        self.value = val

    def get_value(self):
        network = self._layer.network
        if not self.is_no_inputs():
            self.set_value(network.activate_func(self.get_input_sum()))
        return self.value

    def is_no_inputs(self):
        return not self.inputs

    '''Взвешенная сумма'''
    def get_input_sum(self):
        total_sum = sum(curr_input.prev_neuron.get_value() * curr_input.weight for curr_input in self.inputs)
        return total_sum

    def set_error(self, val):
        if self.is_no_inputs():
            return
        w_delta = val * self._layer.network.derivate_func(self.get_input_sum())
        for curr_input in self.inputs:
            curr_input.weight -= curr_input.prev_neuron.get_value() * w_delta * self._layer.network.learning_rate
            curr_input.prev_neuron.set_error(curr_input.weight * w_delta)

class Input:
    def __init__(self, prev_neuron: Neuron, weight):
        self.prev_neuron = prev_neuron
        self.weight = weight

class NeuronNetwork:
    def __init__(self, input_l_size, output_l_size, hidden_layer_size=1, learning_rate=0.5):
        self.selected_layer = None
        self.l_count = hidden_layer_size + 2
        hidden_l_size = min(input_l_size * 2 - 1, ceil(input_l_size * 2 / 3 + output_l_size))
        self.layers = [self.add_layer(i, input_l_size, output_l_size, hidden_l_size) for i in range(self.l_count)]
        self.selected_layer = None
        self.activate_func = NeuronNetwork.sigmoid
        self.derivate_func = NeuronNetwork.sigmoid_derivative
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuronNetwork.sigmoid(x) * (1 - NeuronNetwork.sigmoid(x))

    def add_layer(self, i, in_size, out_size, hl_size):
        count = i + 1
        if 1 < count < self.l_count:
            self.selected_layer = Layer(hl_size, self.selected_layer, self)
            return self.selected_layer
        if count == 1: #input
            self.selected_layer = Layer(in_size, None, self)
            return self.selected_layer
        self.selected_layer = Layer(out_size, self.selected_layer, self)
        return self.selected_layer

    def train(self, dataset, iters=1000):
        print(f'\nНачало обучения({iters} итераций)...')
        for i in range(iters):
            self.train_once(dataset)
        print(f'\nОбучение завершено!\n')

    def train_once(self, dataset):
        for case in dataset:
            datacase = {'in_data': case[0], 'res': case[1]}
            self.set_input_data(datacase['in_data'])
            curr_res = self.get_prediction()
            for i in range(len(curr_res)):
                self.layers[self.l_count - 1].neurons[i].set_error(curr_res[i] - datacase['res'])

    def set_input_data(self, val_list):
        self.layers[0].set_input_data(val_list)

    def get_prediction(self):
        layers = self.layers
        output_layer = layers[len(layers) - 1]
        out_data = [neuron.get_value() for neuron in output_layer.neurons]
        return out_data

    def test(self, data, op_name):
        for case in data:
            self.set_input_data(case)
            res = self.get_prediction()
            print(f'{case[0]} {op_name} {case[1]} ~ {res[0]}')