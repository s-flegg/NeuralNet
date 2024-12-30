import random

from data import layer_list
from data import learning_rate as lr

import input_parsing
training_images = input_parsing.Data.training_images
training_labels = input_parsing.Data.training_labels
test_images = input_parsing.Data.test_images
test_labels = input_parsing.Data.test_labels


class HiddenNeuron:
    """An individual neuron in a hidden layer"""

    def __init__(self, weights, bias, activation_function):
        """
        :param weights: the list of weights into the neuron, in order
        :param bias: the offset to be added to the output
        :param activation_function: the activation function to be used
        :type weights: list
        :type bias: float
        :type activation_function: object
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

        self.inputs = []
        """The inputs to the neuron, used for backpropagation, list of floats
        :type: list"""
        self.output = 0
        """The output of the neuron
        :type: float"""
        self.error = 0
        """The error of the neuron
        :type: float"""

    def forward(self, inputs):
        """
        Calculates the output of the neuron

        :param inputs: the inputs to the neuron
        :type inputs: list
        """
        self.inputs = inputs
        self.output = (
            sum(self.weights[i] * inputs[i] for i in range(len(self.weights)))
            + self.bias
        )
        self.output = self.activation_function.run(self.output)

    def backward(self, neurons):
        """
        Calculates the error of the neuron, and updates the weights and bias

        :param neurons: the next layers neurons
        :type neurons: list
        """
        # calculate error
        self.error = sum(
            neurons[i].error * neurons[i].weights[i]  # weight to that neuron
            for i in range(len(neurons))
        )
        self.error *= self.activation_function.derivative(self.output)

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.inputs[i] * self.error * lr

        # update bias
        self.bias -= self.error * lr


class OutputNeuron(HiddenNeuron):
    """
    An individual neuron in the output layer

    This class is a special case of the HiddenNeuron class.
    It inherits the parameters and forward function,
    but overwrites the backward function,
    to accommodate for the error being calculated based on the target output.
    """

    def __init__(self, weights, bias, activation_function):
        """
        :param weights: the list of weights into the neuron, in order
        :param bias: the offset to be added to the output
        :param activation_function: the activation function to be used
        :type weights: list
        :type bias: float
        :type activation_function: object
        """
        super().__init__(weights, bias, activation_function)

    def backward(self, target):
        """
        Calculates the error of the neuron, and updates the weights and bias

        :param target: the target output
        :type target: float
        """
        # calculate error
        self.error = self.output - target
        self.error *= self.activation_function.derivative(self.output)

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.inputs[i] * self.error * lr

        # update bias
        self.bias -= self.error * lr

class ReLU:
    @staticmethod
    def run(x):
        return max(0, x)
    @staticmethod
    def derivative(x):
        return 1 if x > 0 else 0

class Sigmoid:
    @staticmethod
    def run(x):
        return 1 / (1 + pow(2.71828, -x))
    @staticmethod
    def derivative(x):
        return Sigmoid.run(x) * (1 - Sigmoid.run(x))


class Layer:
    """
    A layer of neurons

    This class manages the forward and backward propagation of the every neuron in the
    layer, which involves passing data about the next layer
    and having unique processes for the input and output layer
    """

    def __init__(
        self, neuron_count, input_count, activation_function, layer_number, output=False
    ):
        """
        :param neuron_count: the number of neurons in the layer
        :param input_count: the number of inputs to the layer
        :param activation_function: the activation function to be used
        :param layer_number: the number of the layer in data.layer_list
        :param output: if the layer is the output layer
        :type neuron_count: int
        :type input_count: int
        :type activation_function: object
        :type layer_number: int
        :type output: bool
        """
        self.neuron_count = neuron_count
        self.input_count = input_count
        self.activation_function = activation_function
        self.layer_number = layer_number
        self.output = output
        self.neurons = []
        """The list of neurons in the layer
        :type: list"""

        if output:
            for _ in range(neuron_count):
                self.neurons.append(
                    OutputNeuron(
                        weights=[random.uniform(-0.05, 0.05) for _ in range(input_count)],
                        bias=random.uniform(-1, 1),
                        activation_function=activation_function,
                    )
                )
        else:
            for _ in range(neuron_count):
                self.neurons.append(
                    HiddenNeuron(
                        weights=[random.uniform(-0.05, 0.05) for _ in range(input_count)],
                        bias=random.uniform(-1, 1),
                        activation_function=activation_function,
                    )
                )

    def forward(self, inputs):
        """
        Calculates the outputs for each neurom of the layer and starts the next layer

        As this starts the next layer, it also passes the inputs to the next layer.
        This means that you only need to start the first layer of the network,
        and all other layers will be started automatically.

        :param inputs: the inputs to the layer
        :type inputs: list
        """
        # forward pass
        for i in range(self.neuron_count):
            self.neurons[i].forward(inputs)

        # start next layer
        if self.layer_number < len(layer_list) - 1:
            layer_list[self.layer_number + 1].forward([neuron.output for neuron in self.neurons])

    def backward(self, targets=None):
        """
        Backpropigates error of each neuron and updates the weights and biases,
        recursively.

        As this starts the previous later,
        you only need to start the last layer of the network,
        and all other layers will be started automatically.

        :param targets: the target outputs of each neuron in the layer,
        only needed for the output layer
        :type targets: list
        """
        # backpropagation
        if self.output:
            for i in range(self.neuron_count):
                self.neurons[i].backward(targets[i])
        else:
            for i in range(self.neuron_count):
                self.neurons[i].backward(layer_list[self.layer_number + 1].neurons)

        # start the previous layer
        if self.layer_number > 0:
            layer_list[self.layer_number - 1].backward()

def MSE(targets, outputs):
    return sum(pow(targets[i] - outputs[i], 2) for i in range(len(targets)))


if __name__ == "__main__":
    layer_list.append(Layer(64, 28*28, ReLU, 0))
    layer_list.append(Layer(10, 64, Sigmoid, 1, output=True))

    train_range = 60000
    successes = 0
    for i in range(train_range):
        layer_list[0].forward(training_images[i])
        # print(i)
        #print([neuron.output for neuron in layer_list[-1].neurons])
        #print(training_labels[i])
        output = [neuron.output for neuron in layer_list[-1].neurons]
        success = output.index(max(output)) == training_labels[i].index(1)
        if success:
            successes += 1
        print(
            str(i) + "    "
            + str(MSE(training_labels[i], output)) + "    "
            + str(success)
        )
        layer_list[-1].backward(training_labels[i])
    print("%i out of %i images were classified correctly"%(successes,train_range))

    test_Range = 10000
    successes = 0
    for i in range(test_Range):
        layer_list[0].forward(test_images[i])
        output = [neuron.output for neuron in layer_list[-1].neurons]
        success = output.index(max(output)) == test_labels[i].index(1)
        if success:
            successes += 1
        print(
            str(i) + "    "
            + str(MSE(test_labels[i], output)) + "    "
            + str(success)
        )
    print("%i out of %i images were classified correctly"%(successes,test_Range))
    print("Success rate: %f %%"%(successes/test_Range))
