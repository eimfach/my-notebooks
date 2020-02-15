import numpy
import scipy.special
import torch
# Naive try to convert numpy to pytorch to enable gpu acceleration

device = torch.device("cpu")

# PyTorch only supports CUDA which means NVIDIA :/
# Also look at https://github.com/pytorch/pytorch/issues/7609 for some API comparison
# This current approach using cpu is way slower than pure NumPy
# CPU based calculations

class NeuralNetwork:

    # initialise the neural network
    def __init__(self, learningRate, iNodes, hNodes, oNodes):
        # set number of nodes in each input, hidden, output layer
        self.inodes = iNodes
        self.hnodes = hNodes
        self.onodes = oNodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = torch.from_numpy(numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))).to(device)
        self.who = torch.from_numpy(numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))).to(device)

        # learning rate
        self.lr = learningRate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = torch.from_numpy(numpy.array(inputs_list, ndmin=2).T).to(device)
        targets = torch.from_numpy(numpy.array(targets_list, ndmin=2).T).to(device)

        # calculate signals into hidden layer
        hidden_inputs = torch.mm(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = torch.mm(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = torch.mm(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * torch.mm((output_errors * final_outputs * (
            1.0 - final_outputs)), torch.transpose(hidden_outputs, 0, 1))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * \
            torch.mm((hidden_errors * hidden_outputs *
                       (1.0 - hidden_outputs)), torch.transpose(inputs, 0, 1))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = torch.from_numpy(numpy.array(inputs_list, ndmin=2).T)

        # calculate signals into hidden layer
        hidden_inputs = torch.mm(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = torch.mm(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
