import numpy
import scipy.special
import GPUTraining

gpu_training = GPUTraining.GPUTraining()

# GPU based calculations

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
        # cpu based
        # matrix: [[0.014, 0.056, -0.032 ... ] ... [0.045, 0.032, -0.067 ... ] ... ]
        self.wih = numpy.array(numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)), dtype=numpy.float32)

        # matrix: [[0.014, 0.056, -0.032 ... ] ... [0.045, 0.032, -0.067 ... ] ... ]
        self.who = numpy.array(numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)), dtype=numpy.float32)

        # learning rate
        self.lr = learningRate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the neural network
    def train(self, inputs, targets):
        # gpu based
        
        updated_wih,updated_who = gpu_training.main(self.lr, self.wih, self.who, inputs, targets)

        self.who = updated_who
        self.wih = updated_wih
        pass

    # query the neural network
    def query(self, inputs):
        # cpu based
        # convert inputs list to 2d array
        inputs = numpy.array(inputs, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih.get(), inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who.get(), hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
