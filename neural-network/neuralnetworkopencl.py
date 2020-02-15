import numpy

# GPU based calculations

class NeuralNetwork:

    # initialise the neural network
    def __init__(self, gpuApi, learningRate, iNodes, hNodes, oNodes):
        self.self.gpu_api = gpuApi

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
        self.wih = self.gpu_api.arr(numpy.array(numpy.random.normal(
            0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes)), dtype=numpy.float32))
        # matrix: [[0.014, 0.056, -0.032 ... ] ... [0.045, 0.032, -0.067 ... ] ... ]
        self.who = self.gpu_api.arr(numpy.array(numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)), dtype=numpy.float32))

        # learning rate
        self.lr = learningRate

        # activation function is the sigmoid function
        self.activation_function = lambda x: self.gpu_api.sigmoid(x)

        pass

    # train the neural network

    def train(self, inputs, targets):

        # calculate signals into hidden layer
        # hidden_inputs = [[0.00038f32], [0.0002f32] ...]
        hidden_inputs = self.gpu_api.dot(self.wih, inputs)
        
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = self.gpu_api.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = self.gpu_api.matsubstract(targets, final_outputs)
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = self.gpu_api.dot(self.who.T, output_errors)

        prod1 = self.gpu_api.matmultiply(self.gpu_api.matmultiply(output_errors, final_outputs), self.gpu_api.lmatsubstract(1.0, final_outputs))
        # update the weights for the links between the hidden and output layers
        self.who = self.gpu_api.matadd(self.who, self.gpu_api.lmatmultiply(self.lr, self.gpu_api.dot(prod1, self.gpu_api.transp(hidden_outputs))))

        prod2 = self.gpu_api.matmultiply(self.gpu_api.matmultiply(hidden_errors, hidden_outputs), self.gpu_api.lmatsubstract(1.0, hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih = self.gpu_api.matadd(self.wih, self.gpu_api.lmatmultiply(self.lr, self.gpu_api.dot(prod2, self.gpu_api.transp(inputs)))) 

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        # cpu based
        inputs = numpy.array(inputs_list, ndmin=2, dtype=numpy.float32).T

        # calculate signals into hidden layer
        hidden_inputs = self.gpu_api.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = self.gpu_api.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # convert to numpy array
        return final_outputs
