import gpu
import numpy

gpu_api = gpu.GPU()

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
        self.activation_function = lambda x: gpu_api.sigmoid(x)

        pass

    # train the neural network

    def train(self, inputs_list, targets_list):
        # convert inputs list and targets list to 2d array
        # cpu based
        # inputs_list = [0.01, 0.01, 0.01, 0.02164706, 0.01 ...] -> [[0.01], [0.01], [0.01], [0.02164706], [0.01] ...]
        inputs = numpy.array(inputs_list, ndmin=2, dtype=numpy.float32).T

        # targets_list = [0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01] -> [[0.01], [0.01], [0.01] ...]
        targets = numpy.array(targets_list, ndmin=2, dtype=numpy.float32).T

        # calculate signals into hidden layer
        # hidden_inputs = [[0.00038f32], [0.0002f32] ...]
        hidden_inputs = gpu_api.matmul(self.wih, inputs)
        
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = gpu_api.matmul(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = gpu_api.matsubstract(targets, final_outputs)
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = gpu_api.matmul(self.who.T, output_errors)

        # print("output_errors",type(output_errors))
        # print("final_outputs",type(final_outputs))
        # print("hidden_outputs",type(hidden_outputs))
        # print("hidden_errors",type(hidden_errors))
        # print("inputs",type(inputs))
        prod1 = gpu_api.multiply2(gpu_api.multiply2(output_errors, final_outputs), gpu_api.lmatsubstract(1.0, final_outputs))

        # update the weights for the links between the hidden and output layers
        self.who = gpu_api.add2(self.who, gpu_api.multiply(gpu_api.matmul(prod1, gpu_api.transp(hidden_outputs)), self.lr))

        prod2 = gpu_api.multiply2(gpu_api.multiply2(hidden_errors, hidden_outputs), gpu_api.substract(1.0, hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih = gpu_api.add2(self.wih, gpu_api.multiply(gpu_api.matmul(prod2, gpu_api.transp(inputs)), self.lr)) 

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        # cpu based
        inputs = numpy.array(inputs_list, ndmin=2, dtype=numpy.float32).T

        # calculate signals into hidden layer
        hidden_inputs = gpu_api.matmul(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = gpu_api.mat_mul(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # convert to numpy array
        return final_outputs.get()
