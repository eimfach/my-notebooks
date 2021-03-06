import neuralnetworkopencl
import time
import zipfile
import numpy

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
# 0.161
learning_rate = float(input("Learning Rate: "))
epochs = int(input("Epochs: "))

# load the mnist training data CSV file into a list
zip_handle = zipfile.ZipFile("mnist_datasets/mnist_train.csv.zip")
training_data_list = [b.decode() for b in zip_handle.open("mnist_train.csv").readlines()]
zip_handle.close()

# load the mnist test data CSV file into a list
test_data_file = open("mnist_datasets/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# epochs is the number of times the training data set is used for training

print("input nodes: ", input_nodes)
print("hidden nodes: ", hidden_nodes)
print("output nodes: ", output_nodes)
print("learning rate: ", learning_rate)
print("testing epochs: ", epochs)

def trainAndTest(n):
    startTime = time.time()
    # train the neural network

    dataInputs = []
    dataTargets = []
    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list[:2]:

            # split the record by the ',' commas
            one_letter = record.split(',')

            imageData = one_letter[1:]
            mark = int(one_letter[0])

            # scale and shift the network training inputs
            inputs = (numpy.asfarray(imageData) / 255.0 * 0.99) + 0.01

            # create the network target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01

            # all_values[0] is the target label for this record
            targets[mark] = 0.99

            inputs = numpy.array(inputs, ndmin=2, dtype=numpy.float32).T
            targets = numpy.array(targets, ndmin=2, dtype=numpy.float32).T

            dataInputs.append(inputs)
            dataTargets.append(targets)
    print(numpy.array(dataInputs, dtype=numpy.float32))
    n.train(numpy.array(dataInputs, dtype=numpy.float32), numpy.array(dataTargets,dtype=numpy.float32))

    endTime = time.time()
    print("-------------------------------------------------------------------")
    print("Time elapsed for last Training Session (in seconds):", endTime - startTime)
    print("Start Testing Network Performance ....")
    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        one_letter = record.split(',')

        # correct answer is first value
        correct_label = int(one_letter[0])

        # scale and shift the inputs
        inputs = (numpy.asfarray(one_letter[1:]) / 255.0 * 0.99) + 0.01

        # query the network
        outputs = n.query(inputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)

        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)


for i in range(5):
    n = neuralnetworkopencl.NeuralNetwork(
        learningRate=learning_rate, iNodes=input_nodes, hNodes=hidden_nodes, oNodes=output_nodes)
    trainAndTest(n)
