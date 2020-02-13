import numpy
import neuralnetwork
import matplotlib.pyplot
# This module provides convenience methods for the jupyter notebook for creating, training and testing the performance of a network

def createNetwork(learningRate):
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    return neuralnetwork.NeuralNetwork(learningRate=learningRate, iNodes=input_nodes, hNodes=hidden_nodes, oNodes=output_nodes)

# train the neural network
def training(network, trainingData, epochs):
# epochs is the number of times the training data set is used for training
    
    for e in range(epochs):
        # go through all records in the training data set
        for record in trainingData:

            # split the record by the ',' commas
            one_letter = record.split(',')

            imageData = one_letter[1:]
            mark = int(one_letter[0])

            # scale and shift the network training inputs
            inputs = (numpy.asfarray(imageData) / 255.0 * 0.99) + 0.01

            # create the network target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(network.onodes) + 0.01

            # all_values[0] is the target label for this record
            targets[mark] = 0.99

            network.train(inputs, targets)
            pass
        pass


def performance(network, testData):
    # test the neural network

    # scorecard for how well the network performs, initially empty
    scorecard = []

    # go through all the records in the test data set
    for record in testData:
        # split the record by the ',' commas
        one_letter = record.split(',')

        # correct answer is first value
        correct_label = int(one_letter[0])

        # scale and shift the inputs
        inputs = (numpy.asfarray(one_letter[1:]) / 255.0 * 0.99) + 0.01

        # query the network
        outputs = network.query(inputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)

        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass
    
    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)

def singleLetterTest(testData, network, index):
    one_letter = testData[index].split(",")
    mark = one_letter[index]

    imageData = one_letter[1:]
    # encode image data
    encodedData = (numpy.asfarray(imageData) / 255 * 0.99) + 0.01
    result = network.query(encodedData)
    guess = translateQueryResult(result=result)
    print("Network says :", guess, "| Expected:")
    normalizedForView = numpy.asfarray(one_letter[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(normalizedForView, cmap="Greys", interpolation="None")


def translateQueryResult(result):
    result = result.flatten().tolist()
    return result.index(max(result))
    