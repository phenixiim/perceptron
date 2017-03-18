import random


class Perceptron:
    def __init__(self, initial_weights, initial_bias=0):
        self.weights = list(initial_weights)
        self.bias = initial_bias
        self.guess_correct = 0
        self.guess_correct_rate = 0
        self.guess_false = 0
        self.guess_false_rate = 0
        self.trainedNumber = None

    def predict(self, to_predict):
        # vytvorime skalarni soucin (dot product) vectoru, tedy vector[0] * vaha[0] + vector[1] * vaha[1]
        dot_product = 0
        # pro kazdy index z predikovaneho vector
        for i, _ in enumerate(to_predict):
            # vynasobime stejnym indexem ve vahach a pricteme do `dot_product`
            dot_product += to_predict[i] * self.weights[i]
        # vysledkem je potencial - tedy aktivace - ke ktere pricteme bias (posun) - v zakladu 0
        activation = dot_product + self.bias
        # vse prevedeme na True/False a vratime
        return self.transfer(activation)

    def transfer(self, val):
        if val >= 0:
            return True
        return False

    def train(self, data_set, learning_rate=0.01):
        for x in range(5):
            for input, expectedOutput in data_set:
                predict = self.predict(input)
                error = expectedOutput - predict
                newWeightsDelta = []
                for index, value in enumerate(input):
                    newWeightsDelta.append(value * error * learning_rate)
                for index, weight in enumerate(list(self.weights)):
                    self.weights[index] = weight + newWeightsDelta[index]
                self.bias += error * learning_rate

    def trainMeToNumber(self, totalSteps: int, learningSample: int, trainedNumber: int):
        testSample = totalSteps - learningSample
        self.trainedNumber = trainedNumber


        listRange = list(range(10))
        imageList = []
        for i in listRange:
            imageList.append(open('./sampleData/data' + str(i), 'rb'))

        for step in range(totalSteps):
            random.shuffle(listRange)
            if step > learningSample:
                for imageStream in listRange:
                    input = tuple(imageList[imageStream].read(28 ** 2))
                    result = self.predict(input)
                    if imageStream == trainedNumber and result is True:
                        self.guess_correct += 1
                    if imageStream != trainedNumber and result is True:
                        self.guess_false += 1
            else:
                for imageStream in listRange:
                    input = tuple(imageList[imageStream].read(28 ** 2))
                    output = imageStream == trainedNumber
                    self.train([[input, output]], 0.01)

        self.guess_correct_rate = self.guess_correct / testSample * 100
        self.guess_fail_rate = self.guess_false / (testSample * 9) * 100

class NeuronLayer:
    def __init__(self):
        self.perceptronList = []
        self.trainArmy()

    def trainArmy(self):
        for i in range(10):
            perceptron = Perceptron(initial_weights=([1]*28**2),initial_bias=0)
            perceptron.trainMeToNumber(1000,750, i)
            self.perceptronList.append(perceptron)
            print('Perceptron number: {}, Ok rate: {}, FAIL rate: {}'.format(i, perceptron.guess_correct_rate, perceptron.guess_fail_rate))

    def guessNumberFromImage(self, imageInput):
        result = 0
        for perceptron in self.perceptronList:
            result = result + perceptron.trainedNumber*(perceptron.predict(imageInput) == True)
        return result

    def to_json(self):
        output = ""
        for p in self.perceptronList:
            pass

import pickle

def save():
    neuronLayer = NeuronLayer()
    with open("network", "wb") as fh:
        pickle.dump(neuronLayer, fh)

def load():
    return pickle.loads(open("network", "rb").read())

save()
neuronLayer = load()
for i in range(10):
    guess = 0
    fail = 0
    stream = open('./sampleData/data'+str(i),'rb')
    for step in range(1000):
        input = stream.read(28**2)
        guessedNumber = neuronLayer.guessNumberFromImage(input)
        if(guessedNumber == i):
            guess += 1
        else:
            fail += 1
    print('{}: ok: {}, fail: {}'.format(i, guess/1000*100, fail/1000*100))

