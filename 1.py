import random

class Perceptron:

    def __init__(self, initial_weights, initial_bias=0):
        self.weights = list(initial_weights)
        self.bias = initial_bias

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

    def train(self, data_set, learning_rate=0.00009):
        for input, expectedOutput in data_set:
            predict = self.predict(input)
            error = expectedOutput-predict
            newWeightsDelta = []
            for index,value in enumerate(input):
                newWeightsDelta.append(value*error*learning_rate)
            for index, weight in enumerate(list(self.weights)):
                self.weights[index] = weight + newWeightsDelta[index]
            self.bias += error * learning_rate
            # print(self.weights, self.bias)
            return self.weights, self.bias


p = Perceptron(initial_weights=([1]*28**2),initial_bias=0)

data_set = [
    ((2.7810836, 2.550537003), 0),
    ((1.465489372, 2.362125076), 0),
    ((3.396561688, 4.400293529), 0),
    ((1.38807019, 1.850220317), 0),
    ((3.06407232, 3.005305973), 0),
    ((7.627531214, 2.759262235), 1),
    ((5.332441248, 2.088626775), 1),
    ((6.922596716, 1.77106367), 1),
    ((8.675418651, -0.242068655), 1),
    ((7.673756466, 3.508563011), 1),
]

def hidden():
    prev = None
    succes = 0
    for i in range(5000000):
        random.shuffle(data_set)
        perceptron_state = p.train(data_set)
        if prev == perceptron_state:
            succes += 1
        else:
            prev = perceptron_state
            succes = 0
        if succes == 200:
            print(perceptron_state)
            print(i - succes)
            break

import pprint
from PIL import Image


guess_correct = guess_false = 0
totalSteps = 1000
learningSample = 800
testSample = totalSteps-learningSample

p = Perceptron(initial_weights=([1]*28**2),initial_bias=0)

listRange = list(range(9))
imageList = []
for i in listRange:
    imageList.append(open('./sampleData/data'+str(i), 'rb'))

for step in range(totalSteps):
    random.shuffle(listRange)
    if step > learningSample:
        for imageStream in listRange:
            input = tuple(imageList[imageStream].read(28 ** 2))
            result = p.predict(input)
            if imageStream == 7 and result is True:
                guess_correct += 1
            if imageStream != 7 and result is True:
                guess_false += 1
    else:
        for imageStream in listRange:
            input = tuple(imageList[imageStream].read(28 ** 2))
            output = imageStream == 7
            p.train([[input, output]], 0.01)

print(guess_correct/testSample*100, guess_false/(testSample*9)*100)

minimal = min(p.weights)
maximal = max(p.weights)
print("min {}, max {}".format(minimal, maximal))
i = [int((i+abs(minimal))*4) for i in p.weights]
print(i)
image = Image.frombuffer('L', (28, 28), bytearray(i) , 'raw', 'L', 0, 1)
image.show()
exit
