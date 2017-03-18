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
            #print(self.weights, self.bias)

p = Perceptron(initial_weights=([1]*2),initial_bias=0)

data_set = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0),
]

for i in range(50):
    p.train(data_set,0.001)

print(p.weights, p.bias)

ok = 0
false = 0
for input, expectedOutput in data_set:
    guess = p.predict(input)
    if guess == expectedOutput:
        ok += 1
    else:
        false += 1

print(ok/4*100)
print(false/4*100)


p = Perceptron(initial_weights=([1]*2),initial_bias=0)

data_set = [
    ((0, 0), 0),
    ((0, 1), 0),
    ((1, 0), 0),
    ((1, 1), 1),
]

for i in range(50):
    p.train(data_set,0.01)

print(p.weights, p.bias)

ok = 0
false = 0
for input, expectedOutput in data_set:
    guess = p.predict(input)
    if guess == expectedOutput:
        ok += 1
    else:
        false += 1

print(ok/4*100)
print(false/4*100)