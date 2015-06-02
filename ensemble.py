import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import random
import copy
import math
import sys

from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

class Bagging:
    def __init__(self):
        self.classifiers = []
    def bootstrapping(self, x, y):
        samplex, sampley = [], []
        for i in range(len(x)):
            rand = random.randint(0, len(x) - 1)
            samplex.append(x[rand].copy().tolist())
            sampley.append(y[rand].copy())
        samplex = np.array(samplex)
        sampley = np.array(sampley)
        return samplex, sampley
    def train(self, x, y, basic_classifier):
        samplex, sampley = self.bootstrapping(x, y)
        cl = basic_classifier.newInstance()
        cl.train(samplex, sampley)
        self.classifiers.append(cl)
    def predict(self, x):
        positive, negative = 0, 0
        for classifier in self.classifiers:
            if classifier.predict(x) == 1:
                positive += 1;
            else:
                negative += 1;
        return 1 if positive > negative else 0
    def name(self):
        return "bagging"

class AdaBoostM1:
    def __init__(self, sampleNum):
        self.weight = [1.0 / sampleNum for i in range(sampleNum)]
        self.voting = []
        self.classifiers = []
    def train(self, x, y, basic_classifier):
        cl = basic_classifier.newInstance()
        cl.train(x, y, self.weight)
        Et = 0.0
        for i in range(len(x)):
            if cl.predict(x[i]) != y[i]:
                Et += self.weight[i]
        beta = Et / (1 - Et)
        for i in range(len(x)):
            if cl.predict(x[i]) == y[i]:
                self.weight[i] *= beta
        # normalize
        weightSum = sum(self.weight)
        if weightSum > 0:
            self.weight = [w / weightSum for w in self.weight]
        else:
            self.weight = [1.0 / len(self.weight) for w in self.weight]
        self.voting.append(math.log(1.0 / beta) if beta > 0 else math.log(1e7))
        self.classifiers.append(cl)
    def predict(self, x):
        positive, negative = 0.0, 0.0
        for i in range(len(self.classifiers)):
            if self.classifiers[i].predict(x) == 1:
                positive += self.voting[i]
                # positive += 1
            else:
                negative += self.voting[i]
                # negative += 1
        return 1 if positive > negative else 0
    def name(self):
        return "adaboostM1"

class DecisionTree:
    def train(self, x, y, weight = None):
        self.classifier = tree.DecisionTreeClassifier()
        self.classifier.fit(x, y, sample_weight = weight)
    def predict(self, x):
        return self.classifier.predict(x)[0]
    def newInstance(self):
        return DecisionTree()
    def name(self):
        return "decisionTree"

class SVM:
    def train(self, x, y, weight = None):
        self.classifier = sklearn.svm.SVC(kernel = 'linear')
        self.classifier.fit(x, y, sample_weight = weight)
    def predict(self, x):
        return self.classifier.predict(x)[0]
    def newInstance(self):
        return SVM()
    def name(self):
        return "svm"

class NaiveBayes:
    def train(self, x, y, weight = None):
        self.classifier = BernoulliNB()
        self.classifier.fit(x, y, sample_weight = weight)
    def predict(self, x):
        return self.classifier.predict(x)[0]
    def newInstance(self):
        return NaiveBayes()
    def name(self):
        return "NaiveBayes"

class KNeighbors:
    def train(self, x, y, weight = None):
        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.classifier.fit(x, y)
    def predict(self, x):
        return self.classifier.predict(x)[0]
    def newInstance(self):
        return KNeighbors()
    def name(self):
        return "KNeighbors"
        

def sample(rate, x, y):
    trainx, trainy, testx, testy = [], [], [], []
    for i in range(len(x)):
        if random.random() < rate:
            trainx.append(x[i].copy().tolist())
            trainy.append(y[i].copy())
        else:
            testx.append(x[i].copy().tolist())
            testy.append(y[i].copy())
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    testx = np.array(testx)
    testy = np.array(testy)
    return trainx, trainy, testx, testy

def balancedSample(rate, x, y):
    trainx, trainy, testx, testy = [], [], [], []
    # Stratrified Sampling
    df = pd.DataFrame(x) 
    posSample = df[y == 1].values
    negSample = df[y == 0].values
    rate = 0.8
    for i in range(len(negSample)):
        if random.random() < rate:
            trainx.append(negSample[i].copy().tolist())
            trainy.append(0)
        else:
            testx.append(negSample[i].copy().tolist())
            testy.append(0)

    trainPos = []
    for i in range(len(posSample)):
        if random.random() < rate:
            trainx.append(posSample[i].copy().tolist())
            trainPos.append(posSample[i].copy().tolist())
            trainy.append(1)
        else:
            testx.append(posSample[i].copy().tolist())
            testy.append(1)


    # Multiply positive training samples
    delta = int((len(negSample) - len(posSample)) * rate)
    for i in range(delta):
        rand  = random.randint(0, len(trainPos) - 1)
        trainx.append(trainPos[rand])
        trainy.append(1)

    trainx = np.array(trainx)
    trainy = np.array(trainy)
    testx = np.array(testx)
    testy = np.array(testy)
    return trainx, trainy, testx, testy


def preprocessing(normalize, hasLink):
    df = pd.read_csv('./ContentNewLinkAllSample.csv')
    df['class'] = df['class'].map(lambda c: 1 if c == 'spam' else 0)
    y = df['class']
    if hasLink:
        x = df[df.columns[:-1]].values
    else:
        x = df[df.columns[:-139]].values
    if normalize:
        x = sklearn.preprocessing.normalize(x, axis = 0)
    return balancedSample(0.8, x, y)
    # return sample(0.8, x, y)
    
def test(testx, testy, classifier):
    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0    
    for i in range(len(testx)):
        prediction = classifier.predict(testx[i])
        if prediction == 1 and testy[i] == 1:
            truePos += 1
        elif prediction == 0 and testy[i] == 1:
            falseNeg += 1
        elif prediction == 1 and testy[i] == 0:
            falsePos += 1
        elif prediction == 0 and testy[i] == 0:
            trueNeg += 1
    accuracy = float(truePos + trueNeg) / len(testx)
    precision = float(truePos) / (truePos + falsePos) if truePos + falsePos > 0 else 0.0
    recall = float(truePos) / (truePos + falseNeg) if truePos + falseNeg > 0 else 0.0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return accuracy, precision, recall, F1


def run(normalize = True, hasLink = True, rounds = 50):
    trainx, trainy, testx, testy = preprocessing(normalize, hasLink)
    for ensemble in [Bagging()]:
        for basic_classifier in [DecisionTree()]:
            filename  = './result/' + ensemble.name() + "_" + basic_classifier.name()
            if normalize:
                filename += "_norm"
            if hasLink:
                filename += "_link"
            filename += '.csv'
            fout = open(filename, 'w')
            fout.write('round,accuracy,precision,recall,F1score\n')
            for i in range(0, rounds):
                ensemble.train(trainx, trainy, basic_classifier)
                res = test(testx, testy, ensemble)
                print ensemble.name(), basic_classifier.name(), i, res
                fout.write(str(i) + ',')
                fout.write(str(res[0]) + ',' + str(res[1]) + ',' + str(res[2]) + ',' + str(res[3]) + '\n')
            fout.close()


if __name__ == '__main__':
    run()
    run(normalize = False)
    run(hasLink = False)
