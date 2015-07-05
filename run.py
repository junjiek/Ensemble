from sklearn import tree
import sklearn.svm as svm_pack
import sklearn.preprocessing as prep
from sklearn import linear_model

import random
import copy
import math
import sys

class svm:
    def copy(self):
        return svm()

    def train(self, x, y, weight = None):
        self.c = svm_pack.SVC(kernel = 'linear', max_iter = 2000)
        self.c.fit(x, y, sample_weight = weight)

    def predict(self, x):
        return self.c.predict(x)[0]

    def name(self):
        return 'svm'

class dtree:
    def copy(self):
        return dtree()

    def train(self, x, y, weight = None):
        self.c = tree.DecisionTreeClassifier()
        self.c.fit(x, y, sample_weight = weight)

    def predict(self, x):
        return self.c.predict(x)[0]

    def name(self):
        return 'dtree'

class perceptron:
    def copy(self):
        return perceptron()

    def train(self, x, y, weight = None):
        self.c = linear_model.Perceptron(shuffle = True, n_iter = 50)
        self.c.fit(x, y, sample_weight = weight)

    def predict(self, x):
        return self.c.predict(x)[0]

    def name(self):
        return 'perceptron'

class bagging:
    def train(self, t, x, y, base_classifier):
        if t == 1:
            self.ensemble = []
        self.t = t
        xt, yt = self.sample(x, y)
        ct = base_classifier.copy()
        ct.train(xt, yt)
        self.ensemble.append(ct)

    def predict(self, x):
        pos, neg = 0, 0
        for i in range(self.t):
            if self.ensemble[i].predict(x) == 1:
                pos += 1
            else:
                neg += 1
        return 1 if pos > neg else 0

    def sample(self, x, y):
        rx, ry = [], []
        for i in range(len(x)):
            rand = random.randint(0, len(x) - 1)
            rx.append(copy.deepcopy(x[rand]))
            ry.append(copy.deepcopy(y[rand]))
        return rx, ry

    def name(self):
        return 'bagging'

class adaboost:
    def train(self, t, x, y, base_classifier):
        if t == 1:
            self.weight = [1.0 / len(x) for i in range(len(x))]
            self.voting = []
            self.ensemble = []
        self.t = t
        ct = base_classifier.copy()
        ct.train(x, y, [e * len(x) for e in self.weight])
        et = 0.0
        for j in range(len(x)):
            if ct.predict(x[j]) != y[j]:
                et += self.weight[j]
        beta = et / (1 - et)
        for j in range(len(x)):
            if ct.predict(x[j]) == y[j]:
                self.weight[j] *= beta
        self.weight = self.normalize(self.weight)
        self.voting.append(math.log(1.0 / beta) if beta > 0 else math.log(1e7))
        self.ensemble.append(ct)

    def predict(self, x):
        pos, neg = 0.0, 0.0
        for i in range(self.t):
            if self.ensemble[i].predict(x) == 1:
                #pos += self.voting[i]
                pos += 1
            else:
                #neg += self.voting[i]
                neg += 1
        return 1 if pos > neg else 0

    def name(self):
        return 'adaboost'

    def normalize(self, w):
        return [e / sum(w) for e in w] if sum(w) > 0 else [1.0 / len(w) for i in range(len(w))]

def sample(rate, x, y):
    trainx, trainy, testx, testy = [], [], [], []
    for i in range(len(x)):
        if random.random() < rate:
            trainx.append(copy.deepcopy(x[i]))
            trainy.append(copy.deepcopy(y[i]))
        else:
            testx.append(copy.deepcopy(x[i]))
            testy.append(copy.deepcopy(y[i]))
    return trainx, trainy, testx, testy

def process(argv1, argv2, normalize = True, nolink = False):
    x, y = [], []
    cnt = 0
    for line in open('data.csv'):
        cnt += 1
        if cnt == 1:
            continue
        inputs = line.strip().split(',')
        if not nolink:
            x.append([float(e) for e in inputs[:-1]])
        else:
            x.append([float(e) for e in inputs[:-139]])
        y.append(1 if inputs[-1] == 'spam' else 0)
    if normalize:
        x = prep.normalize(x, axis = 0)
    trainx, trainy, testx, testy = sample(0.8, x, y)
    for ensemble_classifier in [adaboost(), bagging()]:
        for base_classifier in [svm(), dtree(), perceptron()]:
            if ensemble_classifier.name() != argv1 or base_classifier.name() != argv2:
                continue
            for t in range(1, 60):
                ensemble_classifier.train(t, trainx, trainy, base_classifier)
                tp, tn, fp, fn = 0, 0, 0, 0
                for i in range(len(testx)):
                    py = ensemble_classifier.predict(testx[i])
                    if py == 1 and testy[i] == 1:
                        tp += 1
                    if py == 1 and testy[i] == 0:
                        fp += 1
                    if py == 0 and testy[i] == 1:
                        fn += 1
                    if py == 0 and testy[i] == 0:
                        tn += 1
                print ensemble_classifier.name(), base_classifier.name(), t
                accu = float(tp + tn) / (tp + tn + fn + fp)
                prec = float(tp) / (tp + fp) if tp + fp > 0 else 0.0
                recall = float(tp) / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
                print 'Accuracy', accu
                print 'Precision', prec
                print 'Recall', recall
                print 'F1', f1
                filename = argv1 + '_' + argv2
                if not  normalize:
                    filename += '_nonorm'
                if nolink:
                    filename += '_nolink'
                fout = open(filename + '.out', 'a+')
                fout.write(ensemble_classifier.name() + ' ')
                fout.write(base_classifier.name() + ' ')
                fout.write(str(t) + ' ')
                fout.write(str(accu) + ' ' + str(prec) + ' ' + str(recall))
                fout.write(' ' + str(f1) + '\n')
                fout.close()

if __name__ == '__main__':
    nl = True if '-nolink' in sys.argv else False
    norm = False if '-nonorm' in sys.argv else True
    process(sys.argv[1], sys.argv[2], nolink = nl, normalize = norm)
