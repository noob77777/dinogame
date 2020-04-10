import numpy as np
from PIL import ImageGrab
import cv2
import time
import pickle
import keyboard

from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

f = open('dataset.list', 'rb')
data = pickle.load(f)
f.close()

def preprocess(_X):
    scale_percent = 50 # percent of original size
    width = int(_X.shape[1] * scale_percent / 100)
    height = int(_X.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    _X = cv2.resize(_X, dim, interpolation = cv2.INTER_AREA)

    ##cv2.imshow('window', _X)
        
    _X = _X/255.0
    _X = np.ravel(_X)

    return _X


def Xy(data):
    X = []
    y = []
    
    for _X, _y in data:

        _X = preprocess(_X)
        X.append(_X)
        y.append(_y)

    X = np.array(X)
    y = np.array(y)
    return X, y


def train():
    X, y = Xy(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    clf = RandomForestClassifier(n_estimators = 3)
    clf2 = svm.SVC(kernel = 'linear', gamma = 'auto')
    clf3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)

    confidence = clf.score(X_test, y_test)
    confidence2 = clf2.score(X_test, y_test)
    confidence3 = clf3.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    y_pred2 = clf2.predict(X_test)
    y_pred3 = clf3.predict(X_test)

    y_final = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        y_final[i] = y_pred[i] | y_pred2[i]

    print(y_test.tolist().count(1), y_final.tolist().count(1))
    matrix1 = confusion_matrix(y_test, y_pred)
    print(matrix1)
    matrix2 = confusion_matrix(y_test, y_pred2)
    print(matrix2)
    matrix3 = confusion_matrix(y_test, y_pred3)
    print(matrix3)
    matrix = confusion_matrix(y_test, y_final)
    print(matrix)
    print(confidence, confidence2, confidence3)

    f = open('randomforest.classifier', 'wb')
    pickle.dump(clf, f)
    f.close()
    f = open('svm.classifier', 'wb')
    pickle.dump(clf2, f)
    f.close()
    f = open('knn.classifier', 'wb')
    pickle.dump(clf3, f)
    f.close()

train()
