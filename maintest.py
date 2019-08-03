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

import classifier

f = open('main.classifier', 'rb')
clf = pickle.load(f)
f.close()

while(True):
    
    x = 400
    y = 150
    screen =  np.array(ImageGrab.grab(bbox=(x,y,x+300,y+150)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    scale_percent = 60 # percent of original size
    width = int(screen.shape[1] * scale_percent / 100)
    height = int(screen.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    screen = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA)

    X = classifier.preprocess(screen)
    y = clf.predict(np.array([X]))

    print(y)
    res = y[0]

    
    if(res == 1):
        keyboard.press('space')
    
    cv2.imshow('window',screen)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
