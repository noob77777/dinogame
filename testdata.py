import numpy as np
from PIL import ImageGrab
import cv2
import time
import pickle
import keyboard
import matplotlib.pyplot as plt

f = open('dataset.list', 'rb')
data = pickle.load(f)
f.close()

cnt = 0

for i in range(len(data)):
    y = data[i][1]
    if y == 1:
        cnt = i
        break

cv2.imshow('window', data[i][0])
print(data[i][1])
