import numpy as np
from PIL import ImageGrab
import cv2
import time
import pickle
import keyboard

def generate_data(): 
##    last_time = time.time()
    record = 0
    data = []
    while(True):
        # 800x600 windowed mode for GTA 5, at the top left position of your main screen.
        # 40 px accounts for title bar.

        if keyboard.is_pressed('q'):
            record = 1
        elif record == 1:
            break
        
        x = 400
        y = 150
        screen =  np.array(ImageGrab.grab(bbox=(x,y,x+300,y+150)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
##        print('loop took {} seconds'.format(time.time()-last_time))
##        last_time = time.time()

        scale_percent = 60 # percent of original size
        width = int(screen.shape[1] * scale_percent / 100)
        height = int(screen.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        screen = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA)

        if(record == 1):
            X = screen
            y = 0
            if keyboard.is_pressed('space'):
                y = 1
            data.append((X, y))
        
        ##cv2.imshow('window',screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    f = open('dataset.list', 'wb')
    pickle.dump(data, f)
    f.close()

generate_data()

