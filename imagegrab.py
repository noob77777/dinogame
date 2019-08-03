import numpy as np
from PIL import ImageGrab
import cv2
import time

def screen_record(): 
    last_time = time.time()
    while(True):

        x = 400
        y = 150
        screen =  np.array(ImageGrab.grab(bbox=(x,y,x+400,y+150)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
##        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        scale_percent = 60 # percent of original size
        width = int(screen.shape[1] * scale_percent / 100)
        height = int(screen.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        screen = cv2.resize(screen, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow('window',screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

screen_record()
