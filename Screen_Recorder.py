import time
import numpy as np
import cv2
from mss import mss

screen = {'left': 70, 'top': 25, 'width': 900, 'height': 1000}
pic_buffer = []
# Lock = True
with mss() as sct:
    start_time = time.time()
    last_time = time.time()
    while "Screen capturing":

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(screen))
        pic_buffer.append(img)

        # 0.01 for 0.1
        # 0.05 for 0.25
        if time.time() - start_time > 0.01:
            cv2.imshow("OpenCV/Numpy normal", pic_buffer[0])
            pic_buffer.pop(0)  # Pop out the activated screen from list
        # print("fps: {}".format(1 / (time.time() - last_time)))

        # If Keyboard Cancelling
        if cv2.waitKey(33) & 0xFF in (
            ord('q'),
            27,
        ):
            break

