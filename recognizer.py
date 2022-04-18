from pip import main
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np

def load():
    model = load_model("./number_model.h5")
    return model


def load_video(model):
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height = frame.shape[0] 
            width = frame.shape[1]
            rect_x = (width//2)-200
            rect_y = (height//2)-200
            rect_w = (width//2) +50
            rect_h = (height//2) + 50
            frame = cv.rectangle(frame, (rect_x, rect_y), (rect_w, rect_h), (0,0,255), 1)
            roi = frame[rect_y:rect_h+1, rect_x:rect_w+1]
            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            roi = cv.medianBlur(roi, 5)
            
            _, thres_roi = cv.threshold(roi, 0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
            thres_roi_resized = cv.resize(thres_roi, (28, 28), interpolation=cv.INTER_AREA)
            value = model.predict(thres_roi_resized.reshape(1, 28,28, 1))
            print(value.argmax(), value[0][value.argmax()])
            cv.imshow("thres", thres_roi)
            cv.imshow("thresre", thres_roi_resized)
            
            cv.imshow("roi", roi)
            # cv.imshow("frame", frame)
        if cv.waitKey(10) & 0xff == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    model = load()
    # model.summary()
    load_video(model)
    