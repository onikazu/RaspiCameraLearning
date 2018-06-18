# source
# http://walking-succession-falls.com/raspberrypi%E3%81%AE%E3%82%AB%E3%83%A1%E3%83%A9%E3%83%A2%E3%82%B8%E3%83%A5%E3%83%BC%E3%83%ABv2%E3%81%8B%E3%82%89%E6%B5%81%E3%82%8C%E3%82%8B%E3%82%B9%E3%83%88%E3%83%AA%E3%83%BC%E3%83%9F%E3%83%B3/

from picamera.array import PiRGBArray
from picamera import PiCamera
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import sys, os
import time
import cv2

print("[INFO] loading model...")
model = MobileNet(weights='imagenet')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(np.uint8(frame))
    image = image.resize((224, 224))
    x = image.img_to_array(image)
    pred_data = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(pred_data))
    results = decode_predictions(preds, top=1)[0]
    for result in results:
        label = result[1]


    # show the frame
    image = cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    # if (True):


    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)



    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break