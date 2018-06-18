# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from gpiozero.tools import random_values
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os


print("[INFO] loading model...")
model = MobileNet(weights='imagenet')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image and initialize the label and
    # probability of the prediction
    preds = model.predict(preprocess_input(image))
    results = decode_predictions(preds, top=1)[0]
    for result in results:
        #print(result)
        label = result[1]
        prob = round(result[2]*100, 2)
    frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)