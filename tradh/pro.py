# https://qiita.com/kamotsuru/items/bc4591daaba0369e5249

import io
import time
import picamera
import picamera.array
import cv2
from keras.applications.mobilenet import MobileNet
from keras.preprocessing.image import array_to_img
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


t1= time.clock()
print("[INFO] loading model...")
model = MobileNet(weights='imagenet')
t2= time.clock()
print('load resnet50: %.3f s' % (t2 - t1))

WINDOW_WIDTH = 640 #854
WINDOW_HEIGHT = 480
ORG_WIDTH = 640 #1280
ORG_HEIGHT = 480 #720

windowName = 'Pi NoIR'

cv2.namedWindow(windowName)
cv2.resizeWindow(windowName, WINDOW_WIDTH, WINDOW_HEIGHT)

w_offset = int((ORG_WIDTH - ORG_HEIGHT)/2)

with picamera.PiCamera() as camera:
        camera.start_preview()
        camera.resolution = (ORG_WIDTH, ORG_HEIGHT)
        time.sleep(2)
#        str = 'not yet classified'

        while True:
            t1= time.clock()
            with picamera.array.PiRGBArray(camera) as stream:
                camera.capture(stream, 'bgr')
                image = stream.array
                if WINDOW_WIDTH != ORG_WIDTH:
                    image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
                #cv2.putText(image, str,(0, CAMERA_HEIGHT - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
                cv2.imshow(windowName, image)

            t2 = time.clock()
#            print('capture image : %.3f s' % (t2 - t1))
            key = cv2.waitKey(12)
            #press Esc(27) to quit, press c(99) to classify
            if key==27:
                break
            elif key==99:
                print('classifying image...')
                t2 = time.clock()
                image = image[: ,w_offset:w_offset + ORG_HEIGHT, :]
                image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)
                x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                array_to_img(x).save('classified.jpg')
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)
                print('Predicted:', decode_predictions(preds))
#                str = '{}'.format(decode_predictions(preds)[0][0][1:])
                t3 = time.clock()
                print('inference :  %.3f s' % (t3 - t2))

cv2.destroyAllWindows()