import picamera
from time import sleep

#omxplayer
with picamera.PiCamera() as camera:
 camera.resolution = (1024, 768)
 camera.start_preview()
 camera.start_recording('output/video.h264')
 # Camera warm-up time
 sleep(5)

 