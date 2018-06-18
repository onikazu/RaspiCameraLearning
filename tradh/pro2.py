# -*- coding: utf-8 -*-
import pygame
import pygame.camera
import io
import os
import subprocess

def main():

    # Init framebuffer/touchscreen environment variables
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.putenv('SDL_FBDEV'      , '/dev/fb1')
    os.putenv('SDL_MOUSEDRV'   , 'TSLIB')
    os.putenv('SDL_MOUSEDEV'   , '/dev/input/touchscreen')
    (x,y) = (0,0)

    pygame.init()
    pygame.camera.init()
    pygame.font.init()
    pygame.mouse.set_visible(False)

    font = pygame.font.SysFont('Comic Sans MS', 24)
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    camera = pygame.camera.Camera("/dev/video0", (640,480))
    camera.start()

    while True:
        snapshot = camera.get_image()
        screen.blit(snapshot, (x, y))

        for event in pygame.event.get():
            # take picture
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                filename = "/tmp/camera.jpg"
                pygame.image.save(snapshot, filename)
                cmd = '/home/pi/src/DeepBeliefSDK/source/jpcnn -i ' + filename + ' -n /home/pi/src/DeepBeliefSDK/networks/jetpac.ntwk  -m s'
                proc = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                stdout, stderr = proc.communicate()
                result = ["0.0",""]
                for line in stdout.splitlines():
                    data = line.split("\t")
                    if float(result[0]) < float(data[0]):
                        result = data
                textsurface = font.render(result[1], False, (0, 0, 0))
                screen.blit(textsurface,(0,0))
                pygame.display.update()
                pygame.time.wait(2000)
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()
        pygame.time.wait(30)

if __name__ == "__main__":
        main()