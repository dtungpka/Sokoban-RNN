from PIL import Image
import cv2
import itertools as it
import numpy as np
import json
import threading



class data:
    terrain = '000000000'

def thrd():
    while True:
        try:
            data.terrain = input('Terrain: ')
            if len(data.terrain) != 9:
                break
        except Exception as e:
            if e == KeyboardInterrupt:
                break

def foo(l,c):
 yield from it.product(*([l]*c))

def get_img(s):
    tmap = np.zeros((3,3))
    index = 0
    for i in range(tmap.shape[0]):
        for j in range(tmap.shape[1]):
            tmap[i,j] = s[index] == '3'
            index+= 1

    img = np.zeros((128*tmap.shape[0],128*tmap.shape[1],3))
    for i in range(tmap.shape[0]):
        for j in range(tmap.shape[1]):
            x = 128*i
            y = 128*j
            if tmap[i,j] == 1:
                img[x:x+128,y:y+128,0] = 200
            else:
                img[x:x+128,y:y+128,1] = 200
    
    return img
grass_img = "D:\\2023-2024\\RNN\\Sokoban\\themes\\Sprout Lands - Sprites - Basic pack\\Tilesets\\grass.png"
grass_tileset = Image.open(grass_img)
#cvt to RGBA
grass_tileset = grass_tileset.convert('RGBA')
grass_tile_images = []
for i in range(0, grass_tileset.width, 128):
        for j in range(0, grass_tileset.height, 128):
            #crop the image to get the tile
            grass_tile_images.append(grass_tileset.crop((i, j, i + 128, j + 128)))
def get_grass_img(index):
    return grass_tile_images[index]

map_list = [''.join(x) for x in foo('30',9)]
thread = threading.Thread(target=thrd)
thread.start()

while True:
    curr_map = get_img(data.terrain)
    cv2.imshow('frame',curr_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
