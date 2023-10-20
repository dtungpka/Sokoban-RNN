from PIL import Image
import cv2
import itertools as it
import numpy as np
import json
import threading
mapped = {'edge':{}, 'coner':{}}


class data:
    tile_choose = 0
    ok = False

def thrd():
    while True:
        try:
            data.tile_choose =int(input('Tile: '))
            data.ok = input('OK?: ').lower() == '.'
        except:
            pass

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



for i,s in enumerate(map_list):
    edge_hash = ''.join([s[c] for c in range(1,len(s),2)])
    coner_hash = ''.join([s[c] for c in range(0,len(s),2)])
    is_conner = False
    
    if edge_hash == '3333':
        continue
    if s[4] == '3' or (edge_hash in mapped['edge'] and not is_conner ) or (coner_hash in mapped['coner'] and is_conner):
        continue
    if edge_hash in mapped['edge']:
        is_conner = True
    print(edge_hash,coner_hash)
    tile_choose = 0
    print(f'Iter {i}/{len(map_list)}: {s} ')
    show_img = np.zeros((128*3,128*3,3))
    curr_map = get_img(s)
    show_img[:128*3,:,:] = curr_map
    cv2.imshow('grass',np.array(get_grass_img(tile_choose))[:,:,0:3])
    cv2.imshow('frame',show_img)
    while True:
        tile_choose = data.tile_choose
        cv2.imshow('grass',np.array(get_grass_img(tile_choose))[:,:,0:3])
        cv2.imshow('frame',show_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if data.ok:
            if not is_conner:
                mapped['edge'][edge_hash] = tile_choose
            else:
                mapped['coner'][coner_hash] = tile_choose
            data.ok = False
            with open('map_terrain.json','w') as f:
                 json.dump(mapped,f)
            break
print("-----------Done-------------")






