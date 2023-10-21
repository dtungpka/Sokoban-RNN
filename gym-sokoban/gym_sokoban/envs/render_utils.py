import numpy as np
import pkg_resources
import imageio
import PIL
import re
from PIL import GifImagePlugin
TILE_EDGE = {"3330": 2, "3303": 6, "3300": 38, "3033": 30, "3030": 46, "3003": 22, "3000": 19, "0333": 5, "0330": 4, "0303": 39, "0300": 12, "0033": 47, "0030": 28, "0003": 21, "0000": 20}

TILE_CONER = {
    '03333':8,
    '30333':2,
    '33303':6,
    '33330':0,
    '33300':-1
}
TILE_ADDITIONAL = {
'.33300.00':11,
'.3.00300.':27,
'00.003.3.':29,
'.00300.33':13,
'00.003303':77,
'30.00300.':53,
'.33000003':60,
'333000300':68,
'003000000':37,
'300300303':69,
'003000333':77,
'300000000':45,
'300000333':53,
'003000303':77,
'033000300':68,
'000300303':69,
'003000030':77,
'030000300':68,
'000000003':36




}

GRASS_VARIATIONS = [20,0,1,8,9,16,17]

BASE_ROOM = None
BASE_WATER = None
water_map = None
def get_tile_hash(neighbor_tile):
    #a 9 tile around the current tile
    #loop and convert to a string
    string = ''
    for i in range(neighbor_tile.shape[0]):
        for j in range(neighbor_tile.shape[1]):
            string += str(neighbor_tile[i,j])
    return string


def get_tile_type(tile_pos, room):
    #3 water
    #0 grass
    tile_type = room[tile_pos[0], tile_pos[1]]
    if tile_type == 3:
        return -1 #water
    if tile_type != 0 and tile_type != 3:
        print('tile type error',tile_type)
    neighbor_tile = np.zeros((3,3), dtype=np.uint8)
    for i in range(3):
        for j in range(3):
            #if the tile is out of bounds, set it to tile_type
            if tile_pos[0] + i - 1 < 0 or tile_pos[0] + i - 1 >= room.shape[0] or tile_pos[1] + j - 1 < 0 or tile_pos[1] + j - 1 >= room.shape[1]:
                neighbor_tile[i,j] = 3
            else:
                neighbor_tile[i,j] = room[tile_pos[0] + i - 1, tile_pos[1] + j - 1]
    #get the hash of the tile
    tile_hash = get_tile_hash(neighbor_tile)
    if tile_hash == '000000000':
        return -2 #land full
    edge_hash = ''.join([tile_hash[s] for s in range(1,len(tile_hash),2)])
    coner_hash = ''.join([tile_hash[s] for s in range(0,len(tile_hash),2)])
    transitions_type = TILE_EDGE[edge_hash] if edge_hash in TILE_EDGE else TILE_CONER[coner_hash]
    for key in TILE_ADDITIONAL:
        if re.search(key,tile_hash):
            transitions_type = TILE_ADDITIONAL[key]
    return transitions_type

            
    
resource_package = __name__
min_row = 999
max_row = 0
min_col = 999
max_col = 0
grass_tileset = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'grass.png')))
grass_tile_images = []
grass_tileset = PIL.Image.open(grass_tileset)
water_tile = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'Water.png')))
water_tile = PIL.Image.open(water_tile)
water_tile_images = []
 #every 128 pixels is a new tile, so we loop through the image in 128 pixel increments
for i in range(0, grass_tileset.width, 128):
    for j in range(0, grass_tileset.height, 128):
        #crop the image to get the tile
        grass_tile_images.append(grass_tileset.crop((i, j, i + 128, j + 128)))
for i in range(0, water_tile.width, 128):
    water_tile_images.append(water_tile.crop((i, 0, i + 128, 128)))

boxb_t = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'boxb_t.png')))
boxb_t = PIL.Image.open(boxb_t)

boxbf_t = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'boxbf_t.png')))
boxbf_t = PIL.Image.open(boxbf_t)

boxOK_f = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'boxOK_f.png')))
boxOK_f = PIL.Image.open(boxOK_f)

fire = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'fire.png')))
fire = PIL.Image.open(fire)

fire_extinguisher = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'fire_extinguisher.png')))
fire_extinguisher = PIL.Image.open(fire_extinguisher)

spike = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'spike.png')))
spike = PIL.Image.open(spike)

aris_path = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'arisu.gif')))
aris = PIL.Image.open(aris_path)

ui = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'ui2.png')))
ui = PIL.Image.open(ui)

def get_random_grass_tile():
    distribution = np.array([100,5,10,20,10,5,5])
    #normalize
    distribution = distribution / np.sum(distribution)
    return np.random.choice(GRASS_VARIATIONS,p=distribution)

def init_base_map(room):
    global BASE_ROOM,BASE_WATER,min_row,max_row,min_col,max_col,water_map
    min_row = 999
    max_row = 0
    min_col = 999
    max_col = 0
    for i in range(room.shape[0]):
        for j in range(room.shape[1]):
            if room[i,j] == 3:
                min_row = min(min_row,i)
                min_col = min(min_col,j)
                max_row = max(max_row,i)
                max_col = max(max_col,j)
    #using PIL to load the image
    room = room.copy()[min_row:max_row+1,min_col:max_col+1]
    #generate BASE_WATER as : np.array([0,1,2],[0,1,2],...) to represent the water, repeat the image to fill the room. the array should contain 3 numbers: 0,1,2
    BASE_WATER = np.zeros((room.shape[0], room.shape[1]), dtype=np.uint8)
    for i in range(room.shape[0]):
        for j in range(room.shape[1]):
            BASE_WATER[i, j] = np.random.choice([0,1,2])
    water_map = BASE_WATER.copy()


    _room = room.copy()
    #replace all 1,2,4,5,6 with 0
    _room[_room == 1] = 0
    _room[_room == 2] = 0
    _room[_room == 4] = 0
    _room[_room == 5] = 0
    _room[_room == 6] = 0
    _room[_room == 7] = 0
    _room[_room == 8] = 0
    terrain_map = np.zeros((room.shape[0], room.shape[1]), dtype=np.uint8)
    terrain_rgb = []
    for i in range(room.shape[0]):
        terrain_rgb.append([])
        for j in range(room.shape[1]):
            terrain_map = get_tile_type((i,j), _room)
            if terrain_map == -1:
                terrain_rgb[i].append(-1)
                continue
            if terrain_map == -2 or terrain_map == 20:
                terrain_rgb[i].append(grass_tile_images[get_random_grass_tile()])
                continue
            terrain_rgb[i].append(grass_tile_images[terrain_map])
    BASE_ROOM = terrain_rgb
def overlap_tile(tile1,tile2):
    #using pil to paste tile2 on tile1
    tile1 = tile1.copy()
    tile2 = tile2.convert('RGBA')
    tile1.paste(tile2,mask=tile2.split()[3])
    return tile1

def room_to_rgb(room,frame, room_structure=None,init=False,item=[],facing_right=True):
    global water_map
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__

    room = np.array(room)
    if BASE_ROOM is None:
        init_base_map(room)
    _min_row = 999
    _max_row = 0
    _min_col = 999
    _max_col = 0
    for i in range(room.shape[0]):
        for j in range(room.shape[1]):
            if room[i,j] == 3:
                _min_row = min(_min_row,i)
                _min_col = min(_min_col,j)
                _max_row = max(_max_row,i)
                _max_col = max(_max_col,j)
    #if any of the min,max row or column is out of the base map, reinit the base map
    if _min_row != min_row or _min_col != min_col or _max_row != max_row or _max_col != max_col:
        init_base_map(room)
    if init:
        init_base_map(room)
    
    #crop
    room = room[min_row:max_row+1,min_col:max_col+1]


    aris.seek(frame % aris.n_frames)
    arisu = aris.copy()
    if not facing_right:
        arisu = arisu.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    #water level -> terrain level -> player and box level
    WATER_UPDATE_INTERVAL = 8
    current_water = (frame // WATER_UPDATE_INTERVAL) % 3
    if frame % WATER_UPDATE_INTERVAL == 0:
        water_map = (BASE_WATER + current_water) % 3


    surfaces_img = [None, arisu, overlap_tile(boxbf_t,arisu), None, boxb_t, boxbf_t, boxOK_f,fire_extinguisher,fire]
    surface = []
    for i in range(room.shape[0]):
        surface.append([])
        for j in range(room.shape[1]):
            surface[i].append(surfaces_img[room[i,j]])
    #surfaces = [floor, player, player_on_target, wall, box, box_target, box_on_target,fire_ex,fire]

    #find the min,max row and column that have value 3 in room
    
    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 128+128, room.shape[1] * 128, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 128
        for j in range(room.shape[1]):
            y_j = j * 128
            base_frame = water_tile_images[water_map[i,j]].copy()
            if BASE_ROOM[i][j] != -1:
                base_frame = overlap_tile(base_frame,BASE_ROOM[i][j])
            if surface[i][j] != None:
                base_frame = overlap_tile(base_frame,surface[i][j])
            room_rgb[x_i:(x_i + 128), y_j:(y_j + 128), :] = np.array(base_frame)
    NUMBER_OF_ITEMS = 3
    for i in range(NUMBER_OF_ITEMS):
        offset = int((128 - ui.width) /2)
        if len(item) > i:
            item_img = overlap_tile(ui,surfaces_img[item[i]])
        else:
            item_img = ui
        room_rgb[room_rgb.shape[0]-ui.height:room_rgb.shape[0],offset+i*ui.width:offset+(i+1)*ui.width,:3] = np.array(item_img)[:,:,:3]

    


    return room_rgb


def room_to_tiny_world_rgb(room, room_structure=None, scale=1):

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 1) & (room_structure == 5)] = 6

    wall = [0, 0, 0]
    floor = [243, 248, 238]
    box_target = [254, 126, 125]
    box_on_target = [254, 95, 56]
    box = [142, 121, 56]
    player = [160, 212, 56]
    player_on_target = [219, 212, 56]

    surfaces = [floor, player_on_target, player, box, wall, box_target, box_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_small_rgb = np.zeros(shape=(room.shape[0]*scale, room.shape[1]*scale, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * scale
        for j in range(room.shape[1]):
            y_j = j * scale
            surfaces_id = int(room[i, j])
            room_small_rgb[x_i:(x_i+scale), y_j:(y_j+scale), :] = np.array(surfaces[surfaces_id])

    return room_small_rgb


def room_to_rgb_FT(room, box_mapping, room_structure=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 1) & (room_structure == 5)] = 6

    # Load images, representing the corresponding situation
    box_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box.bmp')))
    box = imageio.imread(box_filename)

    box_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                             '/'.join(('surface', 'box_on_target.bmp')))
    box_on_target = imageio.imread(box_on_target_filename)

    box_target_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box_target.bmp')))
    box_target = imageio.imread(box_target_filename)

    floor_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'floor.bmp')))
    floor = imageio.imread(floor_filename)

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'player.bmp')))
    player = imageio.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'player_on_target.bmp')))
    player_on_target = imageio.imread(player_on_target_filename)

    wall_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'wall.bmp')))
    wall = imageio.imread(wall_filename)

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 128, room.shape[1] * 128, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 128

        for j in range(room.shape[1]):
            y_j = j * 128

            surfaces_id = room[i, j]
            surface = surfaces[surfaces_id]
            if 1 < surfaces_id < 5:
                try:
                    surface = get_proper_box_surface(surfaces_id, box_mapping, i, j)
                except:
                    pass
            room_rgb[x_i:(x_i + 128), y_j:(y_j + 128), :] = surface

    return room_rgb


def get_proper_box_surface(surfaces_id, box_mapping, i, j):
    # not used, kept for documentation
    # names = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]

    box_id = 0
    situation = ''

    if surfaces_id == 2:
        situation = '_target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        situation = '_on_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface_name = 'box{}{}.bmp'.format(box_id, situation)
    resource_package = __name__
    filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multibox', surface_name)))
    surface = imageio.imread(filename)

    return surface


def room_to_tiny_world_rgb_FT(room, box_mapping, room_structure=None, scale=1):
        room = np.array(room)
        if not room_structure is None:
            # Change the ID of a player on a target
            room[(room == 1) & (room_structure == 5)] = 6

        wall = [0, 0, 0]
        floor = [243, 248, 238]
        box_target = [254, 126, 125]
        box_on_target = [254, 95, 56]
        box = [142, 121, 56]
        player = [160, 212, 56]
        player_on_target = [219, 212, 56]

        surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

        # Assemble the new rgb_room, with all loaded images
        room_small_rgb = np.zeros(shape=(room.shape[0] * scale, room.shape[1] * scale, 3), dtype=np.uint8)
        for i in range(room.shape[0]):
            x_i = i * scale
            for j in range(room.shape[1]):
                y_j = j * scale

                surfaces_id = int(room[i, j])
                surface = np.array(surfaces[surfaces_id])
                if 1 < surfaces_id < 5:
                    try:
                        surface = get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j)
                    except:
                        pass
                room_small_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = surface

        return room_small_rgb


def get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j):

    box_id = 0
    situation = 'box'

    if surfaces_id == 2:
        situation = 'target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        box_id = list(box_mapping.values()).index((i, j))
        box_key = list(box_mapping.keys())[box_id]
        situation = 'on_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface = [255, 255, 255]
    if box_id == 0:
        if situation == 'target':
            surface = [111, 127, 232]
        elif situation == 'on_target':
            surface = [6, 33, 130]
        else:
            # Just the box
            surface = [11, 60, 237]

    elif box_id == 1:
        if situation == 'target':
            surface = [195, 127, 232]
        elif situation == 'on_target':
            surface = [96, 5, 145]
        else:
            surface = [145, 17, 214]

    elif box_id == 2:
        if situation == 'target':
            surface = [221, 113, 167]
        elif situation == 'on_target':
            surface = [140, 5, 72]
        else:
            surface = [239, 0, 55]

    elif box_id == 3:
        if situation == 'target':
            surface = [247, 193, 145]
        elif situation == 'on_target':
            surface = [132, 64, 3]
        else:
            surface = [239, 111, 0]

    return surface


def color_player_two(room_rgb, position, room_structure):
    resource_package = __name__

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multiplayer', 'player1.bmp')))
    player = imageio.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'multiplayer', 'player1_on_target.bmp')))
    player_on_target = imageio.imread(player_on_target_filename)

    x_i = position[0] * 128
    y_j = position[1] * 128

    if room_structure[position[0], position[1]] == 5:
        room_rgb[x_i:(x_i + 128), y_j:(y_j + 128), :] = player_on_target

    else:
        room_rgb[x_i:(x_i + 128), y_j:(y_j + 128), :] = player

    return room_rgb


def color_tiny_player_two(room_rgb, position, room_structure, scale=4):

    x_i = position[0] * scale
    y_j = position[1] * scale

    if room_structure[position[0], position[1]] == 5:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [195, 127, 232]

    else:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [96, 5, 145]

    return room_rgb


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
