import numpy as np
import pkg_resources
import imageio


def room_to_rgb(room, room_structure=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__

    room = np.array(room)


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

    surfaces = [floor, player, player_on_target, wall, box, box_target, box_on_target]

    #find the min,max row and column that have value 3 in room
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
    #crop
    room = room[min_row:max_row+1,min_col:max_col+1]
    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 128, room.shape[1] * 128, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 128

        for j in range(room.shape[1]):
            y_j = j * 128
            surfaces_id = room[i, j]

            room_rgb[x_i:(x_i + 128), y_j:(y_j + 128), :] = surfaces[surfaces_id]

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
