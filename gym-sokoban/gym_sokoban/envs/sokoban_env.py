import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .room_utils import generate_room
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import numpy as np
import numba
import PIL


def inter(arr):
    arr = [list(row) for row in arr]
    arr = [[int(char
                .replace(" ", "0")
                .replace("@", "1")
                .replace("+", "2")
                .replace("#", "3")
                .replace("$", "4")
                .replace(".", "5")
                .replace("*", "6")
                .replace("e", "7")
                .replace("f", "8")
                )
            for char in row] for row in arr]
    return np.array(arr, dtype=np.uint8)

CHAPTER = -1
LEVEL = -1
LEVEL_PATH = "xsbs/0/1.xsb"
directions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
class SokobanEnv(gym.Env):
    a = 1
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=250,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0
        self.items = []
        # Penalties and Rewards
        self.penalty_for_step = -0.001
        self.penalty_box_off_target = -.1
        self.penalty_illegal_move = -.05
        self.reward_push_box = 0.005
        self.reward_pickup_item = 1
        self.reward_extinguish_fire = 5
        self.reward_box_on_target = 8
        self.reward_finished = 20
        self.reward_last = 0
        self.current_frame = 0
        self.last_action = -1
        self.base_room_state = None
        self.base_room_structure = None
        # Other Settings
        self.viewer = None
        self.pending_init_render= False
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 128, dim_room[1] * 128)
        self.observation_space = np.zeros((1, 26, 26))
        self.set_level(0,1)
        if reset:
            # Initialize Room
            _ = self.reset()
            

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None
        if action == 3 or action == 0:
            self.last_action = action


        moved_box = False

        # All push actions are in the range of [0, 3]
        if action < 4:
            moved_player, moved_box = self._push(action)
            if not moved_box and not moved_player:
                self.reward_last += self.penalty_illegal_move
            if moved_box:
                self.reward_last += self.reward_push_box
            else:
                self.reward_last += self.penalty_for_step
        elif action == 4:
            moved_player = self._extinguish()
            if not moved_player:
                self.reward_last += self.penalty_illegal_move
            else:
                self.reward_last += self.penalty_for_step
        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.room_state.reshape((1, 26, 26))

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return self.serialize_state(), self.reward_last, done, info
    def _pickup(self):
        """
        Pick up an item, if available.
        :return: Boolean, indicating a change of the room's state
        """
        if len(self.items) >= 3:
            return True
        current_position = self.player_position.copy()
        if self.room_state[current_position[0], current_position[1]] == 7:
            self.room_state[current_position[0], current_position[1]] = 0
            self.room_fixed[current_position[0], current_position[1]] = 0
            self.reward_last += self.reward_pickup_item
            self.items.append(7)
            return True
        return False
    def _extinguish(self):
        """
        Extinguish a fire, in a 3x3 area around the player.
        :return: Boolean, indicating a change of the room's state
        """
        current_position = self.player_position.copy()
        _extinguished = False
        #check inventory
        if 7 not in self.items:
            return False
        
            
        for direction in directions:
            new_position = current_position + direction
            if self.room_state[new_position[0], new_position[1]] == 8:
                self.room_state[new_position[0], new_position[1]] = 0
                self.room_fixed[new_position[0], new_position[1]] = 0
                _extinguished = True
                self.reward_last += self.reward_extinguish_fire
        if _extinguished:
            self.items.remove(7)
        return _extinguished
    

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] or new_box_position[1] >= self.room_state.shape[1]:
            return False, False
        
        #if fire, don't push
        if self.room_state[new_position[0], new_position[1]] == 8:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [4, 6]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [0, 5]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 1
            self.room_state[current_position[0], current_position[1]] = self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 6 if self.room_fixed[new_box_position[0], new_box_position[1]] == 5 else 4

            self.room_state[new_box_position[0], new_box_position[1]] = box_type


            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [0, 5, 7]:
            self.player_position = new_position
            if self.room_state[new_position[0], new_position[1]] == 7:
                self._pickup()
            self.room_state[(new_position[0], new_position[1])] = 1
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        

        # count boxes off or on the target
        empty_targets = self.room_state == 5
        player_on_target = (self.room_fixed == 5) & (self.room_state == 1)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target
        for bp in np.argwhere(self.room_state == 4):
            if 3 in (self.room_state[bp[0] + 1, bp[1]],
                     self.room_state[bp[0] - 1, bp[1]]):
                if 3 in (self.room_state[bp[0], bp[1] + 1],
                         self.room_state[bp[0], bp[1] - 1]):
                    self.reward_last -= 1

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps() or self._check_if_fire_exist_without_extinguisher()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 5
        player_hiding_target = (self.room_fixed == 5) & (self.room_state == 1)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)
    def _check_if_fire_exist_without_extinguisher(self):
        #count the number of fires
        fires = np.sum((self.room_state == 8)*1)
        extinguishers = np.sum((self.room_state == 7)*1) + sum([1 for i in self.items if i == 7])
        return fires > 0 and (extinguishers == 0)
    
    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            self.room_state = self.base_room_state.copy()
            self.room_fixed = self.base_room_structure.copy()
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player)

        self.player_position = np.argwhere((self.room_state == 1) | (self.room_state == 2))[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0
        self.items = []
        self.last_action = -1

        return self.serialize_state()

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=img.shape[1]*2)
            self.viewer.imshow(img)
            return self.viewer.isopen

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state,self.current_frame, self.room_fixed,self.pending_init_render,item=self.items,facing_right=self.last_action != 3)
            self.pending_init_render = False
            self.current_frame += 1

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP
    def is_finished(self):
        return self._check_if_all_boxes_on_target()
    def set_level(self,chapter,level):
        global CHAPTER, LEVEL
        CHAPTER = chapter
        LEVEL = level
        self.base_room_structure, self.base_room_state= generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=False,
                chapter=CHAPTER,
                level=LEVEL
            )
        self.room_state = self.base_room_state.copy()
        self.room_fixed = self.base_room_structure.copy()
        self.pending_init_render = True
    def serialize_state(self):
        s = ""
        for i in self.room_state:
            for j in i:
                s += str(j)
        #append inventory
        s += ''.join([str(x) for x in self.items])
        return s
ACTION_LOOKUP = {
    0: 'right',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'extinguish',
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

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human']
