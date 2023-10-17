

import pygame
import gym
import gym_sokoban
import time

env_name = 'Sokoban-v0'
env = gym.make(env_name)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


for i_episode in range(1):#20
    observation = env.reset()
    steps = 10000
    last_update = time.time()
    while steps > 0:
        env.render(mode='human')
        if time.time() - last_update < .5:
            continue
        last_update = time.time()
        steps -= 1
        action = env.action_space.sample()
        # Sleep makes the actions visible for users
        observation, reward, done, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished")
            env.render()
            break

    env.close()

time.sleep(10)