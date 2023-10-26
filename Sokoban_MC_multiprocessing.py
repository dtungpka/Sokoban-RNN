
import time
import numpy as np
import pickle
import os
import multiprocessing as mp
import atpbar
from atpbar import atpbar,register_reporter, find_reporter, flush
import gym
import gym_sokoban
EPISODES = 20000 * 8

#create folder value_functions if it doesn't exist
if not os.path.exists('value_functions'):
    os.mkdir('value_functions')

def process(process_name,reporter,queue,every_visit_mc = False,episodes = 300):
        register_reporter(reporter)
        chapter = 0
        level = 4
        if not os.path.exists('Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing'):
            os.makedirs('Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing')
        env_name = 'Sokoban-v1'
        env = gym.make(env_name)
        ACTION_LOOKUP = env.unwrapped.get_action_lookup()
        env.unwrapped.set_level(chapter, level)
        env.seed(os.getpid())
        env.reset()
        V = {}
        total_returns = {}
        N = {}
        finished = 0
        max_reward = 0
        max_actions = 300
        finished_history = []
        for episode in  atpbar(range(episodes),name = process_name):
            
            visited = []
            env.reset()
            state = env.unwrapped.serialize_state()
            done = False
            
            for t in range(max_actions):
                action_time = time.time()
                if done:
                    break
                action = greedy_policy(env,V,state,eps_comp=episode/episodes)
                next_state, reward, done, info = env.step(action)
                
                if reward > max_reward:
                    max_reward = reward
                
                if every_visit_mc or (not every_visit_mc and state not in visited):
                    if not every_visit_mc:
                        visited.append((state,action))
                    if state not in total_returns:
                        total_returns[state] = np.zeros(env.action_space.n)
                    if state not in N:
                        N[state] = np.zeros(env.action_space.n)
                    for _state,_action in visited:
                        total_returns[_state][_action] += reward
                        N[_state][_action] += 1
                        V[_state][_action] = (total_returns[_state][_action] / N[_state][_action])
                    
                if env.unwrapped.is_finished():
                    finished += 1
                    finished_history.append(visited)
                    #V[state][action] = (total_returns[state][action] / N[state][action])
                state = env.unwrapped.serialize_state()
            #pbar.set_description(f'Training MC, finished={finished};Max reward gained={round(max_reward,3)}')
            if episode % 1000 == 0:
                result = {'V':V, 'N':N, 'total_returns':total_returns}
                fname = 'Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing'+'/'+process_name+' MC_'+('every' if every_visit_mc else 'first')+'_'+str(episodes)+'_episodes_temp.bin'
                with open(fname, 'wb') as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                replayf = 'Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing'+'/'+process_name+' MC_'+('every' if every_visit_mc else 'first')+'_'+str(episodes)+'_replay_temp.bin'
                with open(replayf, 'wb') as handle:
                    pickle.dump(finished_history, handle, protocol=pickle.HIGHEST_PROTOCOL)


        result = {'V':V, 'N':N, 'total_returns':total_returns}            
        #save the value function
        fname = 'Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing'+'/'+process_name+' MC_'+('every' if every_visit_mc else 'first')+'_'+str(episodes)+'_episodes.bin'
        with open(fname, 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        replayf = 'Results/Chapter '+str(chapter)+'/Level '+str(level)+'/MultiProcessing'+'/'+process_name+' MC_'+('every' if every_visit_mc else 'first')+'_'+str(episodes)+'_replay.bin'
        with open(replayf, 'wb') as handle:
            pickle.dump(finished_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        queue.put((V,N,total_returns))

        
def map( x,  in_min,  in_max,  out_min,  out_max) :
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

def greedy_policy(env,V,s,eps_comp=1e-8):
    #.9 prob of greedy action
    #.1 prob of random action

    if s not in V:
        V[s] = np.zeros(env.action_space.n)
    r_choice = 1 -map(eps_comp,0,1,0.1,.6)
    if np.random.random() < r_choice:
        return np.random.choice(np.arange(env.action_space.n))
    else:
        max_val = np.max(V[s])
         #find all actions that have the max value and choose one at random
        max_actions = np.argwhere(V[s] == max_val).flatten()
        return np.random.choice(max_actions)
def main():
    
    
    #mp.set_start_method('force')
    nprocesses = mp.cpu_count() - 4
    print("Using {} processes".format(nprocesses))
    #divide episodes by number of processes
    episodes = EPISODES // nprocesses
    reporter = find_reporter()
    queue = mp.Queue()
    processes = []
    for i in range(nprocesses):
        p = mp.Process(target=process, args=("CPU {}".format(i),reporter,queue,False,episodes))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    results = []
    while not queue.empty():
        results.append(queue.get())
    V = results[0][0].copy()
    N = results[0][1].copy()
    total_returns = results[0][2].copy()
    for i in range(1,len(results)):
        for state in results[i][0]:
            if state not in V:
                V[state] = np.zeros(8)
            V[state] += results[i][0][state]
        for state in results[i][1]:
            if state not in N:
                N[state] = np.zeros(8)
            N[state] += results[i][1][state]
        for state in results[i][2]:
            if state not in total_returns:
                total_returns[state] = np.zeros(8)
            total_returns[state] += results[i][2][state]
    flush()
    result = {'V':V, 'N':N, 'total_returns':total_returns}
    with open(f'value_functions/Result.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()

