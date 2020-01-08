
"""
@author: Noor Sajid
"""

import numpy as np
import gym
import sys

def normalise(x):

    return (x - x.min()) / (np.ptp(x))

def argmaxrand(a):

    indices = np.where(np.array(a) == np.max(a))[0]
    return np.random.choice(indices)
    


def play_episode(Q, env, max_steps_per_episode, render = False):  
    
    state = env.reset()
    for step in range(max_steps_per_episode):  
        
        #action = argmaxrand(Q[:,state])     
        action = np.argmax(Q[:,state])  
        new_state, reward, done, info = env.step(action) 
        
        if render:
            env.render()
            
        if done:            
            return reward, step+1,  
                    
        else:  
            state = new_state
        
    return reward, step+1
            
        
def environment_update(one,two, odd, episode):   
         
    # Alternating between goal locations:
    if episode in odd:
        env =  gym.make(one) # goal location 1
    else:         
        env =  gym.make(two) # goal location 2

    return env


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "X"*x, "O"*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()