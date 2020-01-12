
"""
# Standard Q-Learning with exploration-exploitation trade-off
using Semi-Uniform random exploration:
      
    Off-policy reinforcement learning algorithm 
    that seeks to find the best action to take 
    given the current state. Off-policy learns 
    from actions that are outside the current 
    policy, like taking random actions, and 
    therefore a policy isn’t needed.
    
  
Updating Q-values based on the following: 
         Q(s,a) = (1-alpha)*Q(s,a) + (alpha*(R(s,a)+ gamma*V(s)))
         Note that V(s) == max Q(s,) - optimal discount reward achievable from that state
         Storing the best action for a given state: reference table for our agent

@author: Noor Sajid
"""

import numpy as np
import random
import utils as ut


class QLAgent():
    
     def __init__(self, num_episodes, max_steps_per_episode, env1, env2,odd, gamma, alpha, epsilon, maxer, miner, exploration_decay_rate,render=False):
            
            # Trial parametres:
            self.max_steps_per_episode = max_steps_per_episode 
            self.num_episodes = num_episodes
            self.odd = odd
            self.render = render
            
            # Environment set-up:
            self.env1 = env1
            self.env2 = env2       
            
            # Q-Learning hp:
            
            self.alpha = alpha  #learning rate
            self.gamma = gamma  #discount rate
            self.epsilon = epsilon #exploration rate
            self.maxer = maxer #1
            self.miner = miner #0.01
            self.exploration_decay_rate = exploration_decay_rate #0.001
            
            # Saving score:
            self.tr = np.zeros(self.num_episodes)
            self.ts = np.zeros(self.num_episodes)  
            
            self.tr_online = np.zeros(self.num_episodes)
            self.ts_online = np.zeros(self.num_episodes) 
            

     def simulator(self):
                
       
        # Acting based on Q-Learning
        env_shadow = ut.environment_update(self.env1, self.env2, [1], 1) 
        Q = np.zeros((env_shadow.action_space.n, env_shadow.observation_space.n))
                      
        for episode in range(self.num_episodes):
            
            self.env = ut.environment_update(self.env1, self.env2, self.odd, episode)              
            
            state = self.env.reset()
            done = False
            
            for step in range(self.max_steps_per_episode): 
                 
                  exploration_rate_threshold = random.uniform(0, 1)
                  
                  if exploration_rate_threshold > self.epsilon:
                      #action = np.argmax(Q[:, state]) # Exploitation: maximum action based on current Q-table values
                      action = ut.argmaxrand(Q[:, state]) 
                  else:
                      action = self.env.action_space.sample() # Exploration: random action
                  
                  #self.env.render()
                  new_state, reward, done, info = self.env.step(action) #act
                  
                  Q[action,state] = (1-self.alpha)*Q[action,state] + self.alpha * (reward + self.gamma * np.max(Q[:, new_state])) # update Q value
                         
                  state = new_state  # updating position in the environment
                  
                  if done: 
                      break
                  
            self.epsilon = self.miner + (self.maxer - self.miner) * np.exp(-self.exploration_decay_rate*episode)                   
            self.tr_online[episode], self.ts_online[episode]  = reward, step+1
                  
                    
            # Updating the exploration rate: 
              
            self.tr[episode], self.ts[episode] = ut.play_episode(Q, self.env, self.max_steps_per_episode, render = False)
            
        return self.tr, self.ts, Q, self.tr_online, self.ts_online
   
        

