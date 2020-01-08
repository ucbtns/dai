"""
@author: Noor Sajid
"""

import numpy as np
import mdp

def priors(theta_transition, theta_reward, env, discount): 
    
        ''' 
            9 states in total and 4 actions:
            LEFT = 0
            DOWN = 1
            RIGHT = 2
            UP = 3
            
            4 types of states: start, goal (+100), frozen and hole        
             # 'FrozenLakeEnv-v1'
     
               "SFF",
               "FFG",
               "FHF"   
                
            # 'FrozenLakeEnv-v2'   
    
                "SFF",
                "FFH",
                "FGF"    
        '''
    
        action_space_size = env.action_space.n
        state_space_size = env.observation_space.n
        ass = np.zeros([action_space_size, state_space_size, state_space_size]) # Transition function [|A| x |S| x |S'|] 
        
        # Using beta function to formulate the structure of the model based on priors:
        # Splitting each action into three types of moves: intended; lateral; others
        
        # hyper-parametres for beta distribution as the conjugate prior:
        a_t = theta_transition;        
        b_t = (1-theta_transition)/2  
            
        P = env.P
        states = list(range(state_space_size))
        for j in states:
            sa = P[j]
            for i in list(range(action_space_size)):
                    state_next = sa[i][0][1]
                    ass[i][j][state_next] = a_t      # intended move
                    if (state_next == 0) or (state_next == 8):
                        ass[i][j][state_next] = ass[i][j][state_next] + b_t     # intended move
                    if state_next <=7:
                        ass[i][j][state_next+1] = b_t  # lateral move
                    if state_next >=1:
                        ass[i][j][state_next-1] = b_t # lateral move      
                    if state_next == 4:                    
                          ass[i][j] = np.where(ass[i][j] !=0,b_t, 0)
                          ass[i][j][state_next] = a_t # lateral move  
                    if state_next == j:
                        if (state_next==7) or (state_next==5):
                            ass[i][j] = 0
                            ass[i][j][state_next] = 1-(1e-5)      # intended move
        ass[i][7] = 0  
        ass[i][7][7] = 1-(1e-5)
    
        ass[i][5] = 0  
        ass[i][5][5] = 1-(1e-5)              
        
          # reward function: |A| x |S|:
        r = np.ones([action_space_size, state_space_size])*10
            
        # We account uncertainty for the reward:
        a_r = theta_reward
        b_r = (1-theta_reward)/2 
        #b_r = (1-theta_reward)
        
        r[:,5] = 100*(a_r) # belief that this is the true reward location
        r[:,7] = 100*(b_r) # belief that this is the wrong reward location
         
        mdp_env = mdp.MDP(ass,r,discount)
        return mdp_env


def update_prior_transition(a,b,action,state,nextstate, learner):
    
        # treat hyperparametres as pseudo-observations: 
        
        update = learner*1
        if action == 0: # left
            # stay in the same place:
            if (state == 0 and nextstate==0) or (state ==3 and nextstate==3) or (state ==6 and nextstate==6):
                a = a + update
            # move as expected: 
            elif state == nextstate + 1:
                a = a + update
            # random move
            else:
                b = b + update       
    
        if action == 1: # down
            # stay in the same place:
            if (state == 6 and nextstate==6) or (state ==7 and nextstate==7) or (state ==8 and nextstate==8):
                a = a + update
            # move as expected:
            elif state == nextstate-3:
                a = a + update
            # random move
            else:
                b = b + update
                   
        if action == 2: # right
            # stay in the same place:
            if (state == 2 and nextstate==2) or (state ==5 and nextstate==5) or (state ==8 and nextstate==8):
                a = a + update
            # move as expected:
            elif state == nextstate - 1:
                a = a + update
            # random move
            else:
                b = b + update
        
        if action == 3: # up
            # stay in the same place:
            if (state == 0 and nextstate==0) or (state ==1 and nextstate==1) or (state ==2 and nextstate==2):
                a = a + update
            # move as expected:
            elif state == nextstate+3:
                a = a + update
            # random move
            else:
                b = b + update
                           
        return a,b


def update_prior_reward(a,b,reward,state):
    
         # treat hyperparametres as pseudo-observations: 
    
        update = 1
        if state == 5 and reward > 0:
                a = a + update # reward in location 5
               
        if state == 5 and reward == 0:
                b = b + update # reward not in location 5  
            
        if state == 7 and reward > 0: 
                b = b + update # reward in location 7
                
        if state == 7 and reward == 0: 
                a = a + update # reward not in location 7
        
        return a,b

