
"""
@author: Noor Sajid

runs and the code
"""

import numpy as np
from brl import BRLAgent
from ql import QLAgent


def stochastic(odd, num, number_trials=200, num_episodes=500, num_iterations=100, max_steps_per_episode=100,
               env1 = 'FrozenLakeEnv-v1', env2= 'FrozenLakeEnv-v2',
               alpha=0.5, gamma=0.99, max_er=1, min_er=0.01, exploration_decay_rate=0.001):
    
    # Bayesian agent (using thompson sampling):    
    BRL_tr = np.zeros((num_episodes, number_trials))
    BRL_ts = np.zeros((num_episodes, number_trials))
    BRL_tr_online = np.zeros((num_episodes, number_trials))
    BRL_ts_online = np.zeros((num_episodes, number_trials))
    
    # Q-Learning agent with exploration:
    QL_tr = np.zeros((num_episodes, number_trials))
    QL_ts = np.zeros((num_episodes, number_trials))
    QL_tr_online = np.zeros((num_episodes, number_trials))
    QL_ts_online = np.zeros((num_episodes, number_trials))
    
    # Q-Learning agent with no exploration:
    QL_tr_no = np.zeros((num_episodes, number_trials))
    QL_ts_no = np.zeros((num_episodes, number_trials))
    QL_tr_online_no = np.zeros((num_episodes, number_trials))
    QL_ts_online_no = np.zeros((num_episodes, number_trials))  
    
    
    for trial in range(number_trials):    
        
        print('Percent complete:', 100*(trial/number_trials))
        
        brlagent = BRLAgent(num_episodes,num_iterations, max_steps_per_episode, env1, env2,odd, a_t=1, b_t=1, a_r=1, b_r=1, k=2) 
        BRL_tr[:,trial], BRL_ts[:,trial], QBR, BRL_tr_online[:,trial], BRL_ts_online[:,trial]  = brlagent.simulator()
       
        qlagent = QLAgent(num_episodes, max_steps_per_episode, env1, env2,odd, gamma,
                          alpha,1, max_er, min_er, exploration_decay_rate)    
        QL_tr[:,trial], QL_ts[:,trial], QQ , QL_tr_online[:,trial], QL_ts_online[:,trial] = qlagent.simulator()
      
        qlagent = QLAgent(num_episodes, max_steps_per_episode, env1, env2,odd, gamma,
                          alpha,0.1, max_er, min_er, exploration_decay_rate)    
        QL_tr_no[:,trial], QL_ts_no[:,trial], QQ , QL_tr_online_no[:,trial], QL_ts_online_no[:,trial] = qlagent.simulator()
        
    
    np.save('brl_tr_online' + str(num) +'.npy', BRL_tr_online) 
    np.save('brl_ts_online' + str(num) +'.npy', BRL_ts_online) 
           
    np.save('ql_tr_online' + str(num) +'.npy', QL_tr_online) 
    np.save('ql_ts_online' + str(num) +'.npy', QL_ts_online) 
    
    np.save('ql_tr_online_no' + str(num) +'.npy', QL_tr_online_no) 
    np.save('ql_ts_online_no' + str(num) +'.npy', QL_ts_online_no) 
    
    np.save('brl_tr_det_' + str(num) +'.npy', BRL_tr) 
    np.save('ql_tr_det_' + str(num) +'.npy', QL_tr) 
    np.save('ql_tr_det_no' + str(num) +'.npy', QL_tr_no) 
    
    
    print('Simulation completed')
