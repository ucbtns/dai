
"""
@author: vilmarith

Stochastic environment 
"""

import os
os.chdir('D:/PhD/Code/bayes')
import numpy as np
from brl import BRLAgent
from ql import QLAgent
import matplotlib.pyplot as plt
import pandas as pd


# Inital parameters:
number_trials = 200
num_episodes = 500
num_iterations = 100
max_steps_per_episode = 100
odd = list(range(20)) + list(range(120,140)) + list(range(250,450))

# Envirnment details:
env1 = 'FrozenLakeEnv-v1'
env2 = 'FrozenLakeEnv-v2'

BRL_tr = np.zeros((num_episodes, number_trials))
BRL_ts = np.zeros((num_episodes, number_trials))
BRL_tr_online = np.zeros((num_episodes, number_trials))
BRL_ts_online = np.zeros((num_episodes, number_trials))

# Bayesian agent (using thompson sampling):
for trial in range(number_trials):    
    brlagent = BRLAgent(num_episodes,num_iterations, max_steps_per_episode, env1, env2,odd, a_t=1, b_t=1, a_r=1, b_r=1, k=2) 
    BRL_tr[:,trial], BRL_ts[:,trial], QBR, BRL_tr_online[:,trial], BRL_ts_online[:,trial]  = brlagent.simulator()

brl_rewards_means = np.mean(BRL_tr_online, axis = 1)
#br = BRL_tr_online.flatten(order='F')

np.save('brl_tr_online.npy', BRL_tr_online) 
np.save('brl_ts_online.npy', BRL_ts_online) 
       
# Q-Learning agent:
QL_tr = np.zeros((num_episodes, number_trials))
QL_ts = np.zeros((num_episodes, number_trials))
QL_tr_online = np.zeros((num_episodes, number_trials))
QL_ts_online = np.zeros((num_episodes, number_trials))

alpha = 0.5
gamma = 0.99
epsilon = 1
max_er = 1
min_er = 0.01
exploration_decay_rate = 0.001


for trial in range(number_trials):    
    qlagent = QLAgent(num_episodes, max_steps_per_episode, env1, env2,odd, gamma,
                      alpha,epsilon, max_er, min_er, exploration_decay_rate)    
    QL_tr[:,trial], QL_ts[:,trial], QQ , QL_tr_online[:,trial], QL_ts_online[:,trial] = qlagent.simulator()

ql_rewards_means = np.mean(QL_tr_online, axis = 1) 

np.save('ql_tr_online.npy', QL_tr_online) 
np.save('ql_ts_online.npy', QL_ts_online) 


# Q-Learning agent:
QL_tr_no = np.zeros((num_episodes, number_trials))
QL_ts_no = np.zeros((num_episodes, number_trials))
QL_tr_online_no = np.zeros((num_episodes, number_trials))
QL_ts_online_no = np.zeros((num_episodes, number_trials))

alpha = 0.5
gamma = 0.99
epsilon = 0.3
max_er = 1
min_er = 0.01
exploration_decay_rate = 0.001


for trial in range(number_trials):    
    qlagent = QLAgent(num_episodes, max_steps_per_episode, env1, env2,odd, gamma,
                      alpha,epsilon, max_er, min_er, exploration_decay_rate)    
    QL_tr_no[:,trial], QL_ts_no[:,trial], QQ , QL_tr_online_no[:,trial], QL_ts_online_no[:,trial] = qlagent.simulator()

qlno_rewards_means = np.mean(QL_tr_online_no, axis = 1) 

np.save('ql_tr_online_no.npy', QL_tr_online_no) 
np.save('ql_ts_online_no.npy', QL_ts_online_no) 

# active infenrece:
trwp = pd.read_csv('trwp.csv', header=None)
trwp_rewards = np.array(np.where(trwp == 1, 100, 0))
trwp_rewards_means = np.mean(trwp_rewards, axis = 1)


trwop = pd.read_csv('trwop.csv', header=None)
trwop_rewards = np.where(trwop == 1, 100, 0)
trwop_rewards_means = np.mean(trwop_rewards, axis = 1)


plt.rcParams.update({'font.size': 18})
with plt.style.context('seaborn-whitegride'):
    plt.figure(figsize=(12,10))
    
    plt.axvline(20, linestyle='--', color='black')
    plt.axvline(120, linestyle='--', color='black')    
    plt.axvline(140, linestyle='--', color='black')
    plt.axvline(250, linestyle='--', color='black')
    plt.axvline(450, linestyle='--', color='black')
    
    plt.plot(range(499),qlno_rewards_means[:-1], label='Q-Learning', linewidth=2)
    plt.plot(range(499),ql_rewards_means[:-1], label='Q-Learning (exploration)', linewidth=2)   
    plt.plot(range(499),brl_rewards_means[:-1], label='Bayesian RL',linewidth=2)
    plt.plot(range(499),trwp_rewards_means[:-1], label='Active Inference (preferences)', linewidth=2)
    plt.plot(range(499),trwop_rewards_means[:-1], label='Active Inference', linewidth=2)    
    
    plt.grid()
    plt.legend(loc='best', bbox_to_anchor=(1, 0.5), fontsize=15)
    #plt.title("Performance")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.xlim(0, 500)
    plt.ylim(0,101)
plt.show()





