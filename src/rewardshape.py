# -*- coding: utf-8 -*-
"""
@author: Noor Sajid

Test out  multiple different reward functions and see how that shifts learning
for a fully deterministic environment:

"""


import os
os.chdir('D:/PhD/Code/bayes')
import numpy as np
import pandas as pd
import deterministic as dt



# No reward == [log(P(o)) == 0]: 
# How to implement? Directly from the environment function 
# line 132 & 133:
dt.deterministic('no_reward')



# Negative reward for living:
dt.deterministic('_neg_ll_smaller2', num_episodes=1000)

# Negative reward for living & hole:
dt.deterministic('_neg_llh_smaller')

# Negative reward for hole:
dt.deterministic('_neg_h_smaller')



BRL_tr_online = np.load('brl_tr_det_online_neg_h_smaller.npy') 
QL_tr_online = np.load('ql_tr_det_online_neg_h_smaller.npy') 
QL_tr_online_no = np.load('ql_tr_det_online_no_neg_h_smaller.npy') 
trwp = pd.read_csv('trwop_priors_score_h.csv', header=None)
trwp_rewards = np.array(np.where(trwp == 1, 1, np.where(trwp == 3, -.10,0))) 

print('QL:', np.mean(np.mean(QL_tr_online, axis = 1) ))
print('QL e=0.1',np.mean(np.mean(QL_tr_online_no, axis = 1) ))
print('BRL', np.mean(np.mean(BRL_tr_online, axis = 1)))
print('AI', np.mean(np.mean(trwp_rewards, axis = 1)))