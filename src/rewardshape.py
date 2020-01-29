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
# How to implement? Directly from the environment function in Frozen lake & env.py in bayesian RL 
 
# Change to 0,0,0
dt.deterministic('no_reward_updated4',number_trials=100, num_episodes=100,)

# Negative reward for living: change to 100,0,-10
dt.deterministic('_neg_ll_updated4',number_trials=100, num_episodes=100,)

# Negative reward for living & hole: change to 100,-100,-10
dt.deterministic('_neg_llh_updated4',number_trials=100, num_episodes=100,)

# Negative reward for hole: change to 100,-100,0
dt.deterministic('_neg_h_updated4',number_trials=100, num_episodes=100,)

# Negative reward for hole: change to 0, -100, 0
dt.deterministic('_neg_ph_updated4',number_trials=100, num_episodes=100,)

# Baseline: change to 100,0,0
dt.deterministic('_neg_pll_updated4',number_trials=100, num_episodes=100,)

 
# On-policy results
one = np.load('ql_tr_det_nono_reward_updated4.npy') 
onets = np.load('ql_ts_det_nono_reward_updated4.npy') 

two = np.load('ql_tr_det_no_neg_ll_updated4.npy') 
twots = np.load('ql_ts_det_no_neg_ll_updated4.npy') 

three = np.load('ql_tr_det_no_neg_llh_updated4.npy')
threets = np.load('ql_ts_det_no_neg_llh_updated4.npy')

four = np.load('ql_tr_det_no_neg_h_updated4.npy') 
fourts = np.load('ql_ts_det_no_neg_h_updated4.npy') 

five = np.load('ql_tr_det_no_neg_ph_updated4.npy')
fivets = np.load('ql_ts_det_no_neg_ph_updated4.npy')
 
six = np.load('ql_tr_det_no_neg_pll_updated4.npy') 
sixts = np.load('ql_ts_det_no_neg_pll_updated4.npy') 

print('0,0,0', np.mean(np.mean(one, axis = 1) ))
print('100, 0,-10',np.mean(np.mean(two, axis = 1) ))
print('100,-100,-10', np.mean(np.mean(three, axis = 1)))
print('100,-100,0', np.mean(np.mean(four, axis = 1)))
print('0,-100,0', np.mean(np.mean(five, axis = 1)))
print('Baseline', np.mean(np.mean(six, axis = 1)))

print('0,0,0', np.mean(np.mean(onets, axis = 1) ))
print('100, 0,-10',np.mean(np.mean(twots, axis = 1) ))
print('100,-100,-10', np.mean(np.mean(threets, axis = 1)))
print('100,-100,0', np.mean(np.mean(fourts, axis = 1)))
print('0,-100,0', np.mean(np.mean(fivets, axis = 1)))
print('Baseline', np.mean(np.mean(sixts, axis = 1)))

md2 = '_ph'
trwp = pd.read_csv('trwop_priors_score'+ md2 + '_modified2.csv', header=None)
trwp_rewards = np.array(np.where(trwp == 1, 0, np.where(trwp == 3, -.1,0))) 
  



