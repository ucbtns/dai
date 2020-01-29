
"""
@author: Noor Sajid

Stochastic environment 
"""

import os
os.chdir('D:/PhD/Code/bayes')
import numpy as np
import pandas as pd
import stochastic as st
import deterministic as dt
import utils as ut

odd = list(range(20)) + list(range(120,140)) + list(range(250,450))
simulator = True
det = True
sto = True

num = ''
if simulator:
    if sto:
        st.stochastic(odd, num)
    if det:
        dt.deterministic(num)


if sto:
    name = 'stochastic'
    br = 'brl_tr_online' + num + '.npy'
    ql = 'ql_tr_online' + num + '.npy'
    ql_no = 'ql_tr_online_no' + num + '.npy'
    ai = 'trwp.csv'
    ai_no = 'trwop.csv'
if det:
    name = 'deterministic'
    br = 'brl_tr_det_online' + num + '.npy'
    ql = 'ql_tr_det_online' + num + '.npy'
    ql_no = 'ql_tr_det_online_no' + num + '.npy'
    ai = 'trwp_det.csv'
    ai_no = 'trwop_det.csv'
    
    

BRL_tr_online = np.load(br) 
QL_tr_online = np.load(ql) 
QL_tr_online_no = np.load(ql_no) 
trwp = pd.read_csv(ai, header=None)
trwp_rewards = np.array(np.where(trwp == 1, 100, 0))    
trwop = pd.read_csv(ai_no, header=None)
trwop_rewards = np.where(trwop == 1, 100, 0)

# Average:
ci = np.zeros([5,3])
ql = np.mean(QL_tr_online, axis = 1) 
ci[0,0], ci[0,1], ci[0,2] =ut.mean_confidence_interval(ql)
brl = np.mean(BRL_tr_online, axis = 1) 
ci[1,0], ci[1,1], ci[1,2] =ut.mean_confidence_interval(brl)
qlno = np.mean(QL_tr_online_no, axis = 1) 
ci[2,0], ci[2,1], ci[2,2] =ut.mean_confidence_interval(qlno)
trwp = np.mean(trwp_rewards, axis = 1)
ci[3,0], ci[3,1], ci[3,2] =ut.mean_confidence_interval(trwp)
trwop = np.mean(trwop_rewards, axis = 1)
ci[4,0], ci[4,1], ci[4,2] =ut.mean_confidence_interval(trwop)


if det:
    ut.plot_subresultsci('deterministic_plot', QL_tr_online_no, QL_tr_online, BRL_tr_online, trwp_rewards, trwop_rewards, 500, False)
if sto:
    ut.plot_subresultsci('stochastic_plot', QL_tr_online_no, QL_tr_online, BRL_tr_online, trwp_rewards, trwop_rewards, 500, True)
    

