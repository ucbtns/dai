
"""
@author: Noor Sajid

Stochastic environment 
"""

import os
os.chdir('D:/PhD/Code/bayes')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stochastic as st
import deterministic as dt
import utils as ut

odd = list(range(20)) + list(range(120,140)) + list(range(250,450))
simulator = False
det = False
sto = True

if simulator:
    if sto:
        st.stochastic(odd, '1703')
    if det:
        dt.deterministic('1703')


if sto:
    name = 'stochastic'
    br = 'brl_tr_online.npy'
    ql = 'ql_tr_online.npy'
    ql_no = 'ql_tr_online_no.npy'
    ai = 'trwp.csv'
    ai_no = 'trwop.csv'
if det:
    name = 'deterministic'
    br = 'brl_tr_det_online.npy'
    ql = 'ql_tr_det_online.npy'
    ql_no = 'ql_tr_det_online_no.npy'
    ai = 'trwp_det.csv'
    ai_no = 'trwop_Det.csv'
   

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


ut.plot_subresults(name, qlno, ql, brl, trwp, trwop, sto)


