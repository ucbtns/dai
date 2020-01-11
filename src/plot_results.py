import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_subresults(name, qlno, ql,brl,trwp, trwop, sample=500, sto = True):
    sample = 500
    
    brl_rewards_means = np.mean(brl_rewards, axis=1)
    brl_rewards_stddev = np.std(brl_rewards, axis=1)

    qlno_rewards_means = np.mean(qlno_rewards, axis=1)
    qlno_rewards_stddev = np.std(qlno_rewards, axis=1)

    ql_rewards_means = np.mean(ql_rewards, axis=1)
    ql_rewards_stddev = np.std(ql_rewards, axis=1)

    trwp_rewards_means = np.mean(trwp_rewards, axis=1)
    trwp_rewards_stddev = np.std(trwp_rewards, axis=1)

    trwop_rewards_means = np.mean(trwop_rewards, axis=1)
    trwop_rewards_stddev = np.std(trwop_rewards, axis=1)

    plt.rcParams.update({'font.size': 18})    
    fig,a =  plt.subplots(5,1,sharex=True, sharey=True, figsize=(15,15))   
    a[0].plot(range(sample-1),qlno_rewards_means[:-1], label='Q-Learning', linewidth=2)
    a[0].fill_between(range(sample-1), qlno_rewards_means[:-1] + qlno_rewards_stddev[:-1], qlno_rewards_means[:-1] - qlno_rewards_stddev[:-1], alpha=0.3)
    a[0].set(title='Q-Learning ($\epsilon = 0.10$, decaying to 0.01)', ylabel='Average Reward')
    a[1].plot(range(sample-1),ql_rewards_means[:-1], label='Q-Learning (exploration)', linewidth=2)
    a[1].fill_between(range(sample-1), ql_rewards_means[:-1] + ql_rewards_stddev[:-1], ql_rewards_means[:-1] - ql_rewards_stddev[:-1], alpha=0.3)
    a[1].set(title ='Q-Learning ($\epsilon = 1.00$, decaying to 0.01)', ylabel='Average Reward')
    a[2].plot(range(sample-1),brl_rewards_means[:-1], label='Bayesian RL', linewidth=2)
    a[2].fill_between(range(sample-1), brl_rewards_means[:-1] + brl_rewards_stddev[:-1], brl_rewards_means[:-1] - brl_rewards_stddev[:-1], alpha=0.3)
    a[2].set(title ='Bayesian RL', ylabel='Average Reward')
    a[3].plot(range(sample-1),trwp_rewards_means[:-1], label='Active Inference (preferences)', linewidth=2)
    a[3].fill_between(range(sample-1), trwp_rewards_means[:-1] + trwp_rewards_stddev[:-1], trwp_rewards_means[:-1] - trwp_rewards_stddev[:-1], alpha=0.3)
    a[3].set(title='Active Inference with preferences', ylabel='Average Reward')
    a[4].plot(range(sample-1),trwop_rewards_means[:-1], label='Active Inference', linewidth=2)
    a[4].fill_between(range(sample-1), trwop_rewards_means[:-1] + trwop_rewards_stddev[:-1], trwop_rewards_means[:-1] - trwop_rewards_stddev[:-1], alpha=0.3)
    a[4].set(title='Active Inference', xlabel='Episodes', ylabel='Average Reward')
    
    if sto:
        for i in range(5):
            a[i].axvline(20, linestyle='--', color='grey')
            a[i].axvline(120, linestyle='--', color='grey')
            a[i].axvline(140, linestyle='--', color='grey')
            a[i].axvline(250, linestyle='--', color='grey')
            a[i].axvline(450, linestyle='--', color='grey')
    
    fig.tight_layout()
    fig.savefig(str(name) + '.png', dpi=500)

### Deterministic

# load RL data
brl_rewards = np.load('results/deterministic/brl_tr_det_online.npy')
qlno_rewards = np.load('results/deterministic/ql_tr_det_online_no.npy')
ql_rewards = np.load('results/deterministic/ql_tr_det_online.npy')

brl_rewards_means = np.mean(brl_rewards, axis=1)
brl_rewards_stddev = np.std(brl_rewards, axis=1)

qlno_rewards_means = np.mean(qlno_rewards, axis=1)
qlno_rewards_stddev = np.std(qlno_rewards, axis=1)

ql_rewards_means = np.mean(ql_rewards, axis=1)
ql_rewards_stddev = np.std(ql_rewards, axis=1)

# active infenrece:
trwp = pd.read_csv("results/deterministic/trwp_det.csv", header=None)
trwp_rewards = np.array(np.where(trwp == 1, 100, 0))
trwp_rewards_means = np.mean(trwp_rewards, axis=1)
trwp_rewards_stddev = np.std(trwp_rewards, axis=1)

trwop = pd.read_csv('results/deterministic/trwop_det.csv', header=None)
trwop_rewards = np.where(trwop == 1, 100, 0)
trwop_rewards_means = np.mean(trwop_rewards, axis=1)
trwop_rewards_stddev = np.std(trwop_rewards, axis=1)

# with plt.style.context('seaborn-whitegrid'):
#     # plt.figure(figsize=(12,9))
#     plt.plot(range(499),brl_rewards_means[:-1], label='Bayesian RL',linewidth=2)
#     plt.plot(range(499),qlno_rewards_means[:-1], label='Q-Learning (exploration)', linewidth=2)
#     plt.plot(range(499),ql_rewards_means[:-1], label='Q-Learning with exploration', linewidth=2)
#     plt.plot(range(499),trwp_rewards_means[:-1], label='Active Inference (preferences)', linewidth=2)
#     plt.plot(range(499),trwop_rewards_means[:-1], label='Active Inference', linewidth=2)
    
#     plt.grid()
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.title("Deterministic Environment Performance")
#     plt.xlabel("Episode")
#     plt.ylabel("Average Reward")
# plt.show()

plot_subresults('deterministic_plots', qlno_rewards, ql_rewards, brl_rewards, trwp_rewards, trwop_rewards, 500, False)

### Stochastic

# load RL data
brl_rewards = np.load('results/stochastic/brl_tr_online.npy')
qlno_rewards = np.load('results/stochastic/ql_tr_online_no.npy')
ql_rewards = np.load('results/stochastic/ql_tr_online.npy')

brl_rewards_means = np.mean(brl_rewards, axis=1)
brl_rewards_stddev = np.std(brl_rewards, axis=1)

qlno_rewards_means = np.mean(qlno_rewards, axis=1)
qlno_rewards_stddev = np.std(qlno_rewards, axis=1)

ql_rewards_means = np.mean(ql_rewards, axis=1)
ql_rewards_stddev = np.std(ql_rewards, axis=1)

# active infenrece:
trwp = pd.read_csv("results/stochastic/trwp.csv", header=None)
trwp_rewards = np.array(np.where(trwp == 1, 100, 0))
trwp_rewards_means = np.mean(trwp_rewards, axis=1)
trwp_rewards_stddev = np.std(trwp_rewards, axis=1)

trwop = pd.read_csv('results/stochastic/trwop.csv', header=None)
trwop_rewards = np.where(trwop == 1, 100, 0)
trwop_rewards_means = np.mean(trwop_rewards, axis=1)
trwop_rewards_stddev = np.std(trwop_rewards, axis=1)

plot_subresults('stochastic_plots', qlno_rewards, ql_rewards, brl_rewards, trwp_rewards, trwop_rewards, 500, True)

# plt.rcParams.update({'font.size': 18})
# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(12,10))
    
#     plt.axvline(20, linestyle='--', color='black')
#     plt.axvline(120, linestyle='--', color='black')    
#     plt.axvline(140, linestyle='--', color='black')
#     plt.axvline(250, linestyle='--', color='black')
#     plt.axvline(450, linestyle='--', color='black')
    
#     plt.plot(range(499),qlno_rewards_means[:-1], label='Q-Learning', linewidth=2)
#     plt.plot(range(499),ql_rewards_means[:-1], label='Q-Learning (exploration)', linewidth=2)   
#     plt.plot(range(499),brl_rewards_means[:-1], label='Bayesian RL',linewidth=2)
#     plt.plot(range(499),trwp_rewards_means[:-1], label='Active Inference (preferences)', linewidth=2)
#     plt.plot(range(499),trwop_rewards_means[:-1], label='Active Inference', linewidth=2)    
    
#     plt.grid()
#     plt.legend(loc='best', bbox_to_anchor=(1, 0.5), fontsize=15)
#     #plt.title("Performance")
#     plt.xlabel("Episode")
#     plt.ylabel("Average Reward")
#     plt.xlim(0, 500)
#     plt.ylim(0,101)
# plt.show()

