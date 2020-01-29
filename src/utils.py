
"""
@authors: Noor Sajid & Philip Ball
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
import scipy

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
        
        r = np.where(new_state==7, 100,0)
        
        if render:
            env.render()
            
        if done:            
            return r, step+1,  
                    
        else:  
            state = new_state    
    
    return r, step+1
            
        
def environment_update(one,two, odd, episode):   
         
    # Alternating between goal locations:
    if episode in odd:
        env =  gym.make(one) # goal location 1
    else:         
        env =  gym.make(two) # goal location 2

    return env

    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_subresultsci(name, qlno, ql,brl,trwp, trwop, sample=500, sto = True):
    #sample = 500
    
    brl_rewards_means = np.mean(brl, axis=1)
    brl_rewards_stddev = np.std(brl, axis=1)

    qlno_rewards_means = np.mean(qlno, axis=1)
    qlno_rewards_stddev = np.std(qlno, axis=1)

    ql_rewards_means = np.mean(ql, axis=1)
    ql_rewards_stddev = np.std(ql, axis=1)

    trwp_rewards_means = np.mean(trwp, axis=1)
    trwp_rewards_stddev = np.std(trwp, axis=1)

    trwop_rewards_means = np.mean(trwop, axis=1)
    trwop_rewards_stddev = np.std(trwop, axis=1)

    plt.rcParams.update({'font.size': 18})    
    fig,a =  plt.subplots(5,1,sharex=True, sharey=True, figsize=(15,15))   
    a[0].plot(range(sample-1),qlno_rewards_means[:-1], label='Q-Learning', linewidth=2)
    a[0].fill_between(range(sample-1), qlno_rewards_means[:-1] + qlno_rewards_stddev[:-1], qlno_rewards_means[:-1] - qlno_rewards_stddev[:-1], alpha=0.3)
    a[0].set(title='Q-Learning ($\epsilon = 0.10$)', ylabel='Average Reward')
    a[1].plot(range(sample-1),ql_rewards_means[:-1], label='Q-Learning (exploration)', linewidth=2)
    a[1].fill_between(range(sample-1), ql_rewards_means[:-1] + ql_rewards_stddev[:-1], ql_rewards_means[:-1] - ql_rewards_stddev[:-1], alpha=0.3)
    a[1].set(title ='Q-Learning ($\epsilon = 1.00$, decaying to 0.00)', ylabel='Average Reward')
    a[2].plot(range(sample-1),brl_rewards_means[:-1], label='Bayesian RL', linewidth=2)
    a[2].fill_between(range(sample-1), brl_rewards_means[:-1] + brl_rewards_stddev[:-1], brl_rewards_means[:-1] - brl_rewards_stddev[:-1], alpha=0.3)
    a[2].set(title ='Bayesian RL', ylabel='Average Reward')
    a[3].plot(range(sample-1),trwp_rewards_means[:-1], label='Active Inference', linewidth=2)
    a[3].fill_between(range(sample-1), trwp_rewards_means[:-1] + trwp_rewards_stddev[:-1], trwp_rewards_means[:-1] - trwp_rewards_stddev[:-1], alpha=0.3)
    a[3].set(title='Active Inference', ylabel='Average Reward')
    a[4].plot(range(sample-1),trwop_rewards_means[:-1], label='Active Inference (null model)', linewidth=2)
    a[4].fill_between(range(sample-1), trwop_rewards_means[:-1] + trwop_rewards_stddev[:-1], trwop_rewards_means[:-1] - trwop_rewards_stddev[:-1], alpha=0.3)
    a[4].set(title='Active Inference (null model)', xlabel='Episodes', ylabel='Average Reward')
    
    if sto:
        for i in range(5):
            a[i].axvline(20, linestyle='--', color='grey')
            a[i].axvline(120, linestyle='--', color='grey')
            a[i].axvline(140, linestyle='--', color='grey')
            a[i].axvline(250, linestyle='--', color='grey')
            a[i].axvline(450, linestyle='--', color='grey')
    
    fig.tight_layout()
    fig.savefig(str(name) + '.png', dpi=500)


    