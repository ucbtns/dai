
"""
# Modified partial online planning using Thompson sampling from Pascal Poupart's lecture on BRL
# Modelling uncertainty over both the transition model and reward function
            # Note: Transition model is tabular but can be replaced with function approximator 
            #       However, this would result in the distributions changing to normal - gamma i
@author: Noor Sajid
"""

from scipy.stats import beta
import numpy as np
from env import priors, update_prior_transition, update_prior_reward
import utils as ut
import random


class BRLAgent():
    
    def __init__(self, num_episodes, nt, max_steps_per_episode, env1, env2,odd, a_t=1, b_t=1, a_r=1, b_r=1, k=32, render=False):
            
            # Trial parametres:
            self.max_steps_per_episode = max_steps_per_episode 
            self.nt = nt
            self.num_episodes = num_episodes
            self.k = k
            self.odd = odd
            self.render = render
            
            # Environment set-up:
            self.env1 = env1
            self.env2 = env2       
            
            # Belief state hyperparameter:
            self.a_t, self.b_t, self.a_r, self.b_r = a_t, b_t, a_r, b_r
            self.belief_states = [(a_t,b_t,a_r,b_r)]   
    
            # Saving score:
            self.tr = np.zeros(self.num_episodes)
            self.ts = np.zeros(self.num_episodes)         
            
            self.tr_online = np.zeros(self.num_episodes)
            self.ts_online = np.zeros(self.num_episodes)         
        
        
            
    def planner(self): 
        
            # Planning bayesian optimal behaviour based on thompson sampling approximation
            
            self.state = self.env.reset()
            done = False
            
            for i in range(self.nt):
                    
                    # sample from k thetas from belief (i.e. beta) distribution for transition & reward model model:          
                    theta = zip(beta.rvs(self.a_t,self.b_t,size=self.k), beta.rvs(self.a_r, self.b_r, size=self.k))
                    
                    # create k MDP using the sampled thetas
                    MDPs = [priors(t_t, t_r, self.env, 0.9) for t_t, t_r in theta] 
                    
                    # solve the k MDPs using value iteration.
                    Q_functions = [m.valueiteration(np.zeros(self.env.observation_space.n)) for m in MDPs]        
                    
                    # average Q-value 
                    Q_hat = np.mean(Q_functions,axis=0)
                    
                    # get action to play via max(a) Q-hat(s,a):
                    # action = ut.argmaxrand(Q_hat[:,self.state])
                    
                    action = np.argmax(Q_hat[:,self.state])
                    # sample next state:            
                    new_state, reward, done, info = self.env.step(action) 
                    
                    # update priors:
                    self.a_t,self.b_t = update_prior_transition(self.a_t,self.b_t,action,self.state,new_state, 1)
                    self.a_r,self.b_r = update_prior_reward(self.a_r,self.b_r,reward,new_state)
                        
                    self.belief_states.append((self.a_t,self.b_t,self.a_r,self.b_r)) # storing the beta distribution hyper-parametres   
                    
                    if done:
                        break
                    
                    self.state = new_state           
        
            return Q_hat, self.belief_states, reward, i

    def simulator(self):
            
            # Acting bayes optimal
            
            for episode in range(self.num_episodes):
                random.seed(episode)
                
                self.env = ut.environment_update(self.env1, self.env2, self.odd, episode)    
                self.env.seed(episode)
                
                Q, self.belief_states, reward,i = self.planner()                     
              
                # Updating priors:    
                self.a_t = self.belief_states[-1][0]
                self.b_t = self.belief_states[-1][1]
                
                self.a_r = self.belief_states[-1][2]
                self.b_r = self.belief_states[-1][3]
                                        
                # Storing score from play:  
                self.tr_online[episode], self.ts_online[episode] = reward, i+1
                self.tr[episode], self.ts[episode]  = ut.play_episode(Q, self.env, self.max_steps_per_episode, self.render)
                
            return self.tr, self.ts, Q, self.tr_online, self.ts_online
