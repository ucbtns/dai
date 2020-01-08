
"""
@author: Noor Sajid
"""

import numpy as np

class MDP:

    def __init__(self,T,R,discount):
        
        '''Markov decision process: '''

        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        self.T = T
        self.R = R
        self.discount = discount       
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        
    def valueiteration(self,iv,ni=np.inf,tolerance=0.01):
        '''
                Value iteration procedure
                V <-- max_a R^a + gamma T^a V
        '''
        
        V = iv
        iterId = 0
        while iterId < ni:
            Q = self.R + self.discount*np.dot(self.T,V)
            newV = Q.max(0)
            epsilon = np.linalg.norm(newV - V, np.inf)
            V = newV
            iterId += 1
            if epsilon <= tolerance: 
                break
        
        return self.R + self.discount*np.dot(self.T,V)
