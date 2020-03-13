import numpy as np
import copy

class DeltaXGenerator():
    def __init__(self):
        pass
    
    
    def generate(self, prior, delta=1e-1, size=2500):
        
        temp_prior = copy.deepcopy(prior)
        
        while (np.sum(temp_prior) > 1-delta):

            temp_prior[temp_prior == 0] = float("inf")
            min_ind = np.argmin(temp_prior)
            min_prob = prior[min_ind]
            temp_prior[min_ind] = 0
            temp_prior[temp_prior == float("inf")] = 0
            
            if np.sum(temp_prior != 0) == 0:
                break
            
        if prior[min_ind] > 0:
            temp_prior[min_ind] = min_prob
            
        n_possible_loc = np.sum(temp_prior > 0)
        
        
        deltaX = np.zeros((n_possible_loc, size))
        
        state_nos = np.where(temp_prior>0)[0]
        for i, state_no in enumerate(state_nos):
            deltaX[i,state_no] = 1
            
        print("num possible location:", n_possible_loc, state_nos)
            
            
        return state_nos, deltaX