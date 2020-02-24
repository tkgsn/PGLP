import numpy as np
import copy

class DeltaXGenerator():
    def __init__(self):
        pass
    
    
    def generate(self, prior, delta=1e-1, size=2500):
        
        temp_prior = copy.deepcopy(prior)
        
        #print("prior_sum", np.sum(prior > 0))
        
        print("prior_probs",-np.sort(-temp_prior)[0:5])
        
        while (np.sum(temp_prior) > 1-delta):
            if np.sum(temp_prior != 0) == 1:
                break
            temp_prior[temp_prior == 0] = float("inf")
            min_ind = np.argmin(temp_prior)
            min_prob = prior[min_ind]
            temp_prior[min_ind] = 0
            temp_prior[temp_prior == float("inf")] = 0
            
        if prior[min_ind] > 0:
            temp_prior[min_ind] = min_prob
            
        #print("post_sum", np.sum(temp_prior > 0))
        
        #print("prior_probs",-np.sort(-temp_prior)[0:5])
            
        n_possible_loc = np.sum(temp_prior > 0)
        
        print("num possible location:", n_possible_loc)
        
        deltaX = np.zeros((n_possible_loc, size))
        
        state_nos = np.where(temp_prior>0)[0]
        for i, state_no in enumerate(state_nos):
            deltaX[i,state_no] = 1
            
            
        return state_nos, deltaX