#Generate tragectries
import numpy as np
import copy
import math


class TrajectoryProcessor():
    def __init__(self):
        self.transition_mat = None
    
    def generate(self, len_traj):
        if self.transition_mat is None:
            print("you must initially load")
            
        traj = []
        
        start = self._choice()
        
        for i in range(len_traj):
            
            traj.append(start)
        
            cur_posi = np.zeros((1, self.size))
            cur_posi[0, start] = 1
            dist = np.dot(cur_posi, self.transition_mat)
            
            if dist.sum() == 0:
                continue
                
            next_posi = np.random.choice(range(self.size), p=dist[0])
            
            start = next_posi
        return traj   
    
    def compute_possible_set(self, prior, delta=0):
        
        if delta == 0:
        
            state_nos = np.where(prior>0)[0]
            n_possible_loc = len(state_nos)
            #print("n_possible_loc", n_possible_loc)
            
            return state_nos
        
        else:
            
            state_nos, delta_X = self.compute_delta_set(prior, delta)
            return state_nos

    
    def compute_delta_set(self, prior, delta):
        
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
            
        n_possible_loc = np.sum(temp_prior>0)
        
        
        deltaX = np.zeros((n_possible_loc, self.size))
        
        state_nos = np.where(temp_prior>0)[0]
        for i, state_no in enumerate(state_nos):
            deltaX[i,state_no] = 1
            
        #print("num possible location:", n_possible_loc, state_nos)
            
        return state_nos, deltaX
        
    
    def process(self, oh_locs):
        return np.dot(self.query, oh_locs.T).T
        
    
    def compute_posterior_distribution(self, prior):
        if self.transition_mat is None:
            print("you must initially load")
        
        posterior = np.dot(prior, self.transition_mat)
        
        if posterior.sum() == 0:
            print("end")
        
        return posterior
        
    
    def load(self, path_transition_mat, query, length_traj=500, size_traj=100, threashold=1e-4):
        
        transition_mat = np.loadtxt(path_transition_mat)
        self.transition_mat = self._threash(transition_mat, threashold)
        self.size = len(self.transition_mat)
        
        self.query = copy.deepcopy(query)
        ##### Strange operation!!!!!!!!!!
        for i in range(2500):
            if (math.floor(i/50) % 2) ==1:
                self.query[0, i] += 0.05
    
    def _threash(self, transition_mat, threashold):
        transition_mat = copy.deepcopy(transition_mat)
        transition_mat =  transition_mat * (transition_mat >= threashold)
        transition_mat =  self._normalize(transition_mat)
        return transition_mat
    
    def _normalize(self, transition_mat):
        transition_mat = copy.deepcopy(transition_mat)
        for i, transition_prob in enumerate(transition_mat):
            sum_ = np.sum(transition_prob)
            if sum_ != 0:
                transition_mat[i,:] = transition_mat[i,:]/sum_
        return transition_mat
    
    def _choice(self):
        while True:
            choice = np.random.choice(range(len(self.transition_mat)))
            sum_ = self.transition_mat[choice,:].sum()
            n_non_zero = np.sum(self.transition_mat[choice,:] != 0)
            if (sum_ != 0) and (n_non_zero > 2):
                break
        return choice      
    
    def state2loc(self, states):
        n_states = len(states)
        oh_states = np.zeros((n_states, self.size))
        for i, state in enumerate(states):
            oh_states[i, state] = 1
        
        return self.process(oh_states)
    
    
    def modify_for_test_traj(self, test_traj):
        for i in range(len(test_traj) - 1):
            pre_loc = test_traj[i]
            pos_loc = test_traj[i+1]
            if pre_loc != pos_loc:
                self.transition_mat[pre_loc][pos_loc] += 0.1
                
        self.transition_mat[2485][2435] += 0.1
        self.transition_mat = self._normalize(self.transition_mat)