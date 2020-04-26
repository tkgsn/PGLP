import numpy as np
import copy
import math
import map_processor


class TrajectoryProcessor(map_processor.MapProcessor):
    def __init__(self, n_x_lattice):
        super().__init__(n_x_lattice)
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
        
    def update_graph_mat(self, possible_states):
        updated_graph_mat = copy.deepcopy(self.graph_mat)
        for state in range(len(updated_graph_mat)):
            if state not in possible_states:
                updated_graph_mat[state,:] = 0
                updated_graph_mat[:, state] = 0
        
        return updated_graph_mat
    
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
        
    
    def compute_posterior_distribution(self, prior):
        if self.transition_mat is None:
            print("you must initially load")
        
        posterior = np.dot(prior, self.transition_mat)
        
        if posterior.sum() == 0:
            print("end")
        
        return posterior
    
    def traj_to_states(self, traj):
        state_traj = []
        for latlon in traj:
            if not self._is_in_from_latlon(latlon):
                continue
            state = self._find_nearest_state_from_latlon_in_all_states(latlon)
            state_traj.append(state)
        return state_traj
    
    def trajs_to_state_trajs(self, trajs):
        state_trajs = []
        for traj in trajs:
            state_traj = self.traj_to_states(traj)
            if len(state_traj) != 0:
                state_trajs.append(state_traj)
        return state_trajs
        
    def make_transmat_from_state_trajs(self, state_trajs):
        transition_mat = np.zeros((self.n_x_lattice * self.n_y_lattice, self.n_x_lattice * self.n_y_lattice))
        for state_traj in state_trajs:
            pre_state = state_traj[0]
            for state in state_traj[1:]:
                transition_mat[pre_state, state] += 1
                pre_state = state
        self.transition_mat = self._normalize(transition_mat)
        
    def make_transmat_from_trajs(self, trajs):
        transition_mat = np.zeros((self.n_x_lattice * self.n_y_lattice, self.n_x_lattice * self.n_y_lattice))
        for traj in trajs:
            pre_state = self._find_nearest_state_from_latlon_in_all_states(traj[0])
            for latlon in traj:
                if self._is_in_from_latlon(latlon):
                    state = self._find_nearest_state_from_latlon_in_all_states(latlon)
                    transition_mat[pre_state, state] += 1
                    
                    pre_state = state
                else:
                    break
                    
        self.transition_mat = self._normalize(transition_mat)
        
    def _normalize(self, transition_mat):
        transition_mat = copy.deepcopy(transition_mat)
        for i, transition_prob in enumerate(transition_mat):
            sum_ = np.sum(transition_prob)
            if sum_ != 0:
                transition_mat[i,:] = transition_mat[i,:]/sum_
        return transition_mat
    
    
    def load_trans_mat(self, path_transition_mat, traj, threashold=1e-4):
        
        transition_mat = np.loadtxt(path_transition_mat)
        self.transition_mat = self._threash(transition_mat, threashold)
        self._modify_for_test_traj(traj)
        self.size = len(self.transition_mat)
        
    
    def _threash(self, transition_mat, threashold):
        transition_mat = copy.deepcopy(transition_mat)
        transition_mat =  transition_mat * (transition_mat >= threashold)
        transition_mat =  self._normalize(transition_mat)
        return transition_mat
    
    
    def _modify_for_test_traj(self, test_traj):
        for i in range(len(test_traj) - 1):
            pre_loc = test_traj[i]
            pos_loc = test_traj[i+1]
            if pre_loc != pos_loc:
                self.transition_mat[pre_loc][pos_loc] += 0.1
                
        #self.transition_mat[2485][2435] += 0.1
        #self.transition_mat = self._normalize(self.transition_mat)