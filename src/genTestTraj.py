#Generate tragectries
import numpy as np
import copy


class TrajectoryGenerator():
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
    
    def posterior_distribution(self, prior):
        if self.transition_mat is None:
            print("you must initially load")
        
        posterior = np.dot(prior, self.transition_mat)
        
        if posterior.sum() == 0:
            print("end")
        
        return posterior
        
    
    def load(self, path_transition_mat, length_traj=500, size_traj=100, threashold=1e-4, output_path="results/out.txt"):
        
        transition_mat = np.loadtxt(path_transition_mat)
        self.transition_mat = self._threash(transition_mat, threashold)
        self.size = len(self.transition_mat)
    
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
    
    
    def modify_for_test_traj(self, test_traj):
        for i in range(len(test_traj) - 1):
            pre_loc = test_traj[i]
            pos_loc = test_traj[i+1]
            if pre_loc != pos_loc:
                self.transition_mat[pre_loc][pos_loc] += 0.1
                
        self.transition_mat[2485][2435] += 0.1
        self.transition_mat = self._normalize(self.transition_mat)
            
        #??(2486,2436)