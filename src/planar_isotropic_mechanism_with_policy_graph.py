import mechanism
import numpy as np

class PlanarIsotropicMechanismWithPolicyGraph(mechanism.PlanarIsotropicMechanism):
    
    def load(self, coords, state_nos, policy_mat, n_locations=2500):
        super().load(coords, state_nos, n_locations)
        
        self.policy_mat = policy_mat
        
        """
        isolated_nodes = self.find_isolated_nodes()
        while isolated_nodes:
            
            state = isolated_nodes[0]
            coord = self.state2coord[state]
            _, nearest_state = self._find_nearest_loc(coord)
            
            self.policy_mat[state, nearest_state] = 1
            self.policy_mat[nearest_state, state] = 1
            
            isolated_nodes = self.find_isolated_nodes()
        """
        
    def _compute_sensitivity(self, coords, i, j):
        if self.policy_mat[self.state_nos[i],self.state_nos[j]] == 1:
            return [(coords[i] - coords[j]), (coords[j] - coords[i])]
        else:
            return []
        
    def find_isolated_nodes(self):
        isolated_nodes = []
        for state_no in self.state_nos:
            if (np.sum(self.policy_mat[state_no, self.state_nos] != 1) == 1):
                isolated_nodes += [state_no]
                
        return isolated_nodes