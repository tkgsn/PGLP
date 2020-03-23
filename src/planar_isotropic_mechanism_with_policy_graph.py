import mechanism

class PlanarIsotropicMechanismWithPolicyGraph(mechanism.PlanarIsotropicMechanism):
    
    def load(self, coords, state_nos, policy_mat, n_locations=2500):
        super().load(coords, state_nos, n_locations)
        
        self.policy_mat = policy_mat
        
    def _compute_sensitivity(self, coords, i, j):
        if self.policy_mat[self.state_nos[i],self.state_nos[j]] == 1:
            return [(coords[i] - coords[j]), (coords[j] - coords[i])]
        else:
            return []