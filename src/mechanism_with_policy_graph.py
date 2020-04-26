import mechanism
import numpy as np

def make_mechanism_with_policy_graph_class(mechanism_class):
    
    class MechanismWithPolicyGraph(mechanism_class):
        
        def __init__(self):
            super().__init__()
            self.with_policy = True
        
        #def load(self, coords, state_nos, policy_mat):
        #    super().load(coords, state_nos)
        #    
        #    self.policy_mat = policy_mat
            
        def _compute_sensitivity(self, coords, i, j):
            if self.policy_mat[self.state_nos[i],self.state_nos[j]] == 1:
                return [(coords[i] - coords[j]), (coords[j] - coords[i])]
            else:
                return []
            
        def perturb(self, true_coord):
            perturbed_coord = super().perturb(true_coord)
            
            #if len(self.coords) == 1:
            #    return self.coords[0]
            #else:
            mapped_perturbed_loc, perturbed_state = self._find_nearest_loc(perturbed_coord)
            return mapped_perturbed_loc
            #return perturbed_coord
            
    return MechanismWithPolicyGraph



PlanarIsotropicMechanismWithPolicyGraph = make_mechanism_with_policy_graph_class(mechanism.PlanarIsotropicMechanism)
LaplaceMechanismWithPolicyGraph = make_mechanism_with_policy_graph_class(mechanism.LaplaceMechanism)