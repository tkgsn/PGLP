import mechanism
import numpy as np

def make_mechanism_with_policy_graph_class(mechanism_class):
    
    class MechanismWithPolicyGraph(mechanism_class):
        def load(self, coords, state_nos, policy_mat):
            super().load(coords, state_nos)

            self.policy_mat = policy_mat
            
        def _compute_sensitivity(self, coords, i, j):
            if self.policy_mat[self.state_nos[i],self.state_nos[j]] == 1:
                return [(coords[i] - coords[j]), (coords[j] - coords[i])]
            else:
                return []
            
        def perturb(self, true_loc):
            perturbed_loc = super().perturb(true_loc)
            
            mapped_perturbed_loc, perturbed_state = self._find_nearest_loc(perturbed_loc)
            return mapped_perturbed_loc, perturbed_state
            #return perturbed_loc
            
    return MechanismWithPolicyGraph



PlanarIsotropicMechanismWithPolicyGraph = make_mechanism_with_policy_graph_class(mechanism.PlanarIsotropicMechanism)
LaplaceMechanismWithPolicyGraph = make_mechanism_with_policy_graph_class(mechanism.LaplaceMechanism)