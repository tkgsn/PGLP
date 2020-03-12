import src.mechanism as mec
import joblib
import os
import numpy as np

class ScaledPlanarIsotropicMechanism(mec.PlanarIsotropicMechanism):

    def __init__(self, iso_trans_sample_size=5000):
        super(ScaledPlanarIsotropicMechanism, self).__init__()
        self.iso_trans_sample_size = iso_trans_sample_size
        
        print("scaled-planar-isotropic-mechanism")
        
        
    def _compute_sensitivity(self, coords, i, j):
        
        w_ij = self.weight_mat[i,j]
        if np.isnan(w_ij):
            return [[0,0]]
        
        return [(coords[i] - coords[j]) / w_ij, (coords[j] - coords[i]) / w_ij]
    
    def perturb(self, true_loc):
        
        sample = self._sample_point_from_body(self.transformed_vertices)[0]
        noise = np.random.gamma(3, 1/self.epsilon, 1)
        
        z = noise * np.dot(self.T_i, sample.T)
        
        return true_loc + z