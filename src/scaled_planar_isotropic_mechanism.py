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