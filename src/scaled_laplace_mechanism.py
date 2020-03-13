import src.mechanism as mec
import joblib
import os
import numpy as np
import itertools

class ScaledLaplaceMechanism(mec.LaplaceMechanism):
    def __init__(self):
        print("scaled-laplace-mechanism")

    def _compute_sensitivity(self, coords, i, j):
        
        w_ij = self.weight_mat[i,j]
        if np.isnan(w_ij):
            return [[0,0]]
        
        return [(coords[i] - coords[j]) / w_ij, (coords[j] - coords[i]) / w_ij]