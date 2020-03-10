import src.mechanism as Mechanism
import joblib
import os
import numpy as np
import itertools

class ScaledLaplaceMechanism(Mechanism.Mechanism):
    def __init__(self):
        print("scaled-laplace-mechanism")
        
    def perturb(self, location):
        return np.array(location) + self.laplace_generator()
    
    def inference(self):
        pass
        
    def _compute_sensitivity(self):
        n_locations = len(self.locations)
        
        difs = []
        for i in range(n_locations-1):
            for j in range(i+1, n_locations):
                if not np.isnan(self.weight_mat[i,j]):
                    difs.append(np.abs(self.coords[i] - self.coords[j]) / self.weight_mat[i,j])
        difs = np.array(difs)
        
        sensitivity = np.max(np.linalg.norm(difs, ord=1, axis=1))

        self.sensitivity = sensitivity
        
    def build_distribution(self, epsilon):
        self._compute_sensitivity()
        def laplace():
            return np.random.laplace(0, self.sensitivity/epsilon, 2)
        
        self.laplace_generator =  laplace