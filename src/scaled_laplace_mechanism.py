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
    
    def load(self, name):
        data = joblib.load(os.path.join("data", name + ".jbl"))
        self.mat, self.nodes, self.sd = data["weight_mat"], data["nodes"], data["locations"]
        self.locations = np.array(self.nodes)[:,1]
        
    def _compute_sensitivity(self):
        n_nodes = len(self.nodes)
        difs = []
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if not np.isnan(self.mat[i,j]):
                    difs.append(np.abs(self.locations[i] - self.locations[j]) / self.mat[i,j])
        difs = np.array(difs)
        
        x_sensitivity = np.max(difs[:,0])
        y_sensitivity = np.max(difs[:,1])
        
        self.sensitivity = [x_sensitivity, y_sensitivity]
            
        
    def build_distribution(self, epsilon):
        self._compute_sensitivity()
        def laplace():
            x = np.random.laplace(0, self.sensitivity[0]/epsilon, 1)[0]
            y = np.random.laplace(0, self.sensitivity[1]/epsilon, 1)[0]
            return np.array([x, y])
        
        self.laplace_generator =  laplace