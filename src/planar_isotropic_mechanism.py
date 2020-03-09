import src.mechanism as mec
import joblib
import os
import numpy as np

class ScaledPlanarIsotropicMechanism(mec.IsotropicMechanism):
    
    def load(self, name):
        data = joblib.load(os.path.join("data", name + ".jbl"))
        self.mat, self.nodes, self.cell_deltaXs = data["weight_mat"], data["nodes"], data["locations"]
        self.is_load = True
        
        
        self.sensitivity_hull = self._make_sensitivity_hull(self.cell_deltaXs)
        self._make_convex_hull()
        self.T, self.T_i = self._compute_isotropic_transformation()
        
        print("T:", self.T)
        
        self.transformed_vertice = self._transform(self.vertice)
        self.transformed_deltaXs = self._transform(self.cell_deltaXs)
    
    def _make_sensitivity_hull(self, cell_deltaXs):
        
        sensitivity_hull = []
        
        size = len(cell_deltaXs)
        for i in range(size):
            for j in range(i,size):
                if i == j:
                    continue
                    
                w_ij = self.mat[i, j]
                if not np.isnan(w_ij):
                    sensitivity_hull.append((cell_deltaXs[i] - cell_deltaXs[j])/w_ij)
                    sensitivity_hull.append(cell_deltaXs[j] - cell_deltaXs[i]/w_ij)
                
        return np.array(sensitivity_hull)
    
    def perturb(self, epsilon, true_loc):

        self.epsilon = epsilon
        
        choice = np.random.choice(range(len(self.transformed_vertice)))
        choiced_vertex = self.transformed_vertice[choice]
        noise = np.random.gamma(3, 1/epsilon, 1)
        
        z = noise * np.dot(self.T_i, choiced_vertex.T)
        
        return true_loc + z