import src.mechanism as Mechanism
import joblib
import os
import numpy as np
import itertools

class ExponentialLikeMechanism(Mechanism.Mechanism):
    def __init__(self):
        print("exponential-like-mechanism")
        
    def perturb(self, location):
        ind = self._surrogate(np.array(location))
        distribution = self.dist[ind]
        choiced_ind = np.random.choice(range(len(self.nodes)), p=distribution)
        return self.nodes[choiced_ind][1]
        
    
    def inference(self):
        pass
    
    def _surrogate(self, location):
        min_distance = np.float("inf")
        min_ind = 0
        for i, (_, loc) in enumerate(self.nodes):
            distance = np.linalg.norm(loc - location)
            if distance < min_distance:
                min_distance = distance
                min_ind = i
                
        return min_ind
    
    def load(self, name):
        data = joblib.load(os.path.join("data", name + ".jbl"))
        self.mat, self.nodes, self.locations = data["weight_mat"], data["nodes"], data["locations"]
        self._metrize(self.mat)
        
    def build_distribution(self, epsilon):
        alpha = np.exp(- epsilon * self.mat)
        sum_alpha = np.nansum(alpha, axis=1).reshape(-1,1)
        self.dist = alpha / sum_alpha

    def _metrize(self, mat):

        def check_square_rank(mat):
            return mat.shape[0] == mat.shape[1]

        def positivate(mat):
            for i in range(1, mat.shape[0]):
                for j in range(0, i-1):
                    mat[j,i] = np.abs(mat[j,i])

        def parallelize(mat):
            for i in range(1, mat.shape[0]):
                for j in range(0, i-1):
                    mat[i,j] = mat[j,i]

        def list_all_triangles(mat):
            elements = range(mat.shape[0])
            return list(itertools.combinations(elements, 3))

        def trianglize(mat):
            triangles = list_all_triangles(mat)

            def triangle_to_lengths(triangle):
                lengths = []
                for i in range(3):
                    j = i+1 if i != 2 else 0

                    lengths.append(mat[triangle[i],triangle[j]])

                return np.array(lengths)

            for triangle in triangles:
                lengths = triangle_to_lengths(triangle)
                max_ind = np.argmax(lengths)
                max_value = lengths[max_ind]
                sum_other_values = np.sum(lengths[np.array([0,1,2] != max_ind)])
                if max_value > sum_other_values:
                    j = max_ind +1 if max_ind != 2 else 0
                    mat[triangle[j], triangle[max_ind]] = sum_other_values

            parallelize(mat)

        if not check_square_rank(mat):
            raise

        positivate(mat)
        trianglize(mat)