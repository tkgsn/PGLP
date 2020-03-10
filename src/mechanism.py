import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy
from shapely.geometry import Point, Polygon
import math
import copy
import joblib
import os

class Mechanism():
    def __init__(self):
        pass
    
    def perturb(self):
        pass
    
    def inference(self):
        pass
    
    def load_from_jbl(self, name):
        data = joblib.load(os.path.join("data", name + ".jbl"))
        self.weight_mat, self.locations, self.distance_mat = data["weight_mat"], data["locations"], data["distance_mat"]
        self.coords = np.array(self.locations)[:,1]
        self.is_load = True
    
    def load(self, oh_deltaXs, query):
        self.is_load = True
        
        query = copy.deepcopy(query)
        ##### Strange operation!!!!!!!!!!
        for i in range(2500):
            if (math.floor(i/50) % 2) ==1:
                query[0, i] += 0.05
        
        self.data_dim = len(query[:,0])
        
        self.query = query
        
        self.oh_deltaXs = oh_deltaXs
        
        self.coords = self._process(oh_deltaXs)
    
        
    def _check_included(self, oh_true_loc):
        state_no = np.where(oh_true_loc == 1)[0]
        state_no_deltaXs = [np.where(oh_deltaX == 1)[0][0] for oh_deltaX in self.oh_deltaXs]
        
        return state_no in state_no_deltaXs
    
    def _process(self, oh_locs):
        return np.dot(self.query, oh_locs.T).T
        
        
    def _surrogate(self, oh_true_loc):
        
        if not self._check_included(oh_true_loc):
            min_distance = float("inf")
            
            cell_true_loc = self._process(oh_true_loc)
            
            for oh_deltaX in self.oh_deltaXs:
                cell_deltaX = self._process(oh_deltaX)
                distance = np.linalg.norm(cell_true_loc - cell_deltaX)
                if distance < min_distance:
                    cell_true = cell_deltaX
                    oh_true_loc = oh_deltaX
                    
            print("surrogate by:", cell_true)
                    
        return oh_true_loc
        

class LaplaceMechanism(Mechanism):
    
    def __init__(self):
        super(LaplaceMechanism, self).__init__()
        self.sensitivity = self._l1_sensitivity(oh_deltaXs, query)
    
    def perturb(self, epsilon, oh_true_loc):
        if not self.is_load:
            raise("you should initially load")
        
        oh_true_loc = self._surrogate(oh_true_loc)
        cell_true = np.dot(self.query, oh_true_loc.T)
        
        noise = self.sensitivity * self._laplace_noise(epsilon)
        
        z = cell_true + noise
        return z
        
    def inference(self):
        pass

    def _l1_sensitivity(self):
        
        cell_deltas = np.dot(self.query, self.oh_deltaXs.T)
        size = len(self.oh_deltaXs[0])
        
        l1_norms = [np.linalg.norm((cell_deltas[:,i] - cell_deltas[:,j]), ord=1) for i in range(size) for j in range(size)]
        
        return np.max(l1_norms)
    
    def _laplace_noise(self, epsilon): 
        lap = np.random.exponential(1/epsilon, size=(self.data_dim, 1)) - np.random.exponential(1/epsilon, size=(self.data_dim, 1))
        
        return lap
    
    
class PlanarIsotropicMechanism(Mechanism):
    
    def __init__(self, iso_trans_sample_size=5000):
        super(PlanarIsotropicMechanism, self).__init__()
        self.iso_trans_sample_size = iso_trans_sample_size
        
    def inference(self, prior_dist, cell_z):
        
        if len(self.oh_deltaXs) == 1:
            return self.oh_deltaXs[0]
        
        transformed_z = self._transform(cell_z)

        pos_dist = np.sum(self.oh_deltaXs, axis=0)
        state_nos = np.where(pos_dist == 1)[0]
        
        inference_probs = {}

        for state_no, transformed_coord in zip(state_nos, self.transformed_coords):
            k = self._k_norm((transformed_z - transformed_coord))
            inference_probs[state_no] = np.exp(-self.epsilon * k) * prior_dist[state_no]

        inference_sum = np.sum(list(inference_probs.values()))
        
        inference_probs = {state_no: prob/inference_sum for state_no, prob in inference_probs.items()}

        for state_no, prob in inference_probs.items():
            pos_dist[state_no] = prob
        
        return pos_dist

    
    def _transform(self, vertices):
        return np.dot(self.T, vertices.T).T
    
    def _k_norm(self, vec):

        x,y = vec
        vec = Point(vec)
        
        n_vertices = len(self.transformed_vertices)
        if n_vertices == 2:
            ks = (vec / self.transformed_vertices)
            k = ks[0][0]
            if np.isnan(k):
                k = ks[0][1]
            return abs(k)
        
        a = 0
        k = 0
        
        for i in range(n_vertices):
            j = i+1
            if j == n_vertices:
                j = 0
            v1x, v1y = self.transformed_vertices[i][0], self.transformed_vertices[i][1]
            v2x, v2y = self.transformed_vertices[j][0], self.transformed_vertices[j][1]
            
            b = (v2x-(x/y)*v2y)/((x/y)*(v1y-v2y)-v1x+v2x)
            if b>=0 and b<=1 :
                a = b/y*(v1y-v2y)+v2y/y
            if a > 0:
                k = 1/a

        return k
    
    def _make_polygon(self, vertices):
        hull = ConvexHull(vertices)
        return Polygon(vertices[hull.vertices])

    def build_distribution(self, epsilon):
        self.epsilon = epsilon
        
        sensitivities = self._make_sensitivities(self.coords)
        vertices = self._make_convex_hull(sensitivities)
        self.T, self.T_i = self._compute_isotropic_transformation(vertices)
        
        self.transformed_vertices = self._make_convex_hull(self._transform(vertices))
        self.transformed_coords = self._transform(self.coords)
        
        ### For test
        
        self.sensitivities = sensitivities
        self.vertices = vertices
        
    def perturb(self, oh_true_loc):
        if not self.is_load:
            raise("you should initially load")
    
        if len(self.oh_deltaXs) == 1:
            return self._process(oh_true_loc)
        
        oh_true_loc = self._surrogate(oh_true_loc)
        cell_true_loc = self._process(oh_true_loc)
        
        sample = self._sample_point_from_hull(self.transformed_vertices)
        noise = np.random.gamma(3, 1/self.epsilon, 1)
        
        z = noise * np.dot(self.T_i, sample.T)
        
        return cell_true_loc + z
    
    def _sample_point_from_hull(self, vertices, total_length=None, segments=None):
        if total_length is None:
            total_length, segments = self._compute_total_length_of_full(vertices)

        random_length = np.random.rand() * total_length
        temp_total_length = random_length
        for seg_i, segment in enumerate(segments):
            if temp_total_length - segment < 0:
                break
            temp_total_length = temp_total_length - segment
        
        if seg_i == len(segments)-1:
            seg_j = 0
        else:
            seg_j = seg_i+1

        left_vertex = vertices[seg_i]
        right_vertex = vertices[seg_j]

        sample = left_vertex + (right_vertex - left_vertex) * temp_total_length / segment
        
        return sample

    def _compute_isotropic_transformation(self, vertices):
        
        if len(vertices) == 2:
            T = np.array([[1,0],[0,1]])
            mean_vertices = np.average(vertices, axis=0)
            return T, T
        
        samples = np.zeros((self.iso_trans_sample_size, len(vertices[0])))
        total_length, segments = self._compute_total_length_of_full(vertices)
        
        for sample_i in range(self.iso_trans_sample_size):
            samples[sample_i, :] = self._sample_point_from_hull(vertices, total_length, segments)

        mean_samples = np.average(samples, axis=0)
        
        T = np.average([np.dot(sample.reshape(2,-1), sample.reshape(2,-1).T) for sample in samples], axis=0)
        T = np.linalg.inv(T)
        T = scipy.linalg.sqrtm(T)
        T_i = np.linalg.inv(T)
        
        if np.isnan(T).any():
            print("nan")
            T = np.array([[1,0],[0,1]])
            T_i = np.array([[1,0],[0,1]]) 
            
        return T, T_i
    
    def _make_sensitivities(self, coords):
        
        sensitivities = []
        
        size = len(coords)
        for i in range(size-1):
            for j in range(i+1,size):
                sensitivities += self._compute_sensitivity(coords, i, j)
                
        return np.array(sensitivities)
    
    def _compute_sensitivity(self, coords, i, j):
        return [(coords[i] - coords[j]), (coords[j] - coords[i])]
        
        
    def _compute_total_length_of_full(self, vertices):
        
        segments = []
        total_length = 0
        for i in range(len(vertices)):
            if i == len(vertices)-1:
                j = 0
            else:
                j = i+1
            segment = np.linalg.norm(vertices[i] - vertices[j])
            segments.append(segment)
            total_length += segment
            
        return total_length, segments
        

    def _make_convex_hull(self, vertices):
        
        if len(self.coords) == 2:
            return vertices
        
        else:
            try:
                self.hull = ConvexHull(vertices)
                return vertices[self.hull.vertices]
            
            except:
                print("Vertices are on linear (>=3)")
                max_distance = -float("inf")
                for i in range(len(vertices)-1):
                    for j in range(i+1, len(vertices)):
                        distance = np.linalg.norm(vertices[i] - vertices[j])
                        if distance > max_distance:
                            max_distance = distance
                            edges = (i,j)
                return np.array([vertices[edges[0]], vertices[edges[1]]])