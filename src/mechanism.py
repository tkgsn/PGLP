import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy
from shapely.geometry import Point, Polygon

class Mechanism():
    def __init__(self):
        pass
    
    def perturb(self):
        pass
    
    def inference(self):
        pass
    

    def load(self, oh_deltaXs, query):
        self.is_load = True
        
        self.data_dim = len(query[:,0])
        
        self.query = query
        self.oh_deltaXs = oh_deltaXs
        
        self.cell_deltaXs = self._process(oh_deltaXs)
        
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
    
    
class IsotropicMechanism(Mechanism):
    
    def __init__(self, iso_trans_sample_size=5000):
        super(IsotropicMechanism, self).__init__()
        self.iso_trans_sample_size = iso_trans_sample_size
    
    def load(self, oh_deltaXs, query):
        super().load(oh_deltaXs, query)
        
        if len(self.oh_deltaXs)==1:
            return
        
        self._make_sensitivity_hull()
        self._make_convex_hull()
        self.T, self.T_i = self._compute_isotropic_transformation()
        
        self.transformed_vertice = self._transform(self.vertice)
        self.transformed_deltaXs = self._transform(self.cell_deltaXs)
        
    def inference(self, prior_dist, cell_z):
        
        if len(self.oh_deltaXs) == 1:
            return self.oh_deltaXs[0]
        
        transformed_z = self._transform(cell_z)

        pos_dist = np.sum(self.oh_deltaXs, axis=0)
        state_nos = np.where(pos_dist == 1)[0]
        
        inference_probs = {}
        #print("transformed_z", transformed_z)
        #print("transformed_deltaXs",self.transformed_deltaXs)
        for state_no, transformed_deltaX in zip(state_nos, self.transformed_deltaXs):
            k = self._k_norm((transformed_z - transformed_deltaX))
            inference_probs[state_no] = np.exp(-self.epsilon * k) * prior_dist[state_no]

        inference_sum = np.sum(list(inference_probs.values()))
        print("inference_sum", inference_sum)
        inference_probs = {state_no: prob/inference_sum for state_no, prob in inference_probs.items()}

        for state_no, prob in inference_probs.items():
            pos_dist[state_no] = prob
        
        return pos_dist
    
    def _transform(self, vertice):
        return np.dot(self.T, vertice.T).T
    
        
    def _k_norm(self, vec):
        k = 1

        vec = Point(vec)
        
        if len(self.transformed_vertice) == 2:
            ks = (vec / self.transformed_vertice)
            k = ks[0][0]
            if np.isnan(k):
                k = ks[0][1]
            return abs(k)
        
        polygon = self._make_polygon(self.transformed_vertice)

        while not vec.within(polygon):
            k += 1
            polygon = self._make_polygon(k * self.transformed_vertice)

        while vec.within(polygon):
            k -= 0.1
            polygon = self._make_polygon(k * self.transformed_vertice)

        return k

    def _make_polygon(self, vertice):
        hull = ConvexHull(vertice)
        return Polygon(vertice[hull.vertices])


    def perturb(self, epsilon, oh_true_loc):
        if not self.is_load:
            raise("you should initially load")
    
        if len(self.oh_deltaXs) == 1:
            return self._process(oh_true_loc)
        
        self.epsilon = epsilon
        oh_true_loc = self._surrogate(oh_true_loc)
        cell_true_loc = self._process(oh_true_loc)
        
        choice = np.random.choice(range(len(self.transformed_vertice)))
        choiced_vertex = self.transformed_vertice[choice]
        noise = np.random.gamma(3, 1/epsilon, 1)
        
        z = noise * np.dot(self.T_i, choiced_vertex.T)
        
        return cell_true_loc + z

    def _compute_isotropic_transformation(self):
        
        if len(self.vertice) == 2:
            T = np.array([[1,0],[0,1]])
            mean_vertice = np.average(self.vertice, axis=0)
            self.transformed_vertice = self.vertice - mean_vertice
            return T, T
        
        samples = []
        for _ in range(self.iso_trans_sample_size):
            random_length = np.random.rand() * self.total_length
            temp_total_length = random_length
            for i, segment in enumerate(self.segments):
                if temp_total_length - segment < 0:
                    break
                temp_total_length = temp_total_length - segment

            left_vertex = self.simplices[i][0]
            right_vertex = self.simplices[i][1]

            sample = self.sensitivity_hull[left_vertex] + (self.sensitivity_hull[right_vertex] - self.sensitivity_hull[left_vertex]) * temp_total_length / self.segments[i]
            samples.append(sample)
            
        samples = np.array(samples)

        mean_samples = np.average(samples, axis=0)
        
        T = np.average([np.dot((sample - mean_samples).reshape(2,-1), (sample - mean_samples).reshape(-1,2)) for sample in samples], axis=0)
        T = np.linalg.inv(T)
        T = scipy.linalg.sqrtm(T)
        T_i = np.linalg.inv(T)
        
        if np.isnan(T).any():
            print("nan")
            T = np.array([[1,0],[0,1]])
            T_i = np.array([[1,0],[0,1]]) 
        
        return T, T_i
    
    def _make_sensitivity_hull(self):
        
        sensitivity_hull = []
        
        size = len(self.cell_deltaXs)
        for i in range(size):
            for j in range(i,size):
                if i == j:
                    continue
                    
                sensitivity_hull.append(self.cell_deltaXs[i] - self.cell_deltaXs[j])
                sensitivity_hull.append(self.cell_deltaXs[j] - self.cell_deltaXs[i])
                
        self.sensitivity_hull = np.array(sensitivity_hull)

    def _make_convex_hull(self):
        
        if len(self.cell_deltaXs) == 2:
            self.vertice = self.sensitivity_hull
            self.simplices = np.array([[0,1]])
        
        else:
            try:
                self.hull = ConvexHull(self.sensitivity_hull)
            except:
                print("Vertice are on linear (>=3)")
                min_distance = float("inf")
                for i in range(len(self.sensitivity_hull)):
                    for j in range(i, len(self.sensitivity_hull)):
                        distance = np.linalg.norm(self.sensitivity_hull[i] - self.sensitivity_hull[j])
                        if distance > min_distance:
                            edges = (i,j)
                self.vertice = np.array([self.sensitivity_hull[i], self.sensitivity_hull[j]])
                self.simplices = np.array([[0,1]])
                return
                
            self.vertice = self.sensitivity_hull[self.hull.vertices]
            self.simplices = self.hull.simplices
        
        segments = []
        total_length = 0
        for simplex in self.simplices:
            segment = np.linalg.norm(self.sensitivity_hull[simplex[1]] - self.sensitivity_hull[simplex[0]])
            segments.append(segment)
            total_length += segment
            
        self.total_length = total_length
        self.segments = segments
        
    """        
    def _make_convex_hull(self):
        
        if len(self.cell_deltaXs) == 2:
            self.vertice = self.cell_deltaXs
            self.simplices = np.array([[0,1]])
        
        else:
            self.hull = ConvexHull(self.cell_deltaXs)
            self.vertice = self.hull.points
            self.simplices = self.hull.simplices
        
        segments = []
        total_length = 0
        for simplex in self.simplices:
            segment = np.linalg.norm(self.vertice[simplex[1]] - self.vertice[simplex[0]])
            segments.append(segment)
            total_length += segment
            
        self.total_length = total_length
        self.segments = segments
    """
        
    
"""
class LaplaceMechanism():
    def __init__(self):
        pass
    
    def perturb(self, epsilon, oh_true_loc, oh_deltaXs):
        
        oh_true_loc = self._surrogate(oh_true_loc, oh_deltaXs)
        cell_true = np.dot(self.query, oh_true_loc.T)
        
        noise = self.sensitivity * self._laplace_noise(epsilon)
        
        z = cell_true + noise
        return z
                
    
    def load(self, oh_deltaXs, query):
        self.data_dim = len(query[:,0])
        self.sensitivity = self._l1_sensitivity(oh_deltaXs, query)
        self.query = query
        
    def _laplace_noise(self, epsilon): 
        lap = np.random.exponential(1/epsilon, size=(self.data_dim, 1)) - np.random.exponential(1/epsilon, size=(self.data_dim, 1))
        
        return lap
                
                
    def _check_included(self, oh_true_loc, oh_deltaXs):
        state_no = np.where(oh_true_loc == 1)
        state_no_deltaXs = [np.where(oh_deltaX == 1)[0] for oh_deltaX in oh_deltaXs]
        
        return state_no in state_no_deltaXs
    
    def _surrogate(self, oh_true_loc, oh_deltaXs):
        
        cell_true = np.dot(self.query, oh_true_loc.T)
        
        if not _check_included(oh_true_loc, oh_deltaXs):
            min_distance = float("inf")
            
            for oh_deltaX in oh_deltaXs:
                cell_deltaX = np.dot(self.query, oh_deltaX)
                distance = np.linalg.norm(cell_true - cell_deltaX)
                if distance < min_distance:
                    cell_true = cell_deltaX
                    
        return oh_true_loc
        

    def _l1_sensitivity(self, oh_deltaXs, query):
        
        cell_deltas = np.dot(query, oh_deltaXs.T)
        size = len(oh_deltaXs[0])
        
        l1_norms = [np.linalg.norm(cell_deltas(:,i) - cell_deltas(:,j), ord=1) for i in range(size) for j in range(size)]
        
        return np.max(l1_norms)
"""