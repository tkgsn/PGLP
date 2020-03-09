import numpy as np
import networkx as nx
import os
import joblib

class GraphGenerator():
    def __init__(self):
        pass
    
    def make_graph(self):
        pass
    
    def save(self):
        savedir = os.path.join("data", self.name + ".jbl")
        joblib.dump(filename=savedir, value={"weight_mat":self.wig, "nodes":self.nodes, "locations":self.sd})
        
    
    
class GeoIGraphGenerator(GraphGenerator):
    def __init__(self, map_size=4000, sqrt_n_grids=10):
        self.map_size = map_size
        self.sqrt_n_grids = sqrt_n_grids
        self.grid_length = self.map_size / self.sqrt_n_grids
        
        self.n_grids = self.sqrt_n_grids ** 2
        self.cell_ids = np.array(range(self.n_grids))
        self.cells = np.array([(i,j) for i in range(self.sqrt_n_grids) for j in range(self.sqrt_n_grids)])
        self.latlons = np.array([self._cell_to_ll(cell) for cell in self.cells])
        self.nodes = [(cell, latlon) for cell, latlon in zip(self.cells, self.latlons)]
        self.name = f"mapsize{self.map_size}_n_grids{self.n_grids}"
        
    def _cell_to_ll(self, cell):
        return cell * self.grid_length
    
    def _id_to_cell(self, id):
        return np.array([int(id / self.sqrt_n_grids), id % self.sqrt_n_grids])
    
        
    def make_graph(self):
        self.G = nx.complete_graph(self.n_grids)
        for i in self.G.edges(data=True):
            cell1_ll = self._cell_to_ll(self._id_to_cell(i[0]))
            cell2_ll = self._cell_to_ll(self._id_to_cell(i[1]))
            i[2]["length"] = np.linalg.norm(cell1_ll - cell2_ll)
            
    def make_sd(self):
        self.sd = np.zeros((self.n_grids, self.n_grids))
        for i in self.cell_ids[:-1]:
            for j in self.cell_ids[i+1:]:
                self.sd[i,j] = self.G.edges[(i,j)]["length"]
                
        self.sd += self.sd.T
            
    def make_wig(self):
        self.wig = np.zeros((self.n_grids, self.n_grids))
        for i in self.cell_ids[:-1]:
            for j in self.cell_ids[i+1:]:
                self.wig[i,j] = self.G.edges[(i,j)]["length"]
                
        self.wig += self.wig.T