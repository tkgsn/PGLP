import os
import sqlite3
import pyproj
import numpy as np
import matplotlib.pyplot as plt
import math
EPSG4612 = pyproj.Proj("+init=EPSG:4612")
EPSG2451 = pyproj.Proj("+init=EPSG:2451")

def convert_to_cart(lon, lat):
    x,y = pyproj.transform(EPSG4612, EPSG2451, lon,lat)
    return x,y

data_dir = os.path.join("/", "data", "takagi", "globefish")
dbname = os.path.join(data_dir, "peopleflow.sqlite")

class MapProcessor():
    def __init__(self, n_x_lattice):
        self.n_x_lattice = n_x_lattice
        
        
    def make_graph_from_category(self, placeID=3, r=500):
        
        self.r = r
        self.placeID = placeID
        
        latlons = []
        for x in cur.execute(f"select Latitude, Longitude from peopleflow6 where PlaceID={placeID} and Latitude<={self.max_lat} and Latitude>={self.min_lat} and Longitude<={self.max_lon} and Longitude>={self.min_lon}"):
            latlons.append(x)
        latlons = np.array(latlons)
        
        possible_states = np.zeros(len(latlons), dtype=np.int32)
        for i, latlon in enumerate(latlons):
            possible_states[i] = int(self._find_nearest_coord_from_latlon(latlon[::-1], self.all_states))
        self.possible_states = [int(state) for state in list(set(possible_states))]
        
        self._update_graph_according_to_distance(self.possible_states, r)
        
        #def judge_same_id(self, true_state, perurbed_state):
            
            
                    
        #np.savetxt(os.path.join(data_dir, f"food.txt"), self.graph_mat)
        
    def make_graph_from_area(self, n_split=2, r=500):
        if (n_split > self.n_x_lattice) or (n_split > self.n_y_lattice):
            "n_split should be <= self.n_x(y)_lattice"
            raise
            
        self.possible_states = self.all_states
        
        n_x_lattice_in_area = math.ceil(self.n_x_lattice/n_split)
        n_y_lattice_in_area = math.ceil(self.n_y_lattice/n_split)
        
        n_area = n_split ** 2
        
        def state_to_area_state(state):
            coord = self.state_to_coord(state)
            
            area_coord = [math.floor(coord[0]/n_x_lattice_in_area), math.floor(coord[1]/n_y_lattice_in_area)]
            return area_coord[0] + area_coord[1] * n_split
            
        areas = [[] for _ in range(n_area)]
        for state in self.all_states:
            area_state = state_to_area_state(state)
            areas[area_state].append(state)
        
        for states_in_area in areas:
            self._update_graph_according_to_distance(states_in_area, r)
            
        #DEBUG
        self.areas = areas
                        
    def _update_graph_according_to_distance(self, states, r):
        
        for counter, state in enumerate(states[:-1]):
            coord = self.state_to_coord(state)
            for state_ in states[counter+1:]:
                coord_ = self.state_to_coord(state_)

                distance = np.linalg.norm(coord - coord_) * self.lattice_length

                if distance <= r:
                    self.graph_mat[state,state_] = 1
                    self.graph_mat[state_,state] = 1
            
    
    def make_map(self, min_lon, max_lon, min_lat, max_lat):
        con = sqlite3.connect(dbname)
        cur = con.cursor()
        
        self.min_lon, self.max_lon, self.min_lat, self.max_lat = min_lon, max_lon, min_lat, max_lat

        
        self.x0, self.y0 = convert_to_cart(min_lon, min_lat)
        self.x1, self.y1 = convert_to_cart(max_lon, max_lat)

        bottom_length = np.linalg.norm([self.x1-self.x0, 0])
        side_length = np.linalg.norm([0, self.y1-self.y0])
        self.lattice_length = bottom_length / self.n_x_lattice
        self.n_y_lattice = int(side_length / self.lattice_length) + 1

        self.all_states = list(range(self.n_x_lattice * self.n_y_lattice))
        
        n_state = self.n_x_lattice * self.n_y_lattice
        self.graph_mat = np.zeros((n_state, n_state))
        
    def coord_to_state(self, coord):
        return np.array(int(coord[0] + coord[1] * self.n_x_lattice))

    def state_to_coord(self, state):
        return np.array([state % self.n_x_lattice, int(state / self.n_x_lattice)])

    def states_to_coords(self, states):
        return np.array([self.state_to_coord(state) for state in states])

    def coords_to_states(self, coords):
        return np.array([self.coord_to_state(coord) for coord in coords])
    
    def _find_nearest_coord_from_latlon(self, latlon, states):
        coords = self.states_to_coords(states)
        coord = (convert_to_cart(*latlon) - np.array([self.x0, self.y0])) / self.lattice_length
        distances = np.linalg.norm(coords - coord, axis=1)
        return states[np.argmin(distances)]
    
    def plot_map(self):
        coords = self.states_to_coords(self.possible_states)
        plt.scatter(coords[:,0], coords[:, 1], s=10)
        plt.show()
        
    """
        
    def make_set_of_connected_states(self, states, graph_mat):
        n_states = len(states)

        G = nx.Graph()
        G.add_nodes_from(states)

        for i, state in enumerate(states):
            for state_ in states[i+1:]:
                if graph_mat[state, state_] == 1:
                    G.add_edge(state, state_)

        set_of_connected_states = [list(nodes) for nodes in nx.connected_components(G) if len(nodes) != 1]

        return set_of_connected_states
        
    def save(self):
        joblib.dump(f"{self.min_lon}_{self.max_lon}_{self.min_lat}_{self.max_lat}_placeID{self.placeID}.jbl")
        
    """