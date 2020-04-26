import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import math
import networkx as nx
import mechanism_with_policy_graph
import copy


def distance_on_unit_sphere(latlon1, latlon2):
    lat1, long1 = latlon1
    lat2, long2 = latlon2
    
    degrees_to_radians = math.pi/180.0

    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + 
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    
    R = 6373.0 * 1e3

    return arc * R

def repair_graph(mp, constrained_states):
    pim_with_pg = mechanism_with_policy_graph.PlanarIsotropicMechanismWithPolicyGraph()
    
    original_graph_mat = mp.graph_mat
    
    if len(constrained_states) == 1:
        return mp.graph_mat
    
    set_of_connected_states = MapProcessor.make_set_of_connected_states(constrained_states, original_graph_mat)
    isolated_states = [state[0] for state in set_of_connected_states if len(state) == 1]
    
    pim = mechanism_with_policy_graph.PlanarIsotropicMechanismWithPolicyGraph()
    
    graph_mat = copy.deepcopy(mp.graph_mat)

    while isolated_states:
        isolated_state = isolated_states[0]
        print("search for a connection state of", isolated_state)
        
        isolated_coords = mp.states_to_coords(isolated_states)
        isolated_coord = isolated_coords[0]

        min_diff = float("inf")
        
        for i, states in enumerate(set_of_connected_states):
            if states[0] == isolated_state:
                continue
            coords = mp.states_to_coords(states)
            
            pim.load(coords, states)
            pim.policy_mat = graph_mat
            orig_area = pim.compute_area_of_sensitivity_hull()
            
            diff = float("inf")
            for state_ in states:

                temp_graph_mat = copy.deepcopy(graph_mat)
                temp_graph_mat[state_, isolated_state] = 1
                temp_graph_mat[isolated_state, state_] = 1

                temp_coords = np.concatenate([coords, [isolated_coord]])
                temp_states = np.concatenate([states, [isolated_state]])

                pim.load(temp_coords, temp_states)
                pim.policy_mat = temp_graph_mat
                temp_area = pim.compute_area_of_sensitivity_hull()

                if diff > temp_area - orig_area:
                    diff = temp_area - orig_area
                    state = state_

            if diff < min_diff:
                min_diff = diff
                min_state = state
                min_ind = i

        print("connect", isolated_state, min_state, f"{isolated_coord}, {mp.state_to_coord(min_state)}")

        set_of_connected_states[min_ind].append(isolated_state)
        isolated_states.pop(0)
        graph_mat[min_state, isolated_state] = 1
        graph_mat[isolated_state, min_state] = 1
        
    return graph_mat


data_dir = os.path.join("/", "data", "takagi", "globefish")
dbname = os.path.join(data_dir, "peopleflow.sqlite")

class MapProcessor():
    def __init__(self, n_x_lattice):
        self.n_x_lattice = n_x_lattice
        
    @classmethod
    def make_set_of_connected_states(cls, states, graph_mat):
        G = nx.Graph()
        G.add_nodes_from(states)

        for i, state in enumerate(states):
            for state_ in states[i:]:
                if graph_mat[state, state_] == 1:
                    G.add_edge(state, state_)

        set_of_connected_states = [list(nodes) for nodes in nx.connected_components(G)]

        return set_of_connected_states

    @classmethod
    def connected_states(cls, state, set_of_states):
        for states in set_of_states:
            if state in states:
                return states
        
    def load_latlons_of_category(self, placeID):
        con = sqlite3.connect(dbname)
        cur = con.cursor()
        latlons = []
        for x in cur.execute(f"select Latitude, Longitude from peopleflow6 where PlaceID={placeID} and Latitude<={self.max_lat} and Latitude>={self.min_lat} and Longitude<={self.max_lon} and Longitude>={self.min_lon}"):
            latlons.append(x)
        return np.array(latlons)
        
    def make_graph_from_category_and_distance(self, placeID=3, r=500):

        self.r = r
        self.placeID = placeID

        latlons = self.load_latlons_of_category(placeID)
        
        possible_states = np.zeros(len(latlons), dtype=np.int32)
        for i, latlon in enumerate(latlons):
            possible_states[i] = self._find_nearest_state_from_latlon_in_all_states(latlon)
        self.possible_states = [int(state) for state in list(set(possible_states))]
        self.possible_coords = self.states_to_coords(self.possible_states)
        
        self._update_graph_according_to_distance(self.possible_states, r)
        

    def make_graph_from_category_and_area(self, placeID=3, n_split=2):
        
        self.make_area(n_split)
        self.placeID = placeID
        
        latlons = self.load_latlons_of_category(placeID)
        
        possible_states = np.zeros(len(latlons), dtype=np.int32)
        for i, latlon in enumerate(latlons):
            possible_states[i] = self._find_nearest_state_from_latlon_in_all_states(latlon)
        self.possible_states = [int(state) for state in list(set(possible_states))]
        self.possible_coords = self.states_to_coords(self.possible_states)
        
        self._update_graph_according_to_area(self.possible_states)
        
        
    def make_area(self, n_split):
        n_x_lattice_in_area = math.floor((self.n_x_lattice + 1)/n_split)
        
        n_x_area = math.ceil((self.n_x_lattice + 1) / n_x_lattice_in_area)
        n_y_area = math.ceil((self.n_y_lattice + 1) / n_x_lattice_in_area)
        
        n_area = n_x_area * n_y_area
        
        def state_to_area_state(state):
            coord = self.state_to_coord(state)
            
            area_coord = [math.floor(coord[0]/n_x_lattice_in_area), math.floor(coord[1]/n_x_lattice_in_area)]
            return area_coord[0] + area_coord[1] * n_x_area
        
        self.state_to_area_state = state_to_area_state
            
        areas = [[] for _ in range(n_area)]
        for state in self.all_states:
            area_state = state_to_area_state(state)
            areas[area_state].append(state)
            
        self.areas = areas
        
    def is_same_area(self, state1, state2):
        area1 = self.state_to_area_state(state1)
        area2 = self.state_to_area_state(state2)
        return area1 == area2
    
    def is_in(self, state):
        return state in self.possible_states
        
    def make_graph_from_area(self, n_split=2, r=float("inf")):
        if (n_split > self.n_x_lattice) or (n_split > self.n_y_lattice):
            "n_split should be <= self.n_x(y)_lattice"
            raise
            
        self.make_area(n_split)
            
        self.possible_states = self.all_states
        self.possible_coords = self.states_to_coords(self.possible_states)
        
        for states_in_area in self.areas:
            self._update_graph_according_to_distance(states_in_area, r)
            
        
    def cp_n_split(self, n_subgraph_x_nodes):
        return math.floor((self.n_x_lattice + 1)/n_subgraph_x_nodes)
                        
    def _update_graph_according_to_distance(self, states, r):
        
        for counter, state in enumerate(states):
            coord = self.state_to_coord(state)
            for state_ in states[counter:]:
                coord_ = self.state_to_coord(state_)

                distance = np.linalg.norm(coord - coord_) * self.lattice_length

                if distance <= r:
                    self.graph_mat[state,state_] = 1
                    self.graph_mat[state_,state] = 1
                    
    def _update_graph_according_to_area(self, states):
        
        for counter, state in enumerate(states):
            for state_ in states[counter:]:

                if self.is_same_area(state, state_):
                    self.graph_mat[state,state_] = 1
                    self.graph_mat[state_,state] = 1
    
    def make_map_from_latlon(self, min_lon, max_lon, min_lat, max_lat):
        
        self.min_lon, self.max_lon, self.min_lat, self.max_lat = min_lon, max_lon, min_lat, max_lat

        bottom_length = distance_on_unit_sphere((self.min_lat, self.min_lon), (self.min_lat, self.max_lon))
        side_length = distance_on_unit_sphere((self.min_lat, self.min_lon), (self.max_lat, self.min_lon))
        self.lattice_length = bottom_length / self.n_x_lattice
        
        self.n_y_lattice = round(side_length / self.lattice_length)
        self.n_state = (self.n_x_lattice + 1) * (self.n_y_lattice + 1)

        self.all_states = list(range(self.n_state))
        self.all_coords = self.states_to_coords(self.all_states)
        
        self.graph_mat = np.zeros((self.n_state, self.n_state))
        
    def coord_to_state(self, coord):
        return int(coord[0] + coord[1] * (self.n_x_lattice + 1))

    def state_to_coord(self, state):
        return np.array([state % (self.n_x_lattice + 1), int(state / (self.n_x_lattice + 1))])

    def states_to_coords(self, states):
        return np.array([self.state_to_coord(state) for state in states])

    def coords_to_states(self, coords):
        return np.array([self.coord_to_state(coord) for coord in coords])
    
    def _find_nearest_state_from_latlon_in_all_states(self, latlon):
        coord = np.array([self.n_x_lattice, self.n_y_lattice]) * (np.array([latlon[1], latlon[0]]) - np.array([self.min_lon, self.min_lat])) / (np.array([self.max_lon - self.min_lon, self.max_lat - self.min_lat]))
        state = int(self.coord_to_state([round(coord[0]), round(coord[1])]))
        return int(self.coord_to_state([round(coord[0]), round(coord[1])]))
        
    def _find_nearest_state_from_latlon(self, latlon, states):
        coords = self.states_to_coords(states)
        coord = np.array([self.n_x_lattice, self.n_y_lattice]) * (np.array([latlon[1], latlon[0]]) - np.array([self.min_lon, self.min_lat])) / (np.array([self.max_lon - self.min_lon, self.max_lat - self.min_lat]))
        distances = np.linalg.norm(coords - coord, axis=1)
        return states[np.argmin(distances)]
    
    def _is_in_from_latlon(self, latlon):
        return not (any(latlon < np.array([self.min_lat, self.min_lon])) or any(latlon > np.array([self.max_lat, self.max_lon])))
    
    def find_nearest_state(self, coord):
        dists = np.linalg.norm(coord - self.all_coords, axis=1)
        return self.all_states[np.argmin(dists)]
    
    def find_nearest_possible_state_other_than_own(self, state):
        coord = self.state_to_coord(state)
        distance = np.linalg.norm(self.possible_coords - coord, axis=1)
        distance[distance==0] = float("inf")
        return self.possible_states[np.argmin(distance)]
    
    def plot_map(self):
        coords = self.possible_coords
        plt.scatter(coords[:,0], coords[:, 1], s=10)
        plt.show()
        
    def plot_map_by_latlon(self):
        latlons = np.array([self.state_to_latlon(state) for state in self.possible_states])
        plt.rcParams["font.size"] = 20
        plt.scatter(latlons[:,0], latlons[:, 1], s=10)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig("temp.eps", bbox_inches='tight', pad_inches=0)
        plt.show()
        
    def state_to_latlon(self, state):
        coord = self.state_to_coord(state)
        x = self.max_lon - self.min_lon
        y = self.max_lat - self.min_lat
        
        x_multi = x / self.n_x_lattice
        y_multi = y / self.n_y_lattice
        
        return self.min_lon + coord[0] * x_multi, self.min_lat + coord[1] * y_multi