import networkx as nx
import numpy as np
import copy
from scipy.spatial import ConvexHull


def find_nearest_index(coord, coords):
    min_distance = float("inf")
    for i, coord_ in enumerate(coords):
        distance = np.linalg.norm(coord - coord_)
        if distance < min_distance:
            min_distance = distance
            nearest_ind = i
    return nearest_ind

def is_in(coord, coords, graph_mat, map_processor):
    vertices = make_sensitivity_hull(coords, graph_mat, map_processor)
    k_norm = make_k_norm(vertices)

    diffs = coords - coord
    k_norms = [k_norm(diff) <= 1 + 1e-6 for diff in diffs]

    return k_norms

def make_sensitivity_hull(coords, graph_mat, map_processor):
    sensitivities = make_sensitivities(coords, graph_mat, map_processor)
    vertices = make_convex_hull(sensitivities)
    return vertices

def make_sensitivities(coords, graph_mat, map_processor):

    sensitivities = []

    size = len(coords)
    for i in range(size):
        for j in range(i,size):
            state_i = map_processor.coord_to_state(coords[i])
            state_j = map_processor.coord_to_state(coords[j])
            if graph_mat[state_i, state_j] == 1:
                sensitivities += [(coords[i] - coords[j]), (coords[j] - coords[i])]

    return np.array(sensitivities)

def make_convex_hull(sensitivities):

    try:
        hull = ConvexHull(sensitivities)
        return sensitivities[hull.vertices]

    except:
        max_distance = -float("inf")
        for i in range(len(sensitivities)-1):
            for j in range(i+1, len(sensitivities)):
                distance = np.linalg.norm(sensitivities[i] - sensitivities[j])
                if distance > max_distance:
                    max_distance = distance
                    edges = (i,j)
        return np.array([sensitivities[edges[0]], sensitivities[edges[1]]])
    
def make_k_norm(vertices):

    def k_norm(vec):
        x,y = vec

        n_vertices = len(vertices)
        if n_vertices == 2:
            v1x, v1y = vertices[0][0], vertices[0][1]
            v2x, v2y = vertices[1][0], vertices[1][1]
            
            if v2y - v1y == 0:
                if y != 0:
                    return float("inf")
                else:
                    if np.abs(x/v1x) != np.abs(x/v2x):
                        raise
                    return np.abs(x/v1x)
            
            if v2x - v1x == 0:
                if x != 0:
                    return float("inf")
                else:
                    if np.abs(y/v1y) != np.abs(y/v2y):
                        raise
                    return np.abs(y/v1y)

            if np.abs(x/v1x) != np.abs(y/v1y):
                return float("inf")
            
            return np.abs(x/v1x)

        for i in range(n_vertices):
            j = 0 if i == (n_vertices-1) else i+1
            v1x, v1y = vertices[i][0], vertices[i][1]
            v2x, v2y = vertices[j][0], vertices[j][1]
            
            if x == 0 and y == 0:
                return 0
            
            if (v2y-v1y) == 0 and (v2x-v1x) == 0:
                raise
            
            if x*(v2y-v1y) - y*(v2x-v1x) == 0:
                continue
            
            t = (y*v1x - x*v1y)/(x*(v2y-v1y) - y*(v2x-v1x))
            
            if t >= 0 and t <= 1:
                denom = (v1x + t*(v2x-v1x))
                if denom == 0:
                    k = y / (v1y + t*(v2y-v1y))
                else:
                    k = x / (v1x + t*(v2x-v1x))
                if k >= 0:
                    return k
                
        raise
        return k

    return k_norm


def compute_area_of_sensitivity_hull(vertices):
    area = 0

    n_vertices = len(vertices)

    if n_vertices == 1:
        return 0

    if n_vertices == 2:
        return np.linalg.norm(vertices[0] - vertices[1])

    for i in range(n_vertices):
        j = 0 if i == n_vertices-1 else i+1

        coord0 = vertices[i]
        coord1 = vertices[j]

        area += (1/2)*np.abs(np.linalg.det(np.array([coord0,coord1])))

    return area
    

class GraphRepairer():
    def __init__(self, map_processor):
        self.map_processor = map_processor
        self.load()
        
    
    def load(self):
        self.graph_mat = self.map_processor.graph_mat
        self.states = self.map_processor.possible_states
        
        self.set_of_connected_states = self._make_set_of_connected_states(self.states, self.graph_mat)
        self.disconnected_states = self._make_disconnected_states(self.states, self.set_of_connected_states)
        
        
    def repair_graph(self):
        isolated_states = []

        for disconnected_state in self.disconnected_states:
            disconnected_coord = self.map_processor.state_to_coord(disconnected_state)
            
            is_in_ = False
            
            for i, states in enumerate(self.set_of_connected_states):
                coords = self.map_processor.states_to_coords(states)


                is_ins = is_in(disconnected_coord, coords, self.graph_mat, self.map_processor)
                is_in_ = is_in_ | np.any(is_ins)

                if is_in_:
                    self.set_of_connected_states[i].append(disconnected_state)
                    connected_state = self.set_of_connected_states[i][np.where(is_ins)[0][0]]
                    self.graph_mat[connected_state, disconnected_state] = 1
                    self.graph_mat[disconnected_state, connected_state] = 1
                    print("connect", connected_state, disconnected_state)
                    break

            if not is_in_:
                print("isolated", disconnected_state)
                isolated_states.append(disconnected_state)


        while isolated_states:
            isolated_state = isolated_states[0]
            
            print("search for a connect state of", isolated_state)
            isolated_coords = self.map_processor.states_to_coords(isolated_states)
            isolated_coord = isolated_coords[0]

            nearest_ind = find_nearest_index(isolated_coord, isolated_coords[1:]) + 1
            nearest_state = isolated_states[nearest_ind]
            nearest_coord = isolated_coords[nearest_ind]

            distance = np.linalg.norm(isolated_coord - nearest_coord)

            min_diff = float("inf")

            for i, states in enumerate(self.set_of_connected_states):
                coords = self.map_processor.states_to_coords(states)
                orig_area = compute_area_of_sensitivity_hull(make_sensitivity_hull(coords, self.graph_mat, self.map_processor))
                
                temp_coords = np.concatenate([coords, [isolated_coord]])
                for coord in coords:
                    state = self.map_processor.coord_to_state(coord)
                    
                    temp_graph_mat = copy.deepcopy(self.graph_mat)
                    temp_graph_mat[state, isolated_state] = 1
                    temp_graph_mat[isolated_state, state] = 1

                    temp_area = compute_area_of_sensitivity_hull(make_sensitivity_hull(temp_coords, temp_graph_mat, self.map_processor))

                    diff = temp_area - orig_area

                    if diff == 0:
                        print(states, "something wrong?")

                    if diff < min_diff:
                        min_diff = diff
                        min_state = state
                        min_ind = i

            print("distance", distance)
            print("min_diff", min_diff)

            if min_diff < distance:
                print("connect", isolated_state, min_state, f"{isolated_coord}, {self.map_processor.state_to_coord(min_state)}")

                self.set_of_connected_states[min_ind].append(isolated_state)
                isolated_states.pop(0)
                self.graph_mat[min_state, isolated_state] = 1
                self.graph_mat[isolated_state, min_state] = 1
            else:
                self.set_of_connected_states.append([isolated_state, nearest_state])
                self.graph_mat[nearest_state, isolated_state] = 1
                self.graph_mat[isolated_state, nearest_state] = 1
                isolated_states.pop(nearest_ind)
                isolated_states.pop(0)
        
        
        
    def _make_set_of_connected_states(self, states, graph_mat):
        n_states = len(states)

        G = nx.Graph()
        G.add_nodes_from(states)

        for i, state in enumerate(states):
            for state_ in states[i+1:]:
                if graph_mat[state, state_] == 1:
                    G.add_edge(state, state_)

        set_of_connected_states = [list(nodes) for nodes in nx.connected_components(G) if len(nodes) != 1]

        return set_of_connected_states

    def _make_disconnected_states(self, states, set_of_connected_states):
        disconnected_states = []
        for state in states:
            connecteds = self._connected_states(state, set_of_connected_states)
            if connecteds is None:
                disconnected_states.append(state)

        return disconnected_states
    
    def _connected_states(self, state, set_of_states):
        for states in set_of_states:
            if state in states:
                return states