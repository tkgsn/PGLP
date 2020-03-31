import os
import sqlite3
import pyproj
import numpy as np
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
    
    def make_map(self, min_lon, max_lon, min_lat, max_lat, placeID=3):
        con = sqlite3.connect(dbname)
        cur = con.cursor()

        latlons = []
        for x in cur.execute(f"select Latitude, Longitude from peopleflow6 where PlaceID={placeID} and Latitude<={max_lat} and Latitude>={min_lat} and Longitude<={max_lon} and Longitude>={min_lon}"):
            latlons.append(x)
        latlons = np.array(latlons)
        
        self.x0, self.y0 = convert_to_cart(min_lon, min_lat)
        self.x1, self.y1 = convert_to_cart(max_lon, max_lat)

        bottom_length = np.linalg.norm([self.x1-self.x0, 0])
        side_length = np.linalg.norm([0, self.y1-self.y0])
        self.lattice_length = bottom_length / self.n_x_lattice
        self.n_y_lattice = int(side_length / self.lattice_length) + 1

        self.all_states = list(range(self.n_x_lattice * self.n_y_lattice))
        
        possible_states = np.zeros(len(latlons), dtype=np.int32)
        for i, latlon in enumerate(latlons):
            possible_states[i] = int(self._find_nearest_coord_from_latlon(latlon[::-1], self.all_states))
        self.possible_states = [int(state) for state in list(set(possible_states))]
        

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