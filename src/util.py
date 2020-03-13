import numpy as np
import os

#query = np.loadtxt(os.path.join(os.path.dir(__file__), "..", "data", "locQuery.txt"))

def load_traj(datadir):
    temp = np.loadtxt(datadir)
    traj = []
    for i in temp:
        traj.append(int(i))
    return traj

def oh2id(oh_loc):
    return np.where(oh_loc==1)[0][0]


def make_oh_traj(traj, n_locations):
    oh_traj = np.zeros((len(traj),n_locations))
    for i, state_no in enumerate(traj):
        oh_traj[i, state_no] = 1
    return oh_traj


"""
def ohs2coors(oh_locs):
    return np.dot(self.query, oh_locs.T).T

def make_cell_traj(traj):
    oh_traj = make_oh_traj(traj)
    cell_traj = []
    for i in oh_traj:
        cell = np.dot(query, i.T) + 0.5
        cell = cell.reshape(-1)
        cell_traj.append(cell)
    cell_traj = np.array(cell_traj)
    return np.array(cell_traj)
    
"""