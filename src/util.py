import numpy as np

query = np.loadtxt("data/locQuery.txt")
n = 2500

def load_traj(datadir):
    temp = np.loadtxt(datadir)
    traj = []
    for i in temp:
        traj.append(int(i))
    return traj

def oh2id(oh_loc):
    return np.where(oh_loc==1)[0][0]

def ohs2coors(oh_locs):
    return np.dot(self.query, oh_locs.T).T
    
def make_oh_traj(traj):
    oh_traj = []
    for i in traj:
        traj_ = np.zeros((1,n))
        traj_[0,i] = 1
        oh_traj.append(traj_)
    return np.array(oh_traj).reshape(-1, 2500)

def make_cell_traj(traj):
    oh_traj = make_oh_traj(traj)
    cell_traj = []
    for i in oh_traj:
        cell = np.dot(query, i.T) + 0.5
        cell = cell.reshape(-1)
        cell_traj.append(cell)
    cell_traj = np.array(cell_traj)
    return np.array(cell_traj)