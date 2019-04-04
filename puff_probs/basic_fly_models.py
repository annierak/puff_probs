import numpy as np
from utility import compute_Gaussian,compute_sq_puff_val


class Puff(object): #r_sq becomes the square's radius
    def __init__(self,x,y,r_sq,velocity):
        self.x = x
        self.y = y
        self.r_sq = r_sq
        self.velocity = velocity
    def update(self,dt):
        self.x += self.velocity[0]*dt
        self.y += self.velocity[1]*dt

class Flies(object):
    def __init__(self,x,y,velocity,grid_prob,ampl_const,puffs,
        detection_threshold,conc_fun,use_grid=True):
        self.x = x
        self.y = y
        self.velocity = velocity
        self.grid_prob = grid_prob
        self.use_grid = use_grid
        self.mask_caught = np.zeros(np.shape(self.x)).astype(bool)
        self.mask_newly_caught = np.zeros(np.shape(self.x)).astype(bool)
        self.ampl_const = ampl_const
        self.puffs = puffs
        self.detection_threshold = detection_threshold
        self.conc_fun = conc_fun
    def update(self,dt,t,conc_info,conc_sample_rate=None,xmin=None,ymin=None):
        #update positions
        self.x += self.velocity[0]*dt
        self.y += self.velocity[1]*dt

        #find the concentration value for the flies
        if self.use_grid:
            #conc grid pre-computed; conc_info is a conc array
            grid_inds_x = np.floor((self.x-xmin)/conc_sample_rate).astype(int)
            grid_inds_y = np.floor((self.y-ymin)/conc_sample_rate).astype(int)
            try:
                fly_concs = conc_info[grid_inds_y,grid_inds_x]
            except(IndexError):
                fly_concs = np.zeros_like(grid_inds_y)

        else:
            #compute values for each fly anew; conc_info is the puffs list
            try:
                px,py,r_sq = np.array([(puff.x, puff.y, puff.r_sq) for puff in self.puffs]).T
            except(AttributeError): #adjustment to make compatible with pompy plumes
                puffs_reshaped = conc_info.reshape(-1,conc_info.shape[-1])
                px, py, _, r_sq = puffs_reshaped[~np.isnan(puffs_reshaped[:, 0]), :].T


            fly_concs = np.sum(self.conc_fun(self.ampl_const,
                px[:,None],py[:,None],r_sq[:,None],self.x[None,:],self.y[None,:]),axis=0)

        #Determine which flies have been caught
        if self.grid_prob:
            fly_concs[fly_concs>1.] = 1.
            self.mask_newly_caught = np.random.binomial(
                1,fly_concs,size=np.shape(fly_concs)).astype(bool)
            self.mask_caught = (self.mask_caught | self.mask_newly_caught)
        else:
            self.mask_newly_caught = (fly_concs>self.detection_threshold)
            self.mask_caught = (self.mask_caught | self.mask_newly_caught)

        #Report when intersecting
        if np.sum(self.mask_newly_caught)>0.:
            print('t='+str(t)+', intersecting')
