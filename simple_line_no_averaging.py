import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import time
na = np.newaxis


ampl_const = 1.
detection_threshold = 0.5

dt = 0.05
t = 0.
release_rate = 10.

xmin,xmax = 0,1000
ymin,ymax = -5,5


sim_time = 10*600.
puff_x_vel,puff_y_vel = .8,0.
puff_r_sq = 1.
puff_buffer_time = xmax/puff_x_vel


puff_width = 2*np.sqrt(2*puff_r_sq*(np.log(1./detection_threshold)-np.log(puff_r_sq**1.5)))

print(puff_width)


num_flies = 1000
fly_velocity = np.array([0.,-1.])


def compute_Gaussian(px,py,r_sq,x,y):
    return (
        ampl_const / r_sq**1.5 *
        np.exp(-((x - px)**2 + (y - py)**2 )/ (2 * r_sq))
    )

class Puff(object):
    def __init__(self,x,y,r_sq,velocity):
        self.x = x
        self.y = y
        self.r_sq = r_sq
        self.velocity = velocity
    def update(self,dt):
        self.x += self.velocity[0]*dt
        self.y += self.velocity[1]*dt

class Flies(object):
    def __init__(self,x,y,velocity,grid_prob,use_grid=True):
        self.x = x
        self.y = y
        self.velocity = velocity
        self.grid_prob = grid_prob
        self.use_grid = use_grid
        self.mask_caught = np.zeros(np.shape(self.x)).astype(bool)
    def update(self,dt,conc_info,conc_sample_rate=None):
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
            px,py,r_sq = np.array([(puff.x, puff.y, puff.r_sq) for puff in puffs]).T
            fly_concs = np.sum(compute_Gaussian(
                px[:,na],py[:,na],r_sq[:,na],self.x[na,:],self.y[na,:]),axis=0)

        #Determine which flies have been caught
        if self.grid_prob:
            fly_concs[fly_concs>1.] = 1.
            mask_newly_caught = np.random.binomial(
                1,fly_concs,size=np.shape(fly_concs)).astype(bool)
            self.mask_caught = (self.mask_caught | mask_newly_caught)
        else:
            mask_newly_caught = (fly_concs>detection_threshold)
            self.mask_caught = (self.mask_caught | mask_newly_caught)

#puffs = [Puff(x,0,1,[1,0]) for x in np.arange(1,10,1)]
puffs = []

conc_sample_rate = 0.1
conc_sample_grid_x,conc_sample_grid_y = np.meshgrid(
    np.arange(xmin,xmax,conc_sample_rate),
    np.arange(ymin,ymax,conc_sample_rate)
)

plt.ion()
plt.show()

fig = plt.figure()
ax = plt.subplot(3,1,1)
ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])
# ax.set_aspect('equal')

ps = [mpatches.Ellipse((puff.x, puff.y), puff.r, puff.r) for puff in puffs]
for p in ps:
    ax.add_patch(p)

ax1 = plt.subplot(3,1,2)
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymin,ymax])
ax1text = ax1.text(-0.1,0.5,'',fontsize=8,transform=ax1.transAxes)

# conc_im = ax1.imshow(np.zeros(np.shape(conc_sample_grid_x)),aspect=20,
#     extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=1.)
# plt.colorbar(conc_im,ax=ax1)


fly_x_0,fly_y_0 = np.linspace(xmin,xmax-1.,num_flies),ymax*np.ones(num_flies)

flies1 = Flies(fly_x_0,fly_y_0,fly_velocity,False,use_grid=False)


edgecolor_dict = {0 : 'red', 1 : 'white'}
facecolor_dict = {0 : 'red', 1 : 'white'}

fly_edgecolors = [edgecolor_dict[mode] for mode in flies1.mask_caught]
fly_facecolors =  [facecolor_dict[mode] for mode in flies1.mask_caught]

fly_dots1 = plt.scatter(flies1.x, flies1.y,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)




while t<sim_time:
    #add new puffs
    if np.abs(t%release_rate)<dt:
        new_puff = Puff(0,0,puff_r_sq,[puff_x_vel,puff_y_vel])
        puffs.append(new_puff)
        p=mpatches.Ellipse((new_puff.x, new_puff.y), puff_width, puff_width)
        ps.append(p)
        ax1.add_patch(p)

    #move puffs
    for p,puff in zip(ps,puffs):
        puff.update(dt)
        p.set_center((puff.x,puff.y))
        if puff.y>ymax:
            puffs.remove(puff)

    if t>puff_buffer_time:
        #update concentration grid
        # px,py,r_sq = np.array([(puff.x, puff.y, puff.r_sq) for puff in puffs]).T
        # conc_grid =  compute_Gaussian(
        #     px[:,na,na],py[:,na,na],r_sq[:,na,na],
        #         conc_sample_grid_x[na,:,:],conc_sample_grid_y[na,:,:])
        # conc_grid = np.sum(conc_grid,axis=0)
        # conc_im.set_data(conc_grid)

        #update flies
        flies1.update(dt,puffs,conc_sample_rate=conc_sample_rate)

        fly_dots1.set_offsets(np.c_[flies1.x,flies1.y])

        fly_edgecolors = [edgecolor_dict[mode] for mode in flies1.mask_caught]
        fly_facecolors =  [facecolor_dict[mode] for mode in flies1.mask_caught]

        fly_dots1.set_edgecolor(fly_edgecolors)
        fly_dots1.set_facecolor(fly_facecolors)
        ax1text.set_text(
        '{0}/{1}'.format(np.sum(flies1.mask_caught),len(flies1.mask_caught))
        )

        plt.pause(0.1)


    t+=dt



#raw_input()
