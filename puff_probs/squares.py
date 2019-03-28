import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import time
na = np.newaxis
from basic_fly_models import Flies,Puff
from utility import compute_Gaussian,compute_sq_puff_val



ampl_const = 1.
detection_threshold = 0.5

dt = 0.01
t = 0.
release_rate = 2.

x_stretch=4.
xmin,xmax = 0.,x_stretch*16.
ymin,ymax = -1.1,1.1


sim_time = 100*60.
puff_x_vel,puff_y_vel =2.,0. #current setup requires these >=1.
puff_r_sq = .5
puff_buffer_time = xmax/puff_x_vel
stop_puffs = False

num_flies = 1000
fly_velocity = np.array([0.,-1.])

# fly_buffer_time = 60.
# fly_buffer_time = (release_rate*puff_x_vel - 2*puff_r_sq)/puff_x_vel
fly_buffer_time = release_rate

# print('fly buffer time: '+str(fly_buffer_time))
# time.sleep(1)



#puffs = [Puff(x,0,1,[1,0]) for x in np.arange(1,10,1)]
puffs = []

conc_sample_rate = 0.1
conc_sample_grid_x,conc_sample_grid_y = np.meshgrid(
    np.arange(xmin,xmax,conc_sample_rate),
    np.arange(ymin,ymax,conc_sample_rate)
)

plt.ion()
plt.show()


ax1 = plt.subplot(3,1,1)
ax1.set_xlim([xmin,xmax/x_stretch])
ax1.set_ylim([ymin,ymax])
ax1text = ax1.text(-0.2,0.5,'',transform=ax1.transAxes)
timer = ax1.text(0.5,1.2,'',transform=ax1.transAxes)

conc_im = ax1.imshow(np.zeros(np.shape(conc_sample_grid_x)),aspect=xmax/x_stretch/25.,
    extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=1.)
plt.colorbar(conc_im,ax=ax1)


fly_x_0,fly_y_0 = np.linspace(xmin,xmax-0.001,num_flies),ymax*np.ones(num_flies)

flies1 = Flies(fly_x_0,fly_y_0,fly_velocity,False,ampl_const,puffs,
    detection_threshold,compute_sq_puff_val,use_grid=False)
flies2 = Flies(np.copy(fly_x_0),np.copy(fly_y_0),fly_velocity,True,ampl_const,puffs,
    detection_threshold,compute_sq_puff_val,use_grid=True)


edgecolor_dict = {0 : 'red', 1 : 'white'}
facecolor_dict = {0 : 'red', 1 : 'white'}

fly_edgecolors = [edgecolor_dict[mode] for mode in flies1.mask_caught]
fly_facecolors =  [facecolor_dict[mode] for mode in flies1.mask_caught]

fly_dots1 = plt.scatter(flies1.x, flies1.y,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)



ax2 = plt.subplot(3,1,2)
ax2.set_xlim([xmin,xmax/x_stretch])
ax2.set_ylim([ymin,ymax])
ax2text = ax2.text(-0.2,0.5,'',transform=ax2.transAxes)


conc_cumul_im = ax2.imshow(np.zeros(np.shape(conc_sample_grid_x)),aspect=xmax/x_stretch/25.,
    extent=(xmin,xmax,ymin,ymax),vmin=0.,vmax=1.)
conc_cumul_grid = np.zeros(np.shape(conc_sample_grid_x))
plt.colorbar(conc_cumul_im,ax=ax2)

fly_dots2 = plt.scatter(flies2.x, flies2.y,
        edgecolor=fly_edgecolors,facecolor = fly_facecolors,alpha=0.9)

ax3 = plt.subplot(3,1,3)
ax3.set_xlim([xmin,xmax/x_stretch])
ax3.set_ylim([0.,1.])

# print(len(conc_cumul_grid[0,:]))
# print(ymax)
# print(int(ymax/conc_sample_rate))
# input()

line, = ax3.plot(np.arange(xmin,xmax,conc_sample_rate),
    conc_cumul_grid[int(conc_sample_rate*ymax),:])


while t<sim_time:
    print(t)

    if stop_puffs:
        if t<=puff_buffer_time:
            #add new puffs
            if np.abs(t%release_rate)<dt:
                new_puff = Puff(-puff_r_sq,0,puff_r_sq,[puff_x_vel,puff_y_vel])
                puffs.append(new_puff)
            #move puffs
            for puff in puffs:
                puff.update(dt)
                if puff.y>ymax:
                    puffs.remove(puff)
    else:
        if np.abs(t%release_rate)<dt:
            new_puff = Puff(-puff_r_sq,0,puff_r_sq,[puff_x_vel,puff_y_vel])
            puffs.append(new_puff)
        #move puffs
        for puff in puffs:
            puff.update(dt)
            if puff.y>ymax:
                puffs.remove(puff)


    if t>puff_buffer_time:
#    if t>0:
        #update concentration grid
        px,py,r_sq = np.array([(puff.x, puff.y, puff.r_sq) for puff in puffs]).T
        conc_grid =  compute_sq_puff_val(ampl_const,
            px[:,na,na],py[:,na,na],r_sq[:,na,na],
                conc_sample_grid_x[na,:,:],conc_sample_grid_y[na,:,:])
        conc_grid = np.sum(conc_grid,axis=0)

        #update cumulative concentration
        if t<puff_buffer_time+fly_buffer_time:
            conc_cumul_grid += (conc_grid>detection_threshold).astype(float)
            conc_cumul_prob = (conc_cumul_grid/((t-puff_buffer_time)/dt))#*(1/np.abs())


        #Once the average value has been computed, perform the necessary adjustments
        if np.abs(t-(puff_buffer_time+fly_buffer_time))<0.001:
            #step one: moving particle adjustments
            period_length = release_rate*puff_x_vel
            add_factor = (puff_x_vel*((2*puff_r_sq)/(np.abs(fly_velocity[1])))/(period_length))
            conc_cumul_prob[conc_cumul_prob>0.] = conc_cumul_prob[conc_cumul_prob>0.] + add_factor


            #step two: bernoulli trial adjustments
            n = (2*puff_r_sq)/(np.abs(fly_velocity[1]))/dt #Number of intersections
            conc_cumul_prob[conc_cumul_prob>0.] = 1.-(1.-conc_cumul_prob[conc_cumul_prob>0.])**(1./n)


        if t>puff_buffer_time+fly_buffer_time:
            conc_cumul_im.set_data(conc_cumul_prob)



            #time.sleep(0.1)
    #update flies
        if t>puff_buffer_time+fly_buffer_time:
            timer.set_text(str(t)+' s')
            line.set_ydata(conc_cumul_prob[int(ymax/conc_sample_rate),:])
            conc_im.set_data(conc_grid)
            plt.pause(0.001)


            flies1.update(dt,t,conc_grid,conc_sample_rate=conc_sample_rate)

            fly_dots1.set_offsets(np.c_[flies1.x,flies1.y])

            fly_edgecolors = [edgecolor_dict[mode] for mode in flies1.mask_caught]
            fly_facecolors =  [facecolor_dict[mode] for mode in flies1.mask_caught]

            fly_dots1.set_edgecolor(fly_edgecolors)
            fly_dots1.set_facecolor(fly_facecolors)
            # ax1text.set_text(
            # '{0}/{1}'.format(np.sum(flies1.mask_caught),len(flies1.mask_caught))
            # )
            ax1text.set_text(
            str(np.sum(flies1.mask_caught).astype(float)/len(flies1.mask_caught))[0:5]
            )


            flies2.update(dt,t,conc_cumul_prob,
                conc_sample_rate=conc_sample_rate,xmin=xmin,ymin=ymin)

            fly_dots2.set_offsets(np.c_[flies2.x,flies2.y])

            fly_edgecolors = [edgecolor_dict[mode] for mode in flies2.mask_caught]
            fly_facecolors =  [facecolor_dict[mode] for mode in flies2.mask_caught]

            fly_dots2.set_edgecolor(fly_edgecolors)
            fly_dots2.set_facecolor(fly_facecolors)

            # ax2text.set_text(
            # '{0}/{1}'.format(np.sum(flies2.mask_caught),len(flies1.mask_caught))
            # )
            ax2text.set_text(
            str(np.sum(flies2.mask_caught).astype(float)/len(flies2.mask_caught))[0:5]
            )


            if(np.sum(flies1.y<ymin)>1.):
                input()







    t+=dt



#raw_input()
