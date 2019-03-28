import numpy as np

def compute_Gaussian(ampl_const,px,py,r_sq,x,y):
    return (
        ampl_const / r_sq**1.5 *
        np.exp(-((x - px)**2 + (y - py)**2 )/ (2 * r_sq))
    )

def compute_sq_puff_val(ampl_const,px,py,r_sq,x,y):
    return(
        (ampl_const*np.linalg.norm(np.array([x,y])-np.array([px,py]),np.inf,axis=0)<=r_sq
            ).astype(float)
    )

# def compute_sq_puff_val(px,py,r_sq,x,y):
#     print(np.shape(px))
#     print(np.shape(x))
#     return(
#         ampl_const*np.linalg.norm(np.array([x,y])-np.array([px,py]),axis=0)
#     )
