# extremely messy impelementation of
# metapopulation model as described in measles paper?
import pandas as pd
import numpy as np
import math
from scipy.stats import nbinom

# state matrix
num_patches=100
patch_pop=500
initial_inf=10

disease_matrix = np.zeros((num_patches,3))
disease_matrix[:,0]=patch_pop-initial_inf
disease_matrix[:,1]=initial_inf

disease_matrix_ts = [disease_matrix]
delta_I_ts = [disease_matrix[:,1]]
alpha=0.97
# disease infectiousness
beta = 0.5
recovery = 5 #recover after these number of timesteps?
ITERS =20 
for i in range(0,ITERS):
    last_matrix = disease_matrix_ts[i]
    S_t = last_matrix[:,0]
    I_t = last_matrix[:,1]
    R_t = last_matrix[:,2]
    print(S_t,I_t)
    # compute lambda for each row
    lmbda_t = beta*S_t*I_t**(alpha)/patch_pop
    # parameterize negative binomial (can be simplified..)
    variance = lmbda_t + (1/I_t)*(lmbda_t**2)
    p = lmbda_t/variance
    n = I_t
    print(p,n)
    # note: I think this is a delta (change in infections)...
    next_I = np.zeros(num_patches)
    # bad and lazy way for dealing with NaNs once number of infections reaches 0.
    # just do it before the probabilities are sampled
    # to avoid all these errors.
    for j in range(0,num_patches):
        if not math.isnan(p[j]) and not math.isnan(n[j]):
            next_I[j] = nbinom.rvs(n[j],p[j])
        else:
            next_I[j] = 0
    delta_I_ts.append(next_I)
    next_S = S_t - next_I
    if i >= (recovery-1): # infections that were introduced 'recovery' steps ago should now be removed from I
        recovered = delta_I_ts[-(recovery+1)]
    else:
        recovered = 0
    next_matrix = np.array([next_S,next_I+I_t-recovered,R_t+recovered]).T
    disease_matrix_ts.append(next_matrix)

np.array([disease_matrix_ts[k][:,1] for k in range(0,ITERS)])
pd.DataFrame(np.array([disease_matrix_ts[k][:,1] for k in range(0,ITERS)])).plot(legend=False)
