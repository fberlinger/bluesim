import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import umap

sns.set(style='white', context='poster', rc={'figure.figsize':(14,10)})

# load data from simulated experiment
data = np.loadtxt('./logfiles/200604_153757_data.txt', delimiter=',')

# 50 fish, 8 elements long state vector
# each line 1 fish, xyz_phi_pos xyz_phi_vel columns
no_fishes = 50
data_reshape = np.empty([no_fishes,8])
for ii in range(no_fishes):
	data_reshape[ii,:4] = data[40, 4*ii : 4*ii+4]
	data_reshape[ii,4:] = data[40, 4*no_fishes+4*ii : 4*no_fishes+4*ii+4]

# fit 8 dimensional state vector to 2D
fit = umap.UMAP()
u = fit.fit_transform(data_reshape)

plt.scatter(u[:,0], u[:,1])
plt.title('UMAP embedding of xyz_phi position and velolcity')
plt.show()