"""Evaluates circle formation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def calc_circle(pos, no_fish, test_run):
	"""Fits a 2D circle to X,Y data points using the Coope method and linear least squares:
	
	(X-Xc)**2 + (Y-Yc)**2 = R**2
	2 Xc X + 2 Yc Y + R**2 - Xc**2 - Yc**2 = X**2 + Y**2
	AX + BY + C = X**2 + Y**2
	
	Args:
	    pos (np-array): xyzp-positions of each fish, [0:end_of_time, 0:no_fish*4]
	    no_fish (int): number of fish in this circle
	    test_run (int): index of current test run
	
	Returns:
	    floats: fitted circle parameters and convergence time
	"""

	no_samples = 240 # sample and check for convergence
	sample_increment = np.floor(pos.shape[0] / no_samples).astype(int)
	converged = False
	t_convergence = -1

	fig, ax = plt.subplots(4,3)
	ax = ax.ravel()

	for ii in range(no_samples):
		# Input
		pos_ii = np.reshape(pos[ii*sample_increment,:], (-1,4))
		X = np.array([pos_ii[:,0]]).T
		Y = np.array([pos_ii[:,1]]).T

		no_robots = Y.shape[0]
		colors = cm.rainbow(np.linspace(0, 1, no_robots))

		# Linear least squares
		XY_sq = X**2 + Y**2
		XY_1 = np.concatenate((X, Y, np.ones((len(X),1))), axis=1)
		ABC = np.linalg.lstsq(XY_1, XY_sq, rcond=None)[0]

		# Circle parameters
		x_c = ABC[0]/2
		y_c = ABC[1]/2
		r_c = np.sqrt(ABC[2] + x_c**2 + y_c**2)

		# Error
		r = np.sqrt((X-x_c)**2 + (Y-y_c)**2)
		error = np.sum((r - r_c)**2)
		max_r = max(r)
		min_r = min(r)

		# Spread
		spread = np.sqrt((np.sum(X) - no_robots*x_c)**2 + (np.sum(Y) - no_robots*y_c)**2)
		
		# Convergence
		if converged == False and 1.1*r[0] > max_r[0] and 0.9*r[0] < min_r[0] and spread < 1.5*r_c:
			converged = True
			t_convergence = ii

		# Plot
		if ii % 20 == 0: # plot no_samples / no_subplots times
			ax[int(ii/20)].scatter(X, Y, color=colors, s=25)
			c_c = plt.Circle((x_c, y_c), r_c, color='r', linewidth=1.0, fill=False)
			c_max = plt.Circle((x_c, y_c), max_r, linestyle='dashed', linewidth=0.5, fill=False)
			c_min = plt.Circle((x_c, y_c), min_r, linestyle='dashed', linewidth=0.5, fill=False)
			ax[int(ii/20)].add_patch(c_c)
			ax[int(ii/20)].add_patch(c_max)
			ax[int(ii/20)].add_patch(c_min)
			ax[int(ii/20)].set_aspect('equal', adjustable='box')
			ax[int(ii/20)].axis('off')
	
	if test_run == 0:
		plt.savefig('plots/{}_{}.png'.format(no_fish, test_run), dpi = 1000, bbox_inches = "tight", pad_inches = 1, orientation = 'landscape')
	if t_convergence == -1:
		plt.savefig('plots/{}_{}_failed.png'.format(no_fish, test_run), dpi = 1000, bbox_inches = "tight", pad_inches = 1, orientation = 'landscape')

	return x_c, y_c, r_c, error, max_r, min_r, t_convergence
