from __future__ import division
from filterpy.kalman import unscented_transform
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Q_discrete_white_noise

import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import scipy 
import math
from numpy.random import randn


def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    F = np.array([[1,dt,0,0], 
                  [0,1,0,0], 
                  [0,0,1,dt], 
                  [0,0,0,1]]
                ,dtype=float)
    return np.dot(F,x)


def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])



dt = 0.1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

kf.x = np.array([-1., 1., -1., 1]) # initial stateo
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)


zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements

for z in zs:
    kf.predict()
    kf.update(z)
    print(kf.x,'log-likelihood',kf.log_likelihood)

print(kf)