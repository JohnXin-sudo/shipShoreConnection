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
import pandas as pd
import json


def getMeasurementData(cols):
    dataframe = pd.read_csv('data//battery.csv')
    data = np.array(dataframe.get(cols)).reshape(1,-1) # 只要1x1的数据，观测数据为电压
    return data

def fx(x, dt):
    # state transition function - predict next state based
    # on constant velocity model x = vt + x_0
    A = np.array([[0,0,0,0], 
                  [0,-1./9.7154,0,0], 
                  [0,0,-1./66.7955,0], 
                  [0,0,0,-1./649.0073]]
                ,dtype=float)
    B = np.array([-1/(3600*31),0.0007/9.7154,0.0010/66.7955,0.0009/649.0073]).T                       
    return np.dot(A,x.T)


def hx(x):
    # measurement function - convert state into a measurement
    # where measurements are [x_pos, y_pos]
    return np.array([x[0], x[2]])


configs = json.load(open('config.json', 'r'))

dt = 1
# create sigma points to use in the filter. This is standard for Gaussian processes
points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)

kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)

kf.x = np.array([0.5, 0., 0., 0]) # initial stateo [0.5,0,0,0]
kf.P *= 0.2 # initial uncertainty
z_std = 0.1
kf.R = np.diag([1e-4,1e-4,1e-4,1e-4]) # 1 standard
kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.01**2, block_size=2)

measurementDataVoltage = getMeasurementData(configs['data']['columns'][1]) # 获取测量电压值
measurementDataCurrent = getMeasurementData(configs['data']['columns'][2]) # 获取测量电压值



for voltage,current in measurementDataVoltage,measurementDataCurrent:
    kf.predict()
    kf.update(z=voltage)
    print(kf.x,'log-likelihood',kf.log_likelihood)

print(kf)

x = np.array([0.5,0,0,0]) # initial state
Q = np.array([1e-9,1e-9,1e-9,1e-9]).T # process noise covariance
R = 0.01 # measurement noise covariance
P = np.diag([1e-4,1e-4,1e-4,1e-4]) # # intial covariance

tf = 10000 # simulation length

# set the seed in order to duplicate the run exactly
scipy.random.seed(3)

# true state, observation
xArr = [x] # true state
yArr = [x**2/20 + math.sqrt(R)*scipy.random.normal()]


# Initialize the unscented kalman filter.
kappa = 0.
w = UKF.weights(2, kappa)
p_ukf = np.array([[1, 0],
              [0, 1]])
ukf_mean = [x,x**2/20]
xhatukfArr = [x] # state estimates of the unscented Kalman Filter


for k in range(tf):

    ################################################

    # System simulation
    x = 0.5*x + 25*x/(1 + x*x) + 8*math.cos(1.2*k) + math.sqrt(Q)*scipy.random.normal()
    y = x**2/20 + math.sqrt(R)*scipy.random.normal()

    ################################################

    # Unscented Kalman filter
    ukf_mean = ukf_mean
    sigmas = UKF.sigma_points(ukf_mean,p_ukf, kappa)
    sigmas_f = np.zeros((5, 2))
    for i in range(5):
        sigmas_f[i,0] = 0.5 * sigmas[i, 0] + 25 * sigmas[i, 0] / (1 + sigmas[i, 0]* sigmas[i, 0]) + 8 * math.cos(1.2 * k) + math.sqrt(Q) * scipy.random.normal()
        sigmas_f[i,1] = sigmas[i, 0] ** 2 / 20 + math.sqrt(R) * scipy.random.normal()
    ukf_mean, p_ukf = unscented_transform(sigmas_f, w, w, np.zeros((2, 2)))

    ################################################

    # Save data
    xArr.append(x) # true state
    yArr.append(y) # observation

    xhatukfArr.append(ukf_mean[0])# state estimates of the unscented Kalman Filtero



t = range(tf+1)
plt.figure()
plt.grid()
plt.title('State')
plt.plot(t, xArr, 'b', label='True')
plt.plot(t, xhatukfArr, '.g-.',label='UKF')
plt.legend(loc='best')
plt.show()