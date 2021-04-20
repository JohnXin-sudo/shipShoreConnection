from __future__ import division
from filterpy.kalman import unscented_transform
from filterpy.kalman import UnscentedKalmanFilter as UKF
import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
import scipy 
import math

x = 0.1 # initial state
# x = np.array([0.5,0,0,0]) # 修改
Q = 1 # process noise covariance
# Q = np.array([1e-9,1e-9,1e-9,1e-9]).T # 修改
R = 0.1 # measurement noise covariance
# R = 0.01 # 修改
tf = 500 # simulation length
# tf = 10000 # 修改
N = 200 # number of particles in the particle filter
#intial covariance
P = 2
# P = np.diag([1e-4,1e-4,1e-4,1e-4]) # 修改

scipy.random.seed(3)
# true state, observation
xArr = [x] # true state
yArr = [x**2/20 + math.sqrt(R)*scipy.random.normal()]

# Initialize the  Extended Kalman Filter.
xhat = x
xhatArr = [xhat]
PArr = [P]

# Initialize the unscented kalman filter.
kappa = 0.
w = UKF.weights(2, kappa)
p_ukf = np.array([[1, 0],
              [0, 1]])
ukf_mean = [x,x**2/20]
xhatukfArr = [x] # state estimates of the unscented Kalman Filter

# set the seed in order to duplicate the run exactly

# Initialize the particle filter.
xpart = x + math.sqrt(P)*scipy.random.normal(size=N)
xhatPart = x
xhatPartArr = [xhatPart] # state estimates of the Particle Filter

for k in range(tf):

    ################################################

    # System simulation
    x = 0.5*x + 25*x/(1 + x*x) + 8*math.cos(1.2*k) + math.sqrt(Q)*scipy.random.normal()
    y = x**2/20 + math.sqrt(R)*scipy.random.normal()

    ################################################

    # Extended Kalman filter
    F = 0.5 + 25*(1 - xhat**2)/(1 + xhat**2)**2
    P = F*P*F + Q
    H = xhat/10
    K = P*H*(H*P*H + R)**(-1)
    xhat = 0.5*xhat + 25*xhat/(1 + xhat**2) + 8*math.cos(1.2*(k))
    xhat = xhat + K*(y - xhat**2/20)
    P = (1 - K*H)*P

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

    # Particle filter
    xpartminus = 0.5*xpart + 25*xpart/(1 + xpart**2) + 8*math.cos(1.2*(k)) + math.sqrt(Q)*scipy.random.normal(size=N)
    ypart = xpartminus**2/20
    q=(1/math.sqrt(R)/math.sqrt(2*math.pi))*scipy.exp(-(y-ypart)**2/2/R)
    # Normalize the likelihood of each a priori estimate.
    qsum = scipy.sum(q)
    q = [i/qsum for i in q]
    # Resample.
    for i in range(N):
        u = scipy.random.uniform() # uniform random number between 0 and 1
        qtempsum = 0
        for j in range(N):
            qtempsum += q[j]
            if qtempsum >= u:
                xpart[i] = xpartminus[j]
                break

    # The particle filter estimate is the mean of the particles.
    xhatPart = scipy.mean(xpart)

    ################################################

    # Save data
    xArr.append(x) # true state
    yArr.append(y) # observation
    xhatArr.append(xhat) # state estimates of the Extended Kalman Filter
    PArr.append(P)
    xhatPartArr.append(xhatPart) # state estimates of the Particle Filter
    # xhatukfArr.append(ukf_mean[0])# state estimates of the unscented Kalman Filtero



t = range(tf+1)
plt.figure()
#plt.grid()
plt.title('State')
plt.plot(t, xArr, 'b', label='True')
plt.plot(t, xhatArr, '^k:', label='EKF')
plt.plot(t, xhatPartArr, 'or-', label='PF')
plt.plot(t, xhatukfArr, '.g-.',label='UKF')
plt.legend(loc='best')
plt.show()