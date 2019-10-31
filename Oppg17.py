import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
from scipy.integrate import solve_ivp

#CONSTANTS

phi = 1
T = 1 #max time
X = 1 # max value for x
M = 100 #number of timesteps
N = 100 #number of spatial steps

timeArray = np.linpace(0,T,M)
spaceArray = np.linspace(0,X,N)

#function f
def f(s):
    return 0
#function g
def g(s):
    return 0
