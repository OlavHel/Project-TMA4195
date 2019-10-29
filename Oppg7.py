import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
from scipy.integrate import solve_ivp

newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 12}
plt.rcParams.update(newparams)

mu_g = 10
mu_w = 40
u = 1
phi = 0.4
s_l = 1
s_r = 0.1
K = 0.1
P_0 = 5

epsilon = mu_g/mu_w

def plot_solution(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation
    fig = plt.figure(2)
    plt.clf()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t, x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)  # Rotate the figure
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt);

def f(s):
    return s/(s+(1-s)*epsilon)

def dpc(s):
    return -P_0

def g(s):
    return -f(s)*K*(1-s)/mu_w*dpc(s)

def row_val(s,i,h):
    return -1/phi*(u*(f(s[i+1])-f(s[i-1]))/(2*h)-1/h**2*(g((s[i+1]+s[i])/2)*(s[i+1]-s[i])-g((s[i]+s[i-1])/2)*(s[i]-s[i-1])))

def one_row(t,s):
    h = X/n
    outval = np.empty_like(s)
    outval[0] = row_val(np.insert(s,0,alpha),1,h)
    outval[-1] = row_val(np.insert(s,-1,beta),-2,h)
    for i in range(1,len(s)-1):
        outval[i] = row_val(s,i,h)
    return outval

X = 10
n = 100
alpha = 0.9
beta = 0.2

xs = np.linspace(-1,X,n)
t = [0,1]
s = np.zeros(n-2)
s[xs[1:-1] <= 0] = alpha
s[xs[1:-1] > 0] = beta


sol = solve_ivp(one_row, t, s, "RK23")
t = sol.t
U = sol.y
#U = np.vstack((np.ones(len(U[0,:])),U))


print(U[:,-1])
#plot_solution(xs,t,U)

plot_solution(xs[1:-1],t,U)
plt.show()

