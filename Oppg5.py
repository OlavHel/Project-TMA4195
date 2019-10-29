import numpy as np
import matplotlib.pyplot as plt

mu_g = 5
mu_w = 100
u = 1
phi = 0.4
s_l = 0.8
s_r = 0.2

epsilon = mu_g/mu_w

def f(s):
    return s/(s+(1-s)*epsilon)

def df(s):
    return epsilon*1/(s+(1-s)*epsilon)**2

def dfinv(a):
    return (np.sqrt(epsilon*u/(phi*a))-epsilon)/(1-epsilon)

def curve(xs,t):
    u_l = df(s_l)*u/phi
    u_r = df(s_r)*u/phi
    y = np.empty_like(xs)
    if u_l <= u_r:
        left = xs <= u_l*t
        right = xs >= u_r*t
        y[left] = s_l
        y[right] = s_r
        y[~(left | right)] = dfinv(xs[~(left | right)]/t)
    elif u_l > u_r:
        u_shock = (f(s_l)-f(s_r))/(s_l-s_r)
        left = xs <= u_shock*t
        y[left] = s_l
        y[~left] = s_r
    return y

xs = np.linspace(-1,100,1000)
ts = np.linspace(0,10,3)

ys = curve(xs,ts[1])

plt.figure(1)
for t in ts:
    plt.plot(xs,curve(xs,t))
plt.show()




