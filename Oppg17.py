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

#CONSTANTS
mu_g = 5*10**(-5)
mu_w = 5*10**(-4)
u =  1.5*10**(-6)
phi = 0.4
K = 10**(-13)
P_0 = 7.5*10**6

epsilon = mu_g/mu_w

#function for 3D plotting
def plot_solution(x, t, U, txt='Solution'):
    # Plot the solution of the heat equation
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t, x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)  # Rotate the figure
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt)

def f1(s):
    return s/(s+(1-s)*epsilon)

def f2(s):
    return s**2/(s**2+epsilon*(1-s)**2)

# f3 og g3 er kun for å teste script, gir transport equation med travelling waves for riktige startvilkår
def f3(s):
    return s

def dpc(s):
    return -P_0

def g1(s):
    return -f1(s)*K*(1-s)/mu_w*dpc(s)

def g2(s):
    return -f2(s)*K*(1-s)**2/mu_w*dpc(s)

def g3(s):
    return 0

def s_dot_i(s,i,h,f,g):
    
    fux = u*( f(s[i+1])-f(s[i-1]) )/(2*h)
    gsxx = (g( (s[i+1]+s[i])/2 )*(s[i+1] - s[i]) - g( (s[i]+s[i-1])/2 )*(s[i] - s[i-1]) )/(h**2)
    return (gsxx - fux)/phi

def solver(X,xs,n,T,f,g,s,alpha,beta,Dirichlet = True, Neumann = False):
    h = X/n
    t = [0,T]
    if Dirichlet: #dirichletbetingelser
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g)
            s0[0] = s_dot_i(np.insert(s,0,alpha),1,h,f,g)
            s0[-1] = s_dot_i(np.insert(s,-1,beta),-2,h,f,g)
            return s0
        sol = solve_ivp(s_zero, t, s[1:-1], "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        U_sol = np.zeros((n,len(t_list)))
        U_sol[1:-1] = U_grid
        U_sol[0] = alpha
        U_sol[-1] = beta
        plot_solution(xs,t_list,U_sol)
    elif Neumann: #Neumannbetingelser, med den deriverte lik 0
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g)
            s0[0] = s_dot_i(np.insert(s,0,s[0]),1,h,f,g)
            s0[-1] = s_dot_i(np.insert(s,-1,s[-1]),-2,h,f,g)
            return s0
        sol = solve_ivp(s_zero, t, s, "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        plot_solution(xs,t_list,U_grid)
    else: #dirichlet i X = 0, neumann i enden
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g)
            s0[0] = s_dot_i(np.insert(s,0,alpha),1,h,f,g)
            s0[-1] = s_dot_i(np.insert(s,-1,s[-1]),-2,h,f,g)
            return s0
        sol = solve_ivp(s_zero, t, s[1:], "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        U_sol = np.zeros((n,len(t_list)))
        U_sol[1:] = U_grid
        U_sol[0] = alpha
        plot_solution(xs,t_list,U_sol)
    return 0

def question7():    
    X = 1 #length in x-direction
    n = 100 #number of points
    T = 50 #time
    alpha = 1
    beta = 0
    
    xs = np.linspace(0,X,n)
    s = np.zeros(n)
    s[:] = beta
    s[0] = alpha
    solver(X,xs,n,T,f1,g1,s,alpha,beta,Dirichlet = False,Neumann = False)

def question17(case1 = True):
    X = 1 #length in x-direction
    n = 100 #number of points
    T = 100 #time
    alpha = 1 #initial vales
    beta = 0 # --||--
    xs = np.linspace(0,X,n)
    if case1:       
        b0 = 0 #boundary values
        b1 = 0 #--||--
        s = np.zeros(n)
        s[:] = beta
        s[int(len(xs)*0.47):int(len(xs)*0.53)] = alpha
        
        solver(X,xs,n,T,f2,g2,s,b0,b1,Dirichlet = False,Neumann = True)
    else:
        b0 = 1 #boundary values
        b1 = 0 #--||--       
        s = np.zeros(n)
        s[:] = beta
        s[0] = alpha
        
        solver(X,xs,n,T,f2,g2,s,b0,b1,Dirichlet = False,Neumann = False)

question7()

question17()
    
question17(False)