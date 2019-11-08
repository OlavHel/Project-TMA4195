import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
from scipy.integrate import solve_ivp

newparams = {'figure.figsize': (8.0, 4.0), 'axes.grid': True,
             'lines.markersize': 8, 'lines.linewidth': 2,
             'font.size': 15}
plt.rcParams.update(newparams)

#function for 3D plotting
def plot_solution(x, t, U, txt='$S_g(x,t)$'):
    # Plot the solution of the heat equation
    fig = plt.figure()
    plt.clf()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t, x)
    # ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U, cmap=cm.coolwarm)
    ax.view_init(azim=30)  # Rotate the figure
    plt.xlabel('$t$',fontsize = 24)
    plt.ylabel('$x$',fontsize = 24)
    plt.title(txt, fontsize = 24)

#CONSTANTS
mu_g = 5*10**(-5) #
mu_w = 5*10**(-4) # Pascal*s
u =   1.5*10**(-6) #
phi = 0.4 #dimensionless
K = 10**(-13) # m**2
P_0 = 7.5*10**5 #Pascal, N/m**2

#Dimensionless constants

def dimless(X):
    bigL = X
    gamma = K*P_0/mu_w # Pa*m**2 / Pa*s = m**2 / s
    nu = bigL*u/gamma # m*(m/s) / ()
    bigT = bigL**2*phi/gamma
    return nu,bigT

epsilon = mu_g/mu_w

def f1(s):
    return s/(s+(1-s)*epsilon)

def f2(s):
    return s**2/(s**2+epsilon*(1-s)**2)

def f3(s):
    return 0

def g1(s):
    return f1(s)*(1-s)

def g2(s):
    return f2(s)*(1-s)**2

def g3(s):
    return 0

def fgplotter():
    n = 100
    xs = np.linspace(0,1,n)
    plt.figure('f1')
    plt.plot(xs,f1(xs))
    plt.show()
    plt.figure('g1')
    plt.plot(xs,g1(xs))
    plt.show()
    plt.figure('f2')
    plt.plot(xs,f2(xs))
    plt.show()
    plt.figure('g2')
    plt.plot(xs,g2(xs))
    plt.show()        

def s_dot_i(s,i,h,f,g,X):
    
    fux = ( f(s[i+1])-f(s[i-1]) )/(2*h)
    gsxx = (g( (s[i+1]+s[i])/2 )*(s[i+1] - s[i]) - g( (s[i]+s[i-1])/2 )*(s[i] - s[i-1]) )/(h**2)
    nu,bigT = dimless(X)
    return gsxx - nu*fux

def solver(X,xs,n,T,f,g,s,alpha,beta,Dirichlet = True, Neumann = False):
    h = X/n
    t = [0,T]
    if Dirichlet: #dirichletbetingelser
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g,X)
            s0[0] = s_dot_i(np.insert(s,0,alpha),1,h,f,g,X)
            s0[-1] = s_dot_i(np.insert(s,-1,beta),-2,h,f,g,X)
            return s0
        sol = solve_ivp(s_zero, t, s[1:-1], "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        U_sol = np.zeros((n,len(t_list)))
        U_sol[1:-1] = U_grid
        U_sol[0] = alpha
        U_sol[-1] = beta
        plot_solution(xs,t_list/T,U_sol)
    elif Neumann: #Neumannbetingelser, med den deriverte lik 0
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g,X)
            s0[0] = s_dot_i(np.insert(s,0,s[0]),1,h,f,g,X)
            s0[-1] = s_dot_i(np.insert(s,-1,s[-1]),-2,h,f,g,X)
            return s0
        sol = solve_ivp(s_zero, t, s, "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        plot_solution(xs,t_list/T,U_grid)
    else: #dirichlet i X = 0, neumann i enden
        def s_zero(t,s):
            s0 = np.empty_like(s)
            for i in range(1,len(s0)-1):
                s0[i] = s_dot_i(s,i,h,f,g,X)
            s0[0] = s_dot_i(np.insert(s,0,alpha),1,h,f,g,X)
            s0[-1] = s_dot_i(np.insert(s,-1,s[-1]),-2,h,f,g,X)
            return s0
        sol = solve_ivp(s_zero, t, s[1:], "RK23")
        t_list = sol.t #t-values from the ODE solver
        U_grid = sol.y #U-values from the ODE solver
        U_sol = np.zeros((n,len(t_list)))
        U_sol[1:] = U_grid
        U_sol[0] = alpha
        plot_solution(xs,t_list/T,U_sol)
    return 0

def question7():    
    X = 2.3 #x* = X times dimentionless x x* = Lx
    n = 50 #number of points
    T = 1 #time
    alpha = 1
    beta = 0
    
    xs = np.linspace(0,X,n)
    s = np.zeros(n)
    s[:] = beta
    s[0] = alpha
    solver(X,xs,n,T,f1,g1,s,alpha,beta,Dirichlet = False,Neumann = False)

def question17(case1 = True):
    n = 50 #number of points
    T = 1 #time
    alpha = 1 #initial vales
    beta = 0 # --||--
    if case1:
        X = 1 #x* = X times dimentionless x
        xs = np.linspace(0,X,n)
        b0 = 0 #boundary values
        b1 = 0 #--||--
        s = np.zeros(n)
        s[:] = beta
        s[int(len(xs)*0.47):int(len(xs)*0.53)] = alpha
        
        solver(X,xs,n,T,f3,g2,s,b0,b1,Dirichlet = False,Neumann = True)
    else:
        X = 1.15 #x* = X times dimentionless x
        xs = np.linspace(0,X,n)
        b0 = 1 #boundary values
        b1 = 0 #--||--       
        s = np.zeros(n)
        s[:] = beta
        s[0] = alpha
        
        solver(X,xs,n,T,f2,g2,s,b0,b1,Dirichlet = False,Neumann = False)

#fgplotter()
question7()
question17()
question17(False)