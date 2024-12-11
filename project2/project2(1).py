""" 
    Your college id here: 02028431
    Template code for project 2, contains 9 functions:
    simulate1: complete function for part 1
    part1q1a, part1q1b, part1q1c, part1q2: functions to be completed for part 1
    dualfd1,fd2: complete functions for part 2
    part2q1, part2q2: functions to be completed for part 2
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root
#you may use numpy, scipy, and matplotlib as needed


#---------------------------- Code for Part 1 ----------------------------#
def simulate1(n,X0=-1,Nt=2000,tf=1000,g=1.0,mu=0.0,flag_display=False):
    """
    Simulation code for part 1
    Input:
        n: number of ODEs
        X0: initial condition if n-element numpy array. If X0 is an integer, the code will set 
        the initial condition (otherwise an array is expected).
        Nt: number of time steps
        tf: final time
        g,mu: model parameters
        flag_display: will generate a contour plot displaying results if True
    Output:
    t,x: numpy arrays containing times and simulation results
    """

    def model(t, X, n, g, mu):
        """
        Defines the system of ODEs 
        
        Input:
        - t: time 
        - X: array of variables [x_0, x_1, ..., x_{n-1}]
        - g, mu: scalar model parameters
        - n: number of ODEs (size of the system)
         
        Returns:
        - dXdt: array of derivatives [dx_0/dt, dx_1/dt, ..., dx_{n-1}/dt]
        """
        dXdt = np.zeros(n)
        dXdt[0] = X[0]*(1-g*X[-2]-X[0]-g*X[1]+mu*X[-3])
        dXdt[1] = X[1]*(1-g*X[-1]-X[1]-g*X[2])
        dXdt[2:n-1] = X[2:n-1]*(1-g*X[0:n-3]-X[2:n-1]-g*X[3:n])
        dXdt[-1] = X[-1]*(1-g*X[-3]-X[-1]-g*X[0])
        return dXdt

    # Parameters
    t_span = (0, tf)  # Time span for the integration
    if type(X0)==int: #(modified from original version)
        X0 = 1/(2*g+1)+0.001*np.random.rand(n)  # Random initial conditions, modify if needed

    # Solve the system
    solution = solve_ivp(
        fun=model,
        t_span=t_span,
        y0=X0,
        args=(n, g, mu),
        method='BDF',rtol=1e-10,atol=1e-10,
        t_eval=np.linspace(t_span[0], t_span[1], Nt)  # Times to evaluate the solution
    )

    t,x = solution.t,solution.y #(in original version of code this line was inside if-block below)
    if flag_display:
        # Plot the solution
        plt.contour(t,np.arange(n),x,20)
        plt.xlabel('t') #(corrected axis labels)
        plt.ylabel('i')
    return t,x

import numpy as np
from scipy.optimize import root

def part1q1a(n, g, mu, T):
    """Part 1, question 1 (a)
    Use the variable inputs if/as needed.
    Input:
    n: number of ODEs
    g,mu: model parameters
    T: time at which perturbation energy ratio should be maximized
    
    Output:
    xbar: n-element array containing non-trivial equilibrium solution
    xtilde0: n-element array corresponding to computed initial condition
    eratio: computed maximum perturbation energy ratio
    """
    xbar = np.zeros(n)
    xtilde0 = np.zeros(n)
    eratio = 0.0

    def model(X, n, g, mu):
        dXdt = np.zeros(n)
        dXdt[0] = X[0] * (1 - g * X[-2] - X[0] - g * X[1] + mu * X[-3])
        dXdt[1] = X[1] * (1 - g * X[-1] - X[1] - g * X[2])
        dXdt[2:n-1] = X[2:n-1] * (1 - g * X[0:n-3] - X[2:n-1] - g * X[3:n])
        dXdt[-1] = X[-1] * (1 - g * X[-3] - X[-1] - g * X[0])
        return dXdt

    eventus = root(model, np.ones(n), method='krylov', args=(n, g, mu))
    x_bar = eventus.x

    B = np.zeros([n, n])
    for i in range(2, n - 1):
        B[i, i] = -1
        B[i, i - 2] = -g
        B[i, i + 1] = -g

    B[0, 0] = -1
    B[0, n - 2] = -g
    B[0, 1] = -g
    B[0, n - 3] = mu

    B[1, 1] = -1
    B[1, n - 1] = -g
    B[1, 2] = -g

    B[n - 1, n - 1] = -1
    B[n - 1, n - 3] = -g
    B[n - 1, 0] = -g

    M = np.zeros([n, n])

    for i in range(n):
        M[i, i] = 1 + np.matmul(B[i, :], x_bar)

    for i in range(n):
        for j in range(n):
            M[i, j] += B[i, j] * x_bar[i]

    eigenvalores, eigenvectores = np.linalg.eig(M)
    max_eigenvalue_index = np.argmax(np.real(eigenvalores))
    max_ratio = np.real(eigenvalores[max_eigenvalue_index])

    xbar = x_bar

    xtilde0 = eigenvectores[:, max_eigenvalue_index]

    eratio = max_ratio
    return xbar, xtilde0, eratio




def part1q1b():
    """Part 1, question 1(b): 
    Add input/output if/as needed.
    """
    #use/modify code below as needed:
    n = 19
    g = 1.2
    mu = 2.5
    T = 50

    #add code here
    

    return None #modify if needed


def part1q1c():
    """Part 1, question 1(c): 
    Add input/output if/as needed.
    """
    #use/modify code below as needed:
    n = 19
    g = 2
    mu = 0

    #add code here

    return None #modify if needed

def part1q2():
    """Part 1, question 2: 
    Add input/output if/as needed.
    """

    #add code here


    return None #modify if needed


#---------------------------- End code for Part 1 ----------------------------#


#---------------------------- Code for Part 2 ----------------------------#
def dualfd1(f):
    """
    Code implementing implicit finite difference scheme for special case m=1
    Implementation is not efficient.
    Input:
        f: n-element numpy array
    Output:
        df, d2f: computed 1st and 2nd derivatives
    """
    #parameters, grid
    n = f.size
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    
    #fd method coefficients
    #interior points:
    L1 = [7,h,16,0,7,-h]
    L2 = [-9,-h,0,8*h,9,-h]
    
    #boundary points:
    L1b = [1,0,2,-h]
    L2b = [0,h,-6,5*h]

    L1b2 = [2,h,1,0]
    L2b2 = [-6,-5*h,0,-h]

    A = np.zeros((2*n,2*n))
    #iterate filling a row of A each iteration
    for i in range(n):
        #rows 0 and N-1
        if i==0:
            #Set boundary eqn 1
            A[0,0:4] = L1b
            #Set boundary eqn 2
            A[1,0:4] = L2b
        elif i==n-1:
            A[-2,-4:] = L1b2
            A[-1,-4:] = L2b2
        else:
            #interior rows
            #set equation 1
            ind = 2*i
            A[ind,ind-2:ind+4] = L1
            #set equation 2
            A[ind+1,ind-2:ind+4] = L2

    #set up RHS
    b = np.zeros(2*n)
    c31,c22,cb11,cb21,cb31,cb12,cb22,cb32 = 15/h,24/h,-3.5/h,4/h,-0.5/h,9/h,-12/h,3/h
    for i in range(n):
        if i==0:
            b[i] = cb11*f[0]+cb21*f[1]+cb31*f[2]
            b[i+1] = cb12*f[0]+cb22*f[1]+cb32*f[2]
        elif i==n-1:
            b[-2] =-(cb11*f[-1]+cb21*f[-2]+cb31*f[-3])
            b[-1] = -(cb12*f[-1]+cb22*f[-2]+cb32*f[-3])
        else:
            ind = 2*i
            b[ind] = c31*(f[i+1]-f[i-1])
            b[ind+1] = c22*(f[i-1]-2*f[i]+f[i+1])
    out = np.linalg.solve(A,b)
    df = out[::2]
    d2f = out[1::2]
    return df,d2f


def fd2(f):
    """
    Computes the first and second derivatives with respect to x using second-order finite difference methods.
    
    Input:
    f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    
    Output:
     df, d2f: m x n arrays containing 1st and 2nd derivatives of f with respect to x
    """

    m,n = f.shape
    h = 1/(n-1)
    df = np.zeros_like(f) 
    d2f = np.zeros_like(f)
    
    # First derivative 
    # Centered differences for the interior 
    df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * h)

    # One-sided differences at the boundaries
    df[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * h)
    df[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * h)
    
    # Second derivative 
    # Centered differences for the interior 
    d2f[:, 1:-1] = (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / (h**2)
    
    # One-sided differences at the boundaries
    d2f[:, 0] = (2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]) / (h**2)
    d2f[:, -1] = (2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]) / (h**2)
    
    return df, d2f



def part2q1(f,h):
    """
    Part 2, question 1
    Input:
        f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    Output:
        df, d2f: m x n arrays containing 1st and 2nd derivatives of f with respect to x
        computed with implicit fd scheme
    """
    #use code below if/as needed
    m,n = f.shape
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    df = np.zeros_like(f) #modify as needed
    d2f = np.zeros_like(f) #modify as needed

    #Add code here


    return df,d2f 

def part2q2():
    """
    Part 2, question 2
    Add input/output as needed

    """

    return None #modify as needed


#---------------------------- End code for Part 2 ----------------------------#

if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call functions used to generate the figures included in your report.
