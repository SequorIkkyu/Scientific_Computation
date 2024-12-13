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
# you may use numpy, scipy, and matplotlib as needed


# ---------------------------- Code for Part 1 ----------------------------#
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
import scipy
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
        dXdt[2 : n - 1] = X[2 : n - 1] * (
            1 - g * X[0 : n - 3] - X[2 : n - 1] - g * X[3:n]
        )
        dXdt[-1] = X[-1] * (1 - g * X[-3] - X[-1] - g * X[0])
        return dXdt

    eventus = root(
        model,
        1 / (2 * g + 1) + 0.1 * np.random.rand(n),
        method="krylov",
        args=(n, g, mu),
    )
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

    matM = scipy.linalg.expm(M * T)

    U, Sigma, VT = np.linalg.svd(matM)

    eratio = Sigma[0] ** 2

    xtilde0 = VT[0, :]

    xbar = x_bar

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
    xbar, xtilde0, eratio_linear= part1q1a(n, g, mu, T)

    # Results from simulate1
    t, x = simulate1(n=n, X0=xbar+(1e-5)*xtilde0.real, tf=T, g=g, mu=mu, flag_display=False)

    perturbatio = x - xbar[:, None]  
    e_t0 = np.sum(perturbatio[:, 0] ** 2)  
    e_tT = np.sum(perturbatio[:, -1] ** 2)  
    eratio_simulation = e_tT / e_t0  

    print(f"Linear analysis predicted energy ratio: {eratio_linear:.4f}")
    print(f"Simulation computed energy ratio: {eratio_simulation:.4f}")

    e_over_time = np.sum(perturbatio**2, axis=0)/e_t0
    plt.plot(t, e_over_time)
    plt.xlabel("Time")
    plt.ylabel("Perturbation Energy")
    plt.title("Perturbation Energy over Time")
    plt.show()

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
    Ts = np.linspace(0.01, 50, 100)
    rationes = []

    for T in Ts:
        xbar, xtilde0, eratio_linear= part1q1a(n, g, mu, T)
        rationes.append(eratio_linear)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(Ts, rationes, '.', label='Max e(T)/e(0)')
    ax1.set_xlabel('T')
    ax1.set_ylabel('Max e(T)/e(0)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Variation of Max Energy Growth Ratio with T')
    ax1.grid(True)

    fig, ax2 = plt.subplots(figsize=(8, 6))

    ax2.semilogy(Ts, rationes, ".", label="Max e(T)/e(0)")
    ax2.set_xlabel("T")
    ax2.set_ylabel("Max e(T)/e(0)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_title("Variation of Max Energy Growth Ratio with T")
    ax2.grid(True)

    return None #modify if needed

from scipy.signal import welch
from scipy.spatial import KDTree

def part1q2():
    """Part 1, question 2: 
    Add input/output if/as needed.
    """

    # add code here
    def figuram_designare(n, g, mu, T, seco):
        t, X = simulate1(n=n, X0=-1, tf=T, g=g, mu=mu, flag_display=False)
        t = t[seco:]
        X = X[:, seco:]

        # Plotting Time Evolution
        plt.figure(figsize=(12, 6))
        for i in range(n):
            plt.plot(t, X[i], label=f"x_{i}")
        plt.xlabel("Time")
        plt.ylabel("x values")
        plt.title(f"Time Evolution for n={n}, g={g}, mu={mu}")
        plt.grid(True)
        plt.show()

        # Plotting Power Spectral Density using Welch method
        plt.figure(figsize=(12, 6))
        for i in range(n):
            # Use Welch's method for Power Spectral Density estimation
            f, Pxx = welch(X[i], fs=1.0, nperseg=256, noverlap=128)
            plt.semilogy(f, Pxx, label=f"x_{i}")

        plt.xlabel("Frequency")
        plt.ylabel("Power Spectral Density")
        plt.title(f"Power Spectral Density (Welch) for n={n}, g={g}, mu={mu}")
        plt.grid(True)
        plt.show()

    # Function to create time-delay embedding for each vector
    def create_time_delay_embedding(data, m, tau):
        N = len(data)
        embedded = np.array(
            [data[i : i + m * tau : tau] for i in range(N - (m - 1) * tau)]
        )
        return embedded

    # Function to compute the correlation sum using KDTree
    def correlation_sum_with_kdtree(embedded_data, r):
        N = len(embedded_data)
        embedded_data = np.vstack(embedded_data)

        tree = KDTree(embedded_data)

        count = 0
        for i in range(N):
            count += len(tree.query_ball_point(embedded_data[i], r)) - 1

        return 2 * count / (N * (N - 1)) 

    def estimate_fractal_dimension(
        data, m=3, tau=1, r_min=0.1, r_max=10, num_r=50, downsample_factor=10
    ):
        embedded_data = create_time_delay_embedding(data, m, tau)

        downsampled_data = embedded_data[::downsample_factor]

        downsampled_data = np.vstack(downsampled_data)

        r_values = np.logspace(np.log10(r_min), np.log10(r_max), num_r)

        correlation_sums = []
        for r in r_values:
            correlation_sums.append(correlation_sum_with_kdtree(downsampled_data, r))

        plt.figure(figsize=(8, 6))
        plt.plot(np.log10(r_values), np.log10(correlation_sums), marker="o")
        plt.xlabel("log(r)")
        plt.ylabel("log(C(r))")
        plt.title("Correlation Sum for Fractal Dimension Estimation")
        plt.grid(True)
        plt.show()

        p = len(r_values)
        ordo = []
        for i in range(p - 5):
            ordo.append(
                (correlation_sums[i + 5] - correlation_sums[i])
                / (r_values[i + 5] - r_values[i])
            )

        slope = max(ordo)
        print(f"Estimated Correlation Dimension: {slope}")

    t, X = simulate1(n=9, X0=-1, tf=500, Nt=10000, g=1, mu=0, flag_display=False)
    estimate_fractal_dimension(X.T, m=3, tau=1, downsample_factor=20)

    t, X = simulate1(n=30, X0=-1, tf=500, Nt=10000, g=1, mu=0, flag_display=False)
    estimate_fractal_dimension(X.T, m=3, tau=1, downsample_factor=20)

    t, X = simulate1(n=59, X0=-1, tf=500, Nt=10000, g=1, mu=0, flag_display=False)
    estimate_fractal_dimension(X.T, m=3, tau=1, downsample_factor=20)

    figuram_designare(n=9, g=1, mu=0, T=500, seco=900)
    figuram_designare(n=20, g=1, mu=0, T=1000, seco=900)
    figuram_designare(n=59, g=1, mu=0, T=1000, seco=900)

    return None #modify if needed


# ---------------------------- End code for Part 1 ----------------------------#


# ---------------------------- Code for Part 2 ----------------------------#
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

from scipy.linalg import solve_banded

def part2q1(f,h):
    """
    Part 2, question 1
    Input:
        f: m x n array whose 1st and 2nd derivatives will be computed with respect to x
    Output:
        df, d2f: m x n arrays containing 1st and 2nd derivatives of f with respect to x
        computed with implicit fd scheme
    """
    # use code below if/as needed
    m,n = f.shape
    h = 1/(n-1)
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    df = np.zeros_like(f) #modify as needed
    d2f = np.zeros_like(f) #modify as needed

    # Add code here
    # Coefficients for equations
    L1 = [7, h, 16, 0, 7, -h]
    L2 = [-9, -h, 0, 8 * h, 9, -h]
    L1b = [1, 0, 2, -h]
    L2b = [0, h, -6, 5 * h]
    L1b2 = [2, h, 1, 0]
    L2b2 = [-6, -5 * h, 0, -h]

    eqn = 2 * n
    lower = 3
    upper = 3
    ab = np.zeros((lower + upper + 1, eqn))

    # Band matrix
    for i in range(n):
        if i == 0:
            for j, val in enumerate(L1b):
                diag = upper + (0 - j)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, j] += val

            for j, val in enumerate(L2b):
                diag = upper + (1 - j)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, j] += val

        elif i == n - 1:
            for j, val in enumerate(L1b2):
                col = eqn - 4 + j
                diag = upper + ((2 * n - 2) - col)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val

            for j, val in enumerate(L2b2):
                col = eqn - 4 + j
                diag = upper + ((2 * n - 1) - col)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val

        else:
            for j, val in enumerate(L1):
                col = 2 * i - 2 + j
                diag = upper + ((2 * i) - col)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val

            for j, val in enumerate(L2):
                col = 2 * i - 2 + j
                diag = upper + ((2 * i + 1) - col)
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val

    # RHS coefficients
    c31, c22 = 15 / h, 24 / h
    cb11, cb21, cb31 = -3.5 / h, 4 / h, -0.5 / h
    cb12, cb22, cb32 = 9 / h, -12 / h, 3 / h

    df = np.zeros_like(f)
    d2f = np.zeros_like(f)

    for i in range(m):
        b = np.zeros(eqn)
        for j in range(n):
            if j == 0:
                b[0] += cb11 * f[i, j] + cb21 * f[i, j + 1] + cb31 * f[i, j + 2]
                b[1] += cb12 * f[i, j] + cb22 * f[i, j + 1] + cb32 * f[i, j + 2]

            elif j == n - 1:
                b[2 * j] += -(cb11 * f[i, j] + cb21 * f[i, j - 1] + cb31 * f[i, j - 2])
                b[2 * j + 1] += -(
                    cb12 * f[i, j] + cb22 * f[i, j - 1] + cb32 * f[i, j - 2]
                )

            else:
                b[2 * j] += c31 * (f[i, j + 1] - f[i, j - 1])
                b[2 * j + 1] += c22 * (f[i, j - 1] - 2 * f[i, j] + f[i, j + 1])

        res_x = solve_banded((lower, upper), ab, b)

        df[i, :] = res_x[::2]
        d2f[i, :] = res_x[1::2]
    return df, d2f

import time

def part2q2():
    """
    Part 2, question 2
    Add input/output as needed

    """

    def f1(x, y):
        return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def df1_dx(x, y):
        return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def f2(x, y):
        return np.exp(-100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    def df2_dx(x, y):
        return -200 * (x - 0.5) * np.exp(-100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))

    # Multiscale function
    def f_multiscale(x, y):
        return f1(x, y) + f2(x, y)

    def df_multiscale_dx(x, y):
        return df1_dx(x, y) + df2_dx(x, y)

    grid_sizes = [50, 100, 200, 300, 400]
    results = []

    for n in grid_sizes:
        m = n
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, m)
        X, Y = np.meshgrid(x, y)

        f_values = f_multiscale(X, Y)
        df_exact = df_multiscale_dx(X, Y)

        df_part2q1, _ = part2q1(f_values, 1 / (n - 1))
        df_fd2, _ = fd2(f_values)

        # Compute errors
        central_slice = slice(n // 4, 3 * n // 4)
        error_part2q1 = np.linalg.norm(
            df_part2q1[central_slice, central_slice]
            - df_exact[central_slice, central_slice]
        ) / np.linalg.norm(df_exact[central_slice, central_slice])

        error_fd2 = np.linalg.norm(
            df_fd2[central_slice, central_slice]
            - df_exact[central_slice, central_slice]
        ) / np.linalg.norm(df_exact[central_slice, central_slice])

        # Measure computational time
        start_time = time.time()
        part2q1(f_values, 1 / (n - 1))
        time_part2q1 = time.time() - start_time

        start_time = time.time()
        fd2(f_values)
        time_fd2 = time.time() - start_time

        results.append((n, error_part2q1, error_fd2, time_part2q1, time_fd2))

    grid_sizes = [r[0] for r in results]
    errors_part2q1 = [r[1] for r in results]
    errors_fd2 = [r[2] for r in results]
    times_part2q1 = [r[3] for r in results]
    times_fd2 = [r[4] for r in results]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(grid_sizes, errors_part2q1, label="part2q1 Error", marker="o")
    plt.plot(grid_sizes, errors_fd2, label="fd2 Error", marker="s")
    plt.xlabel("Grid size (n)")
    plt.ylabel("Relative Error")
    plt.title("Accuracy Comparison")
    plt.legend()

    # Efficiency plot
    plt.subplot(1, 2, 2)
    plt.plot(grid_sizes, times_part2q1, label="part2q1 Time", marker="o")
    plt.plot(grid_sizes, times_fd2, label="fd2 Time", marker="s")
    plt.xlabel("Grid size (n)")
    plt.ylabel("Execution Time (s)")
    plt.title("Efficiency Comparison")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return None #modify as needed


# ---------------------------- End code for Part 2 ----------------------------#

if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call functions used to generate the figures included in your report.
