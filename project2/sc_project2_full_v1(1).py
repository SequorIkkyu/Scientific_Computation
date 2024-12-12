""" 
    Your college id here:
    Template code for project 2, contains 9 functions:
    simulate1: complete function for part 1
    part1q1a, part1q1b, part1q1c, part1q2: functions to be completed for part 1
    dualfd1,fd2: complete functions for part 2
    part2q1, part2q2: functions to be completed for part 2
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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

def part1q1a(n,g,mu,T):
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
    #use/modify code below as needed:
    xbar = np.zeros(n)
    xtilde0 = np.zeros(n)
    eratio = 0.0 #should be modified below

    #add code here

    def equilibrium_system(X, n, g, mu):
        """System of equations for equilibrium (dx/dt = 0)."""
        F = np.zeros(n)
        F[0] = X[0] * (1 - g * X[-2] - X[0] - g * X[1] + mu * X[-3])
        F[1] = X[1] * (1 - g * X[-1] - X[1] - g * X[2])
        F[2:n-1] = X[2:n-1] * (1 - g * X[0:n-3] - X[2:n-1] - g * X[3:n])
        F[-1] = X[-1] * (1 - g * X[-3] - X[-1] - g * X[0])
        return F

    def compute_jacobian(X, n, g, mu):
        """Computes the Jacobian matrix at equilibrium X."""
        J = np.zeros((n, n))
        for i in range(n):
            if i == 0:
                J[i, i] = 1 - g * X[-2] - 2 * X[i] - g * X[1] + mu * X[-3]
                J[i, -2] = -g * X[i]
                J[i, 1] = -g * X[i]
                J[i, -3] = mu * X[i]
            elif i == 1:
                J[i, i] = 1 - g * X[-1] - 2 * X[i] - g * X[2]
                J[i, -1] = -g * X[i]
                J[i, 2] = -g * X[i]
            elif i == n-1:
                J[i, i] = 1 - g * X[-3] - 2 * X[i] - g * X[0]
                J[i, -3] = -g * X[i]
                J[i, 0] = -g * X[i]
            else:
                J[i, i] = 1 - g * X[i-2] - 2 * X[i] - g * X[i+1]
                J[i, i-2] = -g * X[i]
                J[i, i+1] = -g * X[i]
        return J

    def max_growth_perturbation(n, g, mu, T, xbar):
        """
        Compute the initial perturbation xtilde0 that maximizes e(T)/e(0).
        We have linearized system: dtilde{x}/dt = J tilde{x}.
        Solution: tilde{x}(T) = exp(J T)*tilde{x}(0).

        We want to maximize (tilde{x}(T)^T tilde{x}(T)) / (tilde{x}(0)^T tilde{x}(0)).
        The maximum energy growth is given by the largest singular value of exp(J T).
        The vector that achieves it is the corresponding singular vector of exp(J T).
        """
        # Step 2: Compute the Jacobian at the equilibrium
        J = compute_jacobian(xbar, n, g, mu)
        # Step 3: Compute the matrix exponential e^(J*T)
        M = expm(J*T)  # matrix exponential

        # # Step 4: Perform SVD on e^(J*T)
        U,Sigma,VT = svd(M)
        # The maximum energy growth ratio is the square of the largest singular value
        eratio = Sigma[0]**2  # energy ratio = ||M xtilde0||^2 / ||xtilde0||^2 = sigma_max^2
        # Corresponding initial perturbation vector is the first row of V^T (or column of V)
        xtilde0 = VT[0,:]

        # Normalize xtilde0 so that initial energy e(0)=1 for convenience
        xtilde0 = xtilde0 / np.linalg.norm(xtilde0)
        return xtilde0, eratio

    # Step 1: Find the equilibrium solution
    X0_guess = 1 / (2 * g + 1) + 0.1 * np.random.rand(n)
    solution = root(equilibrium_system, X0_guess, args=(n, g, mu), method='hybr')
    if not solution.success:
        raise ValueError("Equilibrium solver failed.")
    xbar = solution.x
    print(f"non-trivial equilibrium: {xbar}")

    optimal_perturbation, max_energy_ratio = max_growth_perturbation(n, g, mu, T, xbar)
    print(f"optimal perturbation: {optimal_perturbation}")

    return xbar, optimal_perturbation, max_energy_ratio



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
    # Get equilibrium and perturbation
    xbar, xtilde0, eratio_linear = part1q1a(n, g, mu, T)

    # Now test by simulating the full nonlinear system
    # Initial condition: x(0) = xbar + small*xtilde0
    small = 1e-6
    X0 = xbar + small*xtilde0
    t, x = simulate1(n, X0=X0, Nt=2000, tf=T, g=g, mu=mu)

    # Compute energy at 0 and T
    e0 = np.sum((x[:,0]-xbar)**2)
    eT = np.sum((x[:,-1]-xbar)**2)
    eratio_nonlinear = eT/e0

    # Print or plot comparison
    print("Linear prediction of max e(T)/e(0) in part1q1a:", eratio_linear)
    print("Nonlinear simulation e(T)/e(0) in simulate1:", eratio_nonlinear)

    # They won't be exactly the same, but should be consistent in indicating growth.
    return eratio_linear, eratio_nonlinear
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
    # We vary T in (0, 50]
    Ts = np.linspace(0.1, 50, 50)
    ratios = []
    xbar = equilibrium_solution(n, g, mu)

    for T in Ts:
        xtilde0, eratio = max_growth_perturbation(n, g, mu, T, xbar)
        ratios.append(eratio)

    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot max energy growth ratio
    ax1.plot(Ts, ratios, 'o-', label='Max e(T)/e(0)', color='blue')
    ax1.set_xlabel('T')
    ax1.set_ylabel('Max e(T)/e(0)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Variation of Max Energy Growth Ratio with T')
    ax1.grid(True)

    return Ts, ratios
 
def part1q2():
    """Part 1, question 2: 
    Add input/output if/as needed.
    """

    #add code here
    from scipy.signal import welch

    def analyze_case(n, g, mu, Nt, tf):
        """
        Analyzes a single case and produces plots and statistics.
        """
        X0=-1
        # t, X = simulate1(n=n, X0=X0, tf=T, g=g, mu=mu, flag_display=True)
        t, X = simulate1(n=n, X0=X0, tf=T, g=g, mu=mu, flag_display=False)
        # Discard the transient (e.g., first 10% of time)
        transient_cutoff = int(0.1 * len(t))
        t = t[transient_cutoff:]
        X = X[:, transient_cutoff:]

        # Split the figures for better visualization
        num_figures = (n + 19) // 20  # Divide trajectories into groups of 20
        for fig_idx in range(num_figures):
            plt.figure(figsize=(12, 6))
            start_idx = fig_idx * 20 
            end_idx = min(start_idx + 20, n)
            for i in range(start_idx, end_idx):
                plt.plot(t, X[i], label=f"x_{i}")
            plt.xlabel("Time")
            plt.ylabel("x values")
            plt.title(f"Time Evolution for n={n}, g={g}, mu={mu}, Group {fig_idx + 1}")
            plt.grid(True)
            # plt.legend()
            plt.show()

        # Compute and Plot PSD for all trajectories on one figure
        fs = 1 / (t[1] - t[0])  # Sampling frequency
        plt.figure(figsize=(8, 6))  # Single figure for all PSDs

        for i in range(n):
            freqs, psd = welch(X[i], fs=fs, nperseg=256)  # Welch method
            plt.semilogy(freqs, psd, label=f"x_{i}")

        # Add labels, legend, and grid to the single plot
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.title(f"Power Spectral Density for all trajectories (n={n})")
        plt.grid(True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')  # Legend outside the plot
        plt.tight_layout()  # Adjust layout to accommodate the legend
        plt.show()

        return t, X

    g = 1  # gamma
    mu = 0  # mu

    # Case A: n = 9
    print("Analyzing Case A: n = 9")
    t_A, X_A = analyze_case(n=9, g=g, mu=mu, Nt=1000, tf=500)

    # Case B: n = 20
    print("Analyzing Case B: n = 20")
    t_B, X_B = analyze_case(n=20, g=g, mu=mu, Nt=2000, tf=800)

    # Case C: n = 59
    print("Analyzing Case C: n = 59")
    t_C, X_C = analyze_case(n=59, g=g, mu=mu, Nt=3000, tf=1000)

    # Return results for further analysis
    return (t_A, X_A), (t_B, X_B), (t_C, X_C)


#----------------------------
# ----------------------------#


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
     df, d2f: m x n arrays conaining 1st and 2nd derivatives of f with respect to x
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
        df, d2f: m x n arrays conaining 1st and 2nd derivatives of f with respect to x
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
        # # inefficient implementation use dualfd1
    # for i in range(m):
    #     df[i, :], d2f[i, :] = dualfd1(f[i, :])

    # efficient using sparse matrix, solve_banded

    # Finite Difference Method Coefficients
    # Interior points:
    L1 = [7, h, 16, 0, 7, -h]
    L2 = [-9, -h, 0, 8 * h, 9, -h]

    # Boundary points:
    L1b = [1, 0, 2, -h]
    L2b = [0, h, -6, 5 * h]

    L1b2 = [2, h, 1, 0]
    L2b2 = [-6, -5 * h, 0, -h]

    # Number of equations
    eqn = 2 * n

    # Determine the number of upper and lower diagonals
    # From dualfd1:
    # Each interior row affects columns ind-3 to ind+3 (total 7 diagonals)
    # So, lower bandwidth = 3, upper bandwidth = 3
    lower = 3
    upper = 3
    ab = np.zeros((upper + lower + 1, eqn))  # Shape: (7, 2n)

    # Construct the A matrix in banded form
    for i in range(n):
        if i == 0:
            # Boundary equations
            # A[0, 0:4] = L1b
            # A[1, 0:4] = L2b
            # Row indices in A: 0 and 1
            row = 0
            for j, val in enumerate(L1b):
                col = j
                k = row - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val
            row = 1
            for j, val in enumerate(L2b):
                col = j
                k = row - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val
        elif i == n - 1:
            # Boundary equations
            # A[-2, -4:] = L1b2
            # A[-1, -4:] = L2b2
            row = 2 * n - 2
            for j, val in enumerate(L1b2):
                col = eqn - 4 + j
                k = row - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val
            row = 2 * n - 1
            for j, val in enumerate(L2b2):
                col = eqn - 4 + j
                k = row - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val
        else:
            # Interior equations
            # Set equation 1
            row1 = 2 * i
            for j, val in enumerate(L1):
                col = 2 * i - 2 + j
                k = row1 - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val
            # Set equation 2
            row2 = 2 * i + 1
            for j, val in enumerate(L2):
                col = 2 * i - 2 + j
                k = row2 - col
                diag = upper + k
                if 0 <= diag < ab.shape[0]:
                    ab[diag, col] += val

    # Precompute constants for the RHS vector
    c31, c22 = 15 / h, 24 / h
    cb11, cb21, cb31 = -3.5 / h, 4 / h, -0.5 / h
    cb12, cb22, cb32 = 9 / h, -12 / h, 3 / h

    # Efficient computation using banded matrices and solve_banded
    for i in range(m):
        # Initialize RHS vector
        b = np.zeros(eqn)

        for j in range(n):
            if j == 0:
                # Boundary conditions at the left end
                b[j] += cb11 * f[i, j]
                b[j] += cb21 * f[i, j + 1]
                b[j] += cb31 * f[i, j + 2]

                b[j + 1] += cb12 * f[i, j]
                b[j + 1] += cb22 * f[i, j + 1]
                b[j + 1] += cb32 * f[i, j + 2]
            elif j == n - 1:
                # Boundary conditions at the right end
                b[2 * j] += -(cb11 * f[i, j] + cb21 * f[i, j - 1] + cb31 * f[i, j - 2])
                b[2 * j + 1] += -(cb12 * f[i, j] + cb22 * f[i, j - 1] + cb32 * f[i, j - 2])
            else:
                # Interior points
                b[2 * j] += c31 * (f[i, j + 1] - f[i, j - 1])
                b[2 * j + 1] += c22 * (f[i, j - 1] - 2 * f[i, j] + f[i, j + 1])

        # Solve the linear system using solve_banded
        try:
            res_x = solve_banded((lower, upper), ab, b)
        except LinAlgError as e:
            print(f"LinAlgError for function index {i}: {e}")
            return None, None

        # Extract df and d2f from the solution vector
        df[i, :] = res_x[::2]
        d2f[i, :] = res_x[1::2]

    # Inefficient implementation using dualfd1 for verification
    df_inefficient = np.zeros_like(f)
    d2f_inefficient = np.zeros_like(f)
    for i in range(m):
        df_inefficient[i, :], d2f_inefficient[i, :] = dualfd1(f[i, :])

    # Verification: Check if both methods produce similar results
    if np.allclose(df, df_inefficient, atol=1e-8) and np.allclose(d2f, d2f_inefficient, atol=1e-8):
        print("Verification Successful: Banded matrix and inefficient methods agree within tolerance.")
    else:
        max_diff_df = np.max(np.abs(df - df_inefficient))
        max_diff_d2f = np.max(np.abs(d2f - d2f_inefficient))
        print(f"Verification Failed:\nMax difference in df: {max_diff_df}\nMax difference in d2f: {max_diff_d2f}")

    return df, d2f


def part2q2():
    """
    Part 2, question 2
    Add input/output as needed

    """

        import time

    # Define the multiscale test function
    def multiscale_function(x, y, k1=8 * np.pi / 5, k2=2 * np.pi):
        # tmp = np.sin(k1 * x)
        # print(f"tmp shape: {tmp.shape}")
        # return np.sin(k1 * x)
        return np.sin(k1 * x) + np.sin(k2 * y)

    def analytical_first_derivative_multiscale(x, y, k1=8 * np.pi / 5, k2=2 * np.pi):
        return k1 * np.cos(k1 * x) + k2 * np.cos(k2 * y)

    # Grid parameters
    L = 5

    # m_values = [10, 50, 100,]
    # n_values = [10, 50, 100,]
    m_values = [10, 50, 100, 150, 200, 300, 400, ]
    n_values = [10, 50, 100, 150, 200, 300, 400, ]

    times_part2q1 = []
    times_fd2 = []
    errors_part2q1 = []
    errors_fd2 = []

    for m, n in zip(m_values, n_values):
        x = np.linspace(0, L, n)
        y = np.linspace(0, L, m)
        X, Y = np.meshgrid(x, y)
        # print(f"X.shape: {X.shape}, Y.shape: {Y.shape}")
        # X, Y = x, y

        # Generate the function and analytical derivatives
        u = multiscale_function(X, Y)
        # print(f"u: {u} with shape: {u.shape}")
        u_exact_prime = analytical_first_derivative_multiscale(X, Y)

        # Time part2q1
        start = time.time()
        u_prime_part2q1, _ = part2q1(u)
        time_part2q1 = time.time() - start
        times_part2q1.append(time_part2q1)

        # Time fd2
        start = time.time()
        u_prime_fd2 = fd2(u)[0]
        time_fd2 = time.time() - start
        times_fd2.append(time_fd2)

        # Compute errors
        error_part2q1 = np.abs(u_prime_part2q1 - u_exact_prime)
        error_fd2 = np.abs(u_prime_fd2 - u_exact_prime)
        errors_part2q1.append(np.mean(error_part2q1))
        errors_fd2.append(np.mean(error_fd2))


        if print_results:
            print(f"Grid size: m = {m}, n = {n}")
            print(f"Time (part2q1): {time_part2q1:.3e} seconds")
            print(f"Time (fd2): {time_fd2:.3e} seconds")
            print(f"Average Error (part2q1): {np.mean(error_part2q1):.3e}")
            print(f"Average Error (fd2): {np.mean(error_fd2):.3e}")

        # Plot errors
        if plot_results:
            plt.figure(figsize=(12, 6))
            plt.plot(x, error_part2q1[m // 2, :], label="Implicit part2q1", linewidth=1.5)
            plt.plot(x, error_fd2[m // 2, :], label="Explicit fd2", linewidth=1.5)
            # plt.plot(x, error_part2q1, label="Error (part2q1)", linestyle='--', marker='x', color='blue')
            # plt.plot(x, error_fd2, label="Error (fd2)", linestyle='-')
            # plt.yscale("log")
            plt.title(f"Error in Numerical Derivatives (m={m}, n={n})", fontsize=14)
            plt.xlabel("x", fontsize=12)
            plt.ylabel("Error |u'_{computed} - u'_{exact}|", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid()
            plt.show()

    if plot_results:
        # Plotting results
        plt.figure(figsize=(12, 6))

        # Time comparison
        plt.subplot(1, 2, 1)
        plt.plot(n_values, times_part2q1, marker='o', label='Implicit part2q1')
        plt.plot(n_values, times_fd2, marker='o', label='Explicit fd2')
        plt.xlabel('Grid Size (n)')
        plt.ylabel('Time (seconds)')
        plt.title('Time Comparison')
        plt.legend()
        plt.grid(True)

        # Error comparison
        plt.subplot(1, 2, 2)
        plt.plot(n_values, errors_part2q1, marker='o', label='Implicit part2q1')
        plt.plot(n_values, errors_fd2, marker='o', label='Explicit fd2')
        plt.xlabel('Grid Size (n)')
        plt.ylabel('Average Error')
        plt.title('Average Error Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return {
        "grid_sizes": {"m_values": m_values, "n_values": n_values},
        "times_part2q1": times_part2q1,
        "times_fd2": times_fd2,
        "errors_part2q1": errors_part2q1,
        "errors_fd2": errors_fd2
    }


#---------------------------- End code for Part 2 ----------------------------#

if __name__=='__main__':
    x=0 #please do not remove
    #Add code here to call functions used to generate the figures included in your report.
    part1q1b()
    part1q1c()
    part1q2()
    part2q2()
