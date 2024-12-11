
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



def part1q1a(n, g, mu, T):
    """
    Computes a non-trivial equilibrium solution, the optimal initial perturbation vector,
    and the maximum energy growth ratio using the Jacobian and matrix exponential approach.
    """
    def equilibrium_system(X, n, g, mu):
        """System of equations for equilibrium (dx/dt = 0)."""
        F = np.zeros(n)
        F[0] = X[0] * (1 - g * X[-2] - X[0] - g * X[1] + mu * X[-3])
        F[1] = X[1] * (1 - g * X[-1] - X[1] - g * X[2])
        F[2:n-1] = X[2:n-1] * (1 - g * X[0:n-3] - X[2:n-1] - g * X[3:n])
        F[-1] = X[-1] * (1 - g * X[-3] - X[-1] - g * X[0])
        return F


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
    """Part 1, question 1(b)"""
    n = 19
    g = 1.2
    mu = 2.5
    T = 50

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

part1q1b() 

# non-trivial equilibrium: [0.48814185 0.51926129 0.27085226 0.11948127 0.21450432 0.38372748
#  0.39407917 0.29042971 0.2075811  0.26626992 0.32101203 0.35824221
#  0.26852824 0.28854777 0.23463465 0.36927622 0.23705538 0.40115254
#  0.12976333]
# optimal perturbation: [ 0.08485779 -0.11897305  0.01438293  0.25123298 -0.12972617 -0.10314747
#   0.25392272 -0.0856131  -0.34380123  0.29703131  0.09593256 -0.36926707
#   0.16454578  0.26183755 -0.33339597  0.04205753  0.45157156 -0.2270773
#  -0.00447838]
# Linear prediction of max e(T)/e(0) in part1q1a: 26879.88703325684
# Nonlinear simulation e(T)/e(0) in simulate1: 26871.579422032755



def part1q1c():
    """Part 1, question 1(c)"""
    n = 19
    g = 2
    mu = 0
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

part1q1c()