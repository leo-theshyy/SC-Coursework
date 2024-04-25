import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.signal import find_peaks
import time
from scipy.optimize import root


'''
solve_to
'''
# Perform a single Euler step
def euler_step(f, x0, t0, dt):
    return x0 + dt * f(x0, t0)

# Perform a single RK4 step
def rk4_step(f, x0, t0, dt):
    k1 = dt * f(x0, t0)
    k2 = dt * f(x0 + 0.5 * k1, t0 + 0.5 * dt)
    k3 = dt * f(x0 + 0.5 * k2, t0 + 0.5 * dt)
    k4 = dt * f(x0 + k3, t0 + dt)
    return x0 + (k1 + 2*k2 + 2*k3 + k4) / 6

# Solve the ODE x' = f from t0 to t1 using the Euler method or RK4
def solve_to(f, x0, t0, t1, dt_max, method):

    t_values = [t0]
    x_values = [x0]
    t = t0
    x = x0

    while t < t1:
        dt = min(dt_max, t1 - t)   # Prevent t from exceeding t1
        if method == 'euler':   # Choose method
            x = euler_step(f, x, t, dt)
        elif method == 'rk4':
            x = rk4_step(f, x, t, dt)
        t += dt
        t_values.append(t)
        x_values.append(x)

    return np.array(t_values), np.array(x_values)

'''
calculate_error
'''
# Analytical solution of the ODE
def analytical_solution(t):
    return np.exp(t)

# Calculate error based on method parameter
def calculate_error(f, x0, t0, t1, dt_values, method):
    errors = []

    for dt in dt_values:
        t_numerical, x_numerical = solve_to(f, x0, t0, t1, dt, method)
        x_analytical = analytical_solution(t_numerical)
        error = np.abs(x_numerical[-1] - x_analytical[-1])
        errors.append(error)

    return errors

'''
scipy.odeint
ode_solver
'''
def ode_solver(ode_func, initial_conditions, t_span, *args, **kwargs):
    solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
    t = t_span if hasattr(t_span, '__len__') else [t_span[0], t_span[-1]]
    return t, solution

'''
numerical_shooting
'''
def numerical_shooting(ode_func, t_span, initial_conditions, initial_guess, *args, tol=0.1, max_iter=1000, **kwargs):
    
    def objective_function(y, t_span):
        _, y_solution = ode_solver(ode_func, y, t_span, *args)
        return initial_guess - y_solution[-1]   
    
    
    # Perform shooting method
    for _ in range(max_iter):
        # Solve the ODE system with the current initial conditions
        solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
        
        # Check the phase condition
        if np.allclose(solution[-1], initial_guess, atol=tol):
            # If phase condition is met, compute period and return
            period = compute_period(solution[:, 0], t_span)
            return initial_conditions, period
        
        # Use Newton's method to update the initial conditions
        initial_conditions -= (objective_function(initial_conditions, t_span) / np.gradient(ode_func(initial_conditions, t_span, *args)))
    
    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")
'''
has_periodicity
'''
def has_periodicity(sequence):
    highpeaks, _  = find_peaks(sequence)
    lowpeaks, _ = find_peaks(-sequence)
    high = np.diff(highpeaks)
    low = np.diff(lowpeaks)
    print(high)
    print(low)
    print(np.std(high[5:-1]))
    print(np.std(low[5:-1]))
    if np.std(high[5:-1]) < 1 and np.std(low[5:-1]) < 1:
        return True
    else:
        return False

'''
compute the period
'''
def compute_period(solution, t_span):
    peaks, _ = find_peaks(solution)
    print(peaks)
    if len(peaks) < 2:
        # 如果找不到足够的峰值，返回一个默认值或者抛出异常
        return np.nan  
    period = (t_span[1]-t_span[0])*(peaks[1]-peaks[0])
    return period

'''
use scipy.solve_bvp
solve_bvp
'''
def bvp_residuals(ode_func, guess, t_span, x_end):
    sol = solve_ivp(lambda t, xy: ode_func(t, xy), t_span, [guess[0], guess[1]], t_eval=[t_span[1]])
    return sol.y[0][-1] - x_end

def solve_bvp(ode_func, t_span, x_end, guess):
    solution = minimize(bvp_residuals, guess, args=(ode_func, t_span, x_end), method='BFGS')
    return solution.x

'''
numerical continuation
'''
# Step 1: Find the equilibrium points
def equilibrium_points(ode_sys, *args):
    # Define the equations for root finding
    equilibria = []
    sol = root(ode_sys, [0,0], args)
    if sol.success:
        equilibria.append(sol.x)
    return equilibria

# Step 2: Linear stability analysis
def stability_analysis(A, B, equilibria):
    for equilibrium in equilibria:
        x, y = equilibrium
        # Jacobian matrix
        J = [[2 * x * y - (B + 1), x**2], [B - 2 * x * y, -x**2]]
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(J)
        # Check stability
        if all(np.real(eig) < 0 for eig in eigenvalues):
            print(f"Equilibrium point {equilibrium} is stable.")
        else:
            print(f"Equilibrium point {equilibrium} is unstable.")

def has_periodicity(sequence):
    highpeaks, _  = find_peaks(sequence)
    lowpeaks, _ = find_peaks(-sequence)
    high = np.diff(highpeaks)
    low = np.diff(lowpeaks)
    if np.std(high[5:-1]) < 1 and np.std(low[5:-1]) < 1:
        return True
    else:
        return False
    



# Step 4: Plot the limit cycles
def plot_limit_cycles(dydt, A, B):
    # Define the function to solve ODE for a given B
    def solve_ode(B):
        equilibria = equilibrium_points(dydt, A, B)
        x0, y0 = equilibria[0]
        sol = solve_ivp(lambda t, y: dydt(t, y, A, B), [0, 10], [x0, y0], t_eval=np.linspace(0, 10, 100))
        return sol.y

    # Plot the limit cycles for B in range [2, 3]
    B_values = np.linspace(2, 3, 51)
    for B in B_values:
        xy = solve_ode(B)
    plt.plot(xy[0], xy[1], color='b', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Limit Cycles for B in [2, 3]')
    plt.grid(True)
    plt.show()

'''
finite_difference
'''
def finite_difference(ode_fun, grid_num, interval,  *args):
    #Step1 get the grid
    a, b = interval
    grid = np.linspace(a, b, grid_num+1)
    #Step2 Construct equations
    q = [] #-q is the right hand side of the equation
    
    for xi in grid:
        q.append(ode_fun(xi, [0,0], *args)) #the value of u and v is not important here
    
    h = grid[1] - grid[0] #step size
    
    def F(u):
        eq = np.zeros(grid_num-1)
        eq[0] = (u[1] - 2*u[0] + 1) / h**2 - ode_fun(grid[1], [0,0], *args)[1]
        for i in range(1, grid_num-3):
            eq[i] = (u[i+1] - 2*u[i] + u[i-1]) / h**2 - ode_fun(grid[i+1], [0,0], *args)[1]
        eq[grid_num-2] = (-1 - 2*u[grid_num-2] + u[grid_num-3]) / h**2 - ode_fun(grid[grid_num-1], [0,0], *args)[1]
        return eq
    
    #Step3 solve the equation
    u = root(F, np.zeros(grid_num-1)).x
    
    return u


