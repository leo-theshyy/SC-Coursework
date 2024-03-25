import numpy as np
import matplotlib.pyplot as plt
import time


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
def solve_to(method, f, x0, t0, t1, dt_max):

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


# Define the f
def f(x, t):
    return x


# Analytical solution of the ODE
def analytical_solution(t):
    return np.exp(t)


# Calculate error based on method parameter
def calculate_error(method, dt_values):
    errors = []

    for dt in dt_values:
        t_numerical, x_numerical = solve_to(method, f, 1, t0, t1, dt)
        x_analytical = analytical_solution(t_numerical)
        error = np.abs(x_numerical[-1] - x_analytical[-1])
        errors.append(error)

    return errors


# Define time range
t0 = 0
t1 = 1

# Calculate solutions with different time steps
dt_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

# Calculate errors for Euler method
euler_errors = calculate_error('euler', dt_values)

# Calculate errors for RK4 method
rk4_errors = calculate_error('rk4', dt_values)


# Plot errors for both methods
plt.figure(figsize=(8, 6))
plt.loglog(dt_values, euler_errors, marker='o', linestyle='-', label='Euler Method')
plt.loglog(dt_values, rk4_errors, marker='o', linestyle='-', label='RK4 Method')
plt.title('Error vs. Time Step Size')
plt.xlabel('Time Step Size ($\Delta t$)')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

# Find step sizes for each method that give the same error
target_error = 0.001 # You can determine it as you like, let’s set it as 0.001 here

dt_values_find = np.linspace(0.001,0.1,100)  # Generate a range of time step sizes

# First record the time at the beginning, then record the end time after running, and you can get the time the code runs
start_time_euler = time.time()
for dt in dt_values_find:
    error = calculate_error('euler', [dt])[0]
    if error <= target_error:
        break
euler_time = time.time() - start_time_euler
print(f"Euler method: Time step size = {dt}, Time taken = {euler_time} seconds")

start_time_rk4 = time.time()
for dt in dt_values_find:
    error = calculate_error('rk4', [dt])[0]
    if error <= target_error:
        break
rk4_time = time.time() - start_time_rk4
print(f"RK4 method: Time step size = {dt}, Time taken = {rk4_time} seconds")