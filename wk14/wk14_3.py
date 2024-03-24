import numpy as np
import matplotlib.pyplot as plt

def euler_step_system(f, x0, t0, dt):
    return x0 + dt * f(x0, t0)

def rk4_step_system(f, x0, t0, dt):
    k1 = dt * f(x0, t0)
    k2 = dt * f(x0 + 0.5 * k1, t0 + 0.5 * dt)
    k3 = dt * f(x0 + 0.5 * k2, t0 + 0.5 * dt)
    k4 = dt * f(x0 + k3, t0 + dt)
    return x0 + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_to_system(method, f, x0, t0, t1, dt_max):
    t_values = [t0]
    x_values = [x0]
    t = t0
    x = x0

    while t < t1:
        dt = min(dt_max, t1 - t)
        if method == 'euler':
            x = euler_step_system(f, x, t, dt)
        elif method == 'rk4':
            x = rk4_step_system(f, x, t, dt)
        t += dt
        t_values.append(t)
        x_values.append(x)

    return np.array(t_values), np.array(x_values)

def derivative_system(x, t):
    dxdt = x[1]  # x' = y
    dydt = -x[0]  # y' = -x
    return np.array([dxdt, dydt])

# Define initial conditions
x0 = np.array([0, 1])  # x(0) = 0, x'(0) = 1
t0 = 0
t1 = 10
dt_max = 0.1

# Solve the system using Euler method
t_euler, x_euler = solve_to_system('euler', derivative_system, x0, t0, t1, dt_max)

# Solve the system using RK4 method
t_rk4, x_rk4 = solve_to_system('rk4', derivative_system, x0, t0, t1, dt_max)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_euler[:, 0], x_euler[:, 1], label='Euler Method')
plt.plot(x_rk4[:, 0], x_rk4[:, 1], label='RK4 Method')
plt.title('Phase Portrait')
plt.xlabel('x')
plt.ylabel('x\'')
plt.legend()
plt.grid(True)
plt.show()
