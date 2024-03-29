import numpy as np
import matplotlib.pyplot as plt
# Import the module that solves the initial value problem
import ode_solver as ode

'''
Solve the ode
'''
# Define f of the ode that needs to be solved
def f(x, t):
    return x

# Define the initial values and the step
x0 = 1
t0 = 0
t1 = 1
dt = 0.1

# Use the solve_to function to solve the ode
t_numerical, x_numerical = ode.solve_to(f,x0, t0, t1, dt, 'euler')

# Print the answer
print(t_numerical[-1])
print(x_numerical[-1])

'''
Errors and graphics
'''
'''
# Calculate solutions with different time steps
dt_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

# Calculate errors for Euler method
euler_errors = ode.calculate_error(f, x0, t0, t1, dt_values, 'euler')

# Calculate errors for RK4 method
rk4_errors = ode.calculate_error(f, x0, t0, t1, dt_values, 'rk4')

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
'''






'''
1_a
'''
def ode_system(y, t, A, B):
    x, y = y
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# 初始条件和时间范围
A = 1
B = 3
y0 = [1, 1]  # 初始条件
t_span = np.linspace(0, 20, 1001)  # 时间范围

# 求解常微分方程组
t, solution = ode.ode_solver(ode_system, y0, t_span, A, B)

# 提取解的结果
x_solution = solution[:, 0]
y_solution = solution[:, 1]

print(x_solution)
print(y_solution)

# 绘制结果
plt.plot(t, x_solution, label='x(t)')
plt.plot(t, y_solution, label='y(t)')
plt.xlabel('Time')
plt.ylabel('x,y')
plt.title('Brusselator System for B = 3')
plt.legend()
plt.grid(True)
plt.show()

'''
1_b
'''
# 定义初始猜测值
initial_guess = np.array([0.60, 4.72])  # 替换为您的初始猜测值

# 调用 numerical_shooting 函数，获取起始点的坐标和振荡周期
initial_conditions, period = ode.numerical_shooting(ode_system, t_span, initial_guess, A, B)

# 输出起始点的坐标和振荡周期
print("Coordinates of the starting point:", initial_conditions)
print("Oscillation period:", round(period, 2))
