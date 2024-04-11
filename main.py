import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
# Import the module that solves the initial value problem
import ode_solver as ode


'''
1_a
'''
# Define the ode system for Q1a
def a1_ode_system(xy, t, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Set initial values
A = 1
B = 3
xy0 = [1, 1]  # 初始条件
t_span = np.linspace(0, 20, 1001)  # 时间范围

# Solve the ode
t, solution = ode.ode_solver(a1_ode_system, xy0, t_span, A, B)

# Get the solution
x_solution_a = solution[:, 0]
y_solution_a = solution[:, 1]

print(x_solution_a[-1])
print(y_solution_a[-1])

# Plot the solution as x-t and y-t graphs
plt.plot(t, x_solution_a, label='x(t)')
plt.plot(t, y_solution_a, label='y(t)')
plt.xlabel('Time')
plt.ylabel('x,y')
plt.title('1-a')
plt.legend()
plt.grid(True)
plt.show()

# Plot the limit cycle
plt.plot(x_solution_a,y_solution_a,label='1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('1-a')
plt.legend()
plt.grid(True)
plt.show()

'''
1_b
'''
# 定义初始猜测值
initial_conditions = np.array([0.61843, 4.72089])
initial_guess = np.array([0.4091137, 4.1735098])  

# 调用 numerical_shooting 函数，获取起始点的坐标和振荡周期
initial_conditions, period = ode.numerical_shooting(a1_ode_system, t_span, initial_conditions, initial_guess, A, B)

# 输出起始点的坐标和振荡周期
print("Coordinates of the starting point:", initial_conditions)
print("Oscillation period:", round(period, 2))

'''
2_a
'''
#Define the ode system for Q2a
def a2_ode_system(xyz, t, beta):
    x, y, z = xyz
    dxdt = beta * x - y - z + x * (x**2 + y**2 + z**2) - x * (x**2 + y**2 + z**2)**2
    dydt = x + beta * y - z + y * (x**2 + y**2 + z**2) - y * (x**2 + y**2 + z**2)**2
    dzdt = x + y + beta * z + z * (x**2 + y**2 + z**2) - z * (x**2 + y**2 + z**2)**2
    return [dxdt, dydt, dzdt]

#Set initial values
beta = 1
xyz0 = [1, 0, -1]
t_span = np.linspace(0, 20, 1001)

#Solve the ode
t, solution = ode.ode_solver(a2_ode_system, xyz0, t_span, beta)

#Get the solution
x_solution_b = solution[:, 0]
y_solution_b = solution[:, 1]
z_solution_b = solution[:, 2]

print(x_solution_b[-1])
print(y_solution_b[-1])
print(z_solution_b[-1])

#Plot the solution
plt.plot(t, x_solution_b, label='x(t)')
plt.plot(t, y_solution_b, label='y(t)')
plt.plot(t, z_solution_b, label='z(t)')
plt.xlabel('Time')
plt.ylabel('x,y,z')
plt.title('b')
plt.legend()
plt.grid(True)
plt.show()

# Plot the limit cycle
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_solution_b, y_solution_b, z_solution_b)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

'''
2_b
'''
# 定义初始猜测值
initial_conditions = np.array([-0.939629, -0.086618, 0.853011])
initial_guess = np.array([0.973258, 0.172633, -0.800624])  

# 调用 numerical_shooting 函数，获取起始点的坐标和振荡周期
initial_conditions, period = ode.numerical_shooting(a2_ode_system, t_span, initial_conditions, initial_guess, beta)

# 输出起始点的坐标和振荡周期
print("Coordinates of the starting point:", initial_conditions)
print("Oscillation period:", round(period, 2))
