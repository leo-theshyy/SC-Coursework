import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
'''
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义常微分方程组
def ode_system(y, t):
    dydt1 = -2 * y[0] + y[1]
    dydt2 = y[0] - y[1]
    return [dydt1, dydt2]

# 初始条件和时间范围
y0 = [1, 0]  # 初始条件
t_span = np.linspace(0, 10, 100)  # 时间范围

# 求解常微分方程组
y_solution = odeint(ode_system, y0, t_span)

# 绘制结果
plt.plot(t_span, y_solution[:, 0], label='y1(t)')
plt.plot(t_span, y_solution[:, 1], label='y2(t)')
plt.xlabel('Time')
plt.ylabel('y')
plt.title('Solution of the ODE system')
plt.legend()
plt.grid(True)
plt.show()

# 打印结果
print("Solution of the ODE system:")
print(y_solution)
'''
'''
#带*就可以不用以元组形式输入，用的时候带*就拆开成一个一个的，不带就是一个元组
def f(*args):
    print(args)
    print(*args)
    
f(1,2)
'''

'''
def ode_system(x, y):
    A = 1
    B = 3
    dydx = np.vstack((A + y[0]**2 * y[1] - (B + 1) * y[0], B * y[0] - y[0]**2 * y[1]))
    return dydx

def boundary_conditions(ya, yb):
    return np.array([ya[0] - 0, yb[0] - 2])  # 边界条件：y(0) = 0, y(1) = 2

x = np.linspace(0, 1, 10)  # 在 [0, 1] 范围内创建 10 个点作为初始网格
y_initial = np.zeros((2, x.size))  # 初始猜测

# 解决BVP问题
solution = solve_bvp(ode_system, boundary_conditions, x, y_initial)

# 绘制结果
x_plot = np.linspace(0, 1, 100)
y_plot = solution.sol(x_plot)

plt.plot(x_plot, y_plot[0], label='x(t)')
plt.plot(x_plot, y_plot[1], label='y(t)')
plt.xlabel('t')
plt.ylabel('x(t), y(t)')
plt.legend()
plt.grid()
plt.show()


# 定义ODE系统
def ode_system(t, xy, B):
    x, y = xy
    dxdt = 1 + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# 定义相位条件
def phase_condition(y):
    return y[0] - 1  # 设定 y(t=0) = 1 作为相位条件

# 定义数值射击法的迭代算法
def shooting_method(B, initial_guess, tolerance=1e-6, max_iter=100):
    def residual(ya):
        sol = solve_ivp(lambda t, xy: ode_system(t, xy, B), [0, 10], [1, ya], events=phase_condition)
        return sol.y_events[0][0][1] - 1  # 返回满足相位条件的 y(t)

    # 初始化射击初始值
    ya = initial_guess

    # 迭代求解
    for _ in range(max_iter):
        res = residual(ya)
        if abs(res) < tolerance:
            break
        ya -= res  # 使用简单的线性迭代更新

    return ya

# 解决B=3的情况
B = 3
initial_guess = 0.5  # 初始射击猜测

# 使用数值射击法求解相位条件
ya = shooting_method(B, initial_guess)

# 计算振荡周期
sol = solve_ivp(lambda t, xy, B: ode_system(t, xy, B), [0, 10], [1, ya], args=(B,))
period = sol.t_events[0][-1] * 2  # 振荡周期为相位条件满足后的时间

print("起始点坐标 (x, y):", 1, ya)
print("振荡周期:", round(period, 2))




def a1_ode_system(xy, t, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

def numerical_shooting(ode_func, t_span, initial_conditions, initial_guess, *args, tol=1e-6, max_iter=100):
    
    def objective_function(y, t_span):
        _, y_solution = odeint(ode_func, y, t_span, args)
        return initial_guess - y_solution[-1]

    # Perform shooting method
    for _ in range(max_iter):
        # Solve the ODE system with the current initial conditions
        y_solution = odeint(ode_func, initial_conditions, t_span, args)

        # Check the phase condition
        if np.allclose(y_solution[-1], initial_guess, atol=tol):
            # If phase condition is met, compute period and return
            period = compute_period(y_solution, t_span)
            return initial_conditions, period

        # Use Newton's method to update the initial conditions
        initial_conditions -= (objective_function(initial_conditions, t_span) /
                                np.gradient(ode_func(initial_conditions, t_span, *args)))

    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")

def compute_period(solution):
    peak_indices = np.where(np.logical_and(solution[:-1] < solution[1:], solution[1:] < solution[:-1]))[0]
    if len(peak_indices) < 2:
        # 如果找不到足够的峰值，返回一个默认值或者抛出异常
        return np.nan  # 或者 raise ValueError("Cannot compute period: not enough peaks found.")
    period = np.mean(np.diff(peak_indices))
    return period

# 设置初始条件和猜测值
A = 1
B = 3
initial_conditions = np.array([0.61843, 4.72089])
initial_guess = np.array([0.61843, 4.72089])

# 设置时间间隔
t_span = np.linspace(0, 20, 1001)

# 调用数值射击法函数
try:
    result_conditions, period = numerical_shooting(a1_ode_system, t_span, initial_conditions, initial_guess, A, B)
    print("Coordinates of the starting point:", result_conditions)
    print("Oscillation period:", round(period, 2))
except ValueError as e:
    print(e)
    
    '''
    
    
    
from scipy.signal import find_peaks

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
peaks, _ = find_peaks(data)
valleys, _ = find_peaks([-x for x in data])

print(peaks)
print(valleys)