import numpy as np
from scipy import integrate
from scipy.integrate import odeint
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root
import ode_solver as ode
from scipy.signal import find_peaks
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
    
    
    
    
    
from scipy.signal import find_peaks

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
peaks, _ = find_peaks(data)
valleys, _ = find_peaks([-x for x in data])

print(peaks)
print(valleys)




# Define the differential equations
def dydt(t, y, B):
    x, y = y
    dxdt = 1 + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Define the continuation function
def continuation(x, B):
    sol = solve_ivp(lambda t, y: dydt(t, y, B), [0, 100], [x, 0.1], t_eval=np.linspace(0, 100, 1000))
    return sol.y[0][-1] - sol.y[0][0]  # Return the change in x over time

# Find the branch of limit cycles using natural-parameter continuation
B_values = np.linspace(2, 3, 100)
x_values = []
for B in B_values:
    sol = root(continuation, 0.1, args=(B,))
    x_values.append(sol.x[0])

# Plot the results
plt.plot(B_values, x_values)
plt.xlabel('B')
plt.ylabel('x')
plt.title('Branch of Limit Cycles from Hopf Bifurcation')
plt.grid(True)
plt.show()



# Step 1: Define the ODE
def ode(t, y, alpha):
    return alpha * y - y**3

# Step 2: Solve the ODE for a specific alpha
def solve_ode(alpha):
    sol = solve_ivp(lambda t, y: ode(t, y, alpha), [0, 10], [1], t_eval=np.linspace(0, 10, 100))
    return sol.y[0]

# Step 3: Change the parameter alpha
alpha_values = np.linspace(-2, 2, 100)

# Step 4: Continuation analysis
y_values = []
for alpha in alpha_values:
    y = solve_ode(alpha)
    y_values.append(y[-1])  # Taking the final value of y

# Plot the results
plt.plot(alpha_values, y_values)
plt.xlabel('alpha')
plt.ylabel('y')
plt.title('Continuation Analysis')
plt.grid(True)
plt.show()


import numpy as np
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODE system
def dydt(t, xy, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Step 1: Find the equilibrium points
def equilibrium_points(A, B):
    # Define the equations for root finding
    def equations(xy):
        x, y = xy
        return [A + x**2 * y - (B + 1) * x, B * x - x**2 * y]

    # Initial guesses for equilibrium points
    guesses = [[0, 0]]

    # Find equilibrium points using root finding
    equilibria = []
    for guess in guesses:
        sol = root(equations, guess)
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

# Step 3: Find Hopf bifurcation point
def hopf_bifurcation(A, B):
    # Define the function to find Hopf bifurcation
    def continuation(alpha, B):
        equilibria = equilibrium_points(A, B)
        x, y = equilibria[0]
        return x - 1, y - 3

    # Solve for the Hopf bifurcation point
    sol = root(continuation, [0.1, 0.1], args=(B,))
    if sol.success:
        
        print(f"Hopf bifurcation point at B = {B}: {sol.x}")
    else:
        print(f"Hopf bifurcation point not found.{B}")


# Step 4: Plot the limit cycles
def plot_limit_cycles(A, B):
    # Define the function to solve ODE for a given B
    def solve_ode(B):
        equilibria = equilibrium_points(A, B)
        x0, y0 = equilibria[0]
        sol = solve_ivp(lambda t, y: dydt(t, y, A, B), [0, 10], [x0, y0], t_eval=np.linspace(0, 10, 100))
        return sol.y

    # Plot the limit cycles for B in range [2, 3]
    B_values = np.linspace(2, 3, 50)
    for B in B_values:
        xy = solve_ode(B)
        plt.plot(xy[0], xy[1], color='b', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Limit Cycles for B in [2, 3]')
    plt.grid(True)
    plt.show()

# Main function
def main():
    A = 1
    B_values = np.linspace(2, 3, 11)

    for B in B_values:
        hopf_bifurcation(A, B)

    plot_limit_cycles(A, B_values)
    

if __name__ == "__main__":
    main()


def ode_solver(ode_func, initial_conditions, t_span, *args, **kwargs):
    solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
    t = t_span if hasattr(t_span, '__len__') else [t_span[0], t_span[-1]]
    return t, solution

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
t_span = np.linspace(0, 100, 1001)  # 时间范围

# Solve the ode
t, solution = ode.ode_solver(a1_ode_system, xy0, t_span, A, B)

# Get the solution
x_solution_a = solution[:, 0]
y_solution_a = solution[:, 1]

# Plot the solution as x-t and y-t graphs
plt.plot(t, x_solution_a, label='x(t)')
plt.plot(t, y_solution_a, label='y(t)')
plt.xlabel('Time')
plt.ylabel('x,y')
plt.title('1-a')
plt.legend()
plt.grid(True)
plt.show()

def has_periodicity(sequence):
    highpeaks, _  = find_peaks(sequence)
    lowpeaks, _ = find_peaks(-sequence)
    high = np.diff(highpeaks)
    low = np.diff(lowpeaks)
    print(high)
    print(low)
    print(np.std(high[5:-1]))
    print(np.std(low[5:-1]))
    return


has_periodicity(x_solution_a)


# 定义ODE方程
def dx_dt(t, xy, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# 定义约束条件
def constraint(x_T, y_T, x0, y0):
    return np.array([x_T - x0, y_T - y0])


def shooting_method(A, B, x0, y0, T_guess, tol=1e-6, max_iter=100):
    u0_guess = [x0, y0]
    T = T_guess
    for _ in range(max_iter):
        sol = solve_ivp(lambda t, xy: dx_dt(t, xy, A, B), [0, T], u0_guess)
        x_T, y_T = sol.y[:, -1]
        error = constraint(x_T, y_T, x0, y0)
        if np.all(np.abs(error) < tol):
            return sol, T
        # 更新 T 的值，确保分母不为零
        error_new = constraint(x_T, y_T, x0, y0)
        if error[0] - error_new[0] != 0:
            T -= error[0] * (T - T_guess) / (error[0] - error_new[0])
        else:
            # 如果分母为零，直接使用均值更新 T 的值
            T = (T + T_guess) / 2
    raise RuntimeError("射击法未收敛")

# 示例参数和初始条件
A = 1
B = 3
x0 = 0.61843
y0 = 4.72089
T_guess = 20

# 使用射击法求解
sol, T = shooting_method(A, B, x0, y0, T_guess)

# 输出结果
print("最优解 u0:", sol.y[:, -1])
print("最优解 T:", T)



   

# Define the ODE system
def dydt(t, xy, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Step 1: Find the equilibrium points
def equilibrium_points(A, B):
    # Define the equations for root finding
    def equations(xy):
        x, y = xy
        return [A + x**2 * y - (B + 1) * x, B * x - x**2 * y]

    # Initial guess for equilibrium point
    guess = [0, 0]

    # Find equilibrium point using root finding
    sol = root(equations, guess)
    if sol.success:
        return [sol.x]
    else:
        return []

# Main function
def main():
    A = 1
    B_values = np.linspace(2, 3, 50)
    hopf_points = []

    # Step 2: Natural-parameter continuation
    for B in B_values:
        # Find equilibrium point at B = 2
        if B == 2:
            equilibrium = equilibrium_points(A, B)
            if equilibrium:
                initial_guess = equilibrium[0]
            else:
                print("Equilibrium point not found at B = 2.")
                return
            
        # Step 3: Find Hopf bifurcation point using continuation
        def continuation(xy, B):
            x, y = xy
            return [A + x**2 * y - (B + 1) * x - initial_guess[0], 
                    B * x - x**2 * y - initial_guess[1]]

        sol = root(continuation, [0.1, 0.1], args=(B,))
        if sol.success:
            hopf_points.append(sol.x)
        else:
            print(f"Hopf bifurcation point not found at B = {B}.")

    # Plot the results
    hopf_points = np.array(hopf_points)
    plt.plot(B_values, hopf_points[:, 0], label='Hopf bifurcation x')
    plt.plot(B_values, hopf_points[:, 1], label='Hopf bifurcation y')
    plt.xlabel('B')
    plt.ylabel('Values at Hopf bifurcation point')
    plt.title('Hopf bifurcation point vs. B')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()




# Define the ode system for Q1a
def a1_ode_system(xy, t, A, B):
    x, y = xy
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# Set initial values
A = 1
B = 2.9
xy0 = [1, 2]  # 初始条件
t_span = np.linspace(0, 100, 1001)  # 时间范围

# Solve the ode
t, solution = ode.ode_solver(a1_ode_system, xy0, t_span, A, B)

# Get the solution
x_solution_a = solution[:, 0]
y_solution_a = solution[:, 1]

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

def has_periodicity(sequence):
    highpeaks, _  = find_peaks(sequence)
    lowpeaks, _ = find_peaks(-sequence)
    high = np.diff(highpeaks)
    low = np.diff(lowpeaks)
    print(high)
    print(low)
    print(sequence[high])
    print(np.std(high[5:-1]))
    print(np.std(low[5:-1]))
    if np.std(high[5:-1]) < 1 and np.std(low[5:-1]) < 1:
        return True
    else:
        return False
    
if has_periodicity(x_solution_a):
    print('yes')
else:
    print('no')
    
    
    
    




# 定义ODE
def odefunc(t, vars, A, B):
    x, y = vars
    dxdt = A + x**2 * y - (B + 1) * x
    dydt = B * x - x**2 * y
    return [dxdt, dydt]

# 参数设置
A = 1
B_values = np.linspace(2, 3, 100)  # 在2到3之间均匀取100个点

# 存储平衡解或周期解的列表
equilibrium_or_periodic_solutions = []

# 对于每个B值，求解ODE并找到平衡解或周期解
for B in B_values:
    sol = solve_ivp(odefunc, (0, 10), [0.1, 0.1], args=(A, B), dense_output=True)

    # 检查解是否是平衡解或周期解
    x_values = sol.sol(sol.t)[0]
    y_values = sol.sol(sol.t)[1]

    # 判断x和y是否在最后一步附近收敛到一个常数，如果是，则认为是平衡解或周期解
    if np.allclose(x_values[-1], x_values[-2], atol=1e-6) and np.allclose(y_values[-1], y_values[-2], atol=1e-6):
        equilibrium_or_periodic_solutions.append(x_values[-1])

# 绘制分岔曲线
plt.figure(figsize=(10, 6))
plt.title("Bifurcation Diagram")
plt.xlabel("B")
plt.ylabel("x")
plt.grid(True)

# 绘制平衡解或周期解
plt.plot(B_values[:len(equilibrium_or_periodic_solutions)], equilibrium_or_periodic_solutions, 'k.', markersize=1)

plt.show()


def odefunc(y, lamda):
    return y ** 3 + lamda * y

A = 1
B = 2
lamda = -4
result = root(odefunc, 0, lamda)
print(result.x)
''' 


def equation(y, lam):
    return y**3 + lam * y

def find_all_roots(lam):
    initial_guesses = [0.1, -0.1, 1, -1, 10, -10]  # 初始猜测值
    roots = []
    for guess in initial_guesses:
        sol = root(equation, guess, args=(lam,))
        if sol.success:
            roots.append(sol.x[0])
    return roots

lambda_values = [-2, -1, 0, 1, 2]

for lam in lambda_values:
    roots = find_all_roots(lam)
    print("lambda =", lam, "Roots:", roots)

