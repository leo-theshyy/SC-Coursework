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


def f(*args):
    print(args)
    print(*args)
    
f(1,2)