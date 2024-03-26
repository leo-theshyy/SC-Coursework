import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 定义微分方程
def predator_prey(t, state, a, b, d):
    x, y = state
    dxdt = x * (1 - x) - a * x * y / (d + x)
    dydt = b * y * (1 - y / x)
    return [dxdt, dydt]

# 设置参数和初始条件
a = 1
d = 0.1
b_values = np.linspace(0.1, 0.5, 5)
x0 = 0.5
y0 = 0.5
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

# 解决微分方程并绘制结果
for b in b_values:
    sol = solve_ivp(predator_prey, t_span, [x0, y0], args=(a, b, d), t_eval=t_eval)
    plt.plot(sol.t, sol.y.T)
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend([f'b={b}' for b in b_values])
plt.title('Predator-Prey Dynamics')
plt.show()
