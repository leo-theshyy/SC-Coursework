import ode_iv_solver as iv

def f(x,t):
    return x

x0 = 0
t0 = 0
t1 = 1
dt = 0.1


t_numerical, x_numerical = iv.solve_to(f,x0, t0, t1, dt, 'euler')

print(t_numerical[-1])
print(x_numerical[-1])



