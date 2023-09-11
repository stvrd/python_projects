# Linge - Programming for Computations in Python

# Sources: 
# http://hplgit.github.io/Programming-for-Computations/pub/p4c/._p4c-bootstrap-Python014.html
# https://github.com/slgit/prog4comp_2/tree/master/py36-src

#Program for computing the height of a ball in vertical motion
import numpy as np
import matplotlib.pyplot as plt
v0 = 5 # Initial velocity
g = 9.81 # Acceleration of gravity
t = np.linspace(0, 1, 1001)
y = v0*t - 0.5*g*t**2 # Vertical position
plt.plot(t, y) # plots all y coordinates vs. all t coordinates
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.show()


# I am x meters away from Peter who throws a ball y meters in the air. What is the observed angle?
from math import atan, pi
x = 10.0 # Horizontal position
y = 10.0 # Vertical position
angle = atan(y/x)
print(angle)
print((angle/pi)*180)

# 1.6 Plotting and Printing (p.21)
t = np.linspace(-4, 4, 200)

# choose 100 points in time interval
f_values = t**2
g_values = np.exp(t)
plt.plot(t, f_values, 'r', t, g_values, 'b--')
plt.xlabel('t')
plt.ylabel('f and g')
plt.legend(['t**2', 'e**t'])
plt.title('Plotting of two functions (t**2 and e**t)')
plt.grid('on')
plt.axis([-5, 5, -1, 10])
plt.show()


plt.subplot(2, 1, 1)
# 2 rows, 1 column, plot number 1
v0 = 5
g = 9.81
t = np.linspace(0, 1, 11)
y = v0*t - 0.5*g*t**2
plt.plot(t, y, '*')
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.title('Ball moving vertically')
plt.subplot(2, 1, 2)
# 2 rows, 1 column, plot number 2
t = np.linspace(-2, 2, 100)
f_values = t**2
g_values = np.exp(t)
plt.plot(t, f_values, 'r', t, g_values, 'b--')
plt.xlabel('t')
plt.ylabel('f and g')
plt.legend(['t**2', 'e**t'])
plt.title('Plotting of two functions (t**2 and e**t)')
plt.grid('on')
plt.axis([-3, 3, -1, 10])
plt.tight_layout()
plt.show()
plt.savefig('some_plot.jpg')

# Print
v1 = 10
v2 = 20.0
print('v1 is {}, v2 is {}'.format(v1, v2))


from math import sin
t0 = 2
dt = 0.55
t = t0 + 0*dt; g = t*sin(t)
print('{:6.2f} {:8.3f}'.format(t, g))
t = t0 + 1*dt; g = t*sin(t)
print('{:6.2f} {:8.3f}'.format(t, g))
t = t0 + 2*dt; g = t*sin(t)
print('{:6.2f} {:8.3f}'.format(t, g))


name = 'Dorion'
message = 'Hello {:s}! What is your age? '.format(name)
age = int(input(message))
print('Ok, so you’re half way to {}, wow!'.format(age*2))


# Chapter 3: Loops and Branching (p.59)

for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print('{:d}*5 = {:d}'.format(i, i*5))

list(range(11))


v0 = 4.5                     # Initial velocity
g = 9.81                     # Acceleration of gravity
t = np.linspace(0, 1, 1000)  # 1000 points in time interval
y = v0*t - 0.5*g*t**2        # Generate all heights

# Find index where ball approximately has reached y=0
i = 0
while y[i] >= 0:
    i = i + 1

# Since y[i] is the height at time t[i], we do know the 
# time as well when we have the index i...
print('Time of flight (in seconds): {:g}'.format(t[i]))

# We plot the path again just for comparison
import matplotlib.pyplot as plt
plt.plot(t, y)
plt.plot(t, 0*t, 'g--')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.show()

# Ex 3.3.4: Random Walk in 2 Dimensions
import random
N = 1000                   # number of steps
d = 1                      # step length (e.g., in meter)
x = np.zeros(N+1)          # x coordinates 
y = np.zeros(N+1)          # y coordinates
x[0] = 0;  y[0] = 0        # set initial position

for i in range(N):
    r = random.random()         # random number in [0,1)
    if 0 <= r < 0.25:           # move north
        y[i+1] = y[i] + d
        x[i+1] = x[i]
    elif 0.25 <= r < 0.5:       # move east
        x[i+1] = x[i] + d
        y[i+1] = y[i]
    elif 0.5 <= r < 0.75:       # move south
        y[i+1] = y[i] - d
        x[i+1] = x[i]
    else:                       # move west
        x[i+1] = x[i] - d
        y[i+1] = y[i]

# plot path (mark start and stop with blue o and *, respectively)
plt.plot(x, y, 'r--', x[0], y[0], 'bo', x[-1], y[-1], 'b*')
plt.xlabel('x');  plt.ylabel('y')
plt.show()

# Example 3.11: Compute pi

# Leibniz function:

n = 10000
i = list(range(n+1))

res = 0
for k in i:
    res = res + 1/((4*k + 1)*(4*k + 3))
    
pi = res*8
pi

# Euler function:

n = 12000
i = list(range(1,n+1))
i

res = 0
for k in i:
    res = res + 1/k**2
    
pi = (6*res)**0.5
pi

# Chapter 4: Functions (p.79)

print(range(5))
list(range(5))

# Excercise 4.12: Fit a straight line to data (Method of least squares)

# # Wrong function --------------------------------------------

# def compute_error(y, a, b):
#     x     = list(range(len(y)))
#     y_hat = np.zeros(len(y))
#     e     = np.zeros(len(y))
#     for i in x:
#         y_hat[i] = a*x[i] + b
#         e[i] = (y_hat[i] - y[i])**2
#     return y_hat,y,e

# def ask_user(y,a,b):
#         # a, b = [float(x) for x in input("Give me a and b, and I'll show you the error").split(',')]
#         y_hat,y,e = compute_error(y,a,b)
#         print(sum(e))
#         x = np.linspace(0,len(y)-1,len(y))
#         plt.plot(x,y,'.')
#         plt.plot(a*x + b)
#         plt.show()
        
        
# y = [0.5, 2.0, 1.0, 1.5, 7.5]
# ask_user(y,1,0)
# # compute_error(y,0,2)

# # -------------------------------------------------


def f(t,a,b):
    return a*t + b

def compute_error(a, b):
    E = 0
    for i in range(len(time)):
        E += (f(time[i],a,b) - data[i])**2
    return E


def interactive_line_fit():
    one_more = True
    while one_more:
        a = float(input('Give a: '))
        b = float(input('Give b: '))
        print('The error is: {:g}'.format(compute_error(a, b)))
        y = f(time, a, b)
        plt.plot(time, y, time, data, '*') 
        plt.xlabel('Time (s)')
        plt.ylabel('y (stars) and straight line f(t)')
        plt.show()
        answer = input('Do you want another fit (y/n)? ')
        if answer == "n":
            one_more = False

data = np.array([0.5, 2.0, 1.0, 1.5, 7.5])
time = np.array([0, 1, 2, 3, 4])

interactive_line_fit()
f(time,1.5,3)

# Excercise 4.13: Fit Sines to Straight Line ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

def sinesum(t,b):
    S = np.zeros(len(t))
    for i in range(len(t)):
        n = np.array(range(len(b))) + 1
        S[i] = sum(b * np.sin(n*t[i]))
    return S

t = np.array([-pi/2, pi/4])
b = np.array([    4,   -3])
sinesum(t,b)

# Test if above function is correct:
def test_sinesum():
    t = np.zeros(2); t[0] = -pi/2;  t[1] = pi/4
    b = np.zeros(2); b[0] = 4.0;  b[1] = -3
    print(sinesum(t, b))

test_sinesum()

def f(t):
    return (1/pi)*t

def plot_compare(f, N, M, b):
    time = np.linspace(-pi, pi, M)
    y = f(time)
    S = sinesum(time, b)
    plt.plot(time, y, 'b-', time, S, 'r--')
    plt.xlabel('Time')
    plt.ylabel('f (blue) and S (red)')
    plt.show()

def error(b, f, M):
    time = np.linspace(-pi, pi, M)
    y = f(time)
    S = sinesum(time, b)
    E = np.sqrt(sum((y - S)**2))
    return E


def trial(f, N):
    M = 500
    b = np.zeros(N)
    new_trial = True
    while new_trial:
        for i in range(N):
            text = 'Give b' + str(i+1) + ' : '
            b[i] = float(input(text))
        plot_compare(f, N, M, b)
        print('The error is: ', error(b, f, M))
        answer = input('Another trial (y/n)? ')
        if answer == 'n':
            new_trial = False

N = 3
trial(f,N)

def automate():
    b1 = np.linspace(-1,1,21)
    b2 = b1.copy()
    b3 = b1.copy()
    N = len(b1)
    E = np.zeros((N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                b = np.array([b1[i], b2[j], b3[k]])
                E[i,j,k] = error(b,f,M)
    E.min()
    indices = np.where(E == E.min())
    E[indices]
    b_top = np.array([b1[16], b2[7], b3[12]])
    return b_top

M = 500
b = automate()
b
plot_compare(f,3,500,b)

# Exercise 6.12: Find best b_n
# Just copied, not yet understood ...

from numpy import linspace, zeros, pi, sin, exp

def integrate_coeffs(f, N, M):
    b = zeros(N)
    left_end = -pi; right_end = pi
    for n in range(1, N+1):
        f_sin = lambda t: f(t)*sin(n*t)      
        b[n-1] = (1/pi)*trapezoidal(f_sin, left_end, right_end, M)
    return b
    
def test_integrate_coeffs():
    """Check that sin(nt) are integrated exactly by trapezoidal"""
    def f(t):
        return 1
    tol = 1E-14
    N = 10
    M = 100
    b = integrate_coeffs(f, N, M)
    print(b)
    for i in range(0, 10):
        err = abs(b[i])  # Supposed to be zero
        assert err < tol, 'n = %d, err = %g' % (n,err)

def plot_approx(f, N, M, filename):
    def S_N(b,t):
        sum = 0
        for i in range(len(b)):
            sum += b[i]*sin((i+1)*t)
        return sum        
    left_end = -pi;  right_end = pi
    time = linspace(-pi, pi, 100)
    y = f(time)
    b = integrate_coeffs(f, N, M)
    y_approx = S_N(b, time)

    import matplotlib.pyplot as plt
    plt.figure(); plt.plot(time, y, 'k-', time, y_approx, 'k--')
    plt.xlabel('t');  plt.ylabel('f (solid) and S (dashed)')
    plt.savefig(filename)

def f(t):
        return (1/pi)*t
N = 3
M = 100
b = integrate_coeffs(f, N, M)

for N in [3, 6, 12, 24]:
    plot_filename = '~\Documents' + str(N) + '.pdf'
    plot_approx(f, N, M, plot_filename)

def g(t):
    return exp(-(t-pi))
plot_filename = '~\' + str(100) + '.pdf'
plot_approx(g, 100, M, plot_filename)

# Chapter 5.1: Symbolic Computations
import sympy as sym
x, y = sym.symbols('x y')
z = x*y
print(z)
print(2*x + 3*x - y)                    # Algebraic computation
print(sym.diff(x**2, x))                # Differentiates x**2 wrt. x
print(sym.integrate(sym.cos(x), x))     # Integrates cos(x) wrt. x
print(sym.simplify((x**2 + x**3)/x**2)) # Simplifies expression
print(sym.limit(sym.sin(x)/x, x, 0))    # lim of sin(x)/x as x->0
print(sym.solve(5*x - 15, x))           # Solves 5*x = 15

# turn symbolic expressions into functions
f_expr  = 5*x**3 + 2*x**2 - 1
f1_expr = sym.diff(f_expr, x)
f       = sym.lambdify([x], f_expr)  # f = lambda x: 5*x**3 + 2*x**2 - 1
f1      = sym.lambdify([x], f1_expr) # f1 = lambda x: 15*x**2 + 4*x
print(f(1), f1(1))   # call and print, x = 1

# Exercise 5.3: Approximate sin(x) near zero with Taylor Series
# Wrong:
from math import pi
import sympy as sym
import matplotlib.pyplot as plt
x = sym.symbols('x')
f = sym.sin(x)
f_taylor = f.series(x,1,5)
f_5 = sym.lambdify(x, f_taylor)
x = np.linspace(-pi,pi,100)
f_5(10)

plt.plot(x,f_5(x))


# Wrong:
def S(x,n):
    res = np.zeros(n)
    for j in range(n):
        res[j] = (-1)**j * x**(2*j+1) / math.factorial(2*j+1)
    return sum(res)

x = np.linspace(-pi,pi,100)

S(2,15)

# Exercise 5.4: Fibonacci Numbers
import numpy as np
import math
def make_Fibonacci(N):
    res = np.zeros(N) + 1
    for i in np.array(range(N-2)) + 2 :
        res[i] = res[i-2] + res[i-1]
    return (res)

F = make_Fibonacci(20)

# Kepler: Ratio of Fibonacci Numbers converges to Golden Ratio!
golden_ratio = (1 + math.sqrt(5)) / 2
def converging_ratio(F):
    ratio = np.zeros(len(F))
    for i in np.array(range(len(F)-1))+1:
        ratio[i] = F[i] / F[i-1]
    return ratio

converging_ratio(F)
    
# Chapter 6: Integrals
from numpy import e
def f(x):
    return 3*e**(-x**4)

f(4)
x = np.linspace(-2,4,100)
print(f(x))
plt.plot(x,f(x))
plt.show()

# Trapezoidal method:
def trapezoidal(f, a, b, n):
    h = (b-a)/n
    f_sum = 0
    for i in range(1, n, 1):
        x = a + i*h
        f_sum = f_sum + f(x)
    return h*(0.5*f(a) + f_sum + 0.5*f(b))    

import numpy as np
# Vectorized:
def trapezoidal(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a, b, n+1)
    s = sum(f(x)) - 0.5*f(a) - 0.5*f(b)
    return h*s

trapezoidal(f,0,2,20)

# Midpoint method
def midpoint(f, a, b, n):
    h = (b-a)/n
    f_sum = 0
    for i in range(0, n, 1):
        x = (a + h/2.0) + i*h
        f_sum = f_sum + f(x)
    return h*f_sum

# Vectorized:
from numpy import exp
def midpoint(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a + h/2, b - h/2, n)
    return h*sum(f(x))


midpoint(f,0,2,20)

# Compare both methods:

g = lambda y: math.exp(-y**2)
a = 0
b = 2
print('   n       midpoint        trapezoidal')
for i in range(1, 21):
    n = 2**i
    m = midpoint(g, a, b, n)
    t = trapezoidal(g, a, b, n)
    print('{:7d} {:.16f} {:.16f}'.format(n, m, t))


# 6.4: Vectorize your functions!

v = lambda t: 3*t**2*exp(t**3)
midpoint(v,0,1,10)

# Monte Carlo Integration

# How to plot?
def f(x,y):
    return np.sqrt(x**2 + y**2)

f(x,y)
x = np.linspace(-5,5,101)
y = np.linspace(-5,5,101)
plt.plot(x,y,f(x,y))

# Exercise 6.11: Plot sin functions:
from math import pi
from numpy import sin
def f1(x):
    return sin(x) * sin(2*x)

def f2(x):
    return sin(2*x) * sin(3*x)

x = np.linspace(-pi,pi,101)

plt.plot(x, f2(x))
plt.show()


# Chapter 7: Solving equations, finding Roots
import numpy as np

def brute_force_root_finder(f, a, b, n):
    from numpy import linspace
    x = linspace(a, b, n)
    y = f(x)
    roots = []
    for i in range(n-1):
        if y[i]*y[i+1] < 0:
            root = x[i] - (x[i+1] - x[i])/(y[i+1] - y[i])*y[i]
            roots.append(root)
        elif y[i] == 0:
            root = x[i]
            roots.append(root)
    return roots

def f(x):
    return np.exp(x**2)

def f2(x):
    return np.exp(-x**2)*np.cos(4*x)

def f3(x):
    return x**2 - 9

f(np.array([2,3]))
f2(np.array([2,3]))
f3(np.array([-3,-2,-1,0,1,2,3]))

brute_force_root_finder(f=f2, a = 0, b = 4, n = 1001)

# Exercise 7.7: Newtons Method

def f4(x):
    return x**3 + 2*x - np.exp(-x)

myx = np.linspace(-2,2,401)
f4(myx)

plt.plot(myx,f4(myx))
plt.show()

# Chapter 8: Ordinary Differential Equations -------------------------------

# Example: Water tank filling rate increases continuously
def fill_tank(N):
    dt = (3/N)
    V = np.zeros(N+1)
    V[0] = 1
    r = np.zeros(N+1)
    r[0] = 1

    for i in range(N):
        V[i+1] = V[i] + dt*r[i]
        r[i+1] = V[i+1]

    x = np.linspace(0,3,N+1)
    plt.plot(x, V, '*')

    def e(x):
        return np.exp(x)
    
    plt.plot(x, e(x) )
    plt.show()

fill_tank(200)

# Example 8.2: Population Growth
# First order ODE: N'(t) = r*N(t)
# Known solution : N_0 * exp(r*t)
# Difference equation : N(t+1) = N(t) + r*N(t)



def pop_growth(N_0, r, n, N_t):
    t = np.linspace(0, N_t, n+1)
    dt = N_t/n
    N = np.zeros(n+1)
    N[0] = N_0
    for i in range(n):
        N[i+1] = N[i] + r*dt*N[i]

    numerical_sol = 'bo' if n < 70 else 'b-'

    plt.plot(t, N, numerical_sol, t, N_0*np.exp(r*t), 'r-')
    plt.legend(['numerical', 'exact'], loc='upper left')
    plt.xlabel('t'); plt.ylabel('N(t)')
    plt.show()

pop_growth(100, 0.1, 10, 20) # Starting population 100, 10% increase per month, 100 time steps, 20 Months)

# 8.2.2 - 8.2.5 - Forward Euler Method:
# The differential equation gives a direct formula for the further direction 
# of the solution curve. FE Method assumes this direction linear. 
# The smaller the steps, the better the solutions.

def ode_FE(f, U_0, dt, T):
    # This function solves ODE of the form u'(t) = f(u,t), u = U_0
    N_t = int(round(T/dt))
    u = np.zeros(N_t+1)
    t = np.linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f(u[n], t[n])
    return u, t

# Demo of ode_FE() on population growth: 
def f(u, t):
    return 0.1*u

u, t = ode_FE(f=f, U_0=100, dt=0.5, T=20) # 100 animals, growth rate = 10%, dt = time unit, T = 20 months
plt.plot(t, u, t, 100*np.exp(0.1*t)) 
plt.show()

# 8.2.6 - Population growth depends on size -> The logistic model:
import numpy as np
import matplotlib.pyplot as plt
def f(u):
    return 0.1*(1 - u/500.)*u

f(x)
# Plot derivative:
x = np.linspace(0,700,10001)
plt.plot(x, f(x)); plt.show()

# Plot solution:
u, t = ode_FE(f=f, U_0=100, dt=0.5, T=60)
plt.figure() 
plt.plot(t, u, 'b-')
plt.xlabel('t'); plt.ylabel('N(t)')
plt.show()                     
                                
# 8.2.7: Test case: linear u(t)
def f(u, t):
    a = 4; b = -1; m = 6
    return a + (u - (a*t + b))**m

u, t = ode_FE(f, -1, 0.5, 20)
plt.figure() 
plt.plot(t, u, 'b-')
plt.xlabel('t'); plt.ylabel('N(t)')
plt.show()                     

# 8.3 Spreading of disease
# From difference equations to differential equations (p. 228)
# System of three diff eq with three unknown functions


def ode_FE(f, U_0, dt, T):
    # Forward Euler for System of several ODE:
    N_t = int(round(T/dt))
    # Ensure that any list/tuple returned from f_ is wrapped as array
    f_ = lambda u, t: np.asarray(f(u, t))
    u = np.zeros((N_t+1, len(U_0)))
    t = np.linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f_(u[n], t[n])
    return u, t

def f(u, t):
    S, I, R = u
    return [-beta*S*I, beta*S*I - gamma*I, gamma*I]

beta = 10./(40*8*24)
gamma = 3./(15*24)
dt = 0.1            # 6 min
D = 30              # Simulate for D days
N_t = int(D*24/dt)  # Corresponding no of time steps
T = dt*N_t          # End time
U_0 = [50, 1, 0]
u, t = ode_FE(f, U_0, dt, T)
S = u[:,0]
I = u[:,1]
R = u[:,2]
fig = plt.figure()
l1, l2, l3 = plt.plot(t, S, t, I, t, R)
fig.legend((l1, l2, l3), ('S', 'I', 'R'), 'center right')
plt.xlabel('hours')
plt.show()

# 8.4: Second Order ODE: Oscillating springs
# Exact solution: x(t) = X_0 * cos(omega * t)
# Forward Euler Method has a problem with growing amplitudes. 
# Let's find better methods.
def f(t):
    return X_0 * cos(omega * t)


# Exercise 8.14: Understand finite differences via Taylor series


# Chapter 9: Partial Differential Equations (p. 287)
# Method: Reduce a PDE to a System of ODE

# Diffusion equation: 
# E.g. Ions in Cells, Ink in Water, Heat in Solids
