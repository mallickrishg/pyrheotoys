import numpy as np
import matplotlib.pyplot as plt
import rheology
from scipy.integrate import solve_ivp

# simulate earthquake cycles
nyears = 100
Trecur = nyears*3.15e7

# stress perturbation
delsigma = 2 #in MPa
# long-term strain rate
epl = 1e-14 # 1/s

# define parameters for Linear Burgers object
etakval = 1e18*1e-6
etamval = 1e20*1e-6
evl = rheology.linburgers(etak = etakval,etam = etamval,edot_pl=epl)

# solve IVP
Y0 = evl.Y0_initial(delsigma)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl,[0,Trecur],Y0,method="Radau")
emdot,ekdot = evl.get_edot(sol.y[0,:],sol.y[1,:])

## print solution
plt.rcParams['text.usetex'] = False
plt.figure(1,figsize=(8,8))
plt.subplot(211)
plt.plot(sol.t/3.15e7,sol.y[0,:])
plt.ylabel('Stress (MPa)')
plt.grid(True)
plt.xscale('log'), plt.yscale('linear')

plt.subplot(212)
plt.plot(sol.t/3.15e7,(emdot+ekdot)/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.xscale('log'), plt.yscale('log')
plt.grid(True)
# plt.show()

# define parameters for Maxwell object
evl = rheology.Maxwell(n = 1, eta = 1e12, edot_pl=epl)

# initialize and solve the IVP
Y0 = evl.Y0_initial(delsigma)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl,[0,Trecur],Y0,method="Radau")
emdot = sol.y[1,:]
sigma = (emdot/evl.A)**(1/evl.n)

## print solution
plt.rcParams['text.usetex'] = False
plt.figure(2,figsize=(8,8))
plt.subplot(211)
plt.plot(sol.t/3.15e7,sigma)
plt.ylabel('Stress (MPa)')
plt.grid(True)
plt.xscale('log'), plt.yscale('linear')

plt.subplot(212)
plt.plot(sol.t/3.15e7,emdot/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.xscale('log'), plt.yscale('log')
plt.grid(True)
plt.show()