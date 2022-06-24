"""Script to compute relaxation time series for 4 popular rheologies subject to an initial stress change
and held at a constant strain rate
Rishav Mallick, Caltech Seismolab, 2022
"""
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
etamval = 1e19*1e-6
evl = rheology.linburgers(etak = etakval,etam = etamval,edot_pl=epl)

# solve IVP
Y0 = evl.Y0_initial_edot(delsigma,evl.edot_pl)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl, [0,Trecur], Y0, method="Radau", rtol=1e-12, atol=1e-12)
emdot,ekdot = evl.get_edot(sol.y[0,:],sol.y[1,:])
em,ek = evl.get_e(sol.t,emdot,ekdot)

# print solution
plt.figure(1,figsize=(8,8))
"""plt.subplot(221)
plt.plot(sol.t/3.15e7,sol.y[0,:])
plt.ylabel('Stress (MPa)')
plt.grid(True)
plt.xlim((1e-4,nyears))
plt.xscale('log'), plt.yscale('linear')"""

plt.subplot(211)
plt.plot(sol.t/3.15e7,(emdot)/evl.edot_pl)
plt.plot(sol.t/3.15e7,(emdot+ekdot)/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.legend(['Maxwell','Total'])
plt.xscale('log'), plt.yscale('log')
plt.xlim((1e-4,nyears))
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(sol.t/3.15e7,em)
plt.plot(sol.t/3.15e7,ek)
plt.plot(sol.t/3.15e7,em+ek)
plt.ylabel('Strain')
plt.xlabel('t (yrs)')
plt.legend(['Maxwell','Kelvin','Total'])
plt.xscale('log'), plt.yscale('linear')
plt.grid(True)
plt.xlim((1e-4,nyears))

# define parameters for Maxwell object
evl = rheology.Maxwell(n = 1, A = 1e-12, edot_pl=epl)

# initialize and solve the IVP
Y0 = evl.Y0_initial_edot(delsigma,evl.edot_pl)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl,[0,Trecur],Y0,method="Radau", rtol=1e-12, atol=1e-12)
em = sol.y[0,:]
emdot = sol.y[1,:]
sigma = (emdot/evl.A)**(1/evl.n)

## print solution
plt.figure(2,figsize=(8,8))
"""plt.subplot(221)
plt.plot(sol.t/3.15e7,sigma)
plt.ylabel('Stress (MPa)')
plt.xlabel('t (yrs)')
plt.grid(True)
plt.xscale('log'), plt.yscale('linear')"""

plt.subplot(211)
plt.plot(sol.t/3.15e7,emdot/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.xscale('log'), plt.yscale('log')
plt.grid(True)
plt.xlim((1e-4,nyears))

plt.subplot(212)
plt.plot(sol.t/3.15e7,em)
plt.xlabel('t (yrs)')
plt.ylabel('Strain')
plt.grid(True)
plt.xscale('log'), plt.yscale('linear')
plt.xlim((1e-4,nyears))

# define parameters for Rate-friction object
evl = rheology.ratefriction(Asigma=0.5, edot_pl=epl)

# initialize and solve the IVP
Y0 = evl.Y0_initial_edot(delsigma,evl.edot_pl)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl,[0,Trecur],Y0,method="Radau", rtol=1e-12, atol=1e-12)
ef = sol.y[0,:]
efdot = sol.y[1,:]

## print solution
plt.figure(3,figsize=(8,8))
plt.subplot(212)
plt.plot(sol.t/3.15e7,ef)
plt.ylabel('Strain')
plt.xlabel('t (yrs)')
plt.grid(True)
plt.xlim((1e-4,nyears))
plt.xscale('log'), plt.yscale('linear')

plt.subplot(211)
plt.plot(sol.t/3.15e7,efdot/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.xscale('log'), plt.yscale('log')
plt.grid(True)
plt.xlim((1e-4,nyears))

# define parameters for Rate-state-friction object
# epl = 1e-14
# delsigma = 4
evl = rheology.ratestatefriction(dc =1e-6, G = 100e3, b = 0.006, sigma = 30, edot_pl=epl, edot0 = epl*1e3)

# initialize and solve the IVP
thetai = evl.dc/epl
Y0 = evl.Y0_initial_edot(delsigma,evl.edot_pl,thetai)# get initial conditions
sol = solve_ivp(evl.ode_edot_pl,[0,Trecur],Y0,method="Radau", rtol=1e-12, atol=1e-12)
ef = sol.y[0,:]
efdot = evl.edot0*np.exp(sol.y[2,:])

## print solution
plt.figure(4,figsize=(8,8))
plt.subplot(212)
plt.plot(sol.t/3.15e7,ef)
plt.ylabel('Strain')
plt.xlabel('t (yrs)')
plt.grid(True)
plt.xscale('log'), plt.yscale('linear')
plt.xlim((1e-4,nyears))

plt.subplot(211)
plt.plot(sol.t/3.15e7,efdot/evl.edot_pl)
plt.ylabel('Normalized Strain rate')
plt.xlabel('t (yrs)')
plt.xscale('log'), plt.yscale('log')
plt.grid(True)
plt.xlim((1e-4,nyears))
plt.show()
# plt.show(block = False)
# plt.pause(1)
# plt.close()
