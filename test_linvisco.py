import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# define rheology object
class linburgers:
    def __init__(self,G = 100e3, Gk = 100e3, 
    nm = 1, etam = 1e15, nk = 1, etak = 1e12, epl = 1e-14):
        self.G = G
        """ Shear Modulus in Maxwell element (MPa)"""
        self.Gk = Gk
        """ Shear Modulus in Kelvin element (MPa)"""
        self.nm = nm
        """ power exponent in Maxwell dashpot """
        self.etam = etam
        """ Maxwell viscosity (MPa/s) """
        self.nk = nk
        """ power exponent in Kelvin dashpot """
        self.etak = etak
        """ Kelvin viscosity (MPa/s) """
        self.epl = 1e-14
        """ Long-term inelastic strain rate (1/s) """

    # define ode to be solved
    def ode(self,t,Y):
        sigma = Y[0]
        ek = Y[1]
        ekdot = (sigma - self.Gk*ek)/self.etak
        sigmadot = self.G*(self.epl - sigma/self.etam - ekdot)
        return [sigmadot, ekdot]


# simulate earthquake cycles
nyears = 100
Trecur = nyears*3.15e7
Ncycles = 10

# stress perturbation
delsigma = 2 #in MPa
# long-term strain rate
epl = 1e-14 # 1/s

# define parameters for rheological object
etakval = 1e18*1e-6
etamval = 1e20*1e-6
evl = linburgers(etak = etakval,etam = etamval)
#print(evl.etak,evl.etam)


# solve IVP
Y0 = [epl*etamval + delsigma, 0]
sol = solve_ivp(evl.ode,[0,Trecur],Y0)

# print solution
plt.plot(sol.t,sol.y.T)
plt.show()

