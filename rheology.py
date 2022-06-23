""" Definition of various rheological objects (linear and power-law Maxwell, Burgers, 
rate-dependent friction and rate-state friction) and the associated ODE functions
We consider 3 cases for the ODE
(1) When inelastic strain rate is known (edot_pl)
(2) When applied stress is fixed
(3) When total velocity or strain rate (elastic + inelastic) is fixed

Written by Rishav Mallick, Caltech Seismolab, 2022
"""
from scipy import integrate
import numpy as np

class linburgers:
    def __init__(self,G = 100e3, Gk = 100e3, 
    nm = 1, etam = 1e15, nk = 1, etak = 1e12, edot_pl = 1e-14):
        self.G = G
        """ Shear Modulus in Maxwell element (MPa)"""
        self.G_k = Gk
        """ Shear Modulus in Kelvin element (MPa)"""
        self.n_m = nm
        """ power exponent in Maxwell dashpot """
        self.eta_m = etam
        """ Maxwell viscosity (MPa-s) """
        self.n_k = nk
        """ power exponent in Kelvin dashpot """
        self.eta_k = etak
        """ Kelvin viscosity (MPa-s) """
        self.edot_pl = edot_pl
        """ Long-term inelastic strain rate (1/s) """

    # define ode to be solved when edot_pl is known
    def Y0_initial(self,dtau,edot_i,ek=0.):
        return [edot_i*self.eta_m + dtau, ek]

    def ode_edot_pl(self,t,Y):
        """ integrate time derivatives of sigma and kelvin strain"""
        sigma = Y[0]
        ek = Y[1]
        ekdot = (sigma - self.G_k*ek)/self.eta_k
        sigmadot = self.G*(self.edot_pl - sigma/self.eta_m - ekdot)
        return [sigmadot, ekdot]

    def get_edot(self,sigma,ek):
        """ return Maxwell and Kelvin strain rates from stress and Kelvin strain time series"""
        emdot = sigma/self.eta_m
        ekdot = (sigma - self.G_k*ek)/self.eta_k
        return [emdot,ekdot]

    def get_e(self,t,emdot,ekdot):
        em = integrate.cumulative_trapezoid(emdot,t, initial = 0)
        ek = integrate.cumulative_trapezoid(ekdot,t, initial = 0)
        return [em,ek]


class Maxwell:
    def __init__(self,G = 100e3, n = 1, A = 1e-13, eta = 1e13, edot_pl = 1e-14):
        self.G = G
        """ Shear Modulus in Maxwell element (MPa)"""
        self.n = n
        """ power exponent in Maxwell dashpot """
        if n == 1:
            self.A = 1/eta
            """ Reciprocal of Viscosity when n = 1 (MPa-s)"""
        else:
            self.A = A
            """ pre-factor in tau = A*edot^n """
        self.edot_pl = edot_pl
        """ Long-term inelastic strain rate (1/s) """

    # define ode to be solved when edot_pl is known
    def Y0_initial(self,dtau,edot_i):
        """ Return initial strain and strain rate """
        return  [0,self.A*((edot_i/self.A)**(1/self.n) + dtau)**self.n]

    def ode_edot_pl(self,t,Y):
        """ integrate time derivatives of strain and strain rate"""
        edot = Y[1]
        sigmadot = self.G*(self.edot_pl - edot) 
        acc = sigmadot/(edot**(1/self.n-1))*self.n*self.A**(1/self.n)
        return [edot,acc]

class ratefriction:
    def __init__(self,G = 100e3, Asigma = 1, edot_pl = 1e-14):
        self.G = G
        """ Shear Modulus in Maxwell element (MPa)"""
        self.Asigma = Asigma
        """ (a-b)sigma_n for frictional slider (MPa)"""
        self.edot_pl = edot_pl
        """ Long-term inelastic strain rate (1/s) """

    def Y0_initial(self,dtau,vi):
        """ initialize slip and slip rate"""
        return [0,vi*np.exp(dtau/self.Asigma)]

    def ode_edot_pl(self,t,Y):
        """ integrate time derivatives of strain and strain rate"""
        edot = Y[1]
        sigmadot = self.G*(self.edot_pl - edot) 
        acc = sigmadot*edot/self.Asigma
        return [edot,acc]

