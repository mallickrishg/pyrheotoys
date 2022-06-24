""" Definition of various rheological objects (linear and power-law Maxwell, Burgers, 
rate-dependent friction and rate-state friction) and the associated ODE functions
We consider 2 cases for the ODE
(1) When total strain rate is known (edot_pl)
(2) When applied stress is known

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
        """ integrate time derivatives of stress and kelvin strain"""
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
        """ integrate Maxwell and Kelvin strain rates to get respective strain time series"""
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

class ratestatefriction:
    def __init__(self,G = 100e3, a = 0.01, b = 0.005, 
    sigma = 50, dc = 0.01, edot_pl = 1e-14, edot0 = 1e-10, Vs = 3e3):
        self.G = G
        """ Shear Modulus in Maxwell element (MPa)"""
        self.a = a
        """ (a-b)sigma_n for frictional slider (MPa)"""
        self.b = b
        """ (a-b)sigma_n for frictional slider (MPa)"""
        self.sigma = sigma
        """ (a-b)sigma_n for frictional slider (MPa)"""
        self.dc = dc
        """ critical slip distance (m) or critical strain (no unit)"""
        self.edot_pl = edot_pl
        """ Long-term inelastic strain rate (1/s) """
        self.edot0 = edot0
        """ normalization constant for strain rate (1/s)"""
        self.Vs = Vs
        """ Shear wave velocity in the medium (m/s)"""

    def Y0_initial(self, dtau, edoti = 1e-9, thetai = 1.):
        """ initialize slip, log(theta), log(slip rate)"""
        return [0, np.log(self.edot0*thetai/self.dc) , np.log(edoti*np.exp(dtau/self.a/self.sigma)/self.edot0)]

    def ode_edot_pl(self,t,Y):
        """ integrate time derivatives of strain, log(theta) and log(strain rate)
        use transformed variables phi = log(edot0*theta/dc), zeta = log(edot/edot0)"""
        phi = Y[1]
        edot = self.edot0*np.exp(Y[2])
        damping = 0.5*self.G/self.Vs

        dphi = (self.edot0*np.exp(-phi) - edot)/self.dc
        sigmadot = self.G*(self.edot_pl - edot) 
        dzeta = (sigmadot - self.b*self.sigma*dphi)/(self.a*self.sigma + damping*edot)
        return [edot,dphi,dzeta]

    
