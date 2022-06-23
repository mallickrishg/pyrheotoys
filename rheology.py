""" Definition of various rheological objects (linear Maxwell, Burgers, power-law Maxwell, Burgers, 
rate-dependent friction and rate-state friction) and the associated ODE functions
We consider 3 cases for the ODE
(1) When inelastic strain rate is known (edot_pl)
(2) When applied stress is fixed
(3) When total velocity or strain rate (elastic + inelastic) is fixed

Written by Rishav Mallick, Caltech Seismolab, 2022
"""
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
    def Y0_initial(self,dtau):
        return [self.edot_pl*self.eta_m + dtau, 0]

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
    def Y0_initial(self,dtau):
        """ Return initial strain and strain rate """
        return  [0,self.A*((self.edot_pl/self.A)**(1/self.n) + dtau)**self.n]

    def ode_edot_pl(self,t,Y):
        """ integrate time derivatives of strain and strain rate"""
        edot = Y[1]
        sigmadot = self.G*(self.edot_pl - edot) 
        acc = sigmadot/(edot**(1/self.n-1))*self.n*self.A**(1/self.n)
        return [edot,acc]