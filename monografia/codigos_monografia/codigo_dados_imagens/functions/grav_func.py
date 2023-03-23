import numpy as np
#from code.cartography import ellipsoid as ell
from functions.ellipsoid import WGS84
#---------------------------------- Functions redefinition --------------------------------------
pi = np.pi
cos = np.cos
sin = np.sin
tan = np.tan

#-------------------------------------- Constants -----------------------------------------------
G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
SI2MGAL = 1.0e5

def gamma_closedform(h, phi):
    '''This function calculates the normal gravity by using
    a closed-form formula.
    
    input:
    phi: array containing the geodetic latitudes [degree]
    h: array containing the normal heights [m]
    
    output:
    gamma: array containing the values of normal gravity
           on a chosen height relative to the surface of
           the elipsoid for each geodetic latitude [mGal]
    '''
    a, b, GM, e2, k2 = WGS84()
    f = 1.0/298.257223563
    omega = 7292115.*(10**-11)
    b = a*(1.0-f)
    a2 = a**2
    b2 = b**2
    E = np.sqrt(a2 - b2)
    E2 = E**2
    bE = b/E
    Eb = E/b
    atanEb = np.arctan(Eb)
    phirad = np.deg2rad(phi)
    tanphi = np.tan(phirad)
    cosphi = np.cos(phirad)
    sinphi = np.sin(phirad)
    beta = np.arctan(b*tanphi/a)
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)
    zl = b*sinbeta+h*sinphi
    rl = a*cosbeta+h*cosphi
    zl2 = zl**2
    rl2 = rl**2
    dll2 = rl2-zl2
    rll2 = rl2+zl2
    D = dll2/E2
    R = rll2/E2
    cosbetal = np.sqrt(0.5*(1+R) - np.sqrt(0.25*(1+R**2) - 0.5*D))
    cosbetal2 = cosbetal**2
    sinbetal2 = 1-cosbetal2
    bl = np.sqrt(rll2 - E2*cosbetal2)
    bl2 = bl**2
    blE = bl/E
    Ebl = E/bl
    atanEbl = np.arctan(Ebl)
    q0 = 0.5*((1+3*(bE**2))*atanEb - (3*bE))
    q0l = 3.0*(1+(blE**2))*(1-(blE*atanEbl)) - 1.
    W = np.sqrt((bl2+E2*sinbetal2)/(bl2+E2))

    gamma = GM/(bl2+E2) - cosbetal2*bl*omega**2
    gamma += (((omega**2)*a2*E*q0l)/((bl2+E2)*q0))*(0.5*sinbetal2 - 1./6.)
    # the 10**5 converts from m/s**2 to mGal
    gamma = (10**5)*gamma/W
    return gamma
