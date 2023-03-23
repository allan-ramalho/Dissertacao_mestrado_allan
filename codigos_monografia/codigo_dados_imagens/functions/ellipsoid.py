#---------------------------------- Ellipsoid Parameters ----------------------------------------
def WGS84():
    '''This function returns the parameters defining the reference
    elipsoid WGS84. They are the following:
    a = semimajor axis [m]
    f = flattening
    GM = geocentric gravitational constant of the Earth (including
    the atmosphere) [m**3/s**2]
    omega = angular velocity [rad/s]
    
    input >
    
    output >
    a:          float - semimajor axis [m]
    b:          float - semiminor axis [m]
    GM:         float - geocentric gravitational constant
    e2:         float - first eccentricity squared
    k2:         float - second eccentricity squared
    '''
    a = 6378137.0
    f = 1.0/298.257223563 # WGS84
    GM = 3986004.418*(10**8)
    #    omega = 7292115*(10**-11)
    b = a*(1-f)
    e2 = (a**2-b**2)/(a**2)
    k2 = 1-e2
    assert a > b, 'major_semiaxis must be greater than the minor_semiaxis'
    assert a > 0, 'major semiaxis must be nonnull'
    assert b > 0, 'minor semiaxis must be nonnull'    
    assert type(a)  == float, 'Semimajor axis must be a float'
    assert type(b)  == float, 'Semiminor axis must be a float'
    assert type(GM) == float, 'Gravitational constant must be a float'
    assert type(e2) == float, 'First eccentricity must be a float'
    assert type(k2) == float, 'Constant k2 must be a float'
    return a, b, GM, e2, k2#, omega

def GRS80():
    '''This function returns the parameters defining the reference
    elipsoid GRS80. They are the following:
    a = semimajor axis [m]
    f = flattening
    GM = geocentric gravitational constant of the Earth (including
    the atmosphere) [m**3/s**2]
    omega = angular velocity [rad/s]
    
    input >
    
    output >
    a:          float - semimajor axis [m]
    b:          float - semiminor axis [m]
    GM:         float - geocentric gravitational constant
    e2:         float - first eccentricity squared
    k2:         float - second eccentricity squared
    '''
    a = 6378137.0
    f = 1/298.257222101 #GRS80
    # f = 0.003352810681183637418
    GM = 3986005.0*(10**8)
    #    omega = 7292115*(10**-11)
    b = a*(1-f)
    e2 = (a**2-b**2)/(a**2)
    k2 = 1-e2
    assert a > b, 'major_semiaxis must be greater than the minor_semiaxis'
    assert a > 0, 'major semiaxis must be nonnull'
    assert b > 0, 'minor semiaxis must be nonnull'    
    assert type(a)  == float, 'Semimajor axis must be a float'
    assert type(b)  == float, 'Semiminor axis must be a float'
    assert type(GM) == float, 'Gravitational constant must be a float'
    assert type(e2) == float, 'First eccentricity must be a float'
    assert type(k2) == float, 'Constant k2 must be a float'
    return a, b, GM, e2, k2#, omega