import math

# Pasquill-Gifford Dispersion Research Equivalents
PASQUILL_GIFFORD = {
    'D': {'a': 0.128, 'b': 0.90, 'c': 32.093, 'd': 0.81066},
    'F': {'a': 0.067, 'b': 0.90, 'c': 14.823, 'd': 0.54503}
}

def gaussian_concentration(x, y, u, Q, stability='D'):
    """
    Computes Ground-Truth Concentration (C) at downwind grid (x, y).
    Serves as the Teacher Model for the Deep Learning framework.
    u: Wind Speed (m/s)
    Q: Emission Rate (g/s)
    """
    u = max(0.5, u)
    if x <= 0: return 0.0
    
    x_km = max(0.01, x / 1000.0)
    coeffs = PASQUILL_GIFFORD.get(stability, PASQUILL_GIFFORD['D'])
    
    sy = coeffs['a'] * (x_km ** coeffs['b']) * 1000.0
    sz = coeffs['c'] * (x_km ** coeffs['d'])
    
    # Ground-level reflection included (H=0, z=0)
    # C(x,y,0) = Q / (pi * u * sy * sz) * exp( -y^2 / 2sy^2 )
    
    core = Q / (math.pi * u * sy * sz)
    lateral = math.exp(- (y**2) / (2 * (sy**2)))
    
    return core * lateral
