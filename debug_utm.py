import math

a = 6378137.0
f = 1 / 298.257223563
k0 = 0.9996
b = a * (1 - f)
e = math.sqrt(1 - (b/a)**2)

def debug_utm(easting, northing, zone=48):
    print(f"Input: E={easting}, N={northing}")
    
    x = easting - 500000.0
    y = northing
    
    e2 = e**2
    e1sq = e2 / (1 - e2)
    
    print(f"x={x}, y={y}")
    print(f"e2={e2}, e1sq={e1sq}")
    
    M = y / k0
    print(f"M={M}")
    
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    print(f"mu={mu} rad ({math.degrees(mu)} deg)")
    
    e1 = (1 - math.sqrt(1-e2)) / (1 + math.sqrt(1-e2))
    
    phi1Rad = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + \
              (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)
    print(f"phi1Rad={phi1Rad} rad ({math.degrees(phi1Rad)} deg)")
    
    # phi1 should be approx 1.29 deg
    
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1Rad)**2)
    T1 = math.tan(phi1Rad)**2
    C1 = e1sq * math.cos(phi1Rad)**2
    R1 = a * (1 - e2) / math.pow(1 - e2 * math.sin(phi1Rad)**2, 1.5)
    D = x / (N1 * k0)
    
    print(f"N1={N1}, R1={R1}, D={D}")
    
    lat_term1 = (N1 * math.tan(phi1Rad) / R1)
    lat_term2 = (D*D/2)
    
    print(f"Lat Term scalar: {lat_term1}")
    print(f"Lat D^2/2 term: {lat_term2}")
    
    lat = phi1Rad - lat_term1 * lat_term2
    print(f"Result Lat (rad): {lat}")
    print(f"Result Lat (deg): {math.degrees(lat)}")

if __name__ == "__main__":
    debug_utm(362087.17, 143510.20)
