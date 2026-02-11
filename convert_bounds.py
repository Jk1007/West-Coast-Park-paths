
import math

# UTM Constants for WGS84
a = 6378137.0
flattening = 1 / 298.257223563
k0 = 0.9996
from math import sqrt, sin, cos, tan, pi

def utm_to_latlon(easting, northing, zone_number=48, zone_letter='N'):
    try:
        x = easting - 500000.0
        y = northing

        if zone_letter == 'S':
            y -= 10000000.0

        lon_origin = (zone_number - 1) * 6 - 180 + 3
        lon_origin_rad = lon_origin * pi / 180

        # Use 'flattening' instead of 'f' to avoid confusion/shadowing
        e2 = 2 * flattening - flattening ** 2
        e1sq = e2 / (1 - e2)
        
        M = y / k0
        mu = M / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
        
        phi1Rad = mu + (3 * e1sq / 2 - 27 * e1sq ** 3 / 32) * sin(2 * mu) + \
                  (21 * e1sq ** 2 / 16 - 55 * e1sq ** 4 / 32) * sin(4 * mu) + \
                  (151 * e1sq ** 3 / 96) * sin(6 * mu)

        phi1 = phi1Rad
        
        N1 = a / sqrt(1 - e2 * sin(phi1) ** 2)
        T1 = tan(phi1) ** 2
        C1 = e1sq * cos(phi1) ** 2
        R1 = a * (1 - e2) / pow(1 - e2 * sin(phi1) ** 2, 1.5)
        D = x / (N1 * k0)
        
        lat = phi1 - (N1 * tan(phi1) / R1) * (D * D / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * e1sq) * D * D * D * D / 24 + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * e1sq - 3 * C1 * C1) * D * D * D * D * D * D / 720)
        lat_deg = lat * 180 / pi
        
        lon = (D - (1 + 2 * T1 + C1) * D * D * D / 6 + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * e1sq + 24 * T1 * T1) * D * D * D * D * D / 120) / cos(phi1)
        lon_deg = lon_origin + lon * 180 / pi
        
        return lat_deg, lon_deg
    except Exception as e:
        return None, str(e)

coords = {
    "NW": (361961.88, 143719.35), # Min X, Max Y
    "SE": (362758.37, 142808.10)  # Max X, Min Y
}

with open('bounds_output.txt', 'w') as outfile:
    outfile.write("Converting User Coordinates to WGS84:\n")
    for key, (x, y) in coords.items():
        lat, lon = utm_to_latlon(x, y)
        if lat is None:
             outfile.write(f"{key}: ERROR {lon}\n")
        else:
             outfile.write(f"{key}: {lat:.10f}, {lon:.10f}\n")
