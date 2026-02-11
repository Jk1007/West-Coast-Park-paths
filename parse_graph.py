import xml.etree.ElementTree as ET
import json
import math

# UTM Constants for WGS84
a = 6378137.0
f = 1 / 298.257223563
k0 = 0.9996
from math import sqrt, sin, cos, tan, pi

def utm_to_latlon(easting, northing, zone_number=48, zone_letter='N'):
    x = easting - 500000.0
    y = northing

    if zone_letter == 'S':
        y -= 10000000.0

    lon_origin = (zone_number - 1) * 6 - 180 + 3
    lon_origin_rad = lon_origin * pi / 180

    e2 = 2 * f - f ** 2
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

def svy21_to_latlon(y, x):
    return utm_to_latlon(x, y)

def svy21_to_latlon(y, x):
    # Wrapper to match old signature, but actually calls utm
    # x is Easting, y is Northing
    # Note: parsing loop passes (y_val, x_val) which was (d4, d5). d4=y, d5=x.
    # So y=northing, x=easting.
    return utm_to_latlon(x, y)

def parse_graphml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    # Source Bounds (User Provided Raw or Auto-detected)
    # Using the exact values provided by User in Step 617 to ensure alignment
    SRC_MIN_X = 361961.88
    SRC_MAX_X = 362758.37
    SRC_MIN_Y = 142808.10
    SRC_MAX_Y = 143719.35
    
    SRC_WIDTH = SRC_MAX_X - SRC_MIN_X
    SRC_HEIGHT = SRC_MAX_Y - SRC_MIN_Y

    # Target Bounds (Refined Step 585)
    TGT_MIN_LON = 103.75872
    TGT_MAX_LON = 103.76689
    TGT_MIN_LAT = 1.29103
    TGT_MAX_LAT = 1.30124
    
    TGT_WIDTH = TGT_MAX_LON - TGT_MIN_LON
    TGT_HEIGHT = TGT_MAX_LAT - TGT_MIN_LAT

    # Helper: Linear Map
    def map_val(v, src_min, src_len, tgt_min, tgt_len):
        if src_len == 0: return tgt_min
        norm = (v - src_min) / src_len
        # Clamping? No, user might want extrapolation if outside bounds but mostly inside
        return tgt_min + (norm * tgt_len)

    # Extract Nodes
    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        x_val = None
        y_val = None
        
        for data in node.findall('g:data', ns):
            key = data.get('key')
            if key == 'd5': # x
                try: x_val = float(data.text)
                except: pass
            elif key == 'd4': # y
                try: y_val = float(data.text)
                except: pass
                
        if x_val is not None and y_val is not None:
            # Direct Linear Mapping
            lon = map_val(x_val, SRC_MIN_X, SRC_WIDTH, TGT_MIN_LON, TGT_WIDTH)
            lat = map_val(y_val, SRC_MIN_Y, SRC_HEIGHT, TGT_MIN_LAT, TGT_HEIGHT)
            
            nodes.append({
                'id': node_id,
                'lat': lat,
                'lon': lon
            })
            
    print(f"Mapped {len(nodes)} nodes using Linear Projection.")
    print(f"Source X: {SRC_MIN_X} -> {SRC_MAX_X}")
    print(f"Target Lon: {TGT_MIN_LON} -> {TGT_MAX_LON}")

    # Extract Edges
    for edge in root.findall('.//g:edge', ns):
        source = edge.get('source')
        target = edge.get('target')
        length = 0
        
        for data in edge.findall('g:data', ns):
            key = data.get('key')
            if key == 'd12': # length
                try:
                    length = float(data.text)
                except:
                    pass
        
        edges.append({
            'source': source,
            'target': target,
            'length': length
        })

    return {'nodes': nodes, 'edges': edges}

if __name__ == "__main__":
    data = parse_graphml('west_coast_park_walk_clean.graphml')
    
    # Save to JSON
    with open('src/data/ParkGraph.json', 'w') as f:
        json.dump(data, f)
        
    print(f"Exported {len(data['nodes'])} nodes and {len(data['edges'])} edges to src/data/ParkGraph.json")
