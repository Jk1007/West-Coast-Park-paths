
# Target Center
TARGET_LAT = 1.2963792786654849
TARGET_LON = 103.76318668868444

# Setup paths from ParkData.js
PARK_PATHS = [
    [
        [103.7590, 1.2920],
        [103.7610, 1.2930],
        [103.7630, 1.2940],
        [103.7650, 1.2950],
        [103.7665, 1.2960]
    ],
    [
        [103.7590, 1.3000],
        [103.7610, 1.2990],
        [103.7630, 1.2980],
        [103.7650, 1.2970],
        [103.7665, 1.2960]
    ],
    [
        [103.7600, 1.2995], 
        [103.7600, 1.2925]
    ],
    [
        [103.7630, 1.2980], 
        [103.7630, 1.2940]
    ],
    [
        [103.7660, 1.2965], 
        [103.7660, 1.2955]
    ]
]

# Calculate Centroid
total_lat = 0
total_lon = 0
count = 0

for path in PARK_PATHS:
    for point in path:
        lon, lat = point
        total_lat += lat
        total_lon += lon
        count += 1

avg_lat = total_lat / count
avg_lon = total_lon / count

print(f"Current Centroid: {avg_lat}, {avg_lon}")
print(f"Target Centroid: {TARGET_LAT}, {TARGET_LON}")

lat_offset = TARGET_LAT - avg_lat
lon_offset = TARGET_LON - avg_lon

print(f"Offset: Lat {lat_offset}, Lon {lon_offset}")

# Apply Offset
with open('new_paths.js', 'w') as f:
    f.write("export const PARK_PATHS = [\n")
    for i, path in enumerate(PARK_PATHS):
        f.write("    [\n")
        for j, point in enumerate(path):
            lon, lat = point
            new_lat = lat + lat_offset
            new_lon = lon + lon_offset
            comma = "," if j < len(path)-1 else ""
            f.write(f"        [{new_lon:.6f}, {new_lat:.6f}]{comma}\n")
        comma_outer = "," if i < len(PARK_PATHS)-1 else ""
        f.write(f"    ]{comma_outer}\n")
    f.write("];\n")
    
    # Also print new Center
    f.write(f"\nexport const PARK_CENTER = [{TARGET_LON}, {TARGET_LAT}];\n")
