// Center of West Coast Park (Calculated from user bounds)
export const PARK_CENTER = [103.76318668868444, 1.2963792786654849]; // [Lon, Lat]

// Approximate bounding box for West Coast Park (Grass/Path area)
// [MinLon, MinLat, MaxLon, MaxLat]
export const PARK_BOUNDS = [
    103.755, 1.285, // MinLon, MinLat (South West)
    103.780, 1.305  // MaxLon, MaxLat (North East - covers near Clementi Woods)
];

// Exit points (Approximate)
export const EXITS = [
    { id: 'exit_highway', position: [103.7650, 1.3000] }, // North/Highway
    { id: 'exit_shore', position: [103.7600, 1.2960] }   // South/Shore
];

// Walking Paths (Conservative "Safe Zone" - Strictly Central)
// Clustered around 1.2963, 103.7631 (Grand Lawn / Adventure Playground)
// Avoids edges (Highway/Water) completely.
export const PARK_PATHS = [
    // 1. Grand Lawn Loop (Central Oval)
    [
        [103.7625, 1.2955],
        [103.7635, 1.2950],
        [103.7645, 1.2958],
        [103.7635, 1.2965],
        [103.7625, 1.2955] // Closed loop
    ],
    // 2. Playground Linear Path (NW to SE, but central)
    [
        [103.7610, 1.2975], // Near McDonald's (Safe inner side)
        [103.7620, 1.2970],
        [103.7630, 1.2965],
        [103.7640, 1.2960],
        [103.7650, 1.2955]  // Towards Carpark (Inner)
    ],
    // 3. Zig-Zag Connectors (Internal)
    [
        [103.7630, 1.2965],
        [103.7635, 1.2950]
    ],
    [
        [103.7620, 1.2970],
        [103.7625, 1.2955]
    ]
];

export const SIM_CONSTANTS = {
    AGENT_COUNT: 50, // Per user refined prompt
    NORMAL_SPEED: 0.000004, // ~1.5 m/s (Approximate walking speed)
    EVAC_SPEED_MULTIPLIER: 2.5,
    REPULSION_RADIUS: 0.3, // 300 meters (in km)
    REPULSION_STRENGTH: 0.00005,
};
