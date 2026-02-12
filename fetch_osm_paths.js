
import fs from 'fs';
import { PARK_BOUNDS } from './src/data/ParkData.js';

// West Coast Park Bounds (MinLon, MinLat, MaxLon, MaxLat)
const [minLon, minLat, maxLon, maxLat] = PARK_BOUNDS;

// Overpass API Query
// distinct West Coast Park area
const query = `
    [out:json][timeout:25];
    area["name"="West Coast Park"]->.searchArea;
    (
      way["highway"~"^(footway|cycleway|path|pedestrian|track|service|residential|unclassified)$"](area.searchArea);
    );
    out body;
    >;
    out skel qt;
`;

const OVERPASS_URL = 'https://overpass-api.de/api/interpreter';

async function fetchAndBuildGraph() {
    console.log("Fetching OSM data for West Coast Park...");
    console.log(`Bounds: ${minLat}, ${minLon} TO ${maxLat}, ${maxLon}`);

    try {
        const response = await fetch(OVERPASS_URL, {
            method: 'POST',
            body: query
        });

        if (!response.ok) {
            throw new Error(`Overpass API Error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log(`Received ${data.elements.length} elements from OSM.`);

        const nodes = {};
        const ways = [];

        // 1. Process Nodes (Strict Filter)
        data.elements.forEach(el => {
            if (el.type === 'node') {
                // Strict Bounds Check
                if (el.lat >= minLat && el.lat <= maxLat &&
                    el.lon >= minLon && el.lon <= maxLon) {
                    nodes[el.id] = { id: String(el.id), lat: el.lat, lon: el.lon };
                }
            } else if (el.type === 'way') {
                ways.push(el);
            }
        });

        console.log(`Filtered to ${Object.keys(nodes).length} nodes inside park bounds.`);

        const graphNodes = {};
        const graphEdges = [];

        // 2. Process Ways
        ways.forEach(way => {
            for (let i = 0; i < way.nodes.length - 1; i++) {
                const uId = String(way.nodes[i]);
                const vId = String(way.nodes[i + 1]);

                const u = nodes[uId];
                const v = nodes[vId];

                // Only add edge if BOTH nodes are inside validation bounds
                if (u && v) {
                    // Add nodes to graph if not present
                    if (!graphNodes[uId]) graphNodes[uId] = u;
                    if (!graphNodes[vId]) graphNodes[vId] = v;

                    // Calculate rough distance (meters)
                    const dist = getDistanceFromLatLonInKm(u.lat, u.lon, v.lat, v.lon) * 1000;

                    // Add Edge
                    graphEdges.push({
                        source: uId,
                        target: vId,
                        length: dist
                    });
                }
            }
        });

        // 3. Format Output
        const finalGraph = {
            nodes: Object.values(graphNodes),
            edges: graphEdges
        };

        console.log(`Generated Graph: ${finalGraph.nodes.length} nodes, ${finalGraph.edges.length} edges.`);

        // 4. Write to File
        fs.writeFileSync('./src/data/ParkGraph.json', JSON.stringify(finalGraph, null, 2));
        console.log("Successfully updated src/data/ParkGraph.json");

    } catch (error) {
        console.error("Error fetching/processing OSM data:", error);
    }
}

// Distance Helper
function getDistanceFromLatLonInKm(lat1, lon1, lat2, lon2) {
    var R = 6371; // Radius of the earth in km
    var dLat = deg2rad(lat2 - lat1);  // deg2rad below
    var dLon = deg2rad(lon2 - lon1);
    var a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) *
        Math.sin(dLon / 2) * Math.sin(dLon / 2)
        ;
    var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    var d = R * c; // Distance in km
    return d;
}

function deg2rad(deg) {
    return deg * (Math.PI / 180)
}

fetchAndBuildGraph();
