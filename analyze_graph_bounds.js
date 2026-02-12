import fs from 'fs';
import ParkGraph from './src/data/ParkGraph.json' with { type: "json" };

const nodes = ParkGraph.nodes;
let minLat = Infinity, maxLat = -Infinity;
let minLon = Infinity, maxLon = -Infinity;

nodes.forEach(n => {
    if (n.lat < minLat) minLat = n.lat;
    if (n.lat > maxLat) maxLat = n.lat;
    if (n.lon < minLon) minLon = n.lon;
    if (n.lon > maxLon) maxLon = n.lon;
});

console.log(`Nodes: ${nodes.length}`);
console.log(`Lat Range: ${minLat} to ${maxLat}`);
console.log(`Lon Range: ${minLon} to ${maxLon}`);
