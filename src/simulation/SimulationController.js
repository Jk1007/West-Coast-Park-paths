import * as turf from '@turf/turf';
import { PARK_BOUNDS, EXITS, SIM_CONSTANTS } from '../data/ParkData';
import { fetchWindData } from '../services/WindService';
import ParkGraph from '../data/ParkGraph.json';

export class SimulationController {
    constructor() {
        this.agents = [];
        this.incidents = [];
        this.wind = { speed: 10, direction: 45 }; // Default fallback
        this.isRunning = false;
        this.status = 'Clear';
        this.lastWindFetch = 0;

        // Raw Data
        this.rawGraph = ParkGraph;

        // Processed Graph (Adjacency List)
        this.nodes = {};
        this.buildGraph();

        this.safeNodes = [];
        this.safeNodesVersion = 0; // State versioning for React
        // NOTE: Safe nodes are now dynamic based on incidents.
        // Initially empty or all exits? For now, empty until incident.

        // Fetch real wind immediately (throttled)
        this.updateWindFromAPI();

        this.initAgents();
    }

    buildGraph() {
        this.nodes = {};

        // 1. Index Nodes
        this.rawGraph.nodes.forEach(node => {
            this.nodes[node.id] = {
                ...node,
                neighbors: []
            };
        });

        // 2. Build Edges (Undirected)
        this.rawGraph.edges.forEach(edge => {
            const u = this.nodes[edge.source];
            const v = this.nodes[edge.target];

            if (u && v) {
                u.neighbors.push(edge.target);
                v.neighbors.push(edge.source);
            }
        });

        console.log(`Graph Built: ${Object.keys(this.nodes).length} nodes.`);
    }

    identifyDynamicSafeNodes() {
        if (this.incidents.length === 0) {
            this.safeNodes = [];
            this.safeNodesVersion++;
            return;
        }

        // Lat/Lon scaling factors (approx for Singapore lat ~1.3)
        const latScale = 111.32;
        // cos(1.3 deg) is almost 1, but let's be precise enough
        const lonScale = 111.32 * Math.cos(1.3 * (Math.PI / 180));

        const possibleNodes = [];

        Object.entries(this.nodes).forEach(([id, node]) => {
            // Find distance to the NEAREST incident
            let minDistToAnyIncident = Infinity;

            for (const incident of this.incidents) {
                const [iLon, iLat] = incident.position;
                const dLat = (node.lat - iLat) * latScale;
                const dLon = (node.lon - iLon) * lonScale;
                const dist = Math.sqrt(dLat * dLat + dLon * dLon); // km

                if (dist < minDistToAnyIncident) {
                    minDistToAnyIncident = dist;
                }
            }

            // Store the Minimum Distance (Safety Score)
            possibleNodes.push({ id, dist: minDistToAnyIncident });
        });

        // 1. Filter nodes > 0.5km from ALL incidents
        // i.e., their closest incident is at least 0.5km away
        let candidates = possibleNodes.filter(n => n.dist >= 0.5);
        console.log(`[DEBUG] checked ${possibleNodes.length} nodes against ${this.incidents.length} incidents. Found ${candidates.length} safe keys (>0.5km).`);

        // 2. Fallback: If no nodes > 0.5km, take top 5% furthest from their nearest threat
        if (candidates.length === 0) {
            console.warn("[DEBUG] No nodes > 0.5km found. Using furthest available nodes.");
            candidates = possibleNodes.sort((a, b) => b.dist - a.dist).slice(0, Math.ceil(possibleNodes.length * 0.05));
        }

        // 3. Guaranteed Fallback
        if (candidates.length === 0 && possibleNodes.length > 0) {
            candidates.push(possibleNodes[0]);
        }

        // 4. Select Top 20 Furthest
        candidates.sort((a, b) => b.dist - a.dist);
        this.safeNodes = candidates.slice(0, 20).map(n => n.id);
        this.safeNodesVersion++;

        console.log(`[DEBUG] Final Safe Nodes IDs:`, this.safeNodes);
    }

    async updateWindFromAPI(force = false) {
        const now = Date.now();
        // Throttle: Only fetch if forced OR > 60 seconds have passed
        if (!force && (now - this.lastWindFetch < 60000)) {
            // console.log("Wind API: Throttled (using cached data)");
            return;
        }

        const windData = await fetchWindData();
        if (windData) {
            this.wind = windData;
            this.lastWindFetch = Date.now();
        }
    }

    initAgents() {
        this.agents = [];
        const nodeIds = Object.keys(this.nodes);

        if (nodeIds.length === 0) return;

        for (let i = 0; i < 100; i++) {
            // Spawn at random node
            const randId = nodeIds[Math.floor(Math.random() * nodeIds.length)];
            const node = this.nodes[randId];

            this.agents.push({
                id: `agent_${i}`,
                position: [node.lon, node.lat],
                velocity: [0, 0],
                state: 'IDLE',
                // Graph Navigation State
                currentNodeId: randId,
                targetNodeId: null,
                path: [] // List of node IDs to follow
            });
        }
    }

    addIncident(coordinate) {
        const id = `incident_${Date.now()}`;
        this.incidents.push({
            id,
            position: coordinate,
            startTime: Date.now(),
            radius: 50, // Initial radius in meters
        });
        this.status = 'Evacuating';

        // new: Calculate Dynamic Locations
        this.identifyDynamicSafeNodes();

        // Trigger Evacuation Pathfinding for all agents
        this.recalculatePaths();
    }

    reset() {
        this.incidents = [];
        this.status = 'Clear';
        this.safeNodes = [];
        this.safeNodesVersion++; // Force UI update to clear safe zones
        this.initAgents();
        this.updateWindFromAPI(); // Refresh wind on reset
    }

    getStats() {
        const totalAgents = this.agents.length;
        const safeAgents = this.agents.filter(a => a.state === 'ESCAPED').length;
        const safetyIndex = totalAgents > 0 ? (safeAgents / totalAgents) * 100 : 0;

        return {
            activeIncidents: this.incidents.length,
            safetyIndex: Math.round(safetyIndex),
        };
    }

    setWind(speed, direction) {
        this.wind.speed = speed;
        this.wind.direction = direction;
    }

    // A* Pathfinding
    findPath(startNodeId, targetNodeIds) {
        const openSet = [startNodeId];
        const cameFrom = {};

        const gScore = {}; // Cost from start
        const fScore = {}; // Estimated total cost

        Object.keys(this.nodes).forEach(id => {
            gScore[id] = Infinity;
            fScore[id] = Infinity;
        });

        gScore[startNodeId] = 0;

        // Heuristic: Distance to NEAREST target node
        const heuristic = (id) => {
            let minH = Infinity;
            const node = this.nodes[id];
            targetNodeIds.forEach(targetId => {
                const target = this.nodes[targetId];
                const d = Math.sqrt(Math.pow(node.lon - target.lon, 2) + Math.pow(node.lat - target.lat, 2));
                if (d < minH) minH = d;
            });
            return minH;
        };

        fScore[startNodeId] = heuristic(startNodeId);

        while (openSet.length > 0) {
            // Get node with lowest fScore
            let current = openSet[0];
            let minF = fScore[current];

            for (let i = 1; i < openSet.length; i++) {
                if (fScore[openSet[i]] < minF) {
                    minF = fScore[openSet[i]];
                    current = openSet[i];
                }
            }

            // If reached any target
            if (targetNodeIds.includes(current)) {
                return this.reconstructPath(cameFrom, current);
            }

            // Remove current from openSet
            openSet.splice(openSet.indexOf(current), 1);

            // Neighbors
            const neighbors = this.nodes[current].neighbors || [];
            for (const neighbor of neighbors) {
                // Distance between current and neighbor
                const n1 = this.nodes[current];
                const n2 = this.nodes[neighbor];
                const dist = Math.sqrt(Math.pow(n1.lon - n2.lon, 2) + Math.pow(n1.lat - n2.lat, 2));

                const tentativeG = gScore[current] + dist;

                if (tentativeG < gScore[neighbor]) {
                    cameFrom[neighbor] = current;
                    gScore[neighbor] = tentativeG;
                    fScore[neighbor] = gScore[neighbor] + heuristic(neighbor);

                    if (!openSet.includes(neighbor)) {
                        openSet.push(neighbor);
                    }
                }
            }
        }

        return null; // No path found
    }

    reconstructPath(cameFrom, current) {
        const totalPath = [current];
        while (current in cameFrom) {
            current = cameFrom[current];
            totalPath.unshift(current);
        }
        return totalPath;
    }

    recalculatePaths() {
        this.agents.forEach(agent => {
            if (this.status === 'Evacuating') {
                agent.state = 'EVACUATING';
                const path = this.findPath(agent.currentNodeId, this.safeNodes);
                if (path && path.length > 1) {
                    agent.path = path.slice(1); // Remove current node
                    agent.targetNodeId = agent.path[0];
                } else if (this.safeNodes.includes(agent.currentNodeId)) {
                    agent.state = 'ESCAPED';
                }
            }
        });
    }

    update(dt) {
        // Validation Guards
        if (!this.agents) return;
        if (!this.incidents) return;
        if (!this.nodes) return;

        // Update incidents (grow plume)
        this.incidents.forEach(incident => {
            if (incident.radius < 500) {
                incident.radius += (2 + (this.wind.speed / 20)) * dt;
            }
        });

        // Update Agents
        this.agents.forEach(agent => {
            if (agent.state === 'ESCAPED') return;

            // NOTE: Agent Speed Scaling
            // Normal speed ~3m/s.
            // 1 degree lat ~ 111,000 meters.
            // 3 m/s = 3 / 111000 = 0.000027 degrees/s.
            // dt is in seconds.
            // So speed should be CONST * dt.

            const speedPerSec = agent.state === 'EVACUATING' ? SIM_CONSTANTS.NORMAL_SPEED * 2.5 : SIM_CONSTANTS.NORMAL_SPEED;
            // SIM_CONSTANTS.NORMAL_SPEED is 0.00003
            const frameDist = speedPerSec * (dt * 60); // Heuristic: Scale up because 0.00003 was originally tuned for per-frame

            // IDLE Random Logic
            if (agent.state === 'IDLE' && !agent.targetNodeId) {
                // Pick random neighbor
                const currentNode = this.nodes[agent.currentNodeId];
                if (currentNode && currentNode.neighbors.length > 0) {
                    agent.targetNodeId = currentNode.neighbors[Math.floor(Math.random() * currentNode.neighbors.length)];
                } else {
                    // Stuck node? Stay put.
                }
            }

            // Move towards targetNodeId
            if (agent.targetNodeId) {
                const targetNode = this.nodes[agent.targetNodeId];
                if (!targetNode) return; // Error safety

                const dx = targetNode.lon - agent.position[0];
                const dy = targetNode.lat - agent.position[1];
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < frameDist) {
                    // Reached node
                    agent.position = [targetNode.lon, targetNode.lat];
                    agent.currentNodeId = agent.targetNodeId;

                    if (agent.state === 'IDLE') {
                        agent.targetNodeId = null; // Will pick new random one next frame
                    } else if (agent.state === 'EVACUATING') {
                        // Pop next from path
                        if (agent.path.length > 0) {
                            agent.path.shift(); // Remove reached node
                            if (agent.path.length > 0) {
                                agent.targetNodeId = agent.path[0];
                            } else {
                                // Path finished
                                if (this.safeNodes.includes(agent.currentNodeId)) {
                                    agent.state = 'ESCAPED';
                                    agent.targetNodeId = null;
                                } else {
                                    agent.targetNodeId = null;
                                }
                            }
                        } else {
                            if (this.safeNodes.includes(agent.currentNodeId)) {
                                agent.state = 'ESCAPED';
                            }
                            agent.targetNodeId = null;
                        }
                    }
                } else {
                    // Normalize and move
                    const moveX = (dx / dist) * frameDist;
                    const moveY = (dy / dist) * frameDist;
                    agent.position[0] += moveX;
                    agent.position[1] += moveY;
                }
            }
        });
    }
}
