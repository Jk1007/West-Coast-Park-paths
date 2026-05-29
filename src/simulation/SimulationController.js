import * as turf from '@turf/turf';
import { HAZARD_DATABASE } from '../constants/HazardDatabase';
import { ChemicalDispersionEngine } from './ChemicalDispersionEngine';
import { PARK_BOUNDS, EXITS, SIM_CONSTANTS } from '../data/ParkData';
import { fetchWindData } from '../services/WindService';
import ParkGraph from '../data/ParkGraph.json';

export class SimulationController {
    constructor() {
        this.dispersionEngine = new ChemicalDispersionEngine();
        this.isNightMode = false;
        this.agents = [];
        this.incidents = [];
        this.wind = { speed: 10, direction: 45 }; // Default fallback
        this.isRunning = false;
        this.status = 'Clear';
        this.lastWindFetch = 0;

        // --- REPLAY/RECORDING STATE ---
        this.isRecording = false;
        this.isReplaying = false;
        this.elapsedMs = 0;
        this.initialSnapshot = null;
        this.eventLog = [];
        this.nextEventIndex = 0;
        // ------------------------------

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
        
        // 3. Periodic Wind Updates (Every 10 seconds)
        this.windInterval = setInterval(() => {
            this.updateWindFromAPI(true);
        }, 10000);
    }

    destroy() {
        if (this.windInterval) clearInterval(this.windInterval);
    }

    buildGraph() {
        this.nodes = {};

        // Create Turf Polygon from PARK_BOUNDS [minLon, minLat, maxLon, maxLat]
        const polygon = turf.bboxPolygon(PARK_BOUNDS);

        // 1. Index Nodes (Filter by Bounding Box)
        this.rawGraph.nodes.forEach(node => {
            const point = turf.point([node.lon, node.lat]);
            if (turf.booleanPointInPolygon(point, polygon)) {
                this.nodes[node.id] = {
                    ...node,
                    neighbors: []
                };
            }
        });

        // 2. Build Edges (Undirected) - Only if both nodes exist in the filtered set
        this.rawGraph.edges.forEach(edge => {
            const u = this.nodes[edge.source];
            const v = this.nodes[edge.target];

            if (u && v) {
                u.neighbors.push(edge.target);
                v.neighbors.push(edge.source);
            }
        });

        // 3. Find Connected Components and Keep Only the Largest
        let largestComponent = [];
        const visited = new Set();

        const getComponent = (startNodeId) => {
            const component = [];
            const stack = [startNodeId];
            visited.add(startNodeId);

            while (stack.length > 0) {
                const currId = stack.pop();
                component.push(currId);
                const node = this.nodes[currId];
                if (node && node.neighbors) {
                    node.neighbors.forEach(neighborId => {
                        if (!visited.has(neighborId)) {
                            visited.add(neighborId);
                            stack.push(neighborId);
                        }
                    });
                }
            }
            return component;
        };

        const allNodeIds = Object.keys(this.nodes);
        for (const nodeId of allNodeIds) {
            if (!visited.has(nodeId)) {
                const comp = getComponent(nodeId);
                if (comp.length > largestComponent.length) {
                    largestComponent = comp;
                }
            }
        }

        // 4. Filter nodes to only keep the largest component
        const largestSet = new Set(largestComponent);
        const filteredNodes = {};
        let edgeCount = 0;

        for (const nodeId of largestComponent) {
            filteredNodes[nodeId] = this.nodes[nodeId];
            // Recount edges just for logging
            edgeCount += this.nodes[nodeId].neighbors.length;
        }
        edgeCount = edgeCount / 2; // Undirected graph

        this.nodes = filteredNodes;

        console.log(`Graph Built & Filtered: ${Object.keys(this.nodes).length} nodes, ${edgeCount} edges (Largest Connected Component within bounds).`);
    }

    identifyDynamicSafeNodes() {
        // Only count ongoing incidents
        const activeIncidents = this.incidents.filter(inc => inc.details?.status !== 'Resolved');
        if (activeIncidents.length === 0) {
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
                if (incident.details?.status === 'Resolved') continue;
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

        // 1. Filter nodes > 0.4km from ALL incidents
        // 0.4km gives adequate safety time while opening up more areas of the park
        let candidates = possibleNodes.filter(n => n.dist >= 0.4);
        console.log(`[DEBUG] checked ${possibleNodes.length} nodes against ${this.incidents.length} incidents. Found ${candidates.length} safe keys (>0.4km).`);

        // 2. Fallback: If no nodes > 0.4km, take top 10% furthest from their nearest threat
        if (candidates.length === 0) {
            console.warn("[DEBUG] No nodes > 0.4km found. Using furthest available nodes.");
            candidates = possibleNodes.sort((a, b) => b.dist - a.dist).slice(0, Math.ceil(possibleNodes.length * 0.10));
        }

        // 3. Guaranteed Fallback
        if (candidates.length === 0 && possibleNodes.length > 0) {
            candidates.push(possibleNodes[0]);
        }

        // Sort candidates by safety (safest first)
        candidates.sort((a, b) => b.dist - a.dist);

        // 4. Spatially Distribute the Safe Nodes
        // Instead of taking the 20 absolute furthest (which form a clustered clump),
        // we enforce a minimum geographic spacing (0.1km / 100m) between safe nodes.
        const selected = [];
        for (const candidate of candidates) {
            if (selected.length >= 20) break;

            const cNode = this.nodes[candidate.id];
            
            // Check distance to all ALREADY selected safe nodes
            let isSpaced = true;
            for (const selId of selected) {
                const sNode = this.nodes[selId];
                const dLat = (cNode.lat - sNode.lat) * latScale;
                const dLon = (cNode.lon - sNode.lon) * lonScale;
                const distToOtherSafeNode = Math.sqrt(dLat * dLat + dLon * dLon);
                
                if (distToOtherSafeNode < 0.1) { // 100 meters spacing requirement
                    isSpaced = false;
                    break;
                }
            }

            if (isSpaced) {
                selected.push(candidate.id);
            }
        }

        // 5. Fallback Fill
        // If the spacing rule eliminated too many nodes and we have less than 5,
        // just fill the rest with the safest available ones we skipped to ensure enough destinations
        if (selected.length < 5) {
            for (const candidate of candidates) {
                if (selected.length >= 20) break;
                if (!selected.includes(candidate.id)) {
                    selected.push(candidate.id);
                }
            }
        }

        this.safeNodes = selected;
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

            // Assign physically authentic walking speeds (4km/h to 7km/h)
            const speedKmh = Math.random() * (7 - 4) + 4;
            const baseSpeedMs = speedKmh / 3.6; // convert exactly to m/s
            
            let demographic = 'Adult';
            if (speedKmh < 4.5) demographic = 'Elder';
            else if (speedKmh > 6.0) demographic = 'Teenager';

            this.agents.push({
                id: `agent_${i}`,
                position: [node.lon, node.lat],
                velocity: [0, 0],
                state: 'IDLE',
                baseSpeedMs,
                demographic,
                // Graph Navigation State
                currentNodeId: randId,
                targetNodeId: null,
                path: [] // List of node IDs to follow
            });
        }
    }

    // Capture the current state to begin a new recording
    startRecording() {
        this.elapsedMs = 0;
        this.eventLog = [];
        this.isRecording = true;

        // Deep copy the exact current state of the simulation
        this.initialSnapshot = {
            wind: { ...this.wind },
            incidents: this.incidents.map(inc => ({ ...inc, position: [...inc.position] })),
            agents: this.agents.map(a => ({
                id: a.id,
                position: [...a.position],
                velocity: [...a.velocity],
                state: a.state,
                currentNodeId: a.currentNodeId,
                targetNodeId: a.targetNodeId,
                path: [...a.path],
                baseSpeedMs: a.baseSpeedMs,
                demographic: a.demographic
            }))
        };
    }

    // Package the timeline for Firebase
    compileReplayTape() {
        return {
            durationMs: this.elapsedMs,
            initialState: this.initialSnapshot,
            events: this.eventLog,
            finalStats: this.getStats()
        };
    }

    // Load an existing Replay Tape and force the simulation into exact initial conditions
    loadReplayTape(tape) {
        this.isRecording = false;
        this.isReplaying = true;
        this.elapsedMs = 0;
        this.nextEventIndex = 0;

        this.initialSnapshot = tape.initialState || tape.frameData;
        if (!this.initialSnapshot) {
            console.error("Replay tape is corrupted or missing initialState.");
            this.status = 'Clear';
            return;
        }

        this.eventLog = tape.events || [];
        this.eventLog.sort((a, b) => a.timeOffsetMs - b.timeOffsetMs); // Ensure chronological order

        // Force the environment, incidents, and agents to the snapshot
        this.wind = { ...(this.initialSnapshot.wind || { speed: 10, direction: 45 }) };
        this.incidents = this.initialSnapshot.incidents ? this.initialSnapshot.incidents.map(inc => ({ ...inc, position: [...inc.position] })) : [];
        this.agents = (this.initialSnapshot.agents || []).map(a => ({
            id: a.id,
            position: [...a.position],
            velocity: a.velocity ? [...a.velocity] : [0, 0],
            state: a.state || 'IDLE',
            currentNodeId: a.currentNodeId,
            targetNodeId: a.targetNodeId || null,
            path: a.path ? [...a.path] : [],
            baseSpeedMs: a.baseSpeedMs || 1.4, // Fallback to approx 5km/h
            demographic: a.demographic || 'Adult'
        }));

        this.status = this.incidents.length > 0 ? 'Evacuating' : 'Clear';
        this.identifyDynamicSafeNodes();
    }

    addIncident(coordinate, payload = null, isReplayFired = false) {
        const id = `incident_${Date.now()}`;

        // Log the exact time the user manually dropped an incident 
        if (this.isRecording && !isReplayFired) {
            this.eventLog.push({
                timeOffsetMs: this.elapsedMs,
                type: 'ADD_INCIDENT',
                payload: { coordinate, details: payload }
            });
        }

        let initialRadius = 50;
        let color = '#ef4444';

        if (payload) {
            // Apply Database Color if mapped
            if (payload.type && HAZARD_DATABASE[payload.type]) {
                color = HAZARD_DATABASE[payload.type].color || '#ef4444';
            }

            // Radius mathematically proportional to Spilled Volume/Mass (Area = pi * r^2 -> r = sqrt(V))
            if (payload.amount) {
                // Highly amplified scaling literal so 1kg vs 100kg is unmistakably different on zoomed-out maps
                initialRadius = Math.max(10, Math.sqrt(payload.amount) * 15); 
            } else if (payload.severity) {
                // Fallback for older replays
                switch (payload.severity.toLowerCase()) {
                    case 'critical': initialRadius = 150; break;
                    case 'high': initialRadius = 100; break;
                    case 'medium': initialRadius = 60; break;
                    case 'low': initialRadius = 30; break;
                }
            }
        }

        this.incidents.push({
            id,
            position: coordinate,
            startTime: Date.now(),
            elapsedSimSec: 0, // Used by Dispension Engine
            radius: initialRadius,
            opacity: 0.8,
            color: color,
            details: {
                ...payload,
                status: payload?.status || 'In-progress' // default to In-progress
            }
        });
        this.status = 'Evacuating';

        // new: Calculate Dynamic Locations
        this.identifyDynamicSafeNodes();

        // Trigger Evacuation Pathfinding for all agents
        this.recalculatePaths();
    }

    resolveIncident(incidentId) {
        const incident = this.incidents.find(inc => inc.id === incidentId);
        if (incident) {
            if (!incident.details) incident.details = {};
            incident.details.status = 'Resolved';
            incident.color = '#10b981';
            incident.details.resolvedAt = new Date().toISOString();
            incident.resolvedAt = incident.details.resolvedAt;

            // Re-identify safe nodes
            this.identifyDynamicSafeNodes();

            // Check if all incidents are resolved
            const activeCount = this.incidents.filter(inc => inc.details?.status !== 'Resolved').length;
            if (activeCount === 0) {
                this.status = 'Clear';
                // Reset agents back to IDLE
                this.agents.forEach(agent => {
                    if (agent.state === 'EVACUATING') {
                        agent.state = 'IDLE';
                        agent.targetNodeId = null;
                        agent.path = [];
                    }
                });
            } else {
                this.recalculatePaths();
            }
        }
    }

    reset() {
        this.incidents = [];
        this.status = 'Clear';
        this.safeNodes = [];
        this.safeNodesVersion++; // Force UI update to clear safe zones
        this.initAgents();
        this.updateWindFromAPI();
        this.isRecording = false; // Stop recording on reset
    }

    getStats() {
        const totalAgents = this.agents.length;
        const safeAgents = this.agents.filter(a => a.state === 'ESCAPED').length;
        const safetyIndex = totalAgents > 0 ? (safeAgents / totalAgents) * 100 : 0;

        return {
            activeIncidents: this.incidents.filter(inc => inc.details?.status !== 'Resolved').length,
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
        // if (!this.incidents) return; // Allow 0 incidents
        if (!this.nodes) return;

        // Advance global simulation clock
        const dtMs = dt * 1000;
        this.elapsedMs += dtMs;

        // REPLAY ENGINE: Fire deterministic events when the timeline passes them
        if (this.isReplaying) {
            while (
                this.nextEventIndex < this.eventLog.length &&
                this.elapsedMs >= this.eventLog[this.nextEventIndex].timeOffsetMs
            ) {
                const event = this.eventLog[this.nextEventIndex];
                if (event.type === 'ADD_INCIDENT') {
                    // Fire it silently so it doesn't try to log itself
                    this.addIncident(event.payload.coordinate, event.payload.details || null, true);
                }
                this.nextEventIndex++;
            }
        }

        // Time Compression factor: 1 visual second = 15 real seconds
        const TIME_SCALE = 15; 

        // Update incidents (grow plume and apply wind drift)
        this.incidents.forEach(incident => {
            if (incident.details?.status === 'Resolved') {
                return; // Stop growing and stop wind drift
            }
            // Track physical time strictly for Square Root expansion mechanics
            incident.elapsedSimSec += (dt * TIME_SCALE);

            // Fetch dynamic state from ERG 2024 Physics Engine
            const state = this.dispersionEngine.computeState(incident, this.isNightMode, this.wind);
            incident.radius = state.radius;
            incident.opacity = state.opacity;
            incident.padLimit = state.padLimit;

            // Wind Drift Vector (Plume moves in direction of wind)
                // Speed is m/s. 1 knot = 0.514 m/s. Let's use wind.speed as km/h for the UI, which is 0.27 m/s.
                // Or if it's knots from API, 1 knot = 0.5144. Let's assume it's roughly m/s for drift simplicity.
                const windSpeedMs = this.wind.speed * 0.277778; // convert km/h to m/s
                const angleRad = (this.wind.direction + 180) * (Math.PI / 180);

                // Scale drift factor down slightly so it's realistic for a gas plume over 1 minute
                const driftMeters = windSpeedMs * dt * 0.25;

                // Lat/Lon approximations (Singapore)
                const mToLat = 1 / 111320;
                const mToLon = 1 / (111320 * Math.cos(incident.position[1] * Math.PI / 180));

                const dLat = (driftMeters * Math.cos(angleRad)) * mToLat;
                const dLon = (driftMeters * Math.sin(angleRad)) * mToLon;

                incident.position[0] += dLon; // Update X (Longitude)
                incident.position[1] += dLat; // Update Y (Latitude)
        });

        // Update Agents
        this.agents.forEach(agent => {
            if (agent.state === 'ESCAPED') return;

            // Precise Real-World Scaling: 
            // Panic yields ~20% speed boost during evacuations
            const panicMultiplier = agent.state === 'EVACUATING' ? 1.2 : 1.0;
            const currentSpeedMs = agent.baseSpeedMs * panicMultiplier;
            
            // Raw physical distance covered this frame in meters
            const frameDistMeters = currentSpeedMs * (dt * TIME_SCALE);

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
                const distDegrees = Math.sqrt(dx * dx + dy * dy);

                // Geographic coordinate distortion factors for Singapore
                const mToLat = 1 / 111320;
                const mToLon = 1 / (111320 * Math.cos(agent.position[1] * (Math.PI / 180)));
                const avgMToDeg = (mToLat + mToLon) / 2;
                
                // Final frame threshold mapped accurately to Decimal Degrees
                const frameDistDegrees = frameDistMeters * avgMToDeg;

                if (distDegrees < frameDistDegrees) {
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
                    const moveX = (dx / distDegrees) * frameDistDegrees;
                    const moveY = (dy / distDegrees) * frameDistDegrees;
                    agent.position[0] += moveX;
                    agent.position[1] += moveY;
                }
            }
        });
    }
}
