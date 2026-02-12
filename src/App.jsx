import React, { useEffect, useRef, useState } from 'react';
import MapComponent from './components/MapComponent';
import UIOverlay from './components/UIOverlay';
import { SimulationController } from './simulation/SimulationController';

function App() {
    const simulationRef = useRef(null);
    const requestRef = useRef();
    const lastSafeNodesVersion = useRef(-1);

    // Simulation State (Synced for Rendering)
    const [agents, setAgents] = useState([]);
    const [incidents, setIncidents] = useState([]);
    const [wind, setWind] = useState({ speed: 0, direction: 0 });
    const [stats, setStats] = useState({ activeIncidents: 0, safetyIndex: 100 });
    const [status, setStatus] = useState('Clear');
    const [safeNodes, setSafeNodes] = useState([]); // Debugging Safe Zones
    // Theme State: 'light' (Carto Voyager) or 'dark' (Carto Dark Matter)
    const [mapTheme, setMapTheme] = useState('light');

    useEffect(() => {
        // ... (existing useEffect)
        // Initialize Simulation
        simulationRef.current = new SimulationController();
        // ...
        syncState();

        // Start Loop
        let lastTime = performance.now();

        const animate = (time) => {
            const dt = (time - lastTime) / 1000;
            lastTime = time;

            // Update Physics (cap dt to avoid huge jumps if tab inactive)
            const safeDt = Math.min(dt, 0.1);
            simulationRef.current.update(safeDt);

            // Sync State
            syncState();

            requestRef.current = requestAnimationFrame(animate);
        };

        requestRef.current = requestAnimationFrame(animate);

        return () => {
            cancelAnimationFrame(requestRef.current);
        };
    }, []);

    // ... (syncState, handleAddIncident, handleReset)

    const syncState = () => {
        if (!simulationRef.current) return;
        const sim = simulationRef.current;
        setAgents([...sim.agents]);
        setIncidents([...sim.incidents]);
        setWind({ ...sim.wind });
        setStats(sim.getStats());
        setStatus(sim.status);

        // Optimized: Only update safeNodes if changed
        if (sim.safeNodesVersion !== lastSafeNodesVersion.current) {
            setSafeNodes([...sim.safeNodes]);
            lastSafeNodesVersion.current = sim.safeNodesVersion;
            console.log('[DEBUG-App] Synced SafeNodes (Version Change):', sim.safeNodes.length);
        }
    };

    const handleAddIncident = (coordinate) => {
        if (simulationRef.current) {
            simulationRef.current.addIncident(coordinate);
        }
    };

    const handleReset = () => {
        if (simulationRef.current) {
            simulationRef.current.reset();
        }
    };

    const toggleTheme = () => {
        setMapTheme(prev => prev === 'light' ? 'dark' : 'light');
    };

    return (
        <div className="relative w-full h-full bg-gray-900 overflow-hidden">
            {/* Map Layer */}
            <MapComponent
                agents={agents}
                incidents={incidents}
                safeNodes={safeNodes}
                onAddIncident={handleAddIncident}
                theme={mapTheme}
            />

            {/* UI Overlay */}
            <UIOverlay
                stats={stats}
                wind={wind}
                status={status}
                onReset={handleReset}
                theme={mapTheme}
                onToggleTheme={toggleTheme}
            />
        </div>
    );
}

export default App;
