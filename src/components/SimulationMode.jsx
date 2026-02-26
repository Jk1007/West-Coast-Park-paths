import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import MapComponent from './MapComponent';
import UIOverlay from './UIOverlay';
import { SimulationController } from '../simulation/SimulationController';
import { XCircle } from 'lucide-react';

const SimulationMode = ({ onExit }) => {
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
        // Initialize Simulation
        simulationRef.current = new SimulationController();
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
            if (simulationRef.current) {
                // optional cleanup
            }
        };
    }, []);

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
        <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="w-full h-full relative"
        >
            {/* Sandbox Thematic Wrapper */}
            <div className="absolute inset-0 border-4 border-amber-500/30 rounded-lg pointer-events-none z-[60]" />
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-amber-500 text-amber-950 px-6 py-1 rounded-b-lg font-bold text-xs uppercase tracking-widest pointer-events-none z-[60] shadow-xl">
                Sandbox Environment
            </div>

            <button
                onClick={onExit}
                className="absolute top-6 right-6 z-[60] bg-gray-900/80 hover:bg-gray-800 text-gray-300 hover:text-white p-2 rounded-full backdrop-blur border border-gray-700 hover:border-amber-500 transition-colors shadow-xl"
                title="Exit Simulation"
            >
                <XCircle className="w-6 h-6" />
            </button>

            <div className="relative w-full h-full bg-gray-900 overflow-hidden rounded-lg">
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
        </motion.div>
    );
};

export default SimulationMode;
