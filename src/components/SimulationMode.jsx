import React, { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MapComponent from './MapComponent';
import UIOverlay from './UIOverlay';
import { SimulationController } from '../simulation/SimulationController';
import { XCircle, CheckCircle2, ShieldAlert, X, Moon, Sun, Hand, Server } from 'lucide-react';
import supabase from '../supabase';
import { HAZARD_DATABASE } from '../constants/HazardDatabase';
import SimulationWsService from '../services/SimulationWsService';

const SimulationMode = ({ onExit, theme }) => {
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
    const [isRecording, setIsRecording] = useState(false);

    const [showSaveModal, setShowSaveModal] = useState(false);
    const [isHandDetected, setIsHandDetected] = useState(false);
    const [wsBackendStatus, setWsBackendStatus] = useState('disconnected');
    const [simName, setSimName] = useState('');
    const [isSaving, setIsSaving] = useState(false);
    const [saveSuccess, setSaveSuccess] = useState(false);
    const [sessionStartTime, setSessionStartTime] = useState(null);
    const [rating, setRating] = useState(5);
    const [focusLevel, setFocusLevel] = useState('medium');

    const [pendingIncidentCoord, setPendingIncidentCoord] = useState(null);
    const [isNightMode, setIsNightMode] = useState(false);
    const [incidentForm, setIncidentForm] = useState({
        Title: '',
        Type: 'CHLORINE_GAS',
        Amount: 100, // kg or L
        Details: ''
    });

    // Connect to Python plume backend via WebSocket
    useEffect(() => {
        SimulationWsService.connect(setWsBackendStatus);
        return () => SimulationWsService.disconnect(setWsBackendStatus);
    }, []);

    useEffect(() => {
        // Initialize Simulation only once
        if (!simulationRef.current) {
            simulationRef.current = new SimulationController();
            syncState();
        }

        // Start Loop
        let lastTime = performance.now();
        let accumulator = 0;
        const fpsInterval = 1000 / 30; // Target 30 FPS to prevent Mapbox WebWorker lag
        const playbackSpeed = 1;

        const animate = (time) => {
            const dtMs = time - lastTime;
            lastTime = time;
            
            accumulator += dtMs;
            
            // Only fire physics and React DOM updates every ~33ms
            if (accumulator >= fpsInterval) {
                // Calculate safe physics delta in seconds
                const syncDt = Math.min(accumulator / 1000, 0.1) * playbackSpeed;
                
                if (simulationRef.current) {
                    simulationRef.current.update(syncDt);
                    syncState();
                }
                
                // Subtract frame quota cleanly to prevent microstutters
                accumulator = accumulator % fpsInterval; 
            }

            // Only queue next frame if not showing save modal (pause physics)
            if (!showSaveModal && !pendingIncidentCoord) {
                requestRef.current = requestAnimationFrame(animate);
            }
        };

        if (!showSaveModal && !pendingIncidentCoord) {
            requestRef.current = requestAnimationFrame(animate);
        }

        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
            if (simulationRef.current) {
                // optional cleanup
            }
        };
    }, [showSaveModal, pendingIncidentCoord]); // Re-bind loop if save or config modal toggles

    // Sync Night Mode downwards to Physics Engine
    useEffect(() => {
        if (simulationRef.current) {
            simulationRef.current.isNightMode = isNightMode;
            syncState();
        }
    }, [isNightMode]);

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

        setIsRecording(sim.isRecording);
    };

    const handleAddIncident = React.useCallback((coordinate) => {
        setPendingIncidentCoord(coordinate);
    }, []);

    const handleAddIncidentDirect = React.useCallback((coordinate, type) => {
        if (simulationRef.current) {
            const config = HAZARD_DATABASE[type];
            const payload = {
                title: config ? config.name : 'Direct Incident',
                type: type,
                amount: 100, // Default spill amount
                desc: config ? config.description : 'Added via Hand Gestures'
            };
            simulationRef.current.addIncident(coordinate, payload);
            syncState();
        }
    }, []);

    const handleResolveIncident = React.useCallback((incidentId) => {
        if (simulationRef.current) {
            simulationRef.current.resolveIncident(incidentId);
            syncState();
        }
    }, []);

    const confirmAddIncident = React.useCallback((e) => {
        e.preventDefault();
        if (simulationRef.current && pendingIncidentCoord) {
            const payload = {
                title: incidentForm.Title,
                type: incidentForm.Type,
                amount: Number(incidentForm.Amount) || 100,
                desc: incidentForm.Details
            };
            simulationRef.current.addIncident(pendingIncidentCoord, payload);
            
            // Re-sync explicitly so the UI immediately shows the unpaused new array length
            syncState(); 
        }
        setPendingIncidentCoord(null);
        setIncidentForm({ Title: '', Type: 'CHLORINE_GAS', Amount: 100, Details: '' });
    }, [pendingIncidentCoord, incidentForm]);

    const handleWindChange = React.useCallback((newWind) => {
        if (simulationRef.current) {
            simulationRef.current.wind = newWind;
        }
    }, []);

    const cancelAddIncident = () => {
        setPendingIncidentCoord(null);
        setIncidentForm({ Title: '', Type: 'CHLORINE_GAS', Amount: 100, Details: '' });
    };

    const handleStartRecording = () => {
        if (simulationRef.current) {
            simulationRef.current.startRecording();
            setIsRecording(true);
            setSessionStartTime(Date.now());
        }
    };

    const handleReset = () => {
        if (simulationRef.current) {
            simulationRef.current.reset();
        }
        setShowSaveModal(false);
        setSaveSuccess(false);
        setSimName('');
        setSessionStartTime(null);
        setRating(5);
        setFocusLevel('medium');
    };

    const handleDiscardRecording = () => {
        if (simulationRef.current) {
            simulationRef.current.isRecording = false;
            simulationRef.current.eventLog = [];
            simulationRef.current.elapsedMs = 0;
            // This leaves the physics/agents where they are, just throws away the tape
        }
        setShowSaveModal(false);
        setSimName('');
    };

    const handleEndSim = () => {
        // Pauses physics (via useEffect dependency) and shows save dialog
        if (simulationRef.current && simulationRef.current.isRecording) {
            simulationRef.current.isRecording = false; // Stop clock
        }
        setShowSaveModal(true);
    };

    const handleSaveSimulation = async (e) => {
        e.preventDefault();
        setIsSaving(true);
        try {
            // Get currently authenticated user
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user ? user.id : null;

            if (!userId) {
                alert("You must be logged in to save a simulation.");
                setIsSaving(false);
                return;
            }

            const replayTape = simulationRef.current.compileReplayTape();
            const simulationPayload = {
                name: simName || `Simulation ${new Date().toLocaleString()}`,
                duration_ms: Math.floor(Number(replayTape.durationMs)),
                final_stats: replayTape.finalStats,
                frame_data: replayTape.initialState || replayTape.frameData,
                events: replayTape.events,
                logs: replayTape.logs,
                user_id: userId
            };

            // 1. Save Simulation Data to PostgreSQL via Supabase
            const { data: simData, error: simError } = await supabase
                .from('simulations')
                .insert([simulationPayload])
                .select();

            if (simError) {
                console.error("Supabase Error Data:", simError);
                throw new Error("Failed to save simulation replay Data.");
            }

            // 2. Save Session Data to PostgreSQL via Supabase
            if (sessionStartTime) {
                 const sessionPayload = {
                     user_id: userId,
                     start_time: sessionStartTime,
                     end_time: Date.now(),
                     duration: Math.floor((Date.now() - sessionStartTime) / 1000),
                     rating: Number(rating),
                     focus_level: focusLevel
                 };
                 
                 const { error: sessionError } = await supabase
                     .from('sessions')
                     .insert([sessionPayload]);
                 
                 if (sessionError) {
                     console.warn("PostgreSQL session save failed", sessionError.message);
                 }
            }

            setSaveSuccess(true);

            // Auto close/reset after 3s
            setTimeout(() => {
                handleReset();
            }, 3000);

        } catch (error) {
            console.error("Error saving simulation:", error);
            alert("Failed to save simulation. Check console.");
        } finally {
            setIsSaving(false);
        }
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
                className="absolute top-6 right-6 z-[60] bg-white/80 dark:bg-gray-900/80 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white p-2 rounded-full backdrop-blur border border-gray-200 dark:border-gray-700 hover:border-amber-500 transition-colors shadow-xl"
                title="Exit Simulation"
            >
                <XCircle className="w-6 h-6" />
            </button>
            
            <button
                onClick={() => setIsNightMode(!isNightMode)}
                className={`absolute top-20 right-6 z-[60] p-2 rounded-full backdrop-blur border transition-colors shadow-lg flex items-center justify-center ${
                    isNightMode 
                    ? 'bg-amber-500/90 text-amber-950 border-amber-600' 
                    : 'bg-white/80 dark:bg-gray-900/80 text-gray-600 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
                title={isNightMode ? "Night Mode Active (Extended PAD)" : "Enable Night Mode (ERG Padding)"}
            >
                {isNightMode ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>

            <div className="relative w-full h-full bg-gray-100 dark:bg-gray-950 overflow-hidden rounded-lg transition-colors duration-300">
                {/* Map Layer */}
                <MapComponent
                    agents={agents}
                    incidents={incidents}
                    safeNodes={safeNodes}
                    wind={wind}
                    isNightMode={isNightMode}
                    onWindChange={handleWindChange}
                    onLocationSelect={handleAddIncident}
                    onAddIncidentDirect={handleAddIncidentDirect}
                    onResolveIncident={handleResolveIncident}
                    onHandStatusChange={setIsHandDetected}
                    theme={theme}
                />

                {isHandDetected && (
                    <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-500/90 backdrop-blur text-white px-4 py-1.5 rounded-full text-sm font-bold shadow-lg animate-pulse z-50 flex items-center gap-2 border border-blue-400">
                        <Hand className="w-4 h-4" /> Headless Hand Tracking Active
                    </div>
                )}

                {/* Python Backend WebSocket Status */}
                <div className={`absolute top-4 right-20 z-50 flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border shadow-lg backdrop-blur transition-colors ${
                    wsBackendStatus === 'connected'
                        ? 'bg-green-500/20 text-green-400 border-green-500/40'
                        : wsBackendStatus === 'connecting'
                        ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 animate-pulse'
                        : 'bg-gray-800/60 text-gray-500 border-gray-700'
                }`}>
                    <Server className="w-3.5 h-3.5" />
                    Backend WS: {wsBackendStatus === 'connected' ? 'Online' : wsBackendStatus === 'connecting' ? 'Connecting…' : 'Offline'}
                    {wsBackendStatus === 'connected' && (
                        <span className="relative flex h-2 w-2 ml-1">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
                        </span>
                    )}
                </div>

                {/* UI Overlay */}
                <UIOverlay
                    stats={stats}
                    wind={wind}
                    status={status}
                    isRecording={isRecording}
                    onStartRecording={handleStartRecording}
                    onReset={handleReset}
                    onEndSim={handleEndSim}
                    theme={theme}
                />
            </div>

            {/* Save Simulation Modal */}
            <AnimatePresence>
                {pendingIncidentCoord && (
                    <div className="absolute inset-0 z-[70] flex items-center justify-center bg-gray-950/60 backdrop-blur-sm pointer-events-auto p-4">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl p-6 w-full max-w-lg shadow-2xl relative"
                        >
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <div className="w-10 h-10 rounded-lg bg-red-50 dark:bg-red-500/10 flex items-center justify-center border border-red-200 dark:border-red-500/20">
                                        <ShieldAlert className="w-5 h-5 text-red-500" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Configure Hazard</h2>
                                        <p className="text-xs text-gray-500 font-mono">
                                            LOC: [{pendingIncidentCoord[1].toFixed(4)}, {pendingIncidentCoord[0].toFixed(4)}]
                                        </p>
                                    </div>
                                </div>
                                <button onClick={cancelAddIncident} className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-full text-gray-500">
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <form onSubmit={confirmAddIncident} className="space-y-5">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Hazard Type</label>
                                        <select
                                            value={incidentForm.Type}
                                            onChange={e => setIncidentForm({...incidentForm, Type: e.target.value})}
                                            className="w-full bg-white dark:bg-gray-950 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-sm focus:ring-1 focus:ring-red-500 outline-none"
                                        >
                                            {Object.entries(HAZARD_DATABASE).map(([key, config]) => (
                                                <option key={key} value={key}>
                                                    {config.name}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Amount Spilled (kg or L)</label>
                                        <input
                                            type="number"
                                            value={incidentForm.Amount}
                                            onChange={e => setIncidentForm({...incidentForm, Amount: e.target.value})}
                                            min="1"
                                            max="50000"
                                            className="w-full bg-white dark:bg-gray-950 border border-gray-300 dark:border-gray-700 rounded-lg px-3 py-2 text-sm focus:ring-1 focus:ring-red-500 outline-none"
                                            placeholder="e.g. 500"
                                        />
                                    </div>
                                </div>

                                <div className="flex justify-end gap-3 pt-4 border-t border-gray-100 dark:border-gray-800">
                                    <button type="button" onClick={cancelAddIncident} className="px-5 py-2 text-sm font-semibold text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors">
                                        Cancel
                                    </button>
                                    <button type="submit" className="px-5 py-2 text-sm font-semibold text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors flex items-center gap-2 shadow-lg shadow-red-500/30">
                                        <ShieldAlert className="w-4 h-4" /> Drop Initial Payload
                                    </button>
                                </div>
                            </form>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Save Simulation Modal */}
            <AnimatePresence>
                {showSaveModal && (
                    <div className="absolute inset-0 z-[70] flex items-center justify-center bg-gray-950/60 backdrop-blur-sm pointer-events-auto p-4">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl p-6 md:p-8 max-w-md w-full shadow-2xl relative overflow-hidden transition-colors"
                        >
                            {saveSuccess ? (
                                <div className="text-center py-8">
                                    <CheckCircle2 className="w-16 h-16 text-green-500 mx-auto mb-4" />
                                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Simulation Saved!</h3>
                                    <p className="text-gray-500 dark:text-gray-400">Restarting sandbox environment...</p>
                                </div>
                            ) : (
                                <>
                                    <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Recording Ended</h3>
                                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-6">
                                        The simulation physics have been paused. You can discard this recording and resume, or save the tape to the database.
                                    </p>

                                    {/* Stats Summary Bubble */}
                                    <div className="bg-gray-50 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-800 rounded-xl p-4 mb-6 grid grid-cols-2 gap-4">
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider font-semibold">Total Duration</p>
                                            <p className="text-xl font-mono text-gray-900 dark:text-white font-medium">
                                                {simulationRef.current ? (simulationRef.current.elapsedMs / 1000).toFixed(1) : 0}s
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider font-semibold">Final Safety Index</p>
                                            <p className={`text-xl font-mono font-medium ${stats.safetyIndex < 50 ? 'text-red-500' : 'text-green-500'}`}>
                                                {stats.safetyIndex}%
                                            </p>
                                        </div>
                                    </div>

                                    <form onSubmit={handleSaveSimulation}>
                                        <div className="mb-6 space-y-4">
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                    Simulation Name (Optional)
                                                </label>
                                                <input
                                                    type="text"
                                                    value={simName}
                                                    onChange={(e) => setSimName(e.target.value)}
                                                    className="w-full bg-white dark:bg-gray-950 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors placeholder-gray-400 dark:placeholder-gray-600"
                                                    placeholder="e.g. High Wind Chemical Spill Test"
                                                    autoFocus
                                                />
                                            </div>
                                            
                                            <div className="grid grid-cols-2 gap-4">
                                                <div>
                                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                        Rating (1-5)
                                                    </label>
                                                    <select
                                                        value={rating}
                                                        onChange={(e) => setRating(e.target.value)}
                                                        className="w-full bg-white dark:bg-gray-950 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
                                                    >
                                                        {[1, 2, 3, 4, 5].map(num => (
                                                            <option key={num} value={num}>{num} {num === 5 ? '(Excellent)' : ''}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                                        Focus Level
                                                    </label>
                                                    <select
                                                        value={focusLevel}
                                                        onChange={(e) => setFocusLevel(e.target.value)}
                                                        className="w-full bg-white dark:bg-gray-950 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
                                                    >
                                                        <option value="low">Low</option>
                                                        <option value="medium">Medium</option>
                                                        <option value="high">High</option>
                                                    </select>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="flex items-center justify-end gap-3 mt-8">
                                            <button
                                                type="button"
                                                onClick={handleDiscardRecording}
                                                className="px-5 py-2.5 rounded-xl font-medium text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                                            >
                                                Discard Recording
                                            </button>
                                            <button
                                                type="submit"
                                                disabled={isSaving}
                                                className="px-5 py-2.5 rounded-xl font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                            >
                                                {isSaving ? 'Saving...' : 'Save to Database'}
                                            </button>
                                        </div>
                                    </form>
                                </>
                            )}
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default SimulationMode;
