import React, { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MapComponent from './MapComponent';
import { XCircle, Play, Pause, RotateCcw, Wind } from 'lucide-react';
import { SimulationController } from '../simulation/SimulationController';

const SimulationReplayMode = ({ simData, onExit, theme }) => {
    const simulationRef = useRef(null);
    const requestRef = useRef();
    const lastSafeNodesVersion = useRef(-1);

    // Playback Controls
    const [isPlaying, setIsPlaying] = useState(false);
    const [playbackSpeed, setPlaybackSpeed] = useState(1);
    const [playbackProgress, setPlaybackProgress] = useState(0); // 0 to 1

    // Derived Visualization State
    const [agents, setAgents] = useState([]);
    const [incidents, setIncidents] = useState([]);
    const [wind, setWind] = useState({ speed: 0, direction: 0, stabilityClass: 'D', weather: { temp: 28, rain: 0, hum: 80 } });
    const [safeNodes, setSafeNodes] = useState([]);
    const [stats, setStats] = useState({ activeIncidents: 0, safetyIndex: 100 });

    const SIM_DURATION = Number(simData?.durationMs) || 10000;

    const syncState = () => {
        if (!simulationRef.current) return;
        const sim = simulationRef.current;

        setAgents([...sim.agents]);
        setIncidents([...sim.incidents]);
        setWind({ ...sim.wind });
        setStats(sim.getStats());

        if (sim.safeNodesVersion !== lastSafeNodesVersion.current) {
            setSafeNodes([...sim.safeNodes]);
            lastSafeNodesVersion.current = sim.safeNodesVersion;
        }

        // Calculate progress percentage
        const progress = Math.min(sim.elapsedMs / SIM_DURATION, 1);
        setPlaybackProgress(progress);

        if (progress >= 1 && isPlaying) {
            setIsPlaying(false);
        }
    };

    // Initialization & Reset
    const resetReplay = () => {
        setIsPlaying(false);
        if (!simData) return;

        // Create a fresh isolated controller specifically for this replay tape
        simulationRef.current = new SimulationController();
        // Disconnect its internal initialization and force it to load the tape
        simulationRef.current.loadReplayTape(simData);

        syncState();
    };

    // On mount or explicit reset
    useEffect(() => {
        resetReplay();
    }, [simData]);

    // Main Playback Loop
    useEffect(() => {
        if (!isPlaying || !simData || !simulationRef.current) {
            // Ensure any pending animation frame is cancelled when playback stops or data is missing
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
                requestRef.current = null;
            }
            return;
        }

        let lastTime = performance.now();
        let accumulator = 0;
        const fpsInterval = 1000 / 30; // Target 30 FPS to prevent WebWorker backlog

        const animate = (time) => {
            const dtMs = time - lastTime;
            lastTime = time;
            
            accumulator += dtMs;

            if (accumulator >= fpsInterval) {
                // Determine step time mapping for physics
                const syncDt = Math.min(accumulator / 1000, 0.1) * playbackSpeed;

                // Advance the isolated physics & event engine
                simulationRef.current.update(syncDt);
                syncState();

                // Calculate progress percentage
                const progress = Math.min(simulationRef.current.elapsedMs / SIM_DURATION, 1);
                setPlaybackProgress(progress);

                if (progress >= 1 && isPlaying) {
                    setIsPlaying(false);
                }
                
                accumulator = accumulator % fpsInterval;
            }

            if (isPlaying && simulationRef.current.elapsedMs < SIM_DURATION) {
                requestRef.current = requestAnimationFrame(animate);
            } else {
                // Ensure animation stops if playback ends or isPlaying becomes false
                if (requestRef.current) {
                    cancelAnimationFrame(requestRef.current);
                    requestRef.current = null;
                }
            }
        };

        requestRef.current = requestAnimationFrame(animate);

        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        };
    }, [isPlaying, playbackSpeed, SIM_DURATION]);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="w-full h-full relative"
        >
            {/* Replay Thematic Wrapper */}
            <div className="absolute inset-0 border-4 border-blue-500/30 rounded-lg pointer-events-none z-[60]" />
            <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-500 text-blue-950 px-6 py-1 rounded-b-lg font-bold text-xs uppercase tracking-widest pointer-events-none z-[60] shadow-xl flex items-center gap-2">
                <Play className="w-3 h-3" /> Replay Viewer
            </div>

            <button
                onClick={onExit}
                className="absolute top-6 right-6 z-[60] bg-white/80 dark:bg-gray-900/80 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white p-2 rounded-full backdrop-blur border border-gray-200 dark:border-gray-700 hover:border-blue-500 transition-colors shadow-xl"
            >
                <XCircle className="w-6 h-6" />
            </button>

            <div className="relative w-full h-full bg-gray-100 dark:bg-gray-950 overflow-hidden rounded-lg transition-colors duration-300">
                <MapComponent
                    agents={agents}
                    incidents={incidents}
                    safeNodes={safeNodes}
                    wind={wind}
                    theme={theme}
                />

                {/* Recorded Atmospheric Conditions Widget */}
                {wind && (
                    <div className="pointer-events-none absolute top-4 left-4 z-[60] bg-white/90 dark:bg-[#0b1120]/80 backdrop-blur-2xl border border-gray-200/50 dark:border-white/10 p-5 rounded-3xl shadow-[0_8px_32px_rgba(0,0,0,0.12)] w-72 flex flex-col gap-4 overflow-hidden before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/5 before:to-purple-500/5 before:-z-10 group">
                        <div className="flex items-center gap-2 mb-1">
                            <Wind className="w-5 h-5 text-blue-500 dark:text-blue-400 group-hover:scale-110 transition-transform duration-300" />
                            <h3 className="text-xs font-black text-gray-900 dark:text-white uppercase tracking-wider">Recorded Atmosphere</h3>
                        </div>
                        <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                            <div>
                                <p className="text-[10px] font-bold text-gray-400 tracking-widest mb-1.5 uppercase">Wind Speed</p>
                                <div className="flex items-baseline gap-1.5">
                                    <span className={`text-xl font-black ${wind.speed > 25 ? 'text-red-500 dark:text-red-400' : 'text-gray-900 dark:text-white'}`}>{wind.speed?.toFixed(1) || '0.0'}</span>
                                    <span className="text-xs font-bold text-gray-500">km/h</span>
                                </div>
                            </div>
                            <div className="flex flex-col">
                                <p className="text-[10px] font-bold text-gray-400 tracking-widest mb-1.5 uppercase">Direction</p>
                                <div className="flex items-center gap-3 mt-1">
                                    <div className="relative w-9 h-9 rounded-2xl border border-blue-500/20 bg-blue-500/10 flex justify-center items-center transition-transform hover:scale-110 duration-300">
                                        <div
                                            style={{ transform: `rotate(${((wind.direction || 0) + 180) % 360}deg)` }}
                                            className="transition-transform duration-500 ease-out flex items-center justify-center pt-0.5"
                                        >
                                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400">
                                                <line x1="12" y1="19" x2="12" y2="5"></line>
                                                <polyline points="5 12 12 5 19 12"></polyline>
                                            </svg>
                                        </div>
                                    </div>
                                    <span className="text-xl font-black text-gray-900 dark:text-white">{wind.direction?.toFixed(0) || '0'}°</span>
                                </div>
                            </div>
                            {wind.weather && (
                                <>
                                    <div className="relative">
                                        <div className="absolute -left-3 top-2 bottom-2 w-px bg-gray-200 dark:bg-gray-800" />
                                        <p className="text-[10px] font-bold text-gray-400 tracking-widest mb-1.5 uppercase">Temp</p>
                                        <div className="flex items-baseline gap-1.5">
                                            <span className="text-xl font-black text-gray-900 dark:text-white">{wind.weather.temp?.toFixed(1) || '0.0'}</span>
                                            <span className="text-xs font-bold text-gray-500">°C</span>
                                        </div>
                                    </div>
                                    <div className="relative">
                                        <div className="absolute -left-3 top-2 bottom-2 w-px bg-gray-200 dark:bg-gray-800" />
                                        <p className="text-[10px] font-bold text-gray-400 tracking-widest mb-1.5 uppercase">Humidity</p>
                                        <div className="flex items-baseline gap-1.5">
                                            <span className="text-xl font-black text-gray-900 dark:text-white">{wind.weather.hum?.toFixed(0) || '0'}</span>
                                            <span className="text-xs font-bold text-gray-500">%</span>
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                        <div className="mt-2 pt-4 border-t border-gray-200 dark:border-white/5 flex justify-between items-center bg-gray-50/50 dark:bg-transparent -mx-5 px-5 -mb-5 pb-5">
                            <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Pasquill-Gifford</span>
                            <span className="text-[10px] font-black bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-400 px-2.5 py-1 rounded-md border border-blue-200 dark:border-blue-500/30 shadow-sm">
                                Class {wind.stabilityClass || 'D'}
                            </span>
                        </div>
                    </div>
                )}

                {/* Playback Controls Overlay */}
                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border border-gray-200 dark:border-gray-800 rounded-2xl shadow-2xl p-4 w-11/12 max-w-2xl z-[60] flex flex-col gap-3">

                    {/* Header Details */}
                    <div className="flex justify-between items-center px-2">
                        <div className="flex flex-col">
                            <span className="text-xs text-gray-500 uppercase tracking-widest font-semibold">{simData?.name || "Recorded Simulation"}</span>
                            <span className="text-sm font-mono text-blue-600 dark:text-blue-400">
                                Safety Index: {stats?.safetyIndex ?? 0}%
                            </span>
                        </div>
                        <div className="text-xs font-mono text-gray-500 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded-full">
                            {((playbackProgress * SIM_DURATION) / 1000).toFixed(1)}s / {(SIM_DURATION / 1000).toFixed(1)}s
                        </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="w-full h-2 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-blue-500"
                            style={{ width: `${playbackProgress * 100}%` }}
                        />
                    </div>

                    {/* Media Buttons */}
                    <div className="flex items-center justify-between mt-1">
                        <div className="flex gap-2">
                            <button
                                onClick={resetReplay}
                                className="p-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                            >
                                <RotateCcw className="w-5 h-5" />
                            </button>
                            <button
                                onClick={() => playbackProgress < 1 && setIsPlaying(!isPlaying)}
                                className={`p-2 rounded-full flex items-center justify-center transition-colors ${playbackProgress >= 1 ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-100 dark:hover:bg-blue-900/50'} ${isPlaying ? 'text-blue-600 dark:text-blue-400' : 'text-gray-600 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400'}`}
                            >
                                {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
                            </button>
                        </div>

                        {/* Speed Toggle */}
                        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
                            {[1, 2, 5].map(speed => (
                                <button
                                    key={speed}
                                    onClick={() => setPlaybackSpeed(speed)}
                                    className={`px-3 py-1 text-xs font-mono rounded-md transition-colors ${playbackSpeed === speed ? 'bg-white dark:bg-gray-700 shadow-sm text-blue-600 dark:text-blue-400 font-bold' : 'text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'}`}
                                >
                                    {speed}x
                                </button>
                            ))}
                        </div>
                    </div>

                </div>
            </div>
        </motion.div>
    );
};

export default SimulationReplayMode;
