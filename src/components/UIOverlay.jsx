import React from 'react';
import { Wind, AlertCircle, ShieldCheck, RefreshCw, Moon, Sun, Circle, Square, Thermometer, Droplets, CloudRain } from 'lucide-react';

const UIOverlay = ({ stats, wind, status, isRecording, onStartRecording, onReset, onEndSim, theme, onToggleTheme }) => {
    // Rotation logic
    const rotation = (wind.direction + 180) % 360;
    const isLight = theme === 'light';

    // Premium Glassmorphism aesthetic based on Live/Replay standard
    const cardClass = isLight
        ? "bg-white/90 backdrop-blur-2xl border border-gray-200/50 shadow-[0_8px_32px_rgba(0,0,0,0.12)] before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/5 before:to-purple-500/5 before:-z-10 group overflow-hidden"
        : "bg-[#0b1120]/80 backdrop-blur-2xl border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.3)] before:absolute before:inset-0 before:bg-gradient-to-br before:from-blue-500/5 before:to-purple-500/5 before:-z-10 group overflow-hidden";

    return (
        <div className="absolute top-4 left-4 flex flex-col gap-4 pointer-events-none select-none z-50">
            {/* Main Stats Card */}
            <div className={`p-5 rounded-3xl w-72 transition-all duration-300 relative ${cardClass}`}>
                
                {/* Header */}
                <div className="flex justify-between items-start mb-4 border-b border-gray-200/50 dark:border-white/10 pb-3">
                    <h1 className="text-xl font-black tracking-tight bg-gradient-to-r from-blue-600 to-teal-500 dark:from-blue-400 dark:to-teal-300 bg-clip-text text-transparent">
                        CrowdShield
                    </h1>
                    {onToggleTheme && (
                        <button
                            onClick={onToggleTheme}
                            className={`pointer-events-auto p-2 rounded-full transition-colors ${isLight ? 'hover:bg-gray-100 text-gray-500' : 'hover:bg-white/10 text-gray-400'}`}
                        >
                            {isLight ? <Moon size={16} /> : <Sun size={16} />}
                        </button>
                    )}
                </div>

                {/* Status Indicator */}
                <div className="flex items-center gap-2 mb-5">
                    <div className={`w-2.5 h-2.5 rounded-full shadow-sm ${status === 'Evacuating' ? 'bg-red-500 animate-pulse shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]'}`} />
                    <span className={`text-[10px] font-black tracking-widest uppercase ${status === 'Evacuating' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                        {status}
                    </span>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-x-8 gap-y-4 mb-4">
                    <div>
                        <p className={`text-[10px] font-bold uppercase tracking-widest mb-1 ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
                            <span className="flex items-center gap-1.5"><AlertCircle size={10} /> Incidents</span>
                        </p>
                        <p className={`text-xl font-black ${isLight ? 'text-gray-900' : 'text-white'}`}>{stats.activeIncidents}</p>
                    </div>
                    <div>
                        <p className={`text-[10px] font-bold uppercase tracking-widest mb-1 ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
                            <span className="flex items-center gap-1.5"><ShieldCheck size={10} /> Safety Index</span>
                        </p>
                        <p className={`text-xl font-black ${stats.safetyIndex < 50 ? 'text-red-500 dark:text-red-400' : 'text-green-600 dark:text-green-400'}`}>
                            {stats.safetyIndex}%
                        </p>
                    </div>
                </div>

                {/* Weather Grid */}
                <div className="grid grid-cols-3 gap-2 border-t border-gray-200/50 dark:border-white/10 pt-4 mb-2">
                    <div className="text-center">
                        <Thermometer size={14} className="mx-auto mb-1 text-orange-500" />
                        <p className={`text-[10px] font-bold ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>{wind.weather?.temp?.toFixed(1) || '--'}°C</p>
                    </div>
                    <div className="text-center">
                        <Droplets size={14} className="mx-auto mb-1 text-blue-500" />
                        <p className={`text-[10px] font-bold ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>{wind.weather?.hum?.toFixed(0) || '--'}%</p>
                    </div>
                    <div className="text-center">
                        <CloudRain size={14} className="mx-auto mb-1 text-teal-500" />
                        <p className={`text-[10px] font-bold ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>{wind.weather?.rain?.toFixed(1) || '--'}mm</p>
                    </div>
                </div>

                {/* Aesthetic Wind Dashboard Slice */}
                <div className={`mt-5 bg-gradient-to-r ${isLight ? 'from-gray-50/80 to-transparent' : 'from-white/5 to-transparent'} border-t ${isLight ? 'border-gray-200/50' : 'border-white/5'} p-4 -mx-5 -mb-5 flex items-center justify-between group overflow-hidden relative`}>
                    <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    
                    <div className="relative z-10 flex flex-col">
                        <span className={`text-[10px] font-bold uppercase tracking-widest mb-1 ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
                            <span className="flex items-center gap-1.5"><Wind size={10} className={isLight ? 'text-blue-500' : 'text-blue-400'}/> Wind Speed</span>
                        </span>
                        <div className="flex items-baseline gap-1">
                            <span className={`text-lg font-black ${isLight ? 'text-gray-900' : 'text-white'}`}>{wind.speed.toFixed(1)}</span>
                            <span className={`text-[10px] font-bold ${isLight ? 'text-gray-400' : 'text-gray-500'}`}>km/h</span>
                        </div>
                    </div>

                    {/* Minimalist Compass */}
                    <div className="relative z-10">
                        <div className={`relative w-10 h-10 border rounded-2xl flex items-center justify-center transition-transform hover:scale-110 duration-300 ${isLight ? 'border-blue-200 bg-blue-50 dark:border-blue-500/20 dark:bg-blue-500/10' : 'border-blue-500/30 bg-blue-500/20'}`}>
                            <div
                                style={{ transform: `rotate(${rotation}deg)` }}
                                className="transition-transform duration-500 ease-out"
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className={isLight ? "text-blue-600" : "text-blue-400"}>
                                    <line x1="12" y1="19" x2="12" y2="5"></line>
                                    <polyline points="5 12 12 5 19 12"></polyline>
                                </svg>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Floating Action Stack */}
            <div className="flex gap-2 pointer-events-auto mt-1 w-72">
                {!isRecording ? (
                    <button
                        onClick={onStartRecording}
                        className="flex-1 p-3 rounded-2xl backdrop-blur-2xl transition-all flex items-center justify-center gap-2 text-xs font-black tracking-wide bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_20px_rgba(220,38,38,0.5)] border border-red-500/50"
                    >
                        <Circle size={14} fill="currentColor" className="animate-pulse" /> START RECORDING
                    </button>
                ) : (
                    <button
                        onClick={onEndSim}
                        className="flex-1 p-3 rounded-2xl backdrop-blur-2xl transition-all flex items-center justify-center gap-2 text-xs font-black tracking-wide bg-gradient-to-r from-amber-500 to-amber-400 hover:from-amber-400 hover:to-amber-300 text-amber-950 shadow-[0_0_15px_rgba(245,158,11,0.3)] hover:shadow-[0_0_20px_rgba(245,158,11,0.5)] border border-amber-400/50"
                    >
                        <Square size={14} fill="currentColor" /> STOP REPLAY
                    </button>
                )}
                <button
                    onClick={onReset}
                    className={`w-12 h-12 flex-shrink-0 flex items-center justify-center rounded-2xl backdrop-blur-2xl transition-all border ${isLight ? 'bg-white/80 border-gray-200/50 text-gray-500 hover:bg-gray-50 hover:text-gray-900 shadow-sm' : 'bg-[#0b1120]/80 border-white/10 text-gray-400 hover:bg-white/5 hover:text-white shadow-[0_8px_32px_rgba(0,0,0,0.3)]'}`}
                    title="Reset Simulation"
                >
                    <RefreshCw size={16} />
                </button>
            </div>
        </div>
    );
};

export default UIOverlay;
