import React from 'react';
import { Wind, AlertCircle, ShieldCheck, RefreshCw, Moon, Sun } from 'lucide-react';

const UIOverlay = ({ stats, wind, status, onReset, theme, onToggleTheme }) => {
    // Rotation logic
    const rotation = (wind.direction + 180) % 360;

    // --- Dynamic Styles based on Theme ---
    const isLight = theme === 'light';

    // Container: Glassmorphism effect
    // Light: White Glass (bg-white/80) with Dark Text
    // Dark: Black Glass (bg-black/70) with White Text
    const cardClass = isLight
        ? "bg-white/90 backdrop-blur-md text-gray-900 border border-gray-200 shadow-xl"
        : "bg-black/70 backdrop-blur-md text-white border border-white/10 shadow-2xl";

    const labelClass = isLight ? "text-gray-500" : "text-gray-400";
    const valueClass = isLight ? "text-gray-900" : "text-white";

    // Reset Button: 
    // Light Map -> Dark Button for contrast
    // Dark Map -> Light Glass Button
    const buttonClass = isLight
        ? "bg-gray-800 hover:bg-gray-700 text-white border border-gray-700 shadow-lg"
        : "bg-white/10 hover:bg-white/20 text-white border border-white/10";

    return (
        <div className="absolute top-4 left-4 flex flex-col gap-2 pointer-events-none select-none">
            {/* Main Stats Card */}
            <div className={`p-4 rounded-xl w-64 transition-colors duration-300 ${cardClass}`}>
                <div className="flex justify-between items-start mb-2">
                    <h1 className="text-xl font-bold bg-gradient-to-r from-blue-500 to-teal-500 bg-clip-text text-transparent">
                        CrowdShield
                    </h1>
                    {onToggleTheme && (
                        <button
                            onClick={onToggleTheme}
                            className="pointer-events-auto p-1 rounded-md hover:bg-gray-200/20 text-gray-400 hover:text-gray-600 transition-colors"
                        >
                            {isLight ? <Moon size={16} /> : <Sun size={16} />}
                        </button>
                    )}
                </div>

                {/* Status Indicator */}
                <div className="flex items-center gap-2 mb-4">
                    <div className={`w-3 h-3 rounded-full ${status === 'Evacuating' ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`} />
                    <span className={`font-semibold tracking-wide uppercase text-sm ${valueClass}`}>{status}</span>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="flex flex-col">
                        <span className={`text-xs flex items-center gap-1 ${labelClass}`}>
                            <AlertCircle size={12} /> Incidents
                        </span>
                        <span className={`text-2xl font-mono ${valueClass}`}>{stats.activeIncidents}</span>
                    </div>
                    <div className="flex flex-col">
                        <span className={`text-xs flex items-center gap-1 ${labelClass}`}>
                            <ShieldCheck size={12} /> Safety Index
                        </span>
                        <span className={`text-2xl font-mono ${stats.safetyIndex < 50 ? 'text-red-500' : 'text-green-500'}`}>
                            {stats.safetyIndex}%
                        </span>
                    </div>
                </div>

                {/* Wind Widget */}
                <div className={`mt-4 pt-4 border-t ${isLight ? 'border-gray-200' : 'border-white/10'} flex items-center justify-between`}>
                    <div className="flex flex-col">
                        <span className={`text-xs flex items-center gap-1 ${labelClass}`}>
                            <Wind size={12} /> Wind (km/h)
                        </span>
                        <span className={`text-lg font-mono ${valueClass}`}>{wind.speed.toFixed(1)}</span>
                    </div>

                    {/* Wind Compass */}
                    <div className={`relative w-10 h-10 border rounded-full flex items-center justify-center ${isLight ? 'border-gray-300 bg-gray-50' : 'border-white/20 bg-white/5'}`}>
                        <div
                            style={{ transform: `rotate(${rotation}deg)` }}
                            className="transition-transform duration-500 ease-out"
                        >
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={isLight ? "text-blue-600" : "text-blue-300"}>
                                <line x1="12" y1="19" x2="12" y2="5"></line>
                                <polyline points="5 12 12 5 19 12"></polyline>
                            </svg>
                        </div>
                        <span className="absolute text-[8px] top-0.5 text-gray-500">N</span>
                    </div>
                </div>
            </div>

            {/* Actions */}
            <div className="flex gap-2 pointer-events-auto">
                <button
                    onClick={onReset}
                    className={`p-2 rounded-lg backdrop-blur-md transition-all flex items-center gap-2 text-sm font-medium ${buttonClass}`}
                >
                    <RefreshCw size={16} /> Reset Simulation
                </button>
            </div>
        </div>
    );
};

export default UIOverlay;
