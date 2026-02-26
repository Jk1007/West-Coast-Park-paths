import React, { useState } from 'react';
import { Home, ExternalLink, Shield } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Navigation = ({ currentMode, onNavigate, onAttemptNavigate }) => {
    const isLanding = currentMode === 'landing';
    const isSim = currentMode === 'simulation';
    const isLive = currentMode === 'live';

    return (
        <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="w-full h-16 bg-gray-950/80 backdrop-blur-lg border-b border-gray-800 flex items-center justify-between px-6 shrink-0 relative z-50 shadow-md"
        >
            {/* Brand / Logo Area */}
            <div
                className="flex items-center gap-3 cursor-pointer group"
                onClick={() => !isLanding && onAttemptNavigate('landing')}
            >
                <div className="w-8 h-8 rounded-lg bg-gray-900 border border-gray-700 flex items-center justify-center group-hover:border-gray-500 transition-colors">
                    <Shield className="w-4 h-4 text-gray-100" />
                </div>
                <span className="font-bold text-lg text-white tracking-wide">
                    Crowd<span className="text-gray-400">Shield</span>
                </span>
            </div>

            {/* Status Indicator */}
            {!isLanding && (
                <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 flex items-center gap-2 px-4 py-1.5 rounded-full border bg-gray-900/50 backdrop-blur"
                    style={{
                        borderColor: isSim ? 'rgba(245, 158, 11, 0.3)' : 'rgba(59, 130, 246, 0.3)',
                    }}>
                    <div className={`w-2 h-2 rounded-full animate-pulse ${isSim ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]' : 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]'}`} />
                    <span className="text-sm font-medium text-gray-200">
                        {isSim ? "Sandbox Environment" : "Production Entry"}
                    </span>
                </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-4">
                {!isLanding && (
                    <button
                        onClick={() => onAttemptNavigate('landing')}
                        className="flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-white bg-gray-900 hover:bg-gray-800 px-4 py-2 rounded-md transition-colors border border-gray-800"
                    >
                        <Home className="w-4 h-4" />
                        Return Hub
                    </button>
                )}

                {!isLanding && isSim && (
                    <button
                        onClick={() => onAttemptNavigate('live')}
                        className="flex items-center gap-2 text-sm font-medium text-blue-400 hover:text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 px-4 py-2 rounded-md transition-colors border border-blue-500/20"
                    >
                        Switch to Live
                        <ExternalLink className="w-4 h-4" />
                    </button>
                )}

                {!isLanding && isLive && (
                    <button
                        onClick={() => onAttemptNavigate('simulation')}
                        className="flex items-center gap-2 text-sm font-medium text-amber-500 hover:text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 px-4 py-2 rounded-md transition-colors border border-amber-500/20"
                    >
                        Switch to Sandbox
                        <ExternalLink className="w-4 h-4" />
                    </button>
                )}
            </div>
        </motion.div>
    );
};

export default Navigation;
