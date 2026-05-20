import React, { useState } from 'react';
import { Home, ExternalLink, Shield, Moon, Sun, LogOut } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import supabase from '../supabase';

const Navigation = ({ currentMode, onNavigate, onAttemptNavigate, theme, toggleTheme }) => {
    const isLanding = currentMode === 'landing';
    const isSim = currentMode === 'simulation';
    const isLive = currentMode === 'live';

    const handleSignOut = async () => {
        localStorage.setItem('manually_signed_out', 'true');
        await supabase.auth.signOut();
    };

    return (
        <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="w-full h-16 bg-white/80 dark:bg-gray-950/80 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-6 shrink-0 relative z-50 shadow-md transition-colors duration-300"
        >
            {/* Brand / Logo Area */}
            <div
                className="flex items-center gap-3 cursor-pointer group"
                onClick={() => !isLanding && onAttemptNavigate('landing')}
            >
                <div className="w-10 h-10 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 group-hover:border-brand-light dark:group-hover:border-gray-500 transition-colors flex bg-brand-dark justify-center items-center">
                    <img src="/logo.jpg" alt="CrowdShield Logo" className="w-full h-full object-cover object-center scale-[1.3]" />
                </div>
                <span className="font-bold text-lg text-brand-dark dark:text-white tracking-wide transition-colors">
                    Crowd<span className="text-brand-light dark:text-gray-400">Shield</span>
                </span>
            </div>

            {/* Status Indicator */}
            {!isLanding && (
                <div className="absolute left-1/2 -translate-x-1/2 top-1/2 -translate-y-1/2 flex items-center gap-2 px-4 py-1.5 rounded-full border bg-white/50 dark:bg-gray-900/50 backdrop-blur transition-colors"
                    style={{
                        borderColor: isSim ? 'rgba(245, 158, 11, 0.3)' : 'rgba(59, 130, 246, 0.3)',
                    }}>
                    <div className={`w-2 h-2 rounded-full animate-pulse ${isSim ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.6)]' : 'bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)]'}`} />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
                        {isSim ? "Sandbox Environment" : "Production Entry"}
                    </span>
                </div>
            )}

            {/* Actions */}
            <div className="flex items-center gap-4">
                {!isLanding && (
                    <button
                        onClick={() => onAttemptNavigate('landing')}
                        className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white bg-gray-100 dark:bg-gray-900 hover:bg-gray-200 dark:hover:bg-gray-800 px-4 py-2 rounded-md transition-colors border border-gray-200 dark:border-gray-800"
                    >
                        <Home className="w-4 h-4" />
                        Return Hub
                    </button>
                )}

                {!isLanding && isSim && (
                    <button
                        onClick={() => onAttemptNavigate('live')}
                        className="flex items-center gap-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 bg-blue-50 dark:bg-blue-500/10 hover:bg-blue-100 dark:hover:bg-blue-500/20 px-4 py-2 rounded-md transition-colors border border-blue-200 dark:border-blue-500/20"
                    >
                        Switch to Live
                        <ExternalLink className="w-4 h-4" />
                    </button>
                )}

                {!isLanding && isLive && (
                    <button
                        onClick={() => onAttemptNavigate('simulation')}
                        className="flex items-center gap-2 text-sm font-medium text-amber-600 dark:text-amber-500 hover:text-amber-700 dark:hover:text-amber-400 bg-amber-50 dark:bg-amber-500/10 hover:bg-amber-100 dark:hover:bg-amber-500/20 px-4 py-2 rounded-md transition-colors border border-amber-200 dark:border-amber-500/20"
                    >
                        Switch to Sandbox
                        <ExternalLink className="w-4 h-4" />
                    </button>
                )}

                <div className="w-px h-6 bg-gray-300 dark:bg-gray-700 mx-2"></div>

                <button
                    onClick={handleSignOut}
                    className="flex items-center gap-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                    title="Sign Out securely"
                >
                    <LogOut className="w-4 h-4" />
                    <span className="hidden sm:inline">Sign Out</span>
                </button>

                <div className="w-px h-6 bg-gray-300 dark:bg-gray-700 mx-2"></div>

                <button
                    onClick={toggleTheme}
                    className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-white/10 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
                >
                    {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
                </button>
            </div>
        </motion.div>
    );
};

export default Navigation;
