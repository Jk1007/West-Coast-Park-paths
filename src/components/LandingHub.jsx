import React from 'react';
import { motion } from 'framer-motion';
import { Activity, ShieldAlert } from 'lucide-react';
import SavedSimulationsList from './SavedSimulationsList';

const LandingHub = ({ onSelectMode }) => {
    return (
        <div className="w-full h-full flex flex-col items-center py-12 px-6 overflow-y-auto relative transition-colors duration-300">

            {/* Ambient Background Glows */}
            <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-amber-500/20 dark:bg-amber-500/10 rounded-full blur-3xl pointer-events-none transition-colors" />
            <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-brand-light/30 dark:bg-brand-light/10 rounded-full blur-3xl pointer-events-none transition-colors" />
            <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-purple-500/20 dark:bg-purple-500/10 rounded-full blur-3xl pointer-events-none -translate-x-1/2 -translate-y-1/2 transition-colors" />

            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="text-center mb-16 z-10"
            >
                <h1 className="text-5xl font-extrabold text-brand-dark dark:text-white mb-4 tracking-tight transition-colors">
                    Crowd<span className="text-brand-light">Shield</span> Hub
                </h1>
                <p className="text-gray-600 dark:text-gray-300 text-lg max-w-2xl mx-auto transition-colors">
                    Select an operational environment. Run sandbox simulations, report active live incidents, or review ongoing reports mapped directly from the database.
                </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl z-10">

                {/* Simulation Mode Card */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    whileHover={{ scale: 1.02, y: -5 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => onSelectMode('simulation')}
                    className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-800 hover:border-amber-500/50 rounded-2xl p-8 cursor-pointer group transition-colors shadow-xl dark:shadow-2xl relative overflow-hidden flex flex-col items-center text-center"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                    <div className="w-16 h-16 rounded-full bg-amber-50 dark:bg-amber-500/10 flex items-center justify-center mb-6 group-hover:bg-amber-100 dark:group-hover:bg-amber-500/20 shadow-[0_0_30px_rgba(245,158,11,0.15)] group-hover:shadow-[0_0_40px_rgba(245,158,11,0.3)] transition-all">
                        <Activity className="w-8 h-8 text-amber-500" />
                    </div>

                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3 transition-colors">Simulation Sandbox</h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed transition-colors">
                        Test evacuation models and response protocols in an isolated environment.
                    </p>

                    <div className="mt-6 px-6 py-2 bg-gray-100 dark:bg-gray-800 group-hover:bg-amber-50 dark:group-hover:bg-amber-500/10 text-gray-600 dark:text-gray-300 group-hover:text-amber-600 dark:group-hover:text-amber-400 rounded-full text-sm font-medium transition-colors border border-gray-200 dark:border-gray-700 group-hover:border-amber-300 dark:group-hover:border-amber-500/30">
                        Launch Sandbox
                    </div>
                </motion.div>

                {/* Live Incident Card */}
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                    whileHover={{ scale: 1.02, y: -5 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => onSelectMode('live')}
                    className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-800 hover:border-red-500/50 rounded-2xl p-8 cursor-pointer group transition-colors shadow-xl dark:shadow-2xl relative overflow-hidden flex flex-col items-center text-center"
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                    <div className="w-16 h-16 rounded-full bg-red-50 dark:bg-red-500/10 flex items-center justify-center mb-6 group-hover:bg-red-100 dark:group-hover:bg-red-500/20 shadow-[0_0_30px_rgba(239,68,68,0.15)] group-hover:shadow-[0_0_40px_rgba(239,68,68,0.3)] transition-all">
                        <ShieldAlert className="w-8 h-8 text-red-500" />
                    </div>

                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-3 transition-colors">Live Hazard Entry</h2>
                    <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed transition-colors">
                        Access the production form to log new critical incidents directly into the database.
                    </p>

                    <div className="mt-6 px-6 py-2 bg-gray-100 dark:bg-gray-800 group-hover:bg-red-50 dark:group-hover:bg-red-500/10 text-gray-600 dark:text-gray-300 group-hover:text-red-600 dark:group-hover:text-red-400 rounded-full text-sm font-medium transition-colors border border-gray-200 dark:border-gray-700 group-hover:border-red-300 dark:group-hover:border-red-500/30">
                        Enter Production
                    </div>
                </motion.div>
            </div>

            {/* Saved Simulations Feed */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                className="w-full max-w-5xl mt-8"
            >
                <SavedSimulationsList onSelectSimulation={(sim) => onSelectMode('replay', sim)} />
            </motion.div>
        </div>
    );
};

export default LandingHub;
