import React, { useEffect, useState } from 'react';
import supabase from '../supabase';
import { Play, Database, ShieldCheck, Clock, CheckCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const SavedSimulationsList = ({ onSelectSimulation }) => {
    const [simulations, setSimulations] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchSimulations = async () => {
            try {
                // Get authenticated user
                const { data: { user } } = await supabase.auth.getUser();
                if (!user) {
                    setLoading(false);
                    return;
                }

                // Query Supabase for all simulations globally
                const { data, error } = await supabase
                    .from('simulations')
                    .select('*')
                    .order('created_at', { ascending: false });

                if (error) {
                    throw error;
                }

                if (data) {
                    // Map back to expected schema
                    const sims = data.map(record => ({
                        id: record.id,
                        name: record.name,
                        createdAt: record.created_at,
                        durationMs: record.duration_ms,
                        finalStats: record.final_stats,
                        initialState: record.frame_data,
                        events: record.events,
                        logs: record.logs
                    }));
                    setSimulations(sims);
                }
            } catch (error) {
                console.error("Error fetching simulations:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchSimulations();
    }, []);

    const formatDate = (isoString) => {
        const d = new Date(isoString);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className="w-full mt-10 z-10">
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg text-blue-600 dark:text-blue-400">
                    <Database size={20} />
                </div>
                <div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white transition-colors">Saved Replays</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 transition-colors">Review and analyze recorded disaster response scenarios.</p>
                </div>
            </div>

            {loading ? (
                <div className="flex justify-center items-center py-12">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                </div>
            ) : simulations.length === 0 ? (
                <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-8 text-center border border-gray-200 dark:border-gray-700 border-dashed backdrop-blur-md transition-colors">
                    <Database className="w-12 h-12 text-gray-400 mx-auto mb-3 opacity-50" />
                    <p className="text-gray-500 dark:text-gray-400">No recorded simulations found.</p>
                    <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">Run the Sandbox and click "End & Save" to store a replay tape here.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <AnimatePresence>
                        {simulations.map((sim, index) => (
                            <motion.div
                                key={sim.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ delay: index * 0.05 }}
                                onClick={() => onSelectSimulation(sim)}
                                className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-500/50 hover:shadow-lg dark:hover:shadow-blue-500/10 rounded-2xl p-5 cursor-pointer transition-all flex flex-col relative overflow-hidden"
                            >
                                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-transparent to-transparent group-hover:from-blue-500/5 transition-colors" />

                                <div className="flex justify-between items-start mb-3">
                                    <h4 className="font-bold text-gray-900 dark:text-white truncate pr-4 transition-colors">
                                        {sim.name || "Untitled Simulation"}
                                    </h4>
                                    <div className="bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 p-1.5 rounded-full shrink-0 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                                        <Play className="w-4 h-4 ml-0.5" />
                                    </div>
                                </div>

                                <div className="text-xs text-gray-500 dark:text-gray-400 mb-4 flex items-center gap-1 transition-colors">
                                    <Clock className="w-3 h-3" /> {formatDate(sim.createdAt)}
                                </div>

                                <div className="mt-auto grid grid-cols-2 gap-2 relative z-10">
                                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-2 flex flex-col transition-colors border border-transparent dark:border-gray-800">
                                        <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Safety Score</span>
                                        <span className={`font-mono font-medium ${sim.finalStats?.safetyIndex < 50 ? 'text-red-500' : 'text-green-500'}`}>
                                            {sim.finalStats?.safetyIndex || 0}%
                                        </span>
                                    </div>
                                    <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-2 flex flex-col transition-colors border border-transparent dark:border-gray-800">
                                        <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">Duration</span>
                                        <span className="font-mono text-gray-700 dark:text-gray-300 font-medium">
                                            {sim.durationMs ? (sim.durationMs / 1000).toFixed(1) : 0}s
                                        </span>
                                    </div>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            )}
        </div>
    );
};

export default SavedSimulationsList;
