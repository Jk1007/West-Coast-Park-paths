import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MapPin, AlertTriangle, ShieldAlert, X, Activity, Droplets, Wind, Zap, Trash2 } from 'lucide-react';
import MapComponent from './MapComponent';
import supabase from '../supabase';

// Helper to determine plume radius and color based on severity
const getSeverityConfig = (severity) => {
    switch (severity?.toLowerCase()) {
        case 'critical': return { radius: 150, color: '#ef4444' }; // Red
        case 'high': return { radius: 100, color: '#f97316' }; // Orange
        case 'medium': return { radius: 60, color: '#eab308' }; // Yellow
        case 'low': return { radius: 30, color: '#3b82f6' }; // Blue
        default: return { radius: 50, color: '#ffffff' };
    }
};

// Helper to pick an icon based on Type
const getTypeIcon = (type) => {
    if (!type) return <AlertTriangle className="w-5 h-5" />;
    const t = type.toLowerCase();
    if (t.includes('chemical') || t.includes('spill') || t.includes('bio')) return <Droplets className="w-5 h-5 text-purple-400" />;
    if (t.includes('gas') || t.includes('smoke')) return <Wind className="w-5 h-5 text-gray-400" />;
    if (t.includes('radiation')) return <Zap className="w-5 h-5 text-amber-500" />;
    if (t.includes('fire')) return <Activity className="w-5 h-5 text-red-500" />;
    return <AlertTriangle className="w-5 h-5 text-red-400" />;
};

const IncidentViewMode = ({ theme }) => {
    const [incidents, setIncidents] = useState([]);
    const [selectedIncident, setSelectedIncident] = useState(null);
    const [loading, setLoading] = useState(true);

    // Fetch realtime incidents from Supabase
    useEffect(() => {
        const fetchInitialIncidents = async () => {
            const { data, error } = await supabase
                .from('incidents')
                .select('*');

            if (error) {
                console.error("Error fetching incidents:", error);
                setLoading(false);
                return;
            }

            if (data) {
                const fetchedIncidents = data.map(record => {
                    const config = getSeverityConfig(record.severity);
                    
                    let parsedPos = [0, 0];
                    if (Array.isArray(record.coordinates)) {
                        parsedPos = record.coordinates.map(Number);
                    } else if (typeof record.coordinates === 'string') {
                        try { parsedPos = JSON.parse(record.coordinates).map(Number); } catch (e) {}
                    }

                    return {
                        id: record.id,
                        position: parsedPos,
                        radius: config.radius,
                        color: config.color,
                        elapsedSimSec: 300, // Provides required parameter for geometric rendering
                        details: {
                            id: record.id,
                            title: record.title,
                            type: record.type,
                            severity: record.severity,
                            status: record.status,
                            desc: record.description,
                            others: record.others,
                            amount: record.amount || (record.severity === 'Critical' ? 500 : (record.severity === 'High' ? 300 : (record.severity === 'Medium' ? 100 : 50))),
                            timestamp: record.created_at
                        }
                    };
                });
                setIncidents(fetchedIncidents);
            }
            setLoading(false);
        };

        fetchInitialIncidents();

        // Subscribe to Realtime Changes
        const channel = supabase
            .channel('public:incidents_view')
            .on('postgres_changes', { event: '*', schema: 'public', table: 'incidents' }, (payload) => {
                const { eventType, new: newRecord, old: oldRecord } = payload;
                
                setIncidents(prev => {
                    let updated = [...prev];
                    
                    if (eventType === 'INSERT' || eventType === 'UPDATE') {
                        const config = getSeverityConfig(newRecord.severity);
                        
                        let parsedPos = [0, 0];
                        if (Array.isArray(newRecord.coordinates)) {
                            parsedPos = newRecord.coordinates.map(Number);
                        } else if (typeof newRecord.coordinates === 'string') {
                            try { parsedPos = JSON.parse(newRecord.coordinates).map(Number); } catch (e) {}
                        }

                        const mapped = {
                            id: newRecord.id,
                            position: parsedPos,
                            radius: config.radius,
                            color: config.color,
                            elapsedSimSec: 300, // Provides required parameter for geometric rendering
                            details: {
                                id: newRecord.id,
                                title: newRecord.title,
                                type: newRecord.type,
                                severity: newRecord.severity,
                                status: newRecord.status,
                                desc: newRecord.description,
                                others: newRecord.others,
                                amount: newRecord.amount || (newRecord.severity === 'Critical' ? 500 : (newRecord.severity === 'High' ? 300 : (newRecord.severity === 'Medium' ? 100 : 50))),
                                timestamp: newRecord.created_at
                            }
                        };
                        
                        const idx = updated.findIndex(inc => inc.id === newRecord.id);
                        if (idx >= 0) {
                            updated[idx] = mapped; // Update existing
                        } else {
                            updated.push(mapped); // Add new
                        }
                    } else if (eventType === 'DELETE') {
                        updated = updated.filter(inc => inc.id !== oldRecord.id);
                    }
                    
                    return updated;
                });
            })
            .subscribe();

        return () => {
            supabase.removeChannel(channel);
        };
    }, []);

    const handleIncidentClick = (incidentData) => {
        setSelectedIncident(incidentData);
    };

    const handleDeleteClick = async () => {
        if (!window.confirm(`Are you sure you want to permanently delete "${selectedIncident.details.title || selectedIncident.details.type}" from the database?`)) return;

        if (!selectedIncident.id) return;

        try {
            const { data, error } = await supabase
                .from('incidents')
                .delete()
                .eq('id', selectedIncident.id)
                .select();

            if (error) throw error;
            
            if (!data || data.length === 0) {
                alert("Deletion was rejected by the Database Server! You may not have the required Row Level Security permissions, or the item no longer exists.");
                return;
            }

            console.log("Incident successfully deleted from database.");
            setSelectedIncident(null);
            // The existing `postgres_changes` channel subscription automatically removes the marker from the array and re-renders the geoJSON without it!
        } catch (error) {
            console.error("Error deleting incident:", error);
            alert(`Deletion Error: ${error.message}`);
        }
    };

    const closePanel = () => {
        setSelectedIncident(null);
    };

    return (
        <div className="w-full h-full relative overflow-hidden bg-gray-50 dark:bg-gray-950 transition-colors duration-300">
            {/* Map Viewer */}
            <MapComponent
                mode="view"
                theme={theme}
                incidents={incidents}
                onIncidentClick={handleIncidentClick}
                viewportPadding={selectedIncident ? { right: 400 } : {}} // Shift map framing if panel is open
            />

            {/* Top Status Bar */}
            <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border border-gray-200 dark:border-gray-800 px-6 py-3 rounded-full flex items-center gap-4 shadow-xl z-20 pointer-events-auto transition-colors duration-300">
                {loading ? (
                    <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-sm font-medium">
                        <Activity className="w-4 h-4 animate-pulse" />
                        Syncing Database...
                    </div>
                ) : (
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-gray-900 dark:text-gray-100">
                            <MapPin className="w-4 h-4 text-purple-500 dark:text-purple-400" />
                            <span className="text-sm font-bold tracking-wide">Live Dashboard</span>
                        </div>
                        <div className="w-px h-4 bg-gray-300 dark:bg-gray-700"></div>
                        <div className="flex items-center gap-2 text-xs font-mono text-gray-500 dark:text-gray-400">
                            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                            {incidents.length} Active Records
                        </div>
                    </div>
                )}
            </div>

            {/* Incident Details Side Panel */}
            <AnimatePresence>
                {selectedIncident && (
                    <motion.div
                        initial={{ x: '100%', opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: '100%', opacity: 0 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        className="absolute top-4 right-4 bottom-4 w-96 bg-white/95 dark:bg-[#0b1120]/80 backdrop-blur-2xl border border-gray-200 dark:border-white/10 rounded-3xl shadow-[0_8px_32px_0_rgba(0,0,0,0.3)] flex flex-col z-50 overflow-hidden transition-all duration-300"
                    >
                        {/* Header */}
                        <div className="p-6 border-b border-gray-200 dark:border-white/10 flex items-start justify-between bg-gradient-to-b from-gray-50/50 to-white/50 dark:from-white/5 dark:to-transparent transition-colors duration-300 relative">
                            <div className="absolute inset-0 bg-blue-500/5 dark:bg-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                            <div className="flex gap-4 relative z-10 w-full pr-24">
                                <div className="w-12 h-12 rounded-2xl bg-white dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-inner flex items-center justify-center shrink-0 backdrop-blur-md">
                                    {getTypeIcon(selectedIncident.details.type)}
                                </div>
                                <div className="flex-1">
                                    <h2 className="text-gray-900 dark:text-white font-black tracking-tight text-xl leading-tight mb-2 pb-0.5 border-b border-gray-100 dark:border-white/10">
                                        {selectedIncident.details.title || selectedIncident.details.type || "Unknown Incident"}
                                    </h2>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className={`text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full shadow-sm backdrop-blur-md border ${
                                            (selectedIncident.details.severity || 'Unknown').toLowerCase() === 'critical' ? 'bg-red-500/10 text-red-600 dark:bg-red-500/20 dark:text-red-400 border-red-500/30 dark:border-red-500/50 dark:shadow-[0_0_15px_rgba(239,68,68,0.3)]' :
                                            (selectedIncident.details.severity || 'Unknown').toLowerCase() === 'high' ? 'bg-orange-500/10 text-orange-600 dark:bg-orange-500/20 dark:text-orange-400 border-orange-500/30 dark:border-orange-500/50 dark:shadow-[0_0_15px_rgba(249,115,22,0.3)]' :
                                            (selectedIncident.details.severity || 'Unknown').toLowerCase() === 'medium' ? 'bg-yellow-500/10 text-yellow-600 dark:bg-yellow-500/20 dark:text-yellow-400 border-yellow-500/30 dark:border-yellow-500/50 dark:shadow-[0_0_15px_rgba(234,179,8,0.3)]' :
                                            'bg-blue-500/10 text-blue-600 dark:bg-blue-500/20 dark:text-blue-400 border-blue-500/30 dark:border-blue-500/50 dark:shadow-[0_0_15px_rgba(59,130,246,0.3)]'
                                        }`}>
                                            Severity: {selectedIncident.details.severity || 'Unknown'}
                                        </span>
                                        {selectedIncident.details.amount && (
                                            <span className="text-[10px] font-black uppercase tracking-widest px-2.5 py-1 rounded-full backdrop-blur-md bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/40 shadow-sm dark:shadow-[0_0_15px_rgba(16,185,129,0.2)]">
                                                {selectedIncident.details.amount} kg
                                            </span>
                                        )}
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-center gap-1 -mr-2 absolute top-4 right-4 z-20">
                                <button
                                    onClick={handleDeleteClick}
                                    className="p-2 text-red-500 hover:text-red-600 dark:text-red-400/80 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-500/20 rounded-full transition-all"
                                    title="Delete Incident"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                                <button
                                    onClick={closePanel}
                                    className="p-2 text-gray-400 hover:text-gray-900 dark:text-gray-500 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-all"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                        </div>

                        {/* Body Details */}
                        <div className="flex-1 p-6 overflow-y-auto space-y-6">

                            {/* Coordinate Data */}
                            <div className="bg-gradient-to-r from-gray-50 to-white dark:from-white/5 dark:to-transparent border border-gray-200 dark:border-white/10 p-4 rounded-2xl flex items-center justify-between shadow-sm relative overflow-hidden group hover:border-blue-300 dark:hover:border-blue-500/50 transition-all duration-300">
                                <div className="absolute inset-0 bg-gradient-to-r from-transparent to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                                <div className="relative z-10">
                                    <p className="text-[10px] font-bold text-gray-500 dark:text-gray-400 uppercase tracking-widest mb-1">Coordinates (Lng, Lat)</p>
                                    <p className="text-sm font-mono font-bold text-gray-900 dark:text-gray-200">
                                        {selectedIncident.position[0].toFixed(5)}, {selectedIncident.position[1].toFixed(5)}
                                    </p>
                                </div>
                                <div className="relative z-10 w-10 h-10 rounded-xl bg-blue-50 dark:bg-blue-500/20 flex items-center justify-center border border-blue-100 dark:border-blue-500/30 group-hover:scale-110 transition-transform duration-300">
                                    <MapPin className="w-5 h-5 text-blue-500 dark:text-blue-400" />
                                </div>
                            </div>

                            {/* Description Block */}
                            <div>
                                <h3 className="text-[10px] font-black text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-3 flex items-center gap-2">
                                    <ShieldAlert className="w-3 h-3" /> Description
                                </h3>
                                <div className="pl-4 border-l-2 border-gray-200 dark:border-gray-800">
                                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
                                        {selectedIncident.details.desc}
                                    </p>
                                </div>
                            </div>

                            {/* Additional Info Block */}
                            {selectedIncident.details.others && selectedIncident.details.others !== 'None' && (
                                <div>
                                    <h3 className="text-[10px] font-black text-amber-500/80 uppercase tracking-widest mb-3 flex items-center gap-2">
                                        <AlertTriangle className="w-3 h-3" /> Additional Conditions
                                    </h3>
                                    <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-500/10 dark:to-orange-500/5 border border-amber-200 dark:border-amber-500/40 p-4 rounded-2xl shadow-sm dark:shadow-[0_0_20px_rgba(245,158,11,0.15)] relative overflow-hidden group">
                                        <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/10 blur-2xl rounded-full translate-x-12 -translate-y-12" />
                                        <p className="text-sm font-bold text-amber-900 dark:text-amber-300/90 leading-relaxed relative z-10">
                                            {selectedIncident.details.others}
                                        </p>
                                    </div>
                                </div>
                            )}

                        </div>

                        {/* Footer Record Metadata */}
                        <div className="p-5 bg-gray-50/80 dark:bg-black/20 border-t border-gray-200 dark:border-white/5 text-[10px] text-gray-500 font-mono flex items-center justify-between backdrop-blur-md">
                            <span className="truncate pr-4 opacity-50 text-gray-600 dark:text-gray-400">ID: {selectedIncident.details.id}</span>
                            {selectedIncident.details.timestamp && (
                                <span className="shrink-0 font-bold opacity-70 border border-gray-300 dark:border-gray-800 px-2 py-1 rounded-md bg-white dark:bg-gray-900">{new Date(selectedIncident.details.timestamp).toLocaleTimeString()}</span>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default IncidentViewMode;
