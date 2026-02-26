import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MapPin, AlertTriangle, ShieldAlert, X, Activity, Droplets, Wind, Zap } from 'lucide-react';
import MapComponent from './MapComponent';
import { db } from '../firebase';
import { collection, onSnapshot, query } from 'firebase/firestore';

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

const IncidentViewMode = () => {
    const [incidents, setIncidents] = useState([]);
    const [selectedIncident, setSelectedIncident] = useState(null);
    const [loading, setLoading] = useState(true);

    // Fetch realtime incidents from Firestore
    useEffect(() => {
        const q = query(collection(db, 'incident')); // Using "incident" collection as set up previously

        const unsubscribe = onSnapshot(q, (snapshot) => {
            const fetchedIncidents = [];
            snapshot.forEach((doc) => {
                const data = doc.data();
                // Map the Firebase data into the format MapComponent expects for plumes
                if (data.Coordinates) {
                    const config = getSeverityConfig(data.Servertity); // matching the exact db typo schema
                    fetchedIncidents.push({
                        id: doc.id,
                        position: data.Coordinates, // [lng, lat]
                        radius: config.radius,
                        color: config.color,
                        // Store full payload for the details panel
                        details: {
                            id: data.Inc_Id || doc.id,
                            type: data.Type || 'Unknown Hazard',
                            severity: data.Servertity || 'Unspecified',
                            desc: data.Desc || 'No details provided.',
                            others: data.Others || 'None',
                            timestamp: data.Timestamp
                        }
                    });
                }
            });
            setIncidents(fetchedIncidents);
            setLoading(false);

            // If the currently selected incident was updated, refresh the selected state
            setSelectedIncident(prev => {
                if (!prev) return null;
                const updated = fetchedIncidents.find(inc => inc.id === prev.id);
                return updated || null;
            });
        }, (error) => {
            console.error("Error listening to incident collection: ", error);
            setLoading(false);
        });

        return () => unsubscribe();
    }, []);

    const handleIncidentClick = (incidentData) => {
        setSelectedIncident(incidentData);
    };

    const closePanel = () => {
        setSelectedIncident(null);
    };

    return (
        <div className="w-full h-full relative overflow-hidden bg-gray-950">
            {/* Map Viewer */}
            <MapComponent
                mode="view"
                theme="dark"
                incidents={incidents}
                onIncidentClick={handleIncidentClick}
                viewportPadding={selectedIncident ? { right: 400 } : {}} // Shift map framing if panel is open
            />

            {/* Top Status Bar */}
            <div className="absolute top-6 left-1/2 -translate-x-1/2 bg-gray-900/80 backdrop-blur-md border border-gray-800 px-6 py-3 rounded-full flex items-center gap-4 shadow-xl z-20 pointer-events-auto">
                {loading ? (
                    <div className="flex items-center gap-2 text-gray-400 text-sm font-medium">
                        <Activity className="w-4 h-4 animate-pulse" />
                        Syncing Database...
                    </div>
                ) : (
                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-2 text-gray-300">
                            <MapPin className="w-4 h-4 text-purple-400" />
                            <span className="text-sm font-bold tracking-wide">Live Dashboard</span>
                        </div>
                        <div className="w-px h-4 bg-gray-700"></div>
                        <div className="flex items-center gap-2 text-xs font-mono text-gray-400">
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
                        className="absolute top-4 right-4 bottom-4 w-96 bg-gray-900/95 backdrop-blur-xl border border-gray-800 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden"
                    >
                        {/* Header */}
                        <div className="p-6 border-b border-gray-800 flex items-start justify-between bg-gray-900/50">
                            <div className="flex gap-4">
                                <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-xl shrink-0">
                                    {getTypeIcon(selectedIncident.details.type)}
                                </div>
                                <div>
                                    <h2 className="text-white font-bold text-lg leading-tight mb-1">
                                        {selectedIncident.details.type}
                                    </h2>
                                    <div className="flex items-center gap-2">
                                        <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${selectedIncident.details.severity.toLowerCase() === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/30' :
                                                selectedIncident.details.severity.toLowerCase() === 'high' ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30' :
                                                    selectedIncident.details.severity.toLowerCase() === 'medium' ? 'bg-yellow-500/20 text-yellow-500 border border-yellow-500/30' :
                                                        'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                                            }`}>
                                            Severity: {selectedIncident.details.severity}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <button
                                onClick={closePanel}
                                className="p-2 -mr-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Body Details */}
                        <div className="flex-1 p-6 overflow-y-auto space-y-6">

                            {/* Coordinate Data */}
                            <div className="bg-gray-950 border border-gray-800 p-4 rounded-xl flex items-center justify-between">
                                <div>
                                    <p className="text-xs text-gray-500 font-medium mb-1">Coordinates (Lng, Lat)</p>
                                    <p className="text-sm font-mono text-gray-300">
                                        {selectedIncident.position[0].toFixed(5)}, {selectedIncident.position[1].toFixed(5)}
                                    </p>
                                </div>
                                <MapPin className="w-5 h-5 text-gray-600" />
                            </div>

                            {/* Description Block */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Description</h3>
                                <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">
                                    {selectedIncident.details.desc}
                                </p>
                            </div>

                            {/* Additional Info Block */}
                            {selectedIncident.details.others && selectedIncident.details.others !== 'None' && (
                                <div>
                                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Additional Conditions</h3>
                                    <div className="bg-amber-500/5 border border-amber-500/20 p-4 rounded-xl">
                                        <p className="text-sm text-amber-200/80 leading-relaxed">
                                            {selectedIncident.details.others}
                                        </p>
                                    </div>
                                </div>
                            )}

                        </div>

                        {/* Footer Record Metadata */}
                        <div className="p-4 bg-gray-950 border-t border-gray-800 text-xs text-gray-500 font-mono flex items-center justify-between">
                            <span>ID: {selectedIncident.details.id}</span>
                            {selectedIncident.details.timestamp ? (
                                <span>{new Date(selectedIncident.details.timestamp).toLocaleTimeString()}</span>
                            ) : null}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default IncidentViewMode;
