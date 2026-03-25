import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Send, AlertTriangle, CheckCircle2, X, MapPin, Activity, Droplets, Wind, Zap, Eye, EyeOff, List, Search, Filter, Pencil, Trash2, Server } from 'lucide-react';
import MapComponent from './MapComponent';
import supabase from '../supabase';
import { fetchWindData } from '../services/WindService';
import { HAZARD_DATABASE } from '../constants/HazardDatabase';
import SimulationWsService from '../services/SimulationWsService';

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

export const parseCoordinates = (input) => {
    if (!input) return [0, 0];
    if (Array.isArray(input)) return input.map(Number);
    if (typeof input === 'string') {
        const cleaned = input.replace(/[()]/g, ''); // Handles (103.7,1.2)
        const parts = cleaned.split(',').map(s => parseFloat(s.trim()));
        if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
            return parts;
        }
        try { return JSON.parse(input).map(Number); } catch (e) {}
    }
    return [0, 0];
};

const LiveIncidentMode = ({ onFormStateChange, theme }) => {
    // Selected Location from Map Click
    const [selectedLocation, setSelectedLocation] = useState(null);

    // Form State mapped to database schema
    const [formData, setFormData] = useState({
        Title: '',
        Severity_lvl: 'Critical',
        Type: 'CHLORINE_GAS',
        Status: 'Ongoing',
        Details: '',
        Others: '',
        Amount: 100
    });

    const [isSubmitted, setIsSubmitted] = useState(false);
    const [isEditing, setIsEditing] = useState(false);

    // Plume Spread State
    const [activePlume, setActivePlume] = useState(null);

    // Live Atmospheric State
    const [wind, setWind] = useState({ speed: 0, direction: 0, stabilityClass: 'D', weather: { temp: 28, rain: 0, hum: 80 } });
    const [currentTime, setCurrentTime] = useState(Date.now());

    // Merged: Incident Viewer State
    const [incidents, setIncidents] = useState([]);
    const [selectedIncident, setSelectedIncident] = useState(null);
    const [loading, setLoading] = useState(true);
    const [showIncidents, setShowIncidents] = useState(false);

    // Sidebar & Filter State
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [statusFilter, setStatusFilter] = useState('All'); // 'All', 'Ongoing', 'Resolved'
    const [typeFilter, setTypeFilter] = useState('All');
    const [wsBackendStatus, setWsBackendStatus] = useState('disconnected');

    // Connect to Python plume backend via WebSocket
    useEffect(() => {
        SimulationWsService.connect(setWsBackendStatus);
        return () => SimulationWsService.disconnect(setWsBackendStatus);
    }, []);

    // Merged: Fetch realtime incidents from Supabase
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
                    const parsedPos = parseCoordinates(record.coordinates);
                    
                    return {
                        id: record.id,
                        position: parsedPos,
                        radius: config.radius,
                        color: config.color,
                        elapsedSimSec: 300, // Visually triggers native fully-grown Plume state
                        details: {
                            id: record.id,
                            title: record.title,
                            type: record.type,
                            severity: record.severity,
                            status: record.status,
                            desc: record.description,
                            others: record.others,
                            timestamp: record.created_at || new Date().toISOString(),
                            amount: record.amount || (record.severity === 'Critical' ? 500 : (record.severity === 'High' ? 300 : (record.severity === 'Medium' ? 100 : 50)))
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
            .channel('public:incidents')
            .on('postgres_changes', { event: '*', schema: 'public', table: 'incidents' }, (payload) => {
                const { eventType, new: newRecord, old: oldRecord } = payload;
                
                setIncidents(prev => {
                    let updated = [...prev];
                    
                    if (eventType === 'INSERT' || eventType === 'UPDATE') {
                        const config = getSeverityConfig(newRecord.severity);
                        const parsedPos = parseCoordinates(newRecord.coordinates);
                        
                        const mapped = {
                            id: newRecord.id,
                            position: parsedPos,
                            radius: config.radius,
                            color: config.color,
                            elapsedSimSec: 300,
                            details: {
                                id: newRecord.id,
                                title: newRecord.title,
                                type: newRecord.type,
                                severity: newRecord.severity,
                                status: newRecord.status,
                                desc: newRecord.description,
                                others: newRecord.others,
                                timestamp: newRecord.created_at,
                                amount: newRecord.amount || (newRecord.severity === 'Critical' ? 500 : (newRecord.severity === 'High' ? 300 : (newRecord.severity === 'Medium' ? 100 : 50)))
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

        // Mount Live Weather & Real-time System Clock
        const loadWind = async () => {
            const windData = await fetchWindData();
            if (windData) setWind(windData);
        };
        loadWind();

        const clockTick = setInterval(() => setCurrentTime(Date.now()), 1000);
        const windTick = setInterval(loadWind, 60000); // 1 min updates

        return () => {
            supabase.removeChannel(channel);
            clearInterval(clockTick);
            clearInterval(windTick);
        };
    }, []);

    // Track dirty state to warn user on navigation
    useEffect(() => {
        const isDirty = selectedLocation !== null || formData.Title !== '' || formData.Details !== '' || formData.Others !== '' || formData.Amount !== 100;
        onFormStateChange(isDirty);
    }, [formData, selectedLocation, onFormStateChange]);

    const handleMapClick = useCallback((lngLat) => {
        if (!isSubmitted) {
            setSelectedLocation(lngLat);
            setSelectedIncident(null); // Clear selected incident if dropping a new pin
        }
    }, [isSubmitted]);

    const handleIncidentClick = useCallback((incidentData) => {
        setSelectedIncident(incidentData);
        setSelectedLocation(null); // Clear new pin form if viewing an existing incident
    }, []);

    const closePanel = () => {
        setSelectedIncident(null);
    };

    const handleCancel = () => {
        setIsEditing(false);
        setSelectedLocation(null);
        setFormData({ Title: '', Severity_lvl: 'Critical', Type: 'CHLORINE_GAS', Status: 'Ongoing', Details: '', Others: '', Amount: 100 });
    };

    const handleEditClick = () => {
        setIsEditing(true);
        setFormData({
            Title: selectedIncident.details.title || selectedIncident.details.type || '',
            Severity_lvl: selectedIncident.details.severity || 'Critical',
            Type: selectedIncident.details.type || 'CHLORINE_GAS',
            Status: selectedIncident.details.status || 'Ongoing',
            Details: selectedIncident.details.desc || '',
            Others: selectedIncident.details.others === 'None' ? '' : (selectedIncident.details.others || ''),
            Amount: selectedIncident.details.amount || 100
        });
    };

    const handleDeleteClick = async () => {
        if (!window.confirm(`Are you sure you want to permanently delete "${selectedIncident.details.title || selectedIncident.details.type}" from the database?`)) return;

        if (!selectedIncident.id) {
            setActivePlume(null);
            setSelectedIncident(null);
            return;
        }

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
        } catch (error) {
            console.error("Error deleting incident:", error);
            alert(`Deletion Error: ${error.message}`);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        console.log(isEditing ? "Updating Live Incident" : "Submitting Live Incident at", selectedLocation, "Details:", formData);

        try {
            const { data: { user } } = await supabase.auth.getUser();
            const userId = user ? user.id : null;

            if (!userId) {
                alert("You must be logged in to report an incident.");
                return;
            }

            const payload = {
                title: formData.Title,
                description: formData.Details,
                others: formData.Others === '' ? null : formData.Others,
                severity: formData.Severity_lvl,
                type: formData.Type,
                status: formData.Status,
                amount: formData.Amount,
                user_id: userId
            };

            if (!isEditing) {
                payload.coordinates = selectedLocation;
            }

            if (isEditing) {
                const { error } = await supabase
                    .from('incidents')
                    .update(payload)
                    .eq('id', selectedIncident.id);

                if (error) throw error;
                
                console.log("Incident successfully updated in database.");
                setIsEditing(false);
                setIsSubmitted(true);
                onFormStateChange(false);

                setTimeout(() => {
                    setIsSubmitted(false);
                    setFormData({ Title: '', Severity_lvl: 'Critical', Type: 'Chemical Spill', Status: 'Ongoing', Details: '', Others: '' });
                }, 3000);
            } else {
                const { error } = await supabase
                    .from('incidents')
                    .insert([payload]);
                    
                if (error) throw error;
                    
                console.log("Incident successfully written to database.");

                setActivePlume({ 
                    position: selectedLocation, 
                    details: { 
                        title: formData.Title,
                        desc: formData.Details,
                        severity: formData.Severity_lvl,
                        status: formData.Status,
                        others: formData.Others,
                        amount: formData.Amount, 
                        type: formData.Type,
                        startTime: Date.now()
                    } 
                });

                setIsSubmitted(true);
                onFormStateChange(false);

                setTimeout(() => {
                    setIsSubmitted(false);
                    setSelectedLocation(null);
                    setFormData({ Title: '', Severity_lvl: 'Critical', Type: 'Chemical Spill', Status: 'Ongoing', Details: '', Others: '', Amount: 100 });
                    setActivePlume(null);
                }, 5000);
            }
        } catch (error) {
            console.error("Error writing incident to Supabase Database: ", error);
            alert(`Submission Error: ${error.message}\n\nMake sure you have created the schema using the Supabase SQL editor.`);
            return;
        }
    };

    // Derived state for filtering
    const filteredIncidents = incidents.filter(inc => {
        const matchesSearch = (inc.details.type || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
            (inc.details.desc || '').toLowerCase().includes(searchQuery.toLowerCase());
        const matchesStatus = statusFilter === 'All' || inc.details.status === statusFilter;
        const matchesType = typeFilter === 'All' || inc.details.type === typeFilter;

        return matchesSearch && matchesStatus && matchesType;
    });

    const activeInstances = [
        ...(showIncidents ? filteredIncidents : []), 
        ...(activePlume ? [activePlume] : []),
        ...(selectedIncident ? [selectedIncident] : [])
    ];
    const mapIncidents = Array.from(new Map(activeInstances.map(inc => {
        let ageSec = inc.elapsedSimSec || 300; // Default to fully grown (300s)
        
        if (inc.details?.timestamp && !inc.elapsedSimSec) {
            const ts = new Date(inc.details.timestamp).getTime();
            if (!isNaN(ts)) {
                const calculatedAge = (currentTime - ts) / 1000;
                ageSec = Math.max(300, calculatedAge);
            }
        } else if (inc === activePlume && inc.details?.startTime) {
            const st = inc.details.startTime;
            if (!isNaN(st)) {
                ageSec = Math.max(0, (currentTime - st) / 1000);
            }
        }
        
        if (isNaN(ageSec)) ageSec = 300;

        return [inc.id || inc.details?.id || 'anim', { ...inc, elapsedSimSec: ageSec }];
    })).values());

    const handleWindChange = useCallback((newWind) => {
        setWind(prev => ({ ...prev, ...newWind }));
    }, []);

    return (
        <div className="w-full h-full relative overflow-hidden bg-gray-50 dark:bg-gray-950 transition-colors duration-300">

            {/* Base Map Layer */}
            <MapComponent
                mode="live"
                theme={theme}
                wind={wind}
                selectedLocation={selectedLocation}
                onLocationSelect={handleMapClick}
                onIncidentClick={handleIncidentClick}
                onWindChange={handleWindChange}
                incidents={mapIncidents}
                viewportPadding={selectedIncident || selectedLocation ? { right: 400 } : {}}
                selectedIncidentId={selectedIncident?.details?.id || null}
                isNightMode={false} // Could be derived from currentTime if needed
            />

            {/* Atmospheric Conditions Widget */}
            <div className={`absolute bottom-12 right-6 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border border-gray-200 dark:border-gray-800 p-4 rounded-2xl shadow-xl z-20 pointer-events-none transition-all duration-500 ease-in-out ${selectedIncident || isEditing || selectedLocation ? 'translate-y-8 opacity-0 scale-95' : 'translate-y-0 opacity-100 scale-100'}`}>
                <div className="flex items-center gap-2 mb-3">
                    <Wind className="w-4 h-4 text-blue-500" />
                    <h3 className="text-xs font-bold text-gray-900 dark:text-white uppercase tracking-wider">Live Atmosphere</h3>
                </div>
                <div className="grid grid-cols-2 gap-x-8 gap-y-3">
                    <div>
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Wind Speed</p>
                        <p className="text-sm font-mono font-bold text-gray-900 dark:text-gray-100">
                            {wind.speed.toFixed(1)} <span className="text-xs text-gray-400 font-sans font-medium">km/h</span>
                        </p>
                    </div>
                    <div className="flex flex-col">
                        <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1.5">Direction</p>
                        <div className="flex items-center gap-2 mt-1">
                            <div className="relative w-8 h-8 rounded-full border border-gray-200 dark:border-gray-700/50 bg-gray-50/50 dark:bg-gray-800/30 flex justify-center items-center shadow-inner pt-0.5">
                                <div
                                    style={{ transform: "rotate(" + ((wind.direction + 180) % 360) + "deg)" }}
                                    className="transition-transform duration-500 ease-out flex items-center justify-center pt-px"
                                >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500 dark:text-blue-400">
                                        <line x1="12" y1="19" x2="12" y2="5"></line>
                                        <polyline points="5 12 12 5 19 12"></polyline>
                                    </svg>
                                </div>
                            </div>
                            <span className="text-sm font-mono font-bold text-gray-900 dark:text-gray-100">
                                {wind.direction.toFixed(0)}&deg;
                            </span>
                        </div>
                    </div>
                    {wind.weather && (
                        <>
                            <div>
                                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Temp</p>
                                <p className="text-sm font-mono font-bold text-gray-900 dark:text-gray-100">
                                    {wind.weather.temp.toFixed(1)} <span className="text-xs text-gray-400 font-sans font-medium">°C</span>
                                </p>
                            </div>
                            <div>
                                <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-0.5">Humidity</p>
                                <p className="text-sm font-mono font-bold text-gray-900 dark:text-gray-100">
                                    {wind.weather.hum.toFixed(0)} <span className="text-xs text-gray-400 font-sans font-medium">%</span>
                                </p>
                            </div>
                        </>
                    )}
                </div>
                <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-800 flex justify-between items-center">
                    <span className="text-[10px] text-gray-500 uppercase tracking-wider">Pasquill-Gifford</span>
                    <span className="text-[10px] font-bold bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 px-2 py-0.5 rounded-md border border-blue-200 dark:border-blue-500/30">
                        Class {wind.stabilityClass}
                    </span>
                </div>
            </div>

            {/* Python Backend WebSocket Status Badge */}
            <div className={`absolute top-6 right-6 z-30 flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold border shadow-lg backdrop-blur transition-colors pointer-events-none ${
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

            {/* Floating Sidebar Toggle Button */}
            <AnimatePresence>
                {!isSidebarOpen && (
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        className="absolute top-6 left-6 z-30 pointer-events-auto"
                    >
                        <button
                            onClick={() => setIsSidebarOpen(true)}
                            className="bg-white/90 dark:bg-gray-900/90 hover:bg-white dark:hover:bg-gray-800 backdrop-blur-xl border border-gray-200 dark:border-gray-800 text-brand-dark dark:text-white shadow-xl flex items-center gap-3 text-sm font-bold tracking-wide transition-all rounded-2xl px-5 py-4"
                        >
                            <div className="w-8 h-8 rounded-full bg-brand-light/20 flex items-center justify-center">
                                <List className="w-4 h-4 text-brand-dark dark:text-brand-light" />
                            </div>
                            Incident Database
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Instruction Overlay when no location is selected */}
            <AnimatePresence>
                {!selectedLocation && !selectedIncident && !isSubmitted && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="absolute top-10 left-1/2 -translate-x-1/2 bg-white/90 dark:bg-gray-900/90 backdrop-blur-md border border-brand-light/50 px-6 py-3 rounded-full shadow-2xl z-10 pointer-events-none flex items-center gap-3 transition-colors duration-300"
                    >
                        <ShieldAlert className="w-5 h-5 text-brand-light" />
                        <span className="text-brand-dark dark:text-gray-300 font-medium tracking-wide">Select a location on the map to log a live incident</span>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Merged: Left Sidebar List */}
            <AnimatePresence>
                {isSidebarOpen && (
                    <motion.div
                        initial={{ x: '-100%', opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: '-100%', opacity: 0 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        className="absolute top-4 left-4 bottom-4 w-80 bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl border border-gray-200 dark:border-gray-800 rounded-2xl shadow-2xl flex flex-col z-40 overflow-hidden transition-colors duration-300"
                    >
                        {/* Sidebar Header & Filters */}
                        <div className="p-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 flex flex-col gap-3 transition-colors duration-300">
                            <div className="flex items-center justify-between">
                                <h2 className="text-gray-900 dark:text-white font-bold tracking-wide flex items-center gap-2">
                                    <List className="w-5 h-5 text-purple-500" />
                                    Past Incidents
                                </h2>
                                <div className="flex items-center gap-1.5">
                                    <button
                                        onClick={() => {
                                            setShowIncidents(!showIncidents);
                                            if (showIncidents) setSelectedIncident(null);
                                        }}
                                        title={showIncidents ? "Hide Map Pins" : "Show Map Pins"}
                                        className={`p-1.5 rounded-lg transition-colors ${showIncidents ? 'bg-purple-100 text-purple-600 dark:bg-purple-500/20 dark:text-purple-400' : 'text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-white/10'}`}
                                    >
                                        {showIncidents ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                                    </button>
                                    <button
                                        onClick={() => setIsSidebarOpen(false)}
                                        className="p-1.5 text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-white/10 rounded-lg transition-colors flex items-center justify-center bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700"
                                    >
                                        <X className="w-4 h-4" />
                                    </button>
                                </div>
                            </div>

                            {/* Search */}
                            <div className="relative">
                                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                                <input
                                    type="text"
                                    placeholder="Search incidents..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg pl-9 pr-3 py-1.5 text-sm text-gray-900 dark:text-white focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500 transition-colors placeholder-gray-400"
                                />
                            </div>

                            {/* Filters */}
                            <div className="flex gap-2">
                                <div className="flex-1 relative">
                                    <Filter className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-gray-400" />
                                    <select
                                        value={statusFilter}
                                        onChange={(e) => setStatusFilter(e.target.value)}
                                        className="w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg pl-7 pr-2 py-1 text-xs text-gray-700 dark:text-gray-300 focus:outline-none focus:border-purple-500 transition-colors appearance-none"
                                    >
                                        <option value="All">All Status</option>
                                        <option value="Ongoing">Ongoing</option>
                                        <option value="Resolved">Resolved</option>
                                    </select>
                                </div>
                                <div className="flex-1">
                                    <select
                                        value={typeFilter}
                                        onChange={(e) => setTypeFilter(e.target.value)}
                                        className="w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-2 py-1 text-xs text-gray-700 dark:text-gray-300 focus:outline-none focus:border-purple-500 transition-colors appearance-none"
                                    >
                                        <option value="All">All Types</option>
                                        <option value="Chemical Spill">Chemical</option>
                                        <option value="Gas Leak">Gas Leak</option>
                                        <option value="Fire/Smoke">Fire</option>
                                        <option value="Biological">Biological</option>
                                        <option value="Radiation">Radiation</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        {/* Incident List Body */}
                        <div className="flex-1 overflow-y-auto p-2 space-y-2">
                            {filteredIncidents.length === 0 ? (
                                <div className="text-center py-8 text-sm text-gray-500 dark:text-gray-400">
                                    No incidents found matching filters.
                                </div>
                            ) : (
                                filteredIncidents.map((inc) => (
                                    <div
                                        key={inc.id || inc.details.id}
                                        onClick={() => handleIncidentClick(inc)}
                                        className={`p-3 rounded-xl border cursor-pointer transition-all ${selectedIncident?.id === inc.id
                                            ? 'bg-purple-50 dark:bg-purple-500/10 border-purple-200 dark:border-purple-500/30 shadow-md'
                                            : 'bg-white dark:bg-gray-800/50 border-gray-100 dark:border-gray-800 hover:border-purple-200 dark:hover:border-purple-500/30 hover:shadow-sm'
                                            }`}
                                    >
                                        <div className="flex justify-between items-start mb-1">
                                            <h3 className="text-sm font-bold text-gray-900 dark:text-gray-100 line-clamp-1">
                                                {inc.details.title || HAZARD_DATABASE[inc.details.type]?.name || inc.details.type}
                                            </h3>
                                            <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full ${inc.details.status === 'Ongoing' ? 'bg-red-100 text-red-600 dark:bg-red-500/20 dark:text-red-400' : 'bg-green-100 text-green-600 dark:bg-green-500/20 dark:text-green-400'}`}>
                                                {inc.details.status}
                                            </span>
                                        </div>
                                        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 line-clamp-2 leading-relaxed">
                                            {inc.details.desc}
                                        </div>
                                        {inc.details.others && inc.details.others !== 'None' && (
                                            <div className="text-xs text-amber-600 dark:text-amber-500 mt-1.5 font-medium flex items-center gap-1">
                                                <AlertTriangle size={12} /> <span className="truncate">{inc.details.others}</span>
                                            </div>
                                        )}
                                        <div className="flex justify-between items-center text-[10px] text-gray-400 font-mono">
                                            <span>Severity: {inc.details.severity}</span>
                                            {inc.details.timestamp && (
                                                <span>{new Date(inc.details.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Incident Entry Modal overlay */}
            <AnimatePresence>
                {(selectedLocation || isEditing) && (
                    <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/40 backdrop-blur-sm pointer-events-none">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="w-full max-w-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl shadow-2xl overflow-hidden relative flex flex-col max-h-[90vh] transition-colors duration-300 pointer-events-auto"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 transition-colors duration-300">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-lg bg-red-50 dark:bg-red-500/10 flex items-center justify-center border border-red-200 dark:border-red-500/20">
                                        <ShieldAlert className="w-5 h-5 text-red-500" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">{isEditing ? "Edit Live Hazard Report" : "Live Hazard Report"}</h2>
                                        {!isEditing && (
                                            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono mt-0.5">
                                                LOC: [{selectedLocation[1].toFixed(5)}, {selectedLocation[0].toFixed(5)}]
                                            </p>
                                        )}
                                    </div>
                                </div>
                                {!isSubmitted && (
                                    <button onClick={handleCancel} className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-colors">
                                        <X className="w-5 h-5" />
                                    </button>
                                )}
                            </div>

                            {/* Body */}
                            <div className="p-6 overflow-y-auto">
                                {isSubmitted ? (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.9 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        className="bg-green-500/10 border border-green-500/30 rounded-xl p-8 text-center flex flex-col items-center my-8"
                                    >
                                        <CheckCircle2 className="w-16 h-16 text-green-500 mb-4" />
                                        <h2 className="text-2xl font-bold text-green-400 mb-2">{isEditing ? "Hazard Updated" : "Hazard Dispatched"}</h2>
                                        <p className="text-gray-400 mb-4">Response teams are monitoring the Plume Spread radius.</p>
                                        <div className="text-xs font-mono text-green-500/60 bg-green-500/5 px-4 py-2 rounded-md">
                                            Database Record Sync: OK
                                        </div>
                                    </motion.div>
                                ) : (
                                    <form id="incident-form" onSubmit={handleSubmit} className="space-y-6 text-left">

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                                            {/* Title */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Incident Title</label>
                                                <input
                                                    type="text"
                                                    required
                                                    value={formData.Title}
                                                    onChange={(e) => setFormData({ ...formData, Title: e.target.value })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-400 dark:placeholder-gray-500"
                                                    placeholder="e.g. Chloric Gas Release"
                                                />
                                            </div>

                                            {/* Type */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Hazard Type</label>
                                                <select
                                                    value={formData.Type}
                                                    onChange={(e) => setFormData({ ...formData, Type: e.target.value })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors appearance-none"
                                                >
                                                    {Object.entries(HAZARD_DATABASE).map(([key, config]) => (
                                                        <option key={key} value={key}>
                                                            {config.name}
                                                        </option>
                                                    ))}
                                                </select>
                                                <p className="text-[10px] text-gray-500 mt-1 pl-1">
                                                    Base Plume Spread Speed: <span className="font-mono text-purple-500 font-bold">{HAZARD_DATABASE[formData.Type]?.spreadRate}x</span>
                                                </p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                                            {/* Severity */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Severity Level</label>
                                                <select
                                                    value={formData.Severity_lvl}
                                                    onChange={(e) => setFormData({ ...formData, Severity_lvl: e.target.value })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors appearance-none"
                                                >
                                                    <option value="Critical">Critical (Evacuation Required)</option>
                                                    <option value="High">High</option>
                                                    <option value="Medium">Medium</option>
                                                    <option value="Low">Low</option>
                                                </select>
                                            </div>

                                            {/* Status */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Incident Status</label>
                                                <select
                                                    value={formData.Status}
                                                    onChange={(e) => setFormData({ ...formData, Status: e.target.value })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors appearance-none"
                                                >
                                                    <option value="Ongoing">Ongoing</option>
                                                    <option value="Resolved">Resolved</option>
                                                </select>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                                            {/* Amount */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Release Amount (kg)</label>
                                                <input
                                                    type="number"
                                                    min="1"
                                                    required
                                                    value={formData.Amount}
                                                    onChange={(e) => setFormData({ ...formData, Amount: parseInt(e.target.value) || 0 })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-400 dark:placeholder-gray-500"
                                                    placeholder="e.g. 150"
                                                />
                                            </div>

                                            {/* Others */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Other Conditions</label>
                                                <input
                                                    type="text"
                                                    value={formData.Others}
                                                    onChange={(e) => setFormData({ ...formData, Others: e.target.value })}
                                                    className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2.5 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-400 dark:placeholder-gray-500"
                                                    placeholder="e.g. High winds blowing east"
                                                />
                                            </div>
                                        </div>

                                        {/* Details */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">Detailed Description</label>
                                            <textarea
                                                required
                                                rows={4}
                                                value={formData.Details}
                                                onChange={(e) => setFormData({ ...formData, Details: e.target.value })}
                                                className="w-full bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-3 text-gray-900 dark:text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-400 dark:placeholder-gray-500 resize-none"
                                                placeholder="Provide compound details, estimated volume..."
                                            />
                                        </div>
                                    </form>
                                )}
                            </div>

                            {/* Footer Actions */}
                            {!isSubmitted && (
                                <div className="p-6 border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 flex items-center justify-between shrink-0 transition-colors duration-300">
                                    <div className="flex items-center gap-2 text-red-600 dark:text-red-500/80 text-xs font-semibold">
                                        <AlertTriangle className="w-4 h-4" />
                                        Initialize Hazard Spread Protocol
                                    </div>

                                    <div className="flex gap-3">
                                        <button
                                            type="button"
                                            onClick={handleCancel}
                                            className="px-5 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            type="submit"
                                            form="incident-form"
                                            className="flex items-center gap-2 bg-red-600 hover:bg-red-500 text-white text-sm font-medium px-6 py-2 rounded-lg transition-all shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_20px_rgba(220,38,38,0.5)]"
                                        >
                                            <Send className="w-4 h-4" />
                                            {isEditing ? "Save Changes" : "Dispatch & Track Plume"}
                                        </button>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Merged: Incident Details Side Panel */}
            <AnimatePresence>
                {selectedIncident && (
                    <motion.div
                        initial={{ x: '100%', opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        exit={{ x: '100%', opacity: 0 }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        className="absolute top-4 right-4 bottom-4 w-96 bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl border border-gray-200 dark:border-gray-800 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden transition-colors duration-300"
                    >
                        {/* Header */}
                        <div className="p-6 border-b border-gray-200 dark:border-gray-800 flex items-start justify-between bg-gray-50 dark:bg-gray-900/50 transition-colors duration-300">
                            <div className="flex gap-4">
                                <div className="p-3 bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/20 rounded-xl shrink-0">
                                    {getTypeIcon(selectedIncident.details.type)}
                                </div>
                                <div>
                                    <h2 className="text-gray-900 dark:text-white font-bold text-lg leading-tight mb-1">
                                        {selectedIncident.details.title || selectedIncident.details.type}
                                    </h2>
                                    <div className="flex items-center gap-2">
                                        <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                                            selectedIncident.details.severity?.toLowerCase() === 'critical' ? 'bg-red-100 dark:bg-red-500/20 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-500/30' :
                                            selectedIncident.details.severity?.toLowerCase() === 'high' ? 'bg-orange-100 dark:bg-orange-500/20 text-orange-600 dark:text-orange-400 border border-orange-200 dark:border-orange-500/30' :
                                            selectedIncident.details.severity?.toLowerCase() === 'medium' ? 'bg-yellow-100 dark:bg-yellow-500/20 text-yellow-600 dark:text-yellow-500 border border-yellow-200 dark:border-yellow-500/30' :
                                            'bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-500/30'
                                        }`}>
                                            Severity: {selectedIncident.details.severity}
                                        </span>
                                        <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${selectedIncident.details.status === 'Ongoing' ? 'bg-red-100 dark:bg-red-500/10 text-red-600 dark:text-red-400 border border-red-200 dark:border-red-500/30' : 'bg-green-100 dark:bg-green-500/10 text-green-600 dark:text-green-400 border border-green-200 dark:border-green-500/30'}`}>
                                            {selectedIncident.details.status}
                                        </span>
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-center gap-1 -mr-2">
                                <button
                                    onClick={handleEditClick}
                                    className="p-2 text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-500/10 rounded-full transition-colors"
                                    title="Edit Incident"
                                >
                                    <Pencil className="w-5 h-5" />
                                </button>
                                <button
                                    onClick={handleDeleteClick}
                                    className="p-2 text-red-500 hover:text-red-600 dark:text-red-400 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-500/10 rounded-full transition-colors"
                                    title="Delete Incident"
                                >
                                    <Trash2 className="w-5 h-5" />
                                </button>
                                <button
                                    onClick={closePanel}
                                    className="p-2 text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-200 dark:hover:bg-white/10 rounded-full transition-colors"
                                    title="Close Panel"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>
                        </div>

                        {/* Body Details */}
                        <div className="flex-1 p-6 overflow-y-auto space-y-6">

                            {/* Coordinate Data */}
                            <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-4 rounded-xl flex items-center justify-between transition-colors duration-300">
                                <div>
                                    <p className="text-xs text-gray-500 font-medium mb-1">Coordinates (Lng, Lat)</p>
                                    <p className="text-sm font-mono text-gray-900 dark:text-gray-300">
                                        {selectedIncident.position[0].toFixed(5)}, {selectedIncident.position[1].toFixed(5)}
                                    </p>
                                </div>
                                <MapPin className="w-5 h-5 text-gray-400 dark:text-gray-600" />
                            </div>

                            {/* Description Block */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Description</h3>
                                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap">
                                    {selectedIncident.details.desc}
                                </p>
                            </div>

                            {/* Additional Info Block */}
                            {selectedIncident.details.others && selectedIncident.details.others !== 'None' && (
                                <div>
                                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">Additional Conditions</h3>
                                    <div className="bg-amber-50 dark:bg-amber-500/5 border border-amber-200 dark:border-amber-500/20 p-4 rounded-xl transition-colors duration-300">
                                        <p className="text-sm text-amber-700 dark:text-amber-200/80 leading-relaxed">
                                            {selectedIncident.details.others}
                                        </p>
                                    </div>
                                </div>
                            )}

                        </div>

                        {/* Footer Record Metadata */}
                        <div className="p-4 bg-gray-100 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 text-xs text-gray-500 font-mono flex items-center justify-between transition-colors duration-300">
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

export default LiveIncidentMode;
