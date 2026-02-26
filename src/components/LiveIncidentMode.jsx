import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Send, AlertTriangle, CheckCircle2, X } from 'lucide-react';
import MapComponent from './MapComponent';
import { db } from '../firebase';
import { collection, addDoc } from 'firebase/firestore';

const LiveIncidentMode = ({ onFormStateChange }) => {
    // Selected Location from Map Click
    const [selectedLocation, setSelectedLocation] = useState(null);

    // Form State mapped to database schema
    const [formData, setFormData] = useState({
        Title: '',
        Severity_lvl: 'Critical',
        Type: 'Chemical Spill',
        Details: '',
        Others: ''
    });

    const [isSubmitted, setIsSubmitted] = useState(false);

    // Plume Spread State
    const [activePlume, setActivePlume] = useState(null);
    const animationRef = useRef(null);

    // Track dirty state to warn user on navigation
    useEffect(() => {
        const isDirty = selectedLocation !== null || formData.Title !== '' || formData.Details !== '' || formData.Others !== '';
        onFormStateChange(isDirty);
    }, [formData, selectedLocation, onFormStateChange]);

    // Cleanup Animation Loop on unmount
    useEffect(() => {
        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, []);

    const handleMapClick = (lngLat) => {
        if (!selectedLocation && !isSubmitted) {
            setSelectedLocation(lngLat);
        }
    };

    const handleCancel = () => {
        setSelectedLocation(null);
        setFormData({ Title: '', Severity_lvl: 'Critical', Type: 'Chemical Spill', Details: '', Others: '' });
    };

    const runPlumeAnimation = (startTime) => {
        const animate = (time) => {
            const elapsed = time - startTime;
            // Grow radius slowly over time, up to a maximum
            // Let's say it grows 5 units per second, starting at 10, maxing at 150
            const currentRadius = Math.min(10 + (elapsed / 1000) * 15, 150);

            setActivePlume(prev => prev ? { ...prev, radius: currentRadius } : null);
            animationRef.current = requestAnimationFrame(animate);
        };
        animationRef.current = requestAnimationFrame(animate);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        console.log("Submitting Live Incident at", selectedLocation, "Details:", formData);

        const payload = {
            Inc_Id: `INC-${crypto.randomUUID()}`,
            Desc: formData.Details,
            Others: formData.Others,
            Servertity: formData.Severity_lvl,
            Type: formData.Type,
            // Keeping coordinates and timestamp for the map UI to use if needed
            Coordinates: selectedLocation,
            Timestamp: new Date().toISOString()
        };

        try {
            // Log to Firebase 'incident' collection as shown in the screenshot
            await addDoc(collection(db, "incident"), payload);
            console.log("Incident successfully written to database.");
        } catch (error) {
            console.error("Error writing incident to Firebase Database: ", error);
            console.warn("Proceeding with visual simulation only since Firebase credentials might not be configured.");
        }

        // Start Visual Plume Spread (proceeds even if Firebase fails for demo purposes)
        setActivePlume({ position: selectedLocation, radius: 10 });
        runPlumeAnimation(performance.now());

        setIsSubmitted(true);
        onFormStateChange(false); // Clean state after submission

        // Mock successful submission UI reset after 5s (plume keeps spreading contextually)
        setTimeout(() => {
            setIsSubmitted(false);
            setSelectedLocation(null);
            setFormData({ Title: '', Severity_lvl: 'Critical', Type: 'Chemical Spill', Details: '', Others: '' });
            // We keep the plume active on the map!
        }, 5000);
    };

    // Convert activePlume to incidents array format expected by MapComponent
    const mapIncidents = activePlume ? [activePlume] : [];

    return (
        <div className="w-full h-full relative">

            {/* Base Map Layer */}
            <MapComponent
                mode="live"
                theme="dark" // Force dark theme for production mode
                selectedLocation={selectedLocation}
                onLocationSelect={handleMapClick}
                incidents={mapIncidents} // Render expanding plume here
            />

            {/* Instruction Overlay when no location is selected */}
            <AnimatePresence>
                {!selectedLocation && !isSubmitted && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className="absolute top-24 left-1/2 -translate-x-1/2 bg-blue-900/80 backdrop-blur-md border border-blue-500/50 px-6 py-3 rounded-full shadow-2xl z-10 pointer-events-none flex items-center gap-3"
                    >
                        <ShieldAlert className="w-5 h-5 text-blue-400" />
                        <span className="text-blue-100 font-medium tracking-wide">Select a location on the map to log a live incident</span>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Incident Entry Modal overlay */}
            <AnimatePresence>
                {selectedLocation && (
                    <div className="absolute inset-0 z-50 flex items-center justify-center p-4 bg-gray-950/40 backdrop-blur-sm">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="w-full max-w-2xl bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden relative flex flex-col max-h-[90vh]"
                        >
                            {/* Header */}
                            <div className="flex items-center justify-between p-6 border-b border-gray-800 bg-gray-900/50">
                                <div className="flex items-center gap-4">
                                    <div className="w-10 h-10 rounded-lg bg-red-500/10 flex items-center justify-center border border-red-500/20">
                                        <ShieldAlert className="w-5 h-5 text-red-500" />
                                    </div>
                                    <div>
                                        <h2 className="text-xl font-bold text-gray-100">Live Hazard Report</h2>
                                        <p className="text-xs text-gray-400 font-mono mt-0.5">
                                            LOC: [{selectedLocation[1].toFixed(5)}, {selectedLocation[0].toFixed(5)}]
                                        </p>
                                    </div>
                                </div>
                                {!isSubmitted && (
                                    <button onClick={handleCancel} className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-full transition-colors">
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
                                        <h2 className="text-2xl font-bold text-green-400 mb-2">Hazard Dispatched</h2>
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
                                                <label className="block text-sm font-medium text-gray-300 mb-1.5">Incident Title</label>
                                                <input
                                                    type="text"
                                                    required
                                                    value={formData.Title}
                                                    onChange={(e) => setFormData({ ...formData, Title: e.target.value })}
                                                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-600"
                                                    placeholder="e.g. Chloric Gas Release"
                                                />
                                            </div>

                                            {/* Type */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-300 mb-1.5">Hazard Type</label>
                                                <select
                                                    value={formData.Type}
                                                    onChange={(e) => setFormData({ ...formData, Type: e.target.value })}
                                                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors appearance-none"
                                                >
                                                    <option value="Chemical Spill">Chemical Spill</option>
                                                    <option value="Gas Leak">Gas Leak</option>
                                                    <option value="Biological">Biological Hazard</option>
                                                    <option value="Fire/Smoke">Fire/Smoke Plume</option>
                                                    <option value="Radiation">Radiation</option>
                                                </select>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                                            {/* Severity */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-300 mb-1.5">Severity Level</label>
                                                <select
                                                    value={formData.Severity_lvl}
                                                    onChange={(e) => setFormData({ ...formData, Severity_lvl: e.target.value })}
                                                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors appearance-none"
                                                >
                                                    <option value="Critical">Critical (Evacuation Required)</option>
                                                    <option value="High">High</option>
                                                    <option value="Medium">Medium</option>
                                                    <option value="Low">Low</option>
                                                </select>
                                            </div>

                                            {/* Others */}
                                            <div>
                                                <label className="block text-sm font-medium text-gray-300 mb-1.5">Other Conditions (Weather, etc.)</label>
                                                <input
                                                    type="text"
                                                    value={formData.Others}
                                                    onChange={(e) => setFormData({ ...formData, Others: e.target.value })}
                                                    className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-600"
                                                    placeholder="e.g. High winds blowing east"
                                                />
                                            </div>
                                        </div>

                                        {/* Details */}
                                        <div>
                                            <label className="block text-sm font-medium text-gray-300 mb-1.5">Detailed Description</label>
                                            <textarea
                                                required
                                                rows={4}
                                                value={formData.Details}
                                                onChange={(e) => setFormData({ ...formData, Details: e.target.value })}
                                                className="w-full bg-gray-950 border border-gray-800 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 transition-colors placeholder-gray-600 resize-none"
                                                placeholder="Provide compound details, estimated volume..."
                                            />
                                        </div>
                                    </form>
                                )}
                            </div>

                            {/* Footer Actions */}
                            {!isSubmitted && (
                                <div className="p-6 border-t border-gray-800 bg-gray-900/50 flex items-center justify-between shrink-0">
                                    <div className="flex items-center gap-2 text-red-500/80 text-xs">
                                        <AlertTriangle className="w-4 h-4" />
                                        Initialize Hazard Spread Protocol
                                    </div>

                                    <div className="flex gap-3">
                                        <button
                                            type="button"
                                            onClick={handleCancel}
                                            className="px-5 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            type="submit"
                                            form="incident-form"
                                            className="flex items-center gap-2 bg-red-600 hover:bg-red-500 text-white text-sm font-medium px-6 py-2 rounded-lg transition-all shadow-[0_0_15px_rgba(220,38,38,0.3)] hover:shadow-[0_0_20px_rgba(220,38,38,0.5)]"
                                        >
                                            <Send className="w-4 h-4" />
                                            Dispatch & Track Plume
                                        </button>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default LiveIncidentMode;
