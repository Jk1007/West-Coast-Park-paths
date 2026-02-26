import React, { useState } from 'react';
import { AnimatePresence } from 'framer-motion';
import Navigation from './components/Navigation';
import LandingHub from './components/LandingHub';
import SimulationMode from './components/SimulationMode';
import LiveIncidentMode from './components/LiveIncidentMode';
import IncidentViewMode from './components/IncidentViewMode';

function App() {
    // Modes: 'landing', 'simulation', 'live', 'view'
    const [currentMode, setCurrentMode] = useState('landing');
    const [isFormDirty, setIsFormDirty] = useState(false);

    // For Confirmation Modal
    const [pendingMode, setPendingMode] = useState(null);
    const [showConfirm, setShowConfirm] = useState(false);

    const handleAttemptNavigate = (targetMode) => {
        if (targetMode === currentMode) return;

        if (currentMode === 'live' && isFormDirty) {
            setPendingMode(targetMode);
            setShowConfirm(true);
        } else {
            setCurrentMode(targetMode);
        }
    };

    const confirmNavigation = () => {
        if (pendingMode) {
            setCurrentMode(pendingMode);
            setIsFormDirty(false); // Reset dirty state
        }
        setShowConfirm(false);
        setPendingMode(null);
    };

    const cancelNavigation = () => {
        setShowConfirm(false);
        setPendingMode(null);
    };

    return (
        <div className="w-screen h-screen bg-gray-950 flex flex-col font-sans overflow-hidden text-gray-100 selection:bg-blue-500/30">

            {/* Top Navigation */}
            <Navigation
                currentMode={currentMode}
                onAttemptNavigate={handleAttemptNavigate}
            />

            {/* Main Content Area with Routing & Animations */}
            <div className="flex-1 relative w-full h-full overflow-hidden">
                <AnimatePresence mode="wait">
                    {currentMode === 'landing' && (
                        <LandingHub
                            key="landing"
                            onSelectMode={handleAttemptNavigate}
                        />
                    )}

                    {currentMode === 'simulation' && (
                        <SimulationMode
                            key="simulation"
                            onExit={() => handleAttemptNavigate('landing')}
                        />
                    )}

                    {currentMode === 'live' && (
                        <LiveIncidentMode
                            key="live"
                            onFormStateChange={setIsFormDirty}
                        />
                    )}

                    {currentMode === 'view' && (
                        <IncidentViewMode
                            key="view"
                        />
                    )}
                </AnimatePresence>
            </div>

            {/* Unsaved Changes Confirmation Modal */}
            <AnimatePresence>
                {showConfirm && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-gray-950/80 backdrop-blur-sm px-4">
                        <div className="bg-gray-900 border border-gray-800 rounded-2xl p-8 max-w-md w-full shadow-2xl origin-bottom">
                            <h3 className="text-xl font-bold text-white mb-2">Discard Unsaved Changes?</h3>
                            <p className="text-gray-400 mb-8">
                                You have partially filled out a live incident log. Navigating away will discard your changes. Are you sure you want to leave?
                            </p>

                            <div className="flex items-center justify-end gap-3">
                                <button
                                    onClick={cancelNavigation}
                                    className="px-5 py-2.5 rounded-xl font-medium text-gray-300 hover:text-white hover:bg-gray-800 transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={confirmNavigation}
                                    className="px-5 py-2.5 rounded-xl font-medium bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white border border-red-500/20 transition-all"
                                >
                                    Discard & Leave
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </AnimatePresence>

        </div>
    );
}

export default App;
