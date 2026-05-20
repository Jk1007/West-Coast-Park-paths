import React, { useState, useEffect } from 'react';
import { AnimatePresence } from 'framer-motion';
import Navigation from './components/Navigation';
import LandingHub from './components/LandingHub';
import SimulationMode from './components/SimulationMode';
import LiveIncidentMode from './components/LiveIncidentMode';
import SimulationReplayMode from './components/SimulationReplayMode';
import AuthForm from './components/AuthForm';
import supabase from './supabase';

function App() {
    // Theme State
    const [theme, setTheme] = useState(() => {
        return localStorage.getItem('theme') || 'dark';
    });

    useEffect(() => {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
        localStorage.setItem('theme', theme);
    }, [theme]);

    const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'light' : 'dark');

    // Auth State
    const [session, setSession] = useState(null);
    const [isAuthLoading, setIsAuthLoading] = useState(true);
    const [isAutoAuthenticating, setIsAutoAuthenticating] = useState(false);
    const [autoAuthError, setAutoAuthError] = useState(null);

    const performAutoAuth = async () => {
        setIsAutoAuthenticating(true);
        setAutoAuthError(null);
        try {
            const email = 'guest@crowdshield.org';
            const password = 'guestpassword123';

            const { data, error } = await supabase.auth.signInWithPassword({ email, password });
            if (error) {
                // If user doesn't exist, try signing up first
                if (error.message.includes("Invalid login credentials") || error.status === 400) {
                    const { error: signUpError } = await supabase.auth.signUp({ email, password });
                    if (signUpError) {
                        throw signUpError;
                    }
                    // Try logging in again after signup
                    const { error: secondSignInError } = await supabase.auth.signInWithPassword({ email, password });
                    if (secondSignInError) throw secondSignInError;
                } else {
                    throw error;
                }
            }
            // Clear manually signed out flag on successful auto authentication
            localStorage.removeItem('manually_signed_out');
        } catch (err) {
            console.error("Auto authentication failed:", err);
            setAutoAuthError(err.message || "Could not authenticate as guest.");
        } finally {
            setIsAutoAuthenticating(false);
        }
    };

    useEffect(() => {
        // Initial session fetch
        supabase.auth.getSession().then(({ data: { session } }) => {
            setSession(session);
            setIsAuthLoading(false);

            // Trigger auto login if not logged in and not manually signed out
            if (!session && localStorage.getItem('manually_signed_out') !== 'true') {
                performAutoAuth();
            }
        });

        // Listen for auth changes (login/logout)
        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
            setSession(session);
        });

        return () => subscription.unsubscribe();
    }, []);

    // Modes: 'landing', 'simulation', 'live', 'replay'
    const [currentMode, setCurrentMode] = useState('landing');
    const [activeReplayTape, setActiveReplayTape] = useState(null);
    const [isFormDirty, setIsFormDirty] = useState(false);

    // For Confirmation Modal
    const [pendingMode, setPendingMode] = useState(null);
    const [showConfirm, setShowConfirm] = useState(false);

    const handleAttemptNavigate = (targetMode, dataPayload = null) => {
        if (targetMode === currentMode) return;

        if (currentMode === 'live' && isFormDirty) {
            setPendingMode(targetMode);
            setShowConfirm(true);
        } else {
            setCurrentMode(targetMode);
            if (targetMode === 'replay' && dataPayload) {
                setActiveReplayTape(dataPayload);
            } else if (targetMode === 'landing') {
                setActiveReplayTape(null); // Clear tape on exit
            }
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

    if (isAuthLoading || isAutoAuthenticating) {
        return (
            <div className="w-screen h-screen bg-gray-50 dark:bg-gray-950 flex flex-col items-center justify-center gap-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                {isAutoAuthenticating && (
                    <p className="text-sm font-medium text-gray-500 dark:text-gray-400 animate-pulse">
                        Authenticating guest session...
                    </p>
                )}
            </div>
        );
    }

    if (!session) {
        return <AuthForm onGuestAccess={performAutoAuth} />;
    }

    return (
        <div className="w-screen h-screen bg-gray-50 dark:bg-gray-950 flex flex-col font-sans overflow-hidden text-gray-900 dark:text-gray-100 selection:bg-brand-light/30 transition-colors duration-300">

            {/* Top Navigation */}
            <Navigation
                currentMode={currentMode}
                onAttemptNavigate={handleAttemptNavigate}
                theme={theme}
                toggleTheme={toggleTheme}
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
                            theme={theme}
                        />
                    )}

                    {currentMode === 'live' && (
                        <LiveIncidentMode
                            key="live"
                            onFormStateChange={setIsFormDirty}
                            theme={theme}
                        />
                    )}

                    {currentMode === 'replay' && activeReplayTape && (
                        <SimulationReplayMode
                            key="replay"
                            simData={activeReplayTape}
                            onExit={() => handleAttemptNavigate('landing')}
                            theme={theme}
                        />
                    )}
                </AnimatePresence>
            </div>

            {/* Unsaved Changes Confirmation Modal */}
            <AnimatePresence>
                {showConfirm && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-gray-900/40 dark:bg-gray-950/80 backdrop-blur-sm px-4">
                        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-2xl p-8 max-w-md w-full shadow-2xl origin-bottom">
                            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">Discard Unsaved Changes?</h3>
                            <p className="text-gray-600 dark:text-gray-300 mb-8">
                                You have partially filled out a live incident log. Navigating away will discard your changes. Are you sure you want to leave?
                            </p>

                            <div className="flex items-center justify-end gap-3">
                                <button
                                    onClick={cancelNavigation}
                                    className="px-5 py-2.5 rounded-xl font-medium text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
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
