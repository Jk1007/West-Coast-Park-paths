import React, { useRef, useState, useCallback, useMemo, useEffect, forwardRef, useImperativeHandle } from 'react';
import Map, { Source, Layer, NavigationControl, FullscreenControl } from 'react-map-gl/maplibre';
import * as turf from '@turf/turf';
import { PARK_CENTER, PARK_BOUNDS } from '../data/ParkData';
import ParkGraph from '../data/ParkGraph.json';
import maplibregl from 'maplibre-gl';
import { Hand, Terminal } from 'lucide-react';
import { PlumeLayer } from './PlumeLayer';

const MapComponent = forwardRef(({
    agents = [],
    incidents = [],
    safeNodes = [],
    wind = { speed: 0, direction: 0 },
    onAddIncident,
    onAddIncidentDirect,
    theme = 'light',
    mode = 'simulation', 
    selectedLocation = null,
    onLocationSelect = null,
    onIncidentClick = null,
    selectedIncidentId = null,
    isNightMode = false,
    onWindChange = null,
    onHandStatusChange = null
}, ref) => {
    const mapRef = useRef(null);

    // --- WebSocket Gesture Variables ---
    const [gesturesEnabled, setGesturesEnabled] = useState(false);
    const [wsStatus, setWsStatus] = useState('disconnected');
    const wsRef = useRef(null);

    const stateRef = useRef({
        prevIndex: null,
        handDetected: false,
        smoothedIndex: { x: 0, y: 0 },
        prevDistance: null,
        smoothedDistance: 0,
        wasPinching: false,
        lastWindSpeed: -1,
        lastWindDir: -1,
        hoverStart: null
    });

    const fistDebounceRef = useRef(0);

    // --- 3-Step Gesture Pipeline States & Refs ---
    const [gestureMode, setGestureMode] = useState('navigate'); // 'navigate' | 'placement'
    const [pipelineStep, setPipelineStep] = useState(null); // null | 'aiming' | 'selecting'
    const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
    const [lockedPos, setLockedPos] = useState(null);
    const [lockedLngLat, setLockedLngLat] = useState(null);
    const [hoveredBtnId, setHoveredBtnId] = useState(null);
    const [hoverProgress, setHoverProgress] = useState(0);

    const pipelineRef = useRef({
        gestureMode: 'navigate',
        pipelineStep: null,
        cursorPos: { x: 0, y: 0 },
        lockedPos: null,
        lockedLngLat: null,
        hoveredBtnId: null,
        hoverProgress: 0,
        wasPinching: false
    });

    const confirmSelectionRef = useRef(null);
    const containerRef = useRef(null);

    // Sync pipeline state with refs for event loop stability
    useEffect(() => { pipelineRef.current.gestureMode = gestureMode; }, [gestureMode]);
    useEffect(() => { pipelineRef.current.pipelineStep = pipelineStep; }, [pipelineStep]);
    useEffect(() => { pipelineRef.current.cursorPos = cursorPos; }, [cursorPos]);
    useEffect(() => { pipelineRef.current.lockedPos = lockedPos; }, [lockedPos]);
    useEffect(() => { pipelineRef.current.lockedLngLat = lockedLngLat; }, [lockedLngLat]);
    useEffect(() => { pipelineRef.current.hoveredBtnId = hoveredBtnId; }, [hoveredBtnId]);
    useEffect(() => { pipelineRef.current.hoverProgress = hoverProgress; }, [hoverProgress]);

    // Handle pipeline modes cleanup & reset
    useEffect(() => {
        if (gestureMode === 'placement') {
            setPipelineStep('aiming');
        } else {
            setPipelineStep(null);
            setLockedPos(null);
            setLockedLngLat(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
        }
    }, [gestureMode]);

    // Web Audio Synthesizer
    const playSound = (type) => {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();
            
            if (type === 'click') {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime(800, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(300, ctx.currentTime + 0.08);
                gain.gain.setValueAtTime(0.15, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);
                osc.start();
                osc.stop(ctx.currentTime + 0.08);
            } else if (type === 'chime') {
                const now = ctx.currentTime;
                const playTone = (freq, start, duration) => {
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(freq, start);
                    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, start + duration);
                    gain.gain.setValueAtTime(0.1, start);
                    gain.gain.exponentialRampToValueAtTime(0.01, start + duration);
                    osc.start(start);
                    osc.stop(start + duration);
                };
                playTone(523.25, now, 0.3); // C5
                playTone(659.25, now + 0.05, 0.35); // E5
                playTone(783.99, now + 0.1, 0.4); // G5
                playTone(1046.50, now + 0.15, 0.5); // C6
            }
        } catch (e) {
            console.error('Failed to play synthesized sound:', e);
        }
    };

    const handleConfirmSelection = useCallback((btnId) => {
        if (btnId === 'cancel') {
            setPipelineStep('aiming');
            setLockedPos(null);
            setLockedLngLat(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
            playSound('click');
        } else {
            if (onAddIncidentDirect && pipelineRef.current.lockedLngLat) {
                onAddIncidentDirect(pipelineRef.current.lockedLngLat, btnId);
                playSound('chime');
            }
            setPipelineStep('aiming');
            setLockedPos(null);
            setLockedLngLat(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
        }
    }, [onAddIncidentDirect]);

    useEffect(() => {
        confirmSelectionRef.current = handleConfirmSelection;
    }, [handleConfirmSelection]);

    // Provide map access upwards functionally
    useImperativeHandle(ref, () => ({
        getMap: () => mapRef.current?.getMap()
    }));

    // WebSocket Management natively pulling from AI Engine
    useEffect(() => {
        if (!gesturesEnabled) {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            setWsStatus('disconnected');
            if (stateRef.current.handDetected) {
                stateRef.current.handDetected = false;
                if (onHandStatusChange) onHandStatusChange(false);
            }
            return;
        }

        let reconnectTimer;
        const connect = () => {
            if (!gesturesEnabled) return;

            setWsStatus('connecting...');
            console.log('[MapComponent] Attempting to connect to Hand Tracking server at ws://localhost:8000/ws/gestures');
            const ws = new WebSocket('ws://localhost:8000/ws/gestures');
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('Successfully bound to Python Hand Tracking server.');
                setWsStatus('connected');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.wind && onWindChange) {
                        if (data.wind.speed !== stateRef.current.lastWindSpeed || 
                            data.wind.direction !== stateRef.current.lastWindDir) {
                            stateRef.current.lastWindSpeed = data.wind.speed;
                            stateRef.current.lastWindDir = data.wind.direction;
                            onWindChange({
                                ...data.wind,
                                weather: { 
                                    temp: data.wind.temp, 
                                    rain: data.wind.rain, 
                                    hum: data.wind.hum 
                                }
                            });
                        }
                    }

                    if (data.detected) {
                        if (!stateRef.current.handDetected) {
                            stateRef.current.handDetected = true;
                            if (onHandStatusChange) onHandStatusChange(true);
                        }

                        // 1. FIST STOP LOGIC: If fist is closed, we freeze everything
                        if (data.isFist) {
                            stateRef.current.prevIndex = null;
                            stateRef.current.prevDistance = null;
                            return; 
                        }

                        const map = mapRef.current?.getMap();
                        if (!map) return;

                        // Get map container bounding rect to scale coordinates
                        const container = map.getContainer();
                        const rect = container.getBoundingClientRect();
                        const screenX = (data.targetX / 1000) * rect.width;
                        const screenY = (data.targetY / 1000) * rect.height;

                        // Exponential Moving Average (EMA) cursor position smoothing
                        const CURSOR_ALPHA = 0.25;
                        const prevPos = pipelineRef.current.cursorPos;
                        const smoothedX = prevPos.x + CURSOR_ALPHA * (screenX - prevPos.x);
                        const smoothedY = prevPos.y + CURSOR_ALPHA * (screenY - prevPos.y);
                        const smoothedPos = { x: smoothedX, y: smoothedY };
                        setCursorPos(smoothedPos);

                        // Pinch transition detection
                        const isPinching = data.dist < 0.055;
                        const wasPinching = pipelineRef.current.wasPinching;
                        pipelineRef.current.wasPinching = isPinching;
                        const didAirTap = isPinching && !wasPinching;

                        const currentMode = pipelineRef.current.gestureMode;
                        const currentStep = pipelineRef.current.pipelineStep;

                        if (currentMode === 'navigate') {
                            // 2. PAN LOGIC (Hand Index Finger)
                            const ALPHA = 0.1; 
                            stateRef.current.smoothedIndex.x += ALPHA * (data.targetX - stateRef.current.smoothedIndex.x);
                            stateRef.current.smoothedIndex.y += ALPHA * (data.targetY - stateRef.current.smoothedIndex.y);
                            const sIndex = stateRef.current.smoothedIndex;

                            if (stateRef.current.prevIndex) {
                                const dx = sIndex.x - stateRef.current.prevIndex.x;
                                const dy = sIndex.y - stateRef.current.prevIndex.y;
                                const PAN_SENSITIVITY = 0.6; 
                                const offsetX = -dx * PAN_SENSITIVITY;
                                const offsetY = -dy * PAN_SENSITIVITY;

                                if (Math.abs(offsetX) > 1.5 || Math.abs(offsetY) > 1.5) {
                                    map.panBy([offsetX, offsetY], { animate: false });
                                }
                            }
                            stateRef.current.prevIndex = { x: sIndex.x, y: sIndex.y };

                            // 3. ZOOM LOGIC (Thumb to Index Distance)
                            const ZOOM_ALPHA = 0.08; 
                            stateRef.current.smoothedDistance = stateRef.current.smoothedDistance === 0 
                                ? data.dist 
                                : stateRef.current.smoothedDistance + ZOOM_ALPHA * (data.dist - stateRef.current.smoothedDistance);
                            const sDist = stateRef.current.smoothedDistance;

                            if (stateRef.current.prevDistance !== null) {
                                const dDist = sDist - stateRef.current.prevDistance;
                                const ZOOM_SENSITIVITY = 6; 
                                if (Math.abs(dDist) > 0.005) { 
                                    const currentZoom = map.getZoom();
                                    map.jumpTo({ zoom: currentZoom + (dDist * ZOOM_SENSITIVITY), center: map.getCenter() });
                                }
                            }
                            stateRef.current.prevDistance = sDist;
                        } 
                        else if (currentMode === 'placement') {
                            stateRef.current.prevIndex = null;
                            stateRef.current.prevDistance = null;

                            if (currentStep === 'aiming') {
                                if (didAirTap) {
                                    const lngLat = map.unproject([smoothedPos.x, smoothedPos.y]);
                                    playSound('click');
                                    setLockedPos(smoothedPos);
                                    setLockedLngLat([lngLat.lng, lngLat.lat]);
                                    setPipelineStep('selecting');
                                    setHoveredBtnId(null);
                                    setHoverProgress(0);
                                }
                            } 
                            else if (currentStep === 'selecting') {
                                // Determine hovered element
                                const clientX = rect.left + smoothedPos.x;
                                const clientY = rect.top + smoothedPos.y;
                                const elem = document.elementFromPoint(clientX, clientY);

                                let currentHoverBtn = null;
                                if (elem) {
                                    const btn = elem.closest('[data-gesture-btn]');
                                    if (btn) {
                                        currentHoverBtn = btn.getAttribute('data-gesture-btn');
                                    }
                                }

                                const prevHoverBtn = pipelineRef.current.hoveredBtnId;
                                if (currentHoverBtn !== prevHoverBtn) {
                                    setHoveredBtnId(currentHoverBtn);
                                    setHoverProgress(0);
                                    if (currentHoverBtn) {
                                        stateRef.current.hoverStart = Date.now();
                                    } else {
                                        stateRef.current.hoverStart = null;
                                    }
                                } else if (currentHoverBtn) {
                                    if (stateRef.current.hoverStart) {
                                        const elapsed = Date.now() - stateRef.current.hoverStart;
                                        const progress = Math.min((elapsed / 1200) * 100, 100);
                                        setHoverProgress(progress);

                                        if (progress >= 100) {
                                            stateRef.current.hoverStart = null;
                                            if (confirmSelectionRef.current) {
                                                confirmSelectionRef.current(currentHoverBtn);
                                            }
                                        }
                                    }
                                }

                                if (didAirTap && currentHoverBtn) {
                                    stateRef.current.hoverStart = null;
                                    if (confirmSelectionRef.current) {
                                        confirmSelectionRef.current(currentHoverBtn);
                                    }
                                }
                            }
                        }
                    } else {
                        if (stateRef.current.handDetected) {
                            stateRef.current.handDetected = false;
                            if (onHandStatusChange) onHandStatusChange(false);
                        }
                        stateRef.current.prevIndex = null;
                        stateRef.current.prevDistance = null;
                    }
                } catch (err) {
                    console.error("[MapComponent] Payload parser failure: ", err);
                }
            };

            ws.onerror = (err) => {
                console.error("[MapComponent] WebSocket Error: ", err);
            };

            ws.onclose = (event) => {
                console.log(`[MapComponent] Disconnected from tracking server. Code: ${event.code}`);
                setWsStatus('disconnected');
                if (stateRef.current.handDetected) {
                    stateRef.current.handDetected = false;
                    if (onHandStatusChange) onHandStatusChange(false);
                }
                
                if (gesturesEnabled) {
                    reconnectTimer = setTimeout(connect, 3000);
                }
            };
        };

        if (gesturesEnabled) {
            connect();
        }

        return () => {
            if (reconnectTimer) clearTimeout(reconnectTimer);
            if (wsRef.current) {
                wsRef.current.onopen = null;
                wsRef.current.onmessage = null;
                wsRef.current.onerror = null;
                wsRef.current.onclose = null;
                wsRef.current.close();
            }
            if (stateRef.current.handDetected) {
                stateRef.current.handDetected = false;
                if (onHandStatusChange) onHandStatusChange(false);
            }
        };
    }, [gesturesEnabled, onAddIncident, onAddIncidentDirect, onWindChange, onHandStatusChange]);

    const toggleGestures = useCallback(() => {
        setGesturesEnabled(prev => !prev);
    }, []);


    // --- Core Map Configuration (Preserved completely from original structure) ---
    const initialViewState = {
        longitude: PARK_CENTER[0],
        latitude: PARK_CENTER[1],
        zoom: 15.5,
        pitch: 45,
        bearing: 0
    };

    const mapStyleUrl = theme === 'dark' 
        ? 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
        : 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';

    const handleContextMenu = (evt) => {
        if (mode === 'live' && onAddIncident) {
            onAddIncident([evt.lngLat.lng, evt.lngLat.lat]);
        }
    };

    const handleClick = (evt) => {
        if (mode === 'view' || mode === 'live' || mode === 'simulation') {
            if (onLocationSelect) {
                onLocationSelect([evt.lngLat.lng, evt.lngLat.lat]);
            }
            
            const features = mapRef.current?.queryRenderedFeatures(evt.point, {
                layers: ['gaussian-plume-layer']
            });
            if (features && features.length > 0 && onIncidentClick) {
                const featureId = features[0].properties.id;
                onIncidentClick(incidents.find(inc => inc.id === featureId));
            }
        }
    };

    // --- Dynamic Geometric GeoJSON Compilers ---
    const graphGeoJSON = useMemo(() => {
        const features = ParkGraph.edges.map(edge => {
            const start = ParkGraph.nodes.find(n => n.id === edge.source);
            const end = ParkGraph.nodes.find(n => n.id === edge.target);
            if (!start || !end) return null;
            return {
                type: 'Feature',
                geometry: {
                    type: 'LineString',
                    coordinates: [[start.lon, start.lat], [end.lon, end.lat]]
                },
                properties: { id: `${edge.source}-${edge.target}` }
            };
        }).filter(Boolean);

        return { type: 'FeatureCollection', features };
    }, []);

    const agentsGeoJSON = useMemo(() => {
        if (!agents || agents.length === 0) return null;
        const features = agents.map(a => ({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [a.position[0], a.position[1]] },
            properties: { id: a.id, state: a.state }
        }));
        return { type: 'FeatureCollection', features };
    }, [agents]);

    const selectionGeoJSON = useMemo(() => {
        if (!selectedLocation || (mode !== 'view' && mode !== 'live' && mode !== 'simulation')) return null;
        return {
            type: 'FeatureCollection',
            features: [{
                type: 'Feature',
                geometry: { type: 'Point', coordinates: selectedLocation },
                properties: {}
            }]
        };
    }, [selectedLocation, mode]);

    const safeNodesGeoJSON = useMemo(() => {
        if (!safeNodes || safeNodes.length === 0) return null;

        const nodeMap = {};
        ParkGraph.nodes.forEach(n => { nodeMap[n.id] = [n.lon, n.lat]; });

        const features = safeNodes.map(id => {
            const coords = nodeMap[id];
            if (!coords) return null;
            return {
                type: 'Feature',
                geometry: { type: 'Point', coordinates: coords },
                properties: {}
            };
        }).filter(Boolean);

        return { type: 'FeatureCollection', features };
    }, [safeNodes]);

    // --- Map Styles ---
    const pathsLayerStyle = {
        id: 'paths-layer',
        type: 'line',
        paint: {
            'line-color': theme === 'dark' ? '#4b5563' : '#64748b',
            'line-width': 2.5,
            'line-opacity': 0.7
        }
    };

    const agentsLayerStyle = {
        id: 'agents-layer',
        type: 'circle',
        paint: {
            'circle-radius': 5,
            'circle-color': [
                'match',
                ['get', 'state'],
                'IDLE', '#16a34a',
                'EVACUATING', '#ea580c',
                'ESCAPED', '#2563eb',
                '#6b7280'
            ],
            'circle-stroke-width': 1.5,
            'circle-stroke-color': '#ffffff'
        }
    };

    const safeNodesLayerStyle = {
        id: 'safe-nodes-layer',
        type: 'circle',
        paint: {
            'circle-radius': 8,
            'circle-color': '#06b6d4', 
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
            'circle-opacity': 0.8
        }
    };

    const selectionLayerStyle = {
        id: 'selection-layer',
        type: 'circle',
        paint: {
            'circle-radius': 10,
            'circle-color': '#3b82f6',
            'circle-stroke-width': 3,
            'circle-stroke-color': '#ffffff',
            'circle-opacity': 0.9
        }
    };


    return (
        <div ref={containerRef} className="relative w-full h-full">
            <Map
                initialViewState={initialViewState}
                ref={mapRef}
                style={{ width: '100%', height: '100%' }}
                mapStyle={mapStyleUrl}
                onContextMenu={handleContextMenu}
                onClick={handleClick}
                cursor={mode === 'live' ? "pointer" : "crosshair"}
                interactiveLayerIds={mode === 'view' || mode === 'live' ? ['gaussian-plume-layer'] : undefined}
            >
                <NavigationControl position="top-right" />
                <FullscreenControl position="top-right" />

                <Source id="paths-source" type="geojson" data={graphGeoJSON}>
                    <Layer {...pathsLayerStyle} />
                </Source>

                <PlumeLayer 
                    incidents={incidents} 
                    wind={wind} 
                    isNightMode={isNightMode} 
                    selectedIncidentId={selectedIncidentId} 
                />

                <Source id="agents-source" type="geojson" data={agentsGeoJSON}>
                    <Layer {...agentsLayerStyle} />
                </Source>

                {selectionGeoJSON && (
                    <Source id="selection-source" type="geojson" data={selectionGeoJSON}>
                        <Layer {...selectionLayerStyle} />
                    </Source>
                )}

                {safeNodesGeoJSON && (
                    <Source id="safe-nodes-source" type="geojson" data={safeNodesGeoJSON}>
                        <Layer {...safeNodesLayerStyle} />
                    </Source>
                )}
            </Map>

            {/* Custom Sci-Fi Hologram Cursor/Reticle Overlays */}
            {gesturesEnabled && stateRef.current.handDetected && (
                <div
                    className="absolute w-8 h-8 pointer-events-none z-[100] transition-transform duration-75"
                    style={{
                        left: `${cursorPos.x}px`,
                        top: `${cursorPos.y}px`,
                        transform: 'translate(-50%, -50%)'
                    }}
                >
                    {/* Outer glowing pulsing ring */}
                    <div className="absolute inset-0 rounded-full border-2 border-cyan-400/80 bg-cyan-500/10 animate-pulse" />
                    {/* Inner glowing core */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-cyan-400 rounded-full shadow-[0_0_10px_#22d3ee]" />
                </div>
            )}

            {/* Aiming Reticle Phase */}
            {gestureMode === 'placement' && pipelineStep === 'aiming' && (
                <div
                    className="absolute pointer-events-none z-[90]"
                    style={{
                        left: `${cursorPos.x}px`,
                        top: `${cursorPos.y}px`,
                        transform: 'translate(-50%, -50%)'
                    }}
                >
                    <div className="w-16 h-16 border-2 border-dashed border-red-500/60 rounded-full animate-spin [animation-duration:8s]" />
                    <div className="absolute inset-2 border border-red-500/40 rounded-full animate-ping [animation-duration:2s]" />
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-red-600 rounded-full shadow-[0_0_8px_#dc2626]" />
                    
                    <div className="absolute top-12 left-1/2 -translate-x-1/2 bg-slate-900/90 text-white border border-slate-700 px-3 py-1 rounded text-[10px] font-bold tracking-wider whitespace-nowrap shadow-lg uppercase">
                        Pinch (Air Tap) to Lock Position
                    </div>
                </div>
            )}

            {/* Selecting Phase - Locked Reticle and Option Menu */}
            {gestureMode === 'placement' && pipelineStep === 'selecting' && lockedPos && (
                <>
                    {/* Locked geographical marker */}
                    <div
                        className="absolute pointer-events-none z-[90]"
                        style={{
                            left: `${lockedPos.x}px`,
                            top: `${lockedPos.y}px`,
                            transform: 'translate(-50%, -50%)'
                        }}
                    >
                        <div className="w-16 h-16 border-2 border-emerald-500/80 rounded-full shadow-[0_0_12px_rgba(16,185,129,0.4)]" />
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-emerald-500/30 rounded-full animate-ping" />
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2.5 h-2.5 bg-emerald-600 rounded-full shadow-[0_0_8px_#10b981]" />
                    </div>

                    {/* Floating Options Menu */}
                    <div
                        className="absolute z-[95] backdrop-blur-md bg-slate-950/90 border border-slate-800 rounded-2xl p-4 shadow-[0_10px_30px_rgba(0,0,0,0.5)] flex flex-col gap-2 w-64 select-none"
                        style={{
                            left: `${lockedPos.x}px`,
                            top: `${lockedPos.y}px`,
                            transform: 'translate(-50%, -120%)'
                        }}
                    >
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-800/80 pb-2 mb-1">
                            Deploy Spill Payload
                        </div>

                        {[
                            { id: 'CHLORINE_GAS', label: 'Chlorine Gas (Cl2)' },
                            { id: 'AMMONIA', label: 'Anhydrous Ammonia (NH3)' },
                            { id: 'METHANE_LEAK', label: 'Methane Pipeline Leak' },
                            { id: 'INDUSTRIAL_FIRE', label: 'Industrial Toxic Fire' },
                            { id: 'cancel', label: 'Cancel' }
                        ].map((btn) => (
                            <button
                                key={btn.id}
                                data-gesture-btn={btn.id}
                                className={`relative overflow-hidden w-full px-3 py-2 rounded-xl text-left text-xs font-semibold tracking-wide transition-all duration-200 border cursor-pointer ${
                                    hoveredBtnId === btn.id
                                        ? btn.id === 'cancel'
                                            ? 'bg-slate-700/30 text-white border-slate-500 scale-[1.02]'
                                            : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30 scale-[1.02]'
                                        : 'text-slate-300 border-slate-800/80 bg-slate-900/40'
                                }`}
                            >
                                <span className="relative z-10">{btn.label}</span>
                                {hoveredBtnId === btn.id && (
                                    <div 
                                        className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-75"
                                        style={{ width: `${hoverProgress}%` }}
                                    />
                                )}
                            </button>
                        ))}
                    </div>
                </>
            )}

            {/* Websocket Connection UI & Mode Selectors */}
            <div className="absolute bottom-6 left-6 flex gap-3 z-50">
                <button
                    onClick={toggleGestures}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold shadow-lg transition-all border text-xs uppercase tracking-wider ${
                        gesturesEnabled
                            ? 'bg-blue-600 hover:bg-blue-700 text-white border-blue-500'
                            : 'bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-700'
                    }`}
                >
                    <Hand className="w-5 h-5" />
                    {gesturesEnabled ? 'Disconnect AI Tracker' : 'Enable AI Tracker'}
                </button>

                {gesturesEnabled && (
                    <div className="flex bg-slate-900/90 backdrop-blur-md p-1 rounded-full border border-slate-700/50 shadow-lg select-none">
                        <button
                            onClick={() => setGestureMode('navigate')}
                            className={`px-4 py-1.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider transition-all duration-300 cursor-pointer ${
                                gestureMode === 'navigate'
                                    ? 'bg-blue-600 text-white shadow-md'
                                    : 'text-slate-400 hover:text-white'
                            }`}
                        >
                            Navigate
                        </button>
                        <button
                            onClick={() => setGestureMode('placement')}
                            className={`px-4 py-1.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider transition-all duration-300 cursor-pointer ${
                                gestureMode === 'placement'
                                    ? 'bg-red-600 text-white shadow-md'
                                    : 'text-slate-400 hover:text-white'
                            }`}
                        >
                            Place Hazard
                        </button>
                    </div>
                )}
            </div>

            {/* Backend Headless Indicator Box */}
            {gesturesEnabled && (
                <div className="absolute bottom-6 right-6 w-56 h-16 bg-gray-950 border-2 border-gray-800 rounded-xl flex items-center justify-between px-4 shadow-2xl z-50">
                   <div className="flex items-center gap-3">
                       <Terminal className="text-gray-500 w-5 h-5"/>
                       <div className="flex flex-col">
                           <span className="text-gray-300 text-xs font-bold uppercase tracking-wider">AI Python Engine</span>
                           <span className={`text-xs font-mono font-medium ${wsStatus === 'connected' ? 'text-green-400' : 'text-amber-500'}`}>
                               WS: {wsStatus}
                           </span>
                       </div>
                   </div>
                   {wsStatus === 'connected' && (
                       <span className="relative flex h-2.5 w-2.5">
                           <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                           <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500"></span>
                       </span>
                   )}
                </div>
            )}
            
            {gesturesEnabled && wsStatus === 'connected' && (
                <div className="absolute bottom-24 right-6 bg-gray-950/80 backdrop-blur-md px-4 py-2 rounded-lg border border-gray-800 text-gray-300 text-xs shadow-xl z-50 pointer-events-none">
                    WebSocket port 8000 receiving ML tracking boundaries.
                </div>
            )}
        </div>
    );
});

export default MapComponent;
