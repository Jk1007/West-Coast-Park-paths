import React, { useRef, useState, useCallback, useMemo, useEffect, forwardRef, useImperativeHandle } from 'react';
import Map, { Source, Layer, NavigationControl, FullscreenControl } from 'react-map-gl/maplibre';
import * as turf from '@turf/turf';
import { PARK_CENTER, PARK_BOUNDS } from '../data/ParkData';
import ParkGraph from '../data/ParkGraph.json';
import maplibregl from 'maplibre-gl';
import { Hand, Terminal, CheckCircle2, X } from 'lucide-react';
import { PlumeLayer } from './PlumeLayer';
import { PlumePhysics, CHEMICAL_Q_RATES } from '../simulation/PlumePhysics';

const MapComponent = forwardRef(({
    agents = [],
    incidents = [],
    safeNodes = [],
    wind = { speed: 0, direction: 0 },
    onAddIncident,
    onAddIncidentDirect,
    onResolveIncident,
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
    const [pipelineStep, setPipelineStep] = useState(null); // null | 'selecting'
    const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
    const [lockedPos, setLockedPos] = useState(null);
    const [lockedLngLat, setLockedLngLat] = useState(null);
    const [hoveredBtnId, setHoveredBtnId] = useState(null);
    const [hoverProgress, setHoverProgress] = useState(0);
    const [viewportChanged, setViewportChanged] = useState(0);

    const pipelineRef = useRef({
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
    const closeTimeoutRef = useRef(null);

    // Sync pipeline state with refs for event loop stability
    useEffect(() => { pipelineRef.current.pipelineStep = pipelineStep; }, [pipelineStep]);
    useEffect(() => { pipelineRef.current.cursorPos = cursorPos; }, [cursorPos]);
    useEffect(() => { pipelineRef.current.lockedPos = lockedPos; }, [lockedPos]);
    useEffect(() => { pipelineRef.current.lockedLngLat = lockedLngLat; }, [lockedLngLat]);
    useEffect(() => { pipelineRef.current.hoveredBtnId = hoveredBtnId; }, [hoveredBtnId]);
    useEffect(() => { pipelineRef.current.hoverProgress = hoverProgress; }, [hoverProgress]);

    const [hoveredIncident, setHoveredIncident] = useState(null);
    const hoveredIncidentRef = useRef(null);
    useEffect(() => {
        hoveredIncidentRef.current = hoveredIncident;
    }, [hoveredIncident]);
    const [lockedIncidentId, setLockedIncidentId] = useState(null);
    const lockedIncidentIdRef = useRef(null);
    useEffect(() => {
        lockedIncidentIdRef.current = lockedIncidentId;
    }, [lockedIncidentId]);

    const formatDuration = useCallback((inc) => {
        let seconds = 0;
        const startTs = new Date(inc.details?.timestamp || inc.startTime).getTime();
        const endTs = new Date(inc.details?.resolvedAt || inc.resolvedAt).getTime();
        if (!isNaN(startTs) && !isNaN(endTs)) {
            seconds = Math.round((endTs - startTs) / 1000);
        } else if (inc.elapsedSimSec) {
            seconds = Math.round(inc.elapsedSimSec);
        } else if (!isNaN(startTs)) {
            seconds = Math.round((Date.now() - startTs) / 1000);
        }
        
        if (seconds <= 0) return '0s';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        let parts = [];
        if (hrs > 0) parts.push(`${hrs}h`);
        if (mins > 0) parts.push(`${mins}m`);
        if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);
        return parts.join(' ');
    }, []);

    const checkHoverAtPoint = useCallback((point, clientX, clientY) => {
        const map = mapRef.current?.getMap();
        if (!map) return;

        // Check if cursor is over the tooltip first to ensure hover stability
        if (clientX !== undefined && clientY !== undefined) {
            const elem = document.elementFromPoint(clientX, clientY);
            if (elem && elem.closest('[data-hazard-tooltip]')) {
                if (closeTimeoutRef.current) {
                    clearTimeout(closeTimeoutRef.current);
                    closeTimeoutRef.current = null;
                }
                return;
            }
        }

        try {
            const queryPoint = Array.isArray(point) 
                ? point 
                : (point && typeof point.x === 'number' && typeof point.y === 'number' ? [point.x, point.y] : point);

            const lngLat = map.unproject(queryPoint);
            const cursorPt = turf.point([lngLat.lng, lngLat.lat]);
            let foundInc = null;

            for (const inc of incidents) {
                const stability = isNightMode ? 'F' : 'D';
                const baseQ = CHEMICAL_Q_RATES[inc.details?.type] || CHEMICAL_Q_RATES.CHLORINE_GAS; 
                const massRatio = Math.max(0.02, (inc.details?.amount || 100) / 200);
                const Q = baseQ * Math.pow(massRatio, 1.25);
                
                const polyCoords = PlumePhysics.generatePlumePolygon(
                    inc.position, 
                    wind.speed, 
                    wind.direction, 
                    Q, 
                    stability,
                    inc.elapsedSimSec || 0
                );

                if (polyCoords && polyCoords.length > 2) {
                    const poly = turf.polygon([polyCoords]);
                    if (turf.booleanPointInPolygon(cursorPt, poly)) {
                        foundInc = inc;
                        break;
                    }
                }
            }

            if (foundInc) {
                if (closeTimeoutRef.current) {
                    clearTimeout(closeTimeoutRef.current);
                    closeTimeoutRef.current = null;
                }
                setHoveredIncident(foundInc);
                return;
            }
        } catch (e) {
            console.error("Turf hover collision detection failed, falling back to queryRenderedFeatures:", e);
        }

        // Fallback: queryRenderedFeatures
        try {
            const queryPoint = Array.isArray(point) 
                ? point 
                : (point && typeof point.x === 'number' && typeof point.y === 'number' ? [point.x, point.y] : point);

            const features = map.queryRenderedFeatures(queryPoint, {
                layers: ['gaussian-plume-layer']
            });

            if (features && features.length > 0) {
                const featureId = features[0].properties.id;
                const inc = incidents.find(i => i.id === featureId);
                if (inc) {
                    if (closeTimeoutRef.current) {
                        clearTimeout(closeTimeoutRef.current);
                        closeTimeoutRef.current = null;
                    }
                    setHoveredIncident(inc);
                    return;
                }
            }
        } catch (e) {
            // Ignore
        }

        // If not hovering a plume and not over the tooltip, clear hover with debounce
        if (!closeTimeoutRef.current) {
            closeTimeoutRef.current = setTimeout(() => {
                if (lockedIncidentIdRef.current) {
                    const lockedInc = incidents.find(i => i.id === lockedIncidentIdRef.current);
                    if (lockedInc) {
                        setHoveredIncident(lockedInc);
                    } else {
                        setHoveredIncident(null);
                    }
                } else {
                    setHoveredIncident(null);
                }
                closeTimeoutRef.current = null;
            }, 600); // 600ms grace period to cross the gap or interact
        }
    }, [incidents, wind, isNightMode]);

    // Handle pipeline modes cleanup & reset when gestures are toggled off
    useEffect(() => {
        if (!gesturesEnabled) {
            setPipelineStep(null);
            setLockedPos(null);
            setLockedLngLat(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
            setHoveredIncident(null);
            setLockedIncidentId(null);
        }
    }, [gesturesEnabled]);

    useEffect(() => {
        return () => {
            if (closeTimeoutRef.current) {
                clearTimeout(closeTimeoutRef.current);
            }
        };
    }, []);

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

    const handleCloseTooltip = useCallback(() => {
        setLockedIncidentId(null);
        setHoveredIncident(null);
        if (closeTimeoutRef.current) {
            clearTimeout(closeTimeoutRef.current);
            closeTimeoutRef.current = null;
        }
        playSound('click');
    }, []);

    const handleConfirmSelection = useCallback((btnId) => {
        if (btnId === 'close-tooltip') {
            handleCloseTooltip();
        } else if (btnId.startsWith('resolve-')) {
            const incidentId = btnId.replace('resolve-', '');
            if (onResolveIncident) {
                onResolveIncident(incidentId);
            }
            playSound('chime');
            setHoveredIncident(null);
            setLockedIncidentId(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
            if (closeTimeoutRef.current) {
                clearTimeout(closeTimeoutRef.current);
                closeTimeoutRef.current = null;
            }
        } else if (btnId === 'cancel') {
            setPipelineStep(null);
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
            setPipelineStep(null);
            setLockedPos(null);
            setLockedLngLat(null);
            setHoveredBtnId(null);
            setHoverProgress(0);
        }
        // Reset movement states to prevent map panning jumps upon menu closing
        stateRef.current.prevIndex = null;
        stateRef.current.prevDistance = null;
    }, [onAddIncidentDirect, onResolveIncident]);

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
                        const CURSOR_ALPHA = 0.08; // LOWERED sensitivity to filter out hand jitter
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

                        const currentStep = pipelineRef.current.pipelineStep;

                        // Universally check if cursor is over a map plume to trigger hover tooltip
                        checkHoverAtPoint(smoothedPos, rect.left + smoothedPos.x, rect.top + smoothedPos.y);

                        // Universally check if cursor is over a gesture button
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

                        if (currentHoverBtn) {
                            // Freeze panning and zooming when hovering any button
                            stateRef.current.prevIndex = null;
                            stateRef.current.prevDistance = null;
                        } else if (currentStep === null) {
                            // --- NAVIGATION MODE ---
                            // 2. PAN LOGIC (Hand Index Finger)
                            const ALPHA = 0.05; // Smooth index finger coordinate change
                            stateRef.current.smoothedIndex.x += ALPHA * (data.targetX - stateRef.current.smoothedIndex.x);
                            stateRef.current.smoothedIndex.y += ALPHA * (data.targetY - stateRef.current.smoothedIndex.y);
                            const sIndex = stateRef.current.smoothedIndex;

                            if (stateRef.current.prevIndex) {
                                const dx = sIndex.x - stateRef.current.prevIndex.x;
                                const dy = sIndex.y - stateRef.current.prevIndex.y;
                                const PAN_SENSITIVITY = 0.15; // LOWERED sensitivity
                                const offsetX = -dx * PAN_SENSITIVITY;
                                const offsetY = -dy * PAN_SENSITIVITY;

                                if (Math.abs(offsetX) > 0.8 || Math.abs(offsetY) > 0.8) {
                                    map.panBy([offsetX, offsetY], { animate: false });
                                }
                            }
                            stateRef.current.prevIndex = { x: sIndex.x, y: sIndex.y };

                            // 3. ZOOM LOGIC (Thumb to Index Distance)
                            const ZOOM_ALPHA = 0.05; 
                            stateRef.current.smoothedDistance = stateRef.current.smoothedDistance === 0 
                                ? data.dist 
                                : stateRef.current.smoothedDistance + ZOOM_ALPHA * (data.dist - stateRef.current.smoothedDistance);
                            const sDist = stateRef.current.smoothedDistance;

                            if (stateRef.current.prevDistance !== null) {
                                const dDist = sDist - stateRef.current.prevDistance;
                                const ZOOM_SENSITIVITY = 2.5; // LOWERED sensitivity
                                if (Math.abs(dDist) > 0.005) { 
                                    const currentZoom = map.getZoom();
                                    map.jumpTo({ zoom: currentZoom + (dDist * ZOOM_SENSITIVITY), center: map.getCenter() });
                                }
                            }
                            stateRef.current.prevDistance = sDist;

                            // If we pinch (Air Tap) when in navigation mode and NOT hovering a button, check if we are hovering a plume
                            if (didAirTap) {
                                if (hoveredIncidentRef.current) {
                                    // Lock the persistent tooltip for this incident
                                    setLockedIncidentId(hoveredIncidentRef.current.id);
                                    playSound('click');
                                } else {
                                    // Lock the position and open selection menu (small panel)
                                    const lngLat = map.unproject([smoothedPos.x, smoothedPos.y]);
                                    playSound('click');
                                    setLockedPos(smoothedPos);
                                    setLockedLngLat([lngLat.lng, lngLat.lat]);
                                    setPipelineStep('selecting');
                                    setHoveredBtnId(null);
                                    setHoverProgress(0);
                                }
                                // Reset WS references
                                stateRef.current.prevIndex = null;
                                stateRef.current.prevDistance = null;
                            }
                        } else if (currentStep === 'selecting') {
                            // --- SELECTING/MENU MODE ---
                            // Map panning and zooming are frozen, index and distance states reset
                            stateRef.current.prevIndex = null;
                            stateRef.current.prevDistance = null;
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
            const features = mapRef.current?.queryRenderedFeatures(evt.point, {
                layers: ['gaussian-plume-layer']
            });
            if (features && features.length > 0) {
                const featureId = features[0].properties.id;
                const clickedInc = incidents.find(inc => inc.id === featureId);
                if (clickedInc) {
                    setLockedIncidentId(clickedInc.id);
                    setHoveredIncident(clickedInc);
                    if (onIncidentClick) {
                        onIncidentClick(clickedInc);
                    }
                    playSound('click');
                    return;
                }
            }

            // Clicked elsewhere on empty map space
            setLockedIncidentId(null);
            setHoveredIncident(null);

            if (onLocationSelect) {
                onLocationSelect([evt.lngLat.lng, evt.lngLat.lat]);
            }
            playSound('click');
        }
    };

    const currentLockedPos = useMemo(() => {
        if (!lockedLngLat || !mapRef.current) return null;
        try {
            const map = mapRef.current.getMap();
            if (!map) return null;
            return map.project(lockedLngLat);
        } catch (e) {
            return null;
        }
    }, [lockedLngLat, viewportChanged]);

    const currentHoveredIncidentPos = useMemo(() => {
        if (!hoveredIncident || !mapRef.current) return null;
        try {
            const map = mapRef.current.getMap();
            if (!map) return null;
            return map.project(hoveredIncident.position);
        } catch (e) {
            return null;
        }
    }, [hoveredIncident, viewportChanged]);

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
                onMove={() => setViewportChanged(prev => prev + 1)}
                onMouseMove={(evt) => {
                    checkHoverAtPoint(evt.point, evt.originalEvent?.clientX, evt.originalEvent?.clientY);
                }}
                onMouseLeave={() => {
                    if (closeTimeoutRef.current) {
                        clearTimeout(closeTimeoutRef.current);
                        closeTimeoutRef.current = null;
                    }
                    if (!lockedIncidentIdRef.current) {
                        setHoveredIncident(null);
                    }
                }}
                cursor={mode === 'live' ? "pointer" : "crosshair"}
                interactiveLayerIds={['gaussian-plume-layer']}
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

            {/* Premium Glassmorphic Hazard Tooltip */}
            {hoveredIncident && currentHoveredIncidentPos && (
                <div
                    data-hazard-tooltip
                    className="absolute z-[95] backdrop-blur-md bg-slate-950/90 border border-slate-800 rounded-2xl p-4 shadow-[0_10px_30px_rgba(0,0,0,0.5)] flex flex-col gap-2 w-64 select-none text-left"
                    style={{
                        left: `${currentHoveredIncidentPos.x}px`,
                        top: `${currentHoveredIncidentPos.y}px`,
                        transform: 'translate(-50%, -120%)'
                    }}
                >
                    <div className="flex justify-between items-center border-b border-slate-800/80 pb-2 mb-1">
                        <div className="flex items-center gap-2">
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                                Hazard Details
                            </span>
                            <span className={`text-[9px] font-extrabold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                                hoveredIncident.details?.status === 'Resolved'
                                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                    : 'bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse'
                            }`}>
                                {hoveredIncident.details?.status === 'Resolved' ? 'Resolved' : 'In-progress'}
                            </span>
                        </div>
                        <button
                            data-gesture-btn="close-tooltip"
                            onClick={handleCloseTooltip}
                            className={`p-1 rounded-full transition-colors cursor-pointer flex items-center justify-center relative overflow-hidden ${
                                hoveredBtnId === 'close-tooltip'
                                    ? 'bg-slate-800 text-white scale-110 shadow-md'
                                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                            }`}
                            title="Close Panel"
                        >
                            <X className="w-3.5 h-3.5" />
                            {hoveredBtnId === 'close-tooltip' && (
                                <div 
                                    className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-75"
                                    style={{ width: `${hoverProgress}%` }}
                                />
                            )}
                        </button>
                    </div>

                    <div className="flex flex-col gap-1.5 text-xs text-slate-300">
                        <div>
                            <span className="text-slate-500 text-[10px] uppercase tracking-wider block">Chemical Type</span>
                            <span className="font-bold text-slate-100">
                                {hoveredIncident.details?.type?.replace(/_/g, ' ') || 'Unknown'}
                            </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                            <div>
                                <span className="text-slate-500 text-[10px] uppercase tracking-wider block">Spillage Vol</span>
                                <span className="font-semibold text-slate-200">
                                    {hoveredIncident.details?.amount || 100} kg
                                </span>
                            </div>
                            <div>
                                <span className="text-slate-500 text-[10px] uppercase tracking-wider block">Report Time</span>
                                <span className="font-semibold text-slate-200">
                                    {(() => {
                                        const reportTime = hoveredIncident.details?.timestamp || hoveredIncident.startTime;
                                        return reportTime 
                                            ? new Date(reportTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
                                            : 'N/A';
                                    })()}
                                </span>
                            </div>
                        </div>

                        {hoveredIncident.details?.status === 'Resolved' ? (
                            <div className="grid grid-cols-2 gap-2 border-t border-slate-800/50 pt-1.5 mt-0.5">
                                <div>
                                    <span className="text-emerald-500 text-[10px] uppercase tracking-wider block font-medium">Resolved At</span>
                                    <span className="font-semibold text-emerald-300">
                                        {(() => {
                                            const resTime = hoveredIncident.details?.resolvedAt || hoveredIncident.resolvedAt;
                                            return resTime 
                                                ? new Date(resTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) 
                                                : 'N/A';
                                        })()}
                                    </span>
                                </div>
                                <div>
                                    <span className="text-slate-500 text-[10px] uppercase tracking-wider block">Total Duration</span>
                                    <span className="font-semibold text-slate-200">
                                        {formatDuration(hoveredIncident)}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <div className="border-t border-slate-800/50 pt-1.5 mt-0.5">
                                <span className="text-slate-500 text-[10px] uppercase tracking-wider block">Active Duration</span>
                                <span className="font-semibold text-slate-200">
                                    {formatDuration(hoveredIncident)}
                                </span>
                            </div>
                        )}
                    </div>

                    {hoveredIncident.details?.status !== 'Resolved' && (
                        <div className="mt-2 pt-2 border-t border-slate-800/80">
                            <button
                                data-gesture-btn={"resolve-" + hoveredIncident.id}
                                onClick={() => handleConfirmSelection("resolve-" + hoveredIncident.id)}
                                className={`relative overflow-hidden w-full px-3 py-2 rounded-xl text-center text-xs font-bold tracking-wide transition-all duration-200 border cursor-pointer flex items-center justify-center gap-1.5 ${
                                    hoveredBtnId === ("resolve-" + hoveredIncident.id)
                                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 scale-[1.02] shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                                        : 'bg-emerald-600 text-white border-emerald-500 hover:bg-emerald-500 shadow-md hover:shadow-emerald-500/20'
                                }`}
                            >
                                <CheckCircle2 className="w-4 h-4" />
                                <span>Resolve Hazard</span>
                                {hoveredBtnId === ("resolve-" + hoveredIncident.id) && (
                                    <div 
                                        className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-75"
                                        style={{ width: `${hoverProgress}%` }}
                                    />
                                )}
                            </button>
                        </div>
                    )}
                </div>
            )}

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

            {/* Selecting Phase - Locked Reticle and Option Menu */}
            {pipelineStep === 'selecting' && currentLockedPos && (
                <>
                    {/* Locked geographical marker */}
                    <div
                        className="absolute pointer-events-none z-[90]"
                        style={{
                            left: `${currentLockedPos.x}px`,
                            top: `${currentLockedPos.y}px`,
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
                            left: `${currentLockedPos.x}px`,
                            top: `${currentLockedPos.y}px`,
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
                                onClick={() => handleConfirmSelection(btn.id)}
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

            {/* Websocket Connection UI */}
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
