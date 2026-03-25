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
        lastWindDir: -1
    });

    const fistDebounceRef = useRef(0);

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
                        // 1. FIST STOP LOGIC: If fist is closed, we freeze everything
                        if (data.isFist) {
                            stateRef.current.prevIndex = null;
                            stateRef.current.prevDistance = null;
                            if (stateRef.current.handDetected) {
                                stateRef.current.handDetected = true; // Still detected but "stopping"
                            }
                            return; 
                        }

                        if (!stateRef.current.handDetected) {
                            stateRef.current.handDetected = true;
                            if (onHandStatusChange) onHandStatusChange(true);
                        }

                        const map = mapRef.current?.getMap();
                        if (!map) return;

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
                
                // Reconnect after 3 seconds if still enabled
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
    }, [gesturesEnabled, onAddIncident, onWindChange, onHandStatusChange]);

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
        <div className="relative w-full h-full">
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

            {/* Websocket Connection UI */}
            <div className="absolute bottom-6 left-6 flex gap-2 z-50">
                <button
                    onClick={toggleGestures}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full font-medium shadow-lg transition-all border ${
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
