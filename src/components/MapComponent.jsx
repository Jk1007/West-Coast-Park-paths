import React, { useRef, useState, useCallback, useMemo } from 'react';
import Map, { Source, Layer, NavigationControl, FullscreenControl } from 'react-map-gl/maplibre';
import { PARK_CENTER, PARK_BOUNDS } from '../data/ParkData';
import ParkGraph from '../data/ParkGraph.json';
import maplibregl from 'maplibre-gl';

const MapComponent = ({
    agents = [],
    incidents = [],
    safeNodes = [],
    onAddIncident,
    theme = 'light',
    mode = 'simulation', // 'simulation', 'live', or 'view'
    selectedLocation = null,
    onLocationSelect = null,
    onIncidentClick = null
}) => {
    const mapRef = useRef(null);
    const [viewState, setViewState] = useState({
        longitude: PARK_CENTER[0],
        latitude: PARK_CENTER[1],
        zoom: 16,
        pitch: 45,
        bearing: -17.6
    });

    const onMove = useCallback(evt => setViewState(evt.viewState), []);

    // Right Click Handler (Simulation Mode Quick-Add)
    const handleContextMenu = useCallback((event) => {
        if (mode === 'simulation' && onAddIncident) {
            const { lng, lat } = event.lngLat;
            onAddIncident([lng, lat]);
        }
        event.originalEvent.preventDefault();
    }, [mode, onAddIncident]);

    // Left Click Handler (Live Mode Selection & View Mode Details)
    const handleClick = useCallback((event) => {
        if (mode === 'live' && onLocationSelect) {
            const { lng, lat } = event.lngLat;
            onLocationSelect([lng, lat]);
        } else if (mode === 'view' && onIncidentClick) {
            const feature = event.features && event.features.find(f => f.layer.id === 'incidents-layer');
            if (feature) {
                const details = JSON.parse(feature.properties.details || '{}');
                onIncidentClick({
                    id: feature.properties.id,
                    position: feature.geometry.coordinates,
                    details: details
                });
            } else {
                // Clicked on empty map, close panel
                onIncidentClick(null);
            }
        }
    }, [mode, onLocationSelect, onIncidentClick]);

    // Select Style based on Theme
    const mapStyleUrl = theme === 'light'
        ? "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json" // Light/Vibrant
        : "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"; // Dark

    // Adjust Path Color based on Theme (Dark line for Light map, Light line for Dark map)
    const pathColor = theme === 'light' ? '#374151' : '#9ca3af';

    // --- Data Transformation ---

    // 1. Full Graph Network (The actual paths agents walk)
    const graphGeoJSON = useMemo(() => {
        const nodeMap = {};
        ParkGraph.nodes.forEach(n => {
            nodeMap[n.id] = [n.lon, n.lat];
        });

        const features = [];
        ParkGraph.edges.forEach((edge, index) => {
            const start = nodeMap[edge.source];
            const end = nodeMap[edge.target];
            if (start && end) {
                features.push({
                    type: 'Feature',
                    id: index,
                    geometry: {
                        type: 'LineString',
                        coordinates: [start, end]
                    },
                    properties: {}
                });
            }
        });
        return { type: 'FeatureCollection', features };
    }, []);

    // 2. Agents
    const agentsGeoJSON = {
        type: 'FeatureCollection',
        features: agents.map(agent => ({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: agent.position },
            properties: { state: agent.state }
        }))
    };

    // 3. Incidents
    const incidentsGeoJSON = {
        type: 'FeatureCollection',
        features: incidents.map(inc => ({
            type: 'Feature',
            geometry: { type: 'Point', coordinates: inc.position },
            properties: {
                radius: inc.radius,
                color: inc.color || '#ef4444',
                id: inc.id,
                details: inc.details ? JSON.stringify(inc.details) : '{}'
            }
        }))
    };

    // 4. Selected Location (Live Mode)
    const selectionGeoJSON = useMemo(() => {
        if (!selectedLocation) return null;
        return {
            type: 'FeatureCollection',
            features: [{
                type: 'Feature',
                geometry: { type: 'Point', coordinates: selectedLocation },
                properties: {}
            }]
        };
    }, [selectedLocation]);

    // 5. Safe Nodes (Debug Visualization)
    const safeNodesGeoJSON = useMemo(() => {
        // console.log('[DEBUG-Map] safeNodes prop:', safeNodes);
        if (!safeNodes || safeNodes.length === 0) return null;

        const nodeMap = {};
        ParkGraph.nodes.forEach(n => {
            nodeMap[n.id] = [n.lon, n.lat];
        });

        const features = safeNodes.map(id => {
            const coords = nodeMap[id];
            if (!coords) console.warn('[DEBUG-Map] Missing coords for node:', id);
            return {
                type: 'Feature',
                geometry: { type: 'Point', coordinates: coords },
                properties: {}
            };
        }).filter(f => f.geometry.coordinates);

        console.log(`[DEBUG-Map] Generated ${features.length} safe node features.`);
        return {
            type: 'FeatureCollection',
            features
        };
    }, [safeNodes]);


    // --- Layer Styles ---

    const pathsLayerStyle = {
        id: 'paths-layer',
        type: 'line',
        paint: {
            'line-color': pathColor,
            'line-width': 2,
            'line-opacity': 0.5
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
                'IDLE', '#16a34a',      // Green-600
                'EVACUATING', '#ea580c', // Orange-600
                'ESCAPED', '#2563eb',    // Blue-600
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
            'circle-color': '#06b6d4', // Cyan
            'circle-stroke-width': 2,
            'circle-stroke-color': '#ffffff',
            'circle-opacity': 0.8
        }
    };

    // Style for the currently active pinned location dropping from user click
    const selectionLayerStyle = {
        id: 'selection-layer',
        type: 'circle',
        paint: {
            'circle-radius': 10,
            'circle-color': '#3b82f6', // Blue 500
            'circle-stroke-width': 3,
            'circle-stroke-color': '#ffffff',
            'circle-opacity': 0.9
        }
    };

    const incidentsLayerStyle = {
        id: 'incidents-layer',
        type: 'circle',
        paint: {
            'circle-color': ['get', 'color'],
            'circle-opacity': 0.4,
            'circle-radius': [
                'interpolate', ['linear'], ['zoom'],
                10, ['/', ['get', 'radius'], 20],
                15, ['/', ['get', 'radius'], 2],
                20, ['*', ['get', 'radius'], 2]
            ]
        }
    };

    return (
        <Map
            {...viewState}
            onMove={onMove}
            ref={mapRef}
            style={{ width: '100%', height: '100%' }}
            mapStyle={mapStyleUrl}
            onContextMenu={handleContextMenu}
            onClick={handleClick}
            cursor={mode === 'live' ? "pointer" : "crosshair"}
            interactiveLayerIds={mode === 'view' ? ['incidents-layer'] : undefined}
        >
            <NavigationControl position="top-right" />
            <FullscreenControl position="top-right" />



            {/* Paths Layer (Full Graph) */}
            <Source id="paths-source" type="geojson" data={graphGeoJSON}>
                <Layer {...pathsLayerStyle} />
            </Source>

            {/* Incidents Layer */}
            <Source id="incidents-source" type="geojson" data={incidentsGeoJSON}>
                <Layer {...incidentsLayerStyle} />
            </Source>

            {/* Agents Layer */}
            <Source id="agents-source" type="geojson" data={agentsGeoJSON}>
                <Layer {...agentsLayerStyle} />
            </Source>

            {/* Selection Pin (Live Mode) */}
            {selectionGeoJSON && (
                <Source id="selection-source" type="geojson" data={selectionGeoJSON}>
                    <Layer {...selectionLayerStyle} />
                </Source>
            )}

            {/* Safe Nodes (Debug) - Render LAST to be ON TOP */}
            {safeNodesGeoJSON && (
                <Source id="safe-nodes-source" type="geojson" data={safeNodesGeoJSON}>
                    <Layer {...safeNodesLayerStyle} />
                </Source>
            )}
        </Map>
    );
};

export default MapComponent;
