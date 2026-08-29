import React, { useMemo } from 'react';
import { Source, Layer } from 'react-map-gl/maplibre';
import { PlumePhysics, CHEMICAL_Q_RATES } from '../simulation/PlumePhysics';

export const PlumeLayer = ({ incidents, wind, isNightMode, selectedIncidentId }) => {

    const geoJSON = useMemo(() => {
        const features = [];

        incidents.forEach(inc => {
            const stability = wind.stabilityClass || (isNightMode ? 'F' : 'D'); 
            const baseQ = CHEMICAL_Q_RATES[inc.details?.type] || CHEMICAL_Q_RATES.CHLORINE_GAS; 
            const massRatio = Math.max(0.02, (inc.details?.amount || 100) / 200);
            const Q = baseQ * Math.pow(massRatio, 1.25);
            
            const isResolved = inc.details?.status === 'Resolved';
            const elapsed = inc.elapsedSimSec || 0;

            // International Emergency Zones
            const ZONES = [
                { level: 'Cold', limit: 0.2, color: isResolved ? '#059669' : '#22c55e' },   // Green
                { level: 'Warm', limit: 1.0, color: isResolved ? '#047857' : '#eab308' },   // Yellow
                { level: 'Hot',  limit: 3.0, color: isResolved ? '#064e3b' : '#ef4444' }    // Red
            ];

            // Push from largest (Cold) to smallest (Hot) so Hot renders on top
            ZONES.forEach(zone => {
                const polyCoords = PlumePhysics.generatePlumePolygon(
                    inc.position, 
                    wind.speed, 
                    wind.direction, 
                    Q, 
                    stability,
                    elapsed,
                    zone.limit
                );

                if (polyCoords && polyCoords.length > 2) {
                    features.push({
                        type: 'Feature',
                        geometry: {
                            type: 'Polygon',
                            coordinates: [polyCoords]
                        },
                        properties: {
                            color: zone.color,
                            id: inc.id,
                            isResolved: isResolved,
                            zoneLevel: zone.level
                        }
                    });
                }
            });
        });

        return {
            type: 'FeatureCollection',
            features
        };
    }, [incidents, wind, isNightMode]);

    const plumeStyle = {
        id: 'gaussian-plume-layer',
        type: 'fill',
        paint: {
            'fill-color': ['get', 'color'],
            'fill-opacity': [
                'case',
                ['get', 'isResolved'],
                0.2, // Muted opacity for resolved plumes
                selectedIncidentId ? [
                    'case',
                    ['==', ['get', 'id'], selectedIncidentId],
                    0.5, // Highlight specifically 
                    0.1  // Dim others
                ] : [
                    'match',
                    ['get', 'zoneLevel'],
                    'Hot', 0.6,
                    'Warm', 0.4,
                    'Cold', 0.2,
                    0.4
                ] // Staggered opacity for visual depth
            ]
        }
    };
    
    // Smooth outline contour
    const plumeOutlineStyle = {
        id: 'gaussian-plume-outline',
        type: 'line',
        paint: {
            'line-color': [
                'case',
                ['get', 'isResolved'],
                '#10b981', 
                ['get', 'color'] // Match outline to the zone color
            ],
            'line-width': selectedIncidentId ? [
                'case',
                ['==', ['get', 'id'], selectedIncidentId],
                2,
                0.5
            ] : [
                'match',
                ['get', 'zoneLevel'],
                'Hot', 2,
                'Warm', 1.5,
                'Cold', 1,
                1
            ],
            'line-opacity': [
                'case',
                ['get', 'isResolved'],
                0.4,
                0.8
            ]
        }
    };

    return (
        <Source id="gaussian-source" type="geojson" data={geoJSON}>
            <Layer {...plumeStyle} />
            <Layer {...plumeOutlineStyle} />
        </Source>
    );
};
