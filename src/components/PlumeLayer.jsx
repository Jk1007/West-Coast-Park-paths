import React, { useMemo } from 'react';
import { Source, Layer } from 'react-map-gl/maplibre';
import { PlumePhysics, CHEMICAL_Q_RATES } from '../simulation/PlumePhysics';

export const PlumeLayer = ({ incidents, wind, isNightMode, selectedIncidentId }) => {

    const geoJSON = useMemo(() => {
        const features = [];

        incidents.forEach(inc => {
            const stability = isNightMode ? 'F' : 'D'; // Night = Highly Stable
            
            // Base emission rate. We parse the specific chemical's Q, or default to Chlorine.
            // If the user specified an `amount` in kg, we can scale Q proportionally.
            const baseQ = CHEMICAL_Q_RATES[inc.details?.type] || CHEMICAL_Q_RATES.CHLORINE_GAS; 
            const massRatio = Math.max(0.02, (inc.details?.amount || 100) / 200);
            const Q = baseQ * Math.pow(massRatio, 1.25); // Exaggerates visual thickness for smaller spills
            
            // Core physical decay color interpolator (Yellow -> Orange -> Red) mapped to 5 environmental minutes (300s)
            // This aligns visual 'fully grown' with mechanical plume bounding expansion.
            const lifeRatio = Math.min(1, Math.max(0, (inc.elapsedSimSec || 0) / 300));
            
            let r, g, b;
            if (lifeRatio < 0.5) {
                // Dark Amber (#b45309) -> Dark Orange (#c2410c)
                const t = lifeRatio * 2; // scale 0-0.5 to 0-1
                r = Math.round(180 + (194 - 180) * t); // 180 -> 194
                g = Math.round(83 + (65 - 83) * t);    // 83 -> 65
                b = Math.round(9 + (12 - 9) * t);      // 9 -> 12
            } else {
                // Dark Orange (#c2410c) -> Dark Red (#b91c1c)
                const t = (lifeRatio - 0.5) * 2; // scale 0.5-1 to 0-1
                r = Math.round(194 + (185 - 194) * t); // 194 -> 185
                g = Math.round(65 + (28 - 65) * t);    // 65 -> 28
                b = Math.round(12 + (28 - 12) * t);    // 12 -> 28
            }
            const dynamicColor = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
            
            // Map the Gaussian IDLH bounding shape mathematically into geographic space
            const polyCoords = PlumePhysics.generatePlumePolygon(
                inc.position, 
                wind.speed, 
                wind.direction, 
                Q, 
                stability,
                inc.elapsedSimSec || 0
            );

            if (polyCoords && polyCoords.length > 2) {
                features.push({
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [polyCoords]
                    },
                    properties: {
                        color: dynamicColor,
                        id: inc.id
                    }
                });
            }
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
            'fill-opacity': selectedIncidentId ? [
                'case',
                ['==', ['get', 'id'], selectedIncidentId],
                0.8, // Highlight specifically 
                0.3  // Dim others
            ] : 0.7, // Standard opacity
        }
    };
    
    // Smooth outline contour
    const plumeOutlineStyle = {
        id: 'gaussian-plume-outline',
        type: 'line',
        paint: {
            'line-color': '#ffffff',
            'line-width': selectedIncidentId ? [
                'case',
                ['==', ['get', 'id'], selectedIncidentId],
                3,
                1
            ] : 3,
            'line-opacity': 0.9
        }
    };

    return (
        <Source id="gaussian-source" type="geojson" data={geoJSON}>
            <Layer {...plumeStyle} />
            <Layer {...plumeOutlineStyle} />
        </Source>
    );
};
