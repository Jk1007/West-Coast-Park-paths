import { HAZARD_DATABASE } from '../constants/HazardDatabase';

/**
 * ChemicalDispersionEngine
 * Implements physics logic based on ERG 2024 Standards.
 * R = k * sqrt(t)
 */
export class ChemicalDispersionEngine {
    constructor() {
        // ERG 2024 mapped metrics (k-value derived for target PAD over 1 hour, ~3600s)
        this.ERG_DATABASE = {
            'AMMONIA': {
                name: 'Ammonia (Anhydrous)',
                isolationRadius: 30, // meters
                padDay: 1100, // meters
                padNight: 2700, // meters
                k_val: 18.33 // Precomputed to reach 1100m at t=3600
            },
            'CHLORINE_GAS': {
                name: 'Chlorine',
                isolationRadius: 30,
                padDay: 1600,
                padNight: 5100,
                k_val: 26.67 // Precomputed to reach 1600m at t=3600
            },
            'DEFAULT_LIQUID': {
                name: 'Generic Liquid / High Viscosity Acid',
                isolationRadius: 30,
                padDay: 150, // Static pool, nominal PAD
                padNight: 150, 
                k_val: 2.5 // Very slow creep
            }
        };
    }

    /**
     * Calculates the explicit radius and opacity for a given chemical incident over time.
     * @param {Object} incident 
     * @param {Boolean} isNightMode 
     * @param {Object} wind
     * @returns {Object} { radius: Number, opacity: Number, padLimit: Number }
     */
    computeState(incident, isNightMode = false, wind = {speed: 0, direction: 0}) {
        // We track incident lifetime natively.
        // Assuming incident.startTime is absolute, and we compute elapsed visually
        const tElapsedSec = incident.elapsedSimSec || 0; 
        
        let chemKey = incident.details?.type;
        if (!this.ERG_DATABASE[chemKey]) {
            // Map legacy defaults
            chemKey = chemKey === 'CHLORINE_GAS' || chemKey === 'AMMONIA' ? chemKey : 'DEFAULT_LIQUID';
        }
        
        const chemData = this.ERG_DATABASE[chemKey] || this.ERG_DATABASE['DEFAULT_LIQUID'];
        
        // Base PAD (Protective Action Distance Limit) for ~200kg Large Spill
        const basePadLimit = isNightMode ? chemData.padNight : chemData.padDay;
        
        // Optional volumetric scaling: True mass limits dispersion boundary
        const amount = incident.details?.amount || 100;
        const volumeOffset = Math.max(0, Math.sqrt(amount) * 5); 
        
        // Dynamically scale PAD by mass (Sqrt ratio against standard 200kg large spill)
        const massRatio = Math.sqrt(amount / 200);
        // Floor the absolute PAD at the immediate Isolation Radius to prevent disappearing plumes
        const dynamicPadLimit = Math.max(chemData.isolationRadius + volumeOffset, basePadLimit * massRatio);
        
        // Dynamic Wind Acceleration directly drives Initial Plume Speed limit breaking
        const windBoost = 1 + (wind.speed / 20);

        // Physics Core: R = k * sqrt(t)
        // Adding initial isolation/volumetric burst + the expanding root
        let rawRadius = (chemData.isolationRadius + volumeOffset) + (chemData.k_val * windBoost * Math.sqrt(tElapsedSec));

        // Plume realistically stops expanding dynamically when atmospheric dispersion balances
        let cappedRadius = Math.min(rawRadius, dynamicPadLimit);
        
        // Calculate thinning Opacity: inverse proportion to dispersion bounds
        // Starts opaque (0.8) and thins out (0.3) as it hits PAD
        let opacity = 0.8 - (0.5 * (cappedRadius / dynamicPadLimit));
        // Clamp bounds just to be safe
        opacity = Math.max(0.3, Math.min(0.8, opacity));

        return {
            radius: cappedRadius,
            opacity: opacity,
            padLimit: dynamicPadLimit
        };
    }
}
