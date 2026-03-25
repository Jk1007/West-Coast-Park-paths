export const HAZARD_DATABASE = {
    'CHLORINE_GAS': {
        name: 'Chlorine Gas (Cl2)',
        category: 'Chemical',
        spreadRate: 2.8, // meters per frame-tick
        description: 'Heavier-than-air toxic gas. Spreads rapidly along ground level.'
    },
    'AMMONIA': {
        name: 'Anhydrous Ammonia (NH3)',
        category: 'Chemical',
        spreadRate: 3.5,
        description: 'Highly pungent, toxic gas. Extremely rapid atmospheric dispersion.'
    },
    'SULPHURIC_ACID': {
        name: 'Sulphuric Acid Vapour',
        category: 'Chemical',
        spreadRate: 1.2,
        description: 'Corrosive mist. Slower spread but highly critical.'
    },
    'SARIN_GAS': {
        name: 'Sarin Nerve Agent',
        category: 'Biological',
        spreadRate: 1.8,
        description: 'Lethal weaponized nerve gas.'
    },
    'METHANE_LEAK': {
        name: 'Methane Pipeline Leak',
        category: 'Gas Leak',
        spreadRate: 4.5,
        description: 'Highly flammable natural gas. Expands at extreme speeds.'
    },
    'PROPANE_TANK': {
        name: 'Propane Tank Rupture',
        category: 'Gas Leak',
        spreadRate: 3.0,
        description: 'Heavy flammable vapor. Moderate to fast spread.'
    },
    'INDUSTRIAL_FIRE': {
        name: 'Industrial Toxic Smoke / Fire',
        category: 'Fire/Smoke',
        spreadRate: 2.0,
        description: 'Thick black smoke carrying particulate matter.'
    },
    'RADIATION_LEAK': {
        name: 'Uncontained Radiation Isotope',
        category: 'Radiation',
        spreadRate: 0.5,
        description: 'Airborne radioactive particulate. Slower, creeping radius.'
    }
};
