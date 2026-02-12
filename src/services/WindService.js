
// Station Configuration
// Weightage determines influence on the final result for West Coast Park.
// S50 (Clementi Road): Closest (~4km), Highest Weight.
// S60 (Sentosa): Medium Distance (~8km), Medium Weight.
// S117 (Banyan Road): Furthest (~10km), Low Weight.
const STATION_CONFIG = {
    'S50': { name: 'Clementi Road', weight: 3 },
    'S60': { name: 'Sentosa', weight: 2 },
    'S117': { name: 'Banyan Road', weight: 1 }
};

export const fetchWindData = async () => {
    try {
        const [speedRes, dirRes] = await Promise.all([
            fetch('https://api-open.data.gov.sg/v2/real-time/api/wind-speed'),
            fetch('https://api-open.data.gov.sg/v2/real-time/api/wind-direction')
        ]);

        if (!speedRes.ok || !dirRes.ok) {
            console.warn(`Wind API Error: Speed ${speedRes.status} / Dir ${dirRes.status}`);
            return null;
        }

        const speedJson = await speedRes.json();
        const dirJson = await dirRes.json();

        let weightedSpeedSum = 0;
        let totalSpeedWeight = 0;

        let sumSin = 0;
        let sumCos = 0;
        let totalDirWeight = 0;

        const debugInfo = [];

        // 1. Process Speed
        if (speedJson.data?.readings?.[0]?.data) {
            const data = speedJson.data.readings[0].data;
            Object.entries(STATION_CONFIG).forEach(([id, config]) => {
                const reading = data.find(r => r.stationId === id);
                if (reading) {
                    const val = reading.value; // knots
                    weightedSpeedSum += val * config.weight;
                    totalSpeedWeight += config.weight;
                    debugInfo.push(`${config.name} (Speed: ${val} knots, W: ${config.weight})`);
                }
            });
        }

        // 2. Process Direction
        if (dirJson.data?.readings?.[0]?.data) {
            const data = dirJson.data.readings[0].data;
            Object.entries(STATION_CONFIG).forEach(([id, config]) => {
                const reading = data.find(r => r.stationId === id);
                if (reading) {
                    const val = reading.value; // degrees
                    const rad = val * (Math.PI / 180);
                    sumSin += Math.sin(rad) * config.weight;
                    sumCos += Math.cos(rad) * config.weight;
                    totalDirWeight += config.weight;
                    // Note: Just logging direction for debug
                    // debugInfo.push(`${config.name} (Dir: ${val}°, W: ${config.weight})`);
                }
            });
        }

        if (totalSpeedWeight === 0 || totalDirWeight === 0) {
            console.warn("WindService: No matching stations found.");
            return null;
        }

        // Calculate Averages
        const avgKnots = weightedSpeedSum / totalSpeedWeight;
        const avgKmh = avgKnots * 1.852;

        let avgDeg = Math.atan2(sumSin, sumCos) * (180 / Math.PI);
        if (avgDeg < 0) avgDeg += 360;

        console.log(`[WindService] Result: ${avgKnots.toFixed(1)} knots, ${avgDeg.toFixed(0)}°`);
        console.log(`[WindService] Sources:`, debugInfo);

        return {
            speed: avgKmh,
            direction: avgDeg
        };

    } catch (error) {
        console.error("Wind API Network Error:", error);
        return null;
    }
};
