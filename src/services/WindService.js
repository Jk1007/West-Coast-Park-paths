export const fetchWindData = async () => {
    try {
        // Parallel Fetch: Speed and Direction
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

        // Target Station IDs & Weights
        // S50: Clementi Road (Highest Weight: 3)
        // S60: Sentosa (Medium Weight: 2)
        // S117: Banyan Road (Low Weight: 1)
        const targets = {
            'S50': 3,
            'S60': 2,
            'S117': 1
        };

        // --- Calculate Weighted Speed ---
        let weightedSpeedSum = 0;
        let totalSpeedWeight = 0;
        let foundStations = [];

        if (speedJson.data && speedJson.data.readings && speedJson.data.readings[0]) {
            speedJson.data.readings[0].data.forEach(r => {
                if (targets[r.stationId]) {
                    const weight = targets[r.stationId];
                    weightedSpeedSum += r.value * weight;
                    totalSpeedWeight += weight;
                    foundStations.push(r.stationId);
                }
            });
        }

        // --- Calculate Weighted Direction (Vector Averaging) ---
        let sumSin = 0;
        let sumCos = 0;
        let totalDirWeight = 0;

        if (dirJson.data && dirJson.data.readings && dirJson.data.readings[0]) {
            dirJson.data.readings[0].data.forEach(r => {
                if (targets[r.stationId]) {
                    const weight = targets[r.stationId];
                    const rad = r.value * (Math.PI / 180);
                    sumSin += Math.sin(rad) * weight;
                    sumCos += Math.cos(rad) * weight;
                    totalDirWeight += weight;
                }
            });
        }

        if (totalSpeedWeight === 0) return null;

        // Speed Result
        const avgKnots = weightedSpeedSum / totalSpeedWeight;
        const avgKmh = avgKnots * 1.852;

        // Direction Result
        let avgDeg = 0;
        if (totalDirWeight > 0) {
            avgDeg = Math.atan2(sumSin, sumCos) * (180 / Math.PI);
            if (avgDeg < 0) avgDeg += 360;
        }

        console.log(`Wind: ${avgKnots.toFixed(1)} knots, ${avgDeg.toFixed(0)}° (${foundStations.join(', ')})`);

        return {
            speed: avgKmh,
            direction: avgDeg
        };

    } catch (error) {
        console.error("Wind API Network Error:", error);
        return null;
    }
};
