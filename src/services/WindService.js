// WindService.js
// Manages wind telemetry fetching from the Python backend (port 8000).

let globalWindCache = null;
let lastWindFetchTime = 0;

try {
    const saved = localStorage.getItem('crowdshield_wind_cache');
    const savedTime = localStorage.getItem('crowdshield_wind_time');
    if (saved && savedTime) {
        globalWindCache = JSON.parse(saved);
        lastWindFetchTime = parseInt(savedTime, 10);
    }
} catch (e) {
    // Silently ignore localStorage restrictions
}

const CACHE_TTL_MS = 5 * 1000; // 5 second cache (allows 10s React updates)

export const fetchWindData = async (forceRefresh = false) => {
    // 1. Check local memory cache first
    if (!forceRefresh && globalWindCache && (Date.now() - lastWindFetchTime < CACHE_TTL_MS)) {
        return globalWindCache;
    }

    try {
        // 2. Primary Source: Unified Python Backend on Port 8000
        const resp = await fetch('http://localhost:8000/api/wind');
        if (resp.ok) {
            const data = await resp.json();
            const { speed, direction, temp, rain, hum } = data;
            
            // Re-infer Dynamic Pasquill-Gifford Stability Class (A-F)
            let stabilityClass = 'D'; 
            const speedMs = speed * 0.277778;
            const isHot = temp > 31;
            const isRaining = rain > 0.5;

            if (isRaining || speedMs > 6) {
                stabilityClass = 'D';
            } else if (speedMs <= 2.5) {
                if (isHot && hum < 75) stabilityClass = 'A'; 
                else if (temp > 27) stabilityClass = 'B'; 
                else stabilityClass = 'F'; 
            } else if (speedMs <= 5) {
                if (isHot) stabilityClass = 'B'; 
                else if (temp > 27) stabilityClass = 'C'; 
                else stabilityClass = 'D';
            }

            const compiledData = {
                speed,
                direction,
                stabilityClass,
                weather: { temp, rain, hum }
            };

            globalWindCache = compiledData;
            lastWindFetchTime = Date.now();
            
            try {
                localStorage.setItem('crowdshield_wind_cache', JSON.stringify(compiledData));
                localStorage.setItem('crowdshield_wind_time', lastWindFetchTime.toString());
            } catch (e) {}

            return compiledData;
        }
    } catch (err) {
        console.warn("[WindService] Backend fetch failed, falling back to local cache/defaults.");
    }

    // 3. Fallback: Return cached data or environmental defaults
    return globalWindCache || {
        speed: 5.0,
        direction: 210,
        stabilityClass: 'D',
        weather: { temp: 28.0, rain: 0.0, hum: 80 }
    };
};
