# wcp_weather.py
# Handles ALL NEA weather logic (Now / Next 2h / Next 24h)

import math
import requests
from datetime import datetime
import solara as sl

# -----------------------------
# Park reference (West Coast Park)
# -----------------------------
PARK_LAT = 1.298466
PARK_LON = 103.762181

# -----------------------------
# Reactives exposed to UI/core
# -----------------------------
weather_now_str = sl.reactive("—")
forecast_2h_str = sl.reactive("—")
forecast_24h_str = sl.reactive("—")
wind_speed_mps = sl.reactive(None)
wind_to_deg_math = sl.reactive(None)


# Raw values (if core wants them)
temperature_c = sl.reactive(None)
relative_humidity_pct = sl.reactive(None)
nea_wind_kmh = sl.reactive(None)
nea_wind_from_deg = sl.reactive(None)

_last_weather_ts = sl.reactive(0.0)
last_error = sl.reactive("")

# -----------------------------
# Helpers
# -----------------------------
def _dist2(lat1, lon1, lat2, lon2):
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    return dlat * dlat + dlon * dlon

def _nearest_station_id(stations):
    best = None
    best_d = None
    for s in stations:
        loc = s.get("location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            d = _dist2(lat, lon, PARK_LAT, PARK_LON)
            if best_d is None or d < best_d:
                best_d = d
                best = s.get("id")
    return best

def _reading_for_station(readings, sid):
    for r in readings:
        if r.get("station_id") == sid:
            return r.get("value")
    return None

def _deg_to_compass(deg):
    dirs = [
        "North", "NNE", "NE", "ENE",
        "East", "ESE", "SE", "SSE",
        "South", "SSW", "SW", "WSW",
        "West", "WNW", "NW", "NNW"
    ]
    return dirs[int((deg + 11.25) / 22.5) % 16]

# -----------------------------
# Real-time weather (Now)
# -----------------------------
def pull_realtime_weather():
    base = "https://api-open.data.gov.sg/v2/real-time/api/"

    # Temperature
    tj = requests.get(base + "air-temperature", timeout=4).json().get("data") or {}
    sid = _nearest_station_id(tj.get("stations") or [])
    t = _reading_for_station(tj.get("readings") or [], sid)

    # Humidity
    hj = requests.get(base + "relative-humidity", timeout=4).json().get("data") or {}
    sid = _nearest_station_id(hj.get("stations") or [])
    rh = _reading_for_station(hj.get("readings") or [], sid)

    # Wind speed (knots → km/h)
    wj = requests.get(base + "wind-speed", timeout=4).json().get("data") or {}
    sid = _nearest_station_id(wj.get("stations") or [])
    ws_knots = _reading_for_station(wj.get("readings") or [], sid)

    # Wind direction (FROM)
    dj = requests.get(base + "wind-direction", timeout=4).json().get("data") or {}
    sid = _nearest_station_id(dj.get("stations") or [])
    wd = _reading_for_station(dj.get("readings") or [], sid)

    # Store raw values
    temperature_c.value = t
    relative_humidity_pct.value = rh

    if isinstance(ws_knots, (int, float)):
        nea_wind_kmh.value = ws_knots * 1.852
    else:
        nea_wind_kmh.value = None

    nea_wind_from_deg.value = wd

        # speed: km/h -> m/s
    if isinstance(nea_wind_kmh.value, (int, float)):
        wind_speed_mps.value = float(nea_wind_kmh.value) / 3.6
    else:
        wind_speed_mps.value = None

    # direction:
    # NEA/myENV direction is "FROM degrees" (meteorological, 0=N, clockwise)
    # Your sim drift uses a math-style "TO degrees" (what you already had working).
    if isinstance(nea_wind_from_deg.value, (int, float)):
        wd_from = float(nea_wind_from_deg.value) % 360.0
        wd_to = (wd_from + 180.0) % 360.0          # convert FROM -> TO
        wind_to_deg_math.value = (90.0 - wd_to) % 360.0  # TO-bearings -> math degrees
    else:
        wind_to_deg_math.value = None


    # Build "Now" string
    parts = []
    if isinstance(t, (int, float)):
        parts.append("Temp " + str(round(t, 1)) + "°C")
    if isinstance(rh, (int, float)):
        parts.append("RH " + str(int(rh)) + "%")
    if isinstance(nea_wind_kmh.value, (int, float)):
        parts.append("Wind " + str(round(nea_wind_kmh.value, 1)) + " km/h")
    if isinstance(wd, (int, float)):
        parts.append("From " + _deg_to_compass(wd) + " (" + str(int(wd)) + "°)")

    weather_now_str.value = " | ".join(parts) if parts else "—"

# -----------------------------
# Next 2 hours
# -----------------------------
def pull_2hr_forecast():
    url = "https://api-open.data.gov.sg/v2/real-time/api/two-hr-forecast"
    j = requests.get(url, timeout=4).json().get("data") or {}
    items = j.get("items") or []
    if not items:
        forecast_2h_str.value = "—"
        return

    item = items[0]
    fc = None
    for f in item.get("forecasts") or []:
        if (f.get("area") or "").lower() == "west":
            fc = f
            break

    if not fc:
        forecast_2h_str.value = "—"
        return

    sd = item.get("start_period")
    ed = item.get("end_period")
    upd = item.get("update_timestamp")

    parts = []
    if sd and ed:
        parts.append("Forecast for " + sd[-8:-3] + " - " + ed[-8:-3])
    if upd:
        parts.append("Updated: " + upd[-8:-3] + " today")
    if fc.get("forecast"):
        parts.append(fc.get("forecast"))

    forecast_2h_str.value = " | ".join(parts)

# -----------------------------
# Next 24 hours (v1 endpoint)
# -----------------------------
def pull_24hr_forecast():
    url = "https://api.data.gov.sg/v1/environment/24-hour-weather-forecast"
    j = requests.get(url, timeout=6).json()
    items = j.get("items") or []
    if not items:
        forecast_24h_str.value = "—"
        return

    g = items[0].get("general") or {}
    t = g.get("temperature") or {}
    rh = g.get("relative_humidity") or {}
    w = g.get("wind") or {}
    ws = w.get("speed") or {}

    parts = []
    if g.get("forecast"):
        parts.append(g.get("forecast"))
    if t.get("low") is not None:
        parts.append("Temp " + str(t.get("low")) + "–" + str(t.get("high")) + "°C")
    if rh.get("low") is not None:
        parts.append("Humidity " + str(rh.get("low")) + "–" + str(rh.get("high")) + "%")
    if ws.get("low") is not None:
        s = "Wind " + str(ws.get("low")) + "–" + str(ws.get("high")) + " km/h"
        if w.get("direction"):
            s += " • " + w.get("direction")
        parts.append(s)

    forecast_24h_str.value = " | ".join(parts)

# -----------------------------
# Background loop (called by NiceGUI)
# -----------------------------
async def weather_loop():
    first = True
    while True:
        now = datetime.now().timestamp()
        if first or (now - _last_weather_ts.value) > 300:
            try:
                pull_realtime_weather()
                pull_2hr_forecast()
                pull_24hr_forecast()
                _last_weather_ts.value = now
            except Exception as e:
                last_error.value = str(e)
        first = False
        await sl.sleep(1)
