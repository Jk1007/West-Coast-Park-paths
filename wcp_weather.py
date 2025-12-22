# wcp_weather.py
# Handles ALL NEA weather logic (Now / Next 2h / Next 24h)

import math
import requests
from datetime import datetime
import solara as sl
import asyncio


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

def _reading_for_station(readings, station_id):
    for r in readings:
        sid = r.get("station_id")
        if sid is None:
            sid = r.get("stationId")
        if sid == station_id:
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

    def _fetch_data(endpoint):
        url = base + endpoint
        r = requests.get(url, timeout=6)

        # show status codes if not 200
        if r.status_code != 200:
            last_error.value = "realtime " + endpoint + " status " + str(r.status_code) + " body " + r.text[:120]
            return None

        j = r.json()
        d = j.get("data")
        if d is None:
            # show top-level keys if "data" missing
            last_error.value = "realtime " + endpoint + " no data; keys " + str(list(j.keys()))
            return None

        return d


    # Temperature
    tj = _fetch_data("air-temperature")
    if tj is None:
        t = None 
    else:
        t_stations = tj.get("stations") or []
        t_items = tj.get("readings") or []
        if t_items:
            t_readings = t_items[0].get("data") or []
        else:
            t_readings = []
        sid = _nearest_station_id(t_stations)

        if sid is None and len(t_readings) > 0:
            sid = t_readings[0].get("station_id")

        t = _reading_for_station(t_readings, sid)
        if t is None and len(t_readings) > 0:
            t = t_readings[0].get("value")



    # Humidity
    hj = _fetch_data("relative-humidity")
    if hj is None:
        hum = None 
    else:
        h_stations = hj.get("stations") or []
        h_items = hj.get("readings") or []
        if h_items:
            h_readings = h_items[0].get("data") or []
        else:
            h_readings = []
        sid = _nearest_station_id(h_stations)

        if sid is None and len(h_readings) > 0:
            sid = h_readings[0].get("station_id")

        rh = _reading_for_station(h_readings, sid)
        if rh is None and len(h_readings) > 0:
            rh = h_readings[0].get("value")

    # Wind speed (knots → km/h)
    wj = _fetch_data("wind-speed")
    if wj is None:
        w = None 
    else:
        ws_stations = wj.get("stations") or []
        ws_items = wj.get("readings") or []
        if ws_items:
            ws_readings = ws_items[0].get("data") or []
        else:
            ws_readings = []
        sid = _nearest_station_id(ws_stations)

        if sid is None and len(ws_readings) > 0:
            sid = ws_readings[0].get("station_id")

        ws_knots = _reading_for_station(ws_readings, sid)
        if ws_knots is None and len(ws_readings) > 0:
            ws_knots = ws_readings[0].get("value")


    # Wind direction (FROM)
    dj = _fetch_data("wind-direction")
    if dj is None:
        d = None
    else:
        wd_stations = dj.get("stations") or []
        wd_items = dj.get("readings") or []
        if wd_items:
            wd_readings = wd_items[0].get("data") or []
        else:
            wd_readings = []
        sid = _nearest_station_id(wd_stations)

        if sid is None and len(wd_readings) > 0:
            sid = wd_readings[0].get("station_id")

        wd = _reading_for_station(wd_readings, sid) 
        if wd is None and len(wd_readings) > 0:
            wd = wd_readings[0].get("value")


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
    data = requests.get(url, timeout=6).json().get("data") or {}
    items = data.get("items") or []
    if not items:
        forecast_2h_str.value = "—"
        return

    item = items[0]
    forecasts = item.get("forecasts") or []
    meta = item.get("area_metadata") or []

    # Build area -> (lat, lon)
    area_lat = {}
    area_lon = {}
    for m in meta:
        name = (m.get("name") or "").strip().lower()
        loc = m.get("label_location") or {}
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if name and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            area_lat[name] = float(lat)
            area_lon[name] = float(lon)

    # Pick nearest forecast area to West Coast Park (best match)
    best = None
    best_d = None
    for f in forecasts:
        area = ((f.get("area") or "")).strip().lower()
        if area in area_lat and area in area_lon:
            d = _dist2(area_lat[area], area_lon[area], PARK_LAT, PARK_LON)
            if best_d is None or d < best_d:
                best_d = d
                best = f

    if not best and len(forecasts) > 0:
        best = forecasts[0]

    if not best:
        forecast_2h_str.value = "—"
        return

    # Format period + updated + forecast text
    vp = item.get("valid_period") or {}
    sd = vp.get("start")
    ed = vp.get("end")
    upd = item.get("update_timestamp")
    fc_text = best.get("forecast") or "—"

    parts = []

    if sd and ed:
        sd_dt = datetime.fromisoformat(sd)
        ed_dt = datetime.fromisoformat(ed)

        sd_txt = sd_dt.strftime("%I:%M%p").lower()
        ed_txt = ed_dt.strftime("%I:%M%p").lower()
        if sd_txt.startswith("0"):
            sd_txt = sd_txt[1:]
        if ed_txt.startswith("0"):
            ed_txt = ed_txt[1:]

        parts.append("Forecast for " + sd_txt + " - " + ed_txt)

    if upd:
        upd_dt = datetime.fromisoformat(upd)
        upd_txt = upd_dt.strftime("%I:%M%p").lower()
        if upd_txt.startswith("0"):
            upd_txt = upd_txt[1:]
        parts.append("Updated: " + upd_txt + " today")

    parts.append(fc_text)

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
            except Exception as e:
                last_error.value = "realtime: " + repr(e)
                weather_now_str.value = "—"


            try:
                pull_2hr_forecast()
            except Exception as e:
                last_error.value = "2hr: " + repr(e)
                forecast_2h_str.value = "—"

            try:
                pull_24hr_forecast()
            except Exception as e:
                last_error.value = "24hr: " + repr(e)
                forecast_24h_str.value = "—"

            _last_weather_ts.value = now
            first = False

        await asyncio.sleep(1)

            


