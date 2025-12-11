# wcp_solara_test.py — West Coast Park evac with SCDF responder deployment optimiser
# Run:  python -m solara run wcp_solara_test.py

import plotly.graph_objects as go
import asyncio
import math
import numpy as np
import pandas as pd
import solara as sl
import osmnx as ox
import networkx as nx
from scipy.spatial import KDTree
from datetime import datetime
import json
from urllib import request, parse
import re
import os

# -------------------- LOAD WEST COAST PARK WALK GRAPH --------------------
GRAPH_FILE = "west_coast_park_walk_clean.graphml"  # adjust if your file name differs
G = ox.load_graphml(GRAPH_FILE)

UG = nx.Graph()
for u, v, d in G.to_undirected(as_view=False).edges(data=True):
    w = float(d.get("length", 1.0))
    if UG.has_edge(u, v):
        if w < UG[u][v]["weight"]:
            UG[u][v]["weight"] = w
    else:
        UG.add_edge(u, v, weight=w)

nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)[["x", "y"]]
EDGES_GDF = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]]

NODE_POS = nodes_gdf[["x", "y"]].to_numpy()
NODE_IDS = nodes_gdf.index.to_list()
POS_DICT = {nid: (float(nodes_gdf.loc[nid, "x"]), float(nodes_gdf.loc[nid, "y"])) for nid in NODE_IDS}
KD = KDTree(NODE_POS)

X_MIN, Y_MIN = NODE_POS.min(axis=0)
X_MAX, Y_MAX = NODE_POS.max(axis=0)
IS_GEOGRAPHIC = (abs(X_MAX - X_MIN) < 10.0) and (abs(Y_MAX - Y_MIN) < 10.0)
AVG_LAT_RAD = math.radians(float((Y_MIN + Y_MAX) / 2.0))

def meters_to_dx(m): return float(m) / (111_320.0 * math.cos(AVG_LAT_RAD)) if IS_GEOGRAPHIC else float(m)
def meters_to_dy(m): return float(m) / 110_540.0 if IS_GEOGRAPHIC else float(m)
def dx_map_to_m(dx): return float(dx) * 111_320.0 * math.cos(AVG_LAT_RAD) if IS_GEOGRAPHIC else float(dx)
def dy_map_to_m(dy): return float(dy) * 110_540.0 if IS_GEOGRAPHIC else float(dy)

def map_distance_m(a, b):
    ax, ay = float(a[0]), float(a[1]); bx, by = float(b[0]), float(b[1])
    if IS_GEOGRAPHIC:
        dx_m = dx_map_to_m(bx - ax); dy_m = dy_map_to_m(by - ay)
        return math.hypot(dx_m, dy_m)
    return math.hypot(bx - ax, by - ay)

# -------------------- REACTIVE STATE --------------------
tick = sl.reactive(0)
running = sl.reactive(False)
last_error = sl.reactive("")
ui_message = sl.reactive("")
download_link_md = sl.reactive("")

num_people = sl.reactive(200)
pct_cyclists = sl.reactive(15)

hazard_radius = sl.reactive(40.0)     # m (initial radius of a new hazard)
hazard_spread = sl.reactive(1.2)      # m/step (actual circle growth)

expected_growth_m = sl.reactive(3.0)  # m/step
expected_buffer_m = sl.reactive(0.0)  # m

wind_deg = sl.reactive(45.0)          # 0=E, 90=N
wind_speed = sl.reactive(1.0)         # m/s
base_awareness_lag = sl.reactive(8.0)

# plume/risk
plume_show = sl.reactive(True)
plume_Q_gs = sl.reactive(100.0)       # g/s
plume_H_m = sl.reactive(20.0)         # m
plume_stab = sl.reactive("D")         # A..F
plume_range_m = sl.reactive(1500.0)   # m
plume_grid = sl.reactive(90)          # rows

qc_show_paths = sl.reactive(False)
qc_km_margin = sl.reactive(25.0)

risk_show = sl.reactive(True)
risk_grid = sl.reactive(90)
risk_speed_mps = sl.reactive(1.4)

qc_summary = sl.reactive("")
qc_issues  = sl.reactive("")
qc_rows_cache = []

# ---- Live time/date/weather (auto-updating) ----
time_str = sl.reactive("")
date_str = sl.reactive("")
weather_str = sl.reactive("Weather: —")
_last_weather_ts = sl.reactive(0.0)
_runtime_loop_started = sl.reactive(False)

# --- Evacuation timing metrics ---
EVAC_NOTIFICATION_TICK = sl.reactive(None)  # record when hazard notified

# Per-run state
PEOPLE = []
HAZARDS = []   # {"id": int, "pos": np.array([x,y]), "r_m": float}
HAZARD_ID = 0
SAFE_NODES = []
N_SAFE = 12
MANUAL_SAFE = set()
FEATURED_SAFE = set()

# --- SCDF responder deployment state ---
RESPONDERS = set()
N_RESPONDERS = sl.reactive(5)

rng = np.random.default_rng(42)

# -------------------- NOTIFY SHIM --------------------
def _notify(msg, timeout=1500):
    ui_message.value = msg

# -------------------- Weather fetch (stdlib-only) --------------------
def _map_center_latlon():
    if NODE_POS.size and IS_GEOGRAPHIC:
        cx = float((X_MIN + X_MAX) * 0.5)  # lon
        cy = float((Y_MIN + Y_MAX) * 0.5)  # lat
        return cy, cx
    return 1.2926, 103.7635

def _fetch_weather_now():
    lat, lon = _map_center_latlon()
    query = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + parse.urlencode(query)
    try:
        with request.urlopen(url, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        cur = data.get("current") or {}
        t = cur.get("temperature_2m")
        rh = cur.get("relative_humidity_2m")
        ws = cur.get("wind_speed_10m")
        parts = []
        if isinstance(t, (int, float)): parts.append(f"{t:.1f}°C")
        if isinstance(rh, (int, float)): parts.append(f"RH {int(round(rh))}%")
        if isinstance(ws, (int, float)): parts.append(f"Wind {ws:.1f} m/s")
        weather_str.value = ", ".join(parts) if parts else "Weather: unavailable"
        _last_weather_ts.value = datetime.now().timestamp()
    except Exception:
        if not (weather_str.value or "").strip():
            weather_str.value = "Weather: unavailable"

async def _runtime_info_loop():
    _fetch_weather_now()
    while True:
        now = datetime.now()
        time_str.value = now.strftime("%H:%M:%S")
        date_str.value = now.strftime("%A, %d %B %Y")
        if (now.timestamp() - float(_last_weather_ts.value or 0.0)) > 300.0:
            _fetch_weather_now()
        await asyncio.sleep(1.0)

# -------------------- HELPERS --------------------
def _nx_path(u_node, v_node):
    try: return nx.shortest_path(UG, u_node, v_node, weight="weight")
    except nx.NetworkXNoPath: return [u_node, v_node]

def _nx_path_length(u_node, v_node):
    try: return nx.shortest_path_length(UG, u_node, v_node, weight="weight")
    except nx.NetworkXNoPath: return float("inf")

def _nearest_node_idx(x, y):
    _, idx = KD.query([x, y]); return idx, NODE_IDS[idx]

def _unit(v):
    n = np.linalg.norm(v); return v / (n + 1e-9)

def _nearest_hazard_m(pt):
    if not HAZARDS: return None, float("inf")
    best, bestd = None, float("inf")
    for h in HAZARDS:
        d_m = map_distance_m(pt, h["pos"])
        if d_m < bestd: best, bestd = h, d_m
    return best, bestd

def _hazard_radius_at(h):
    return float(h.get("r_m", 0.0))

def _sigmas_yz(x_m, cls):
    x = np.maximum(1.0, np.array(x_m, dtype=float))
    c = str(cls).upper()
    if c == "A":  sig_y = 0.22 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.20 * x
    elif c == "B":sig_y = 0.16 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.12 * x
    elif c == "C":sig_y = 0.11 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.08 * x * (1 + 0.0002 * x) ** (-0.5)
    elif c == "D":sig_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
    elif c == "E":sig_y = 0.06 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.03 * x * (1 + 0.0003 * x) ** (-1.0)
    else:         sig_y = 0.04 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.016 * x * (1 + 0.0003 * x) ** (-1.0)
    sig_y = np.maximum(sig_y, 0.5); sig_z = np.maximum(sig_z, 0.5)
    return sig_y, sig_z

def _point_from_phrase(phrase: str):
    if not phrase:
        return None
    p_raw = phrase
    p = p_raw.strip().lower().replace("-", " ").replace("_", " ")
    p = p.replace("northwest","north west").replace("northeast","north east")
    p = p.replace("southwest","south west").replace("southeast","south east")
    p = p.replace("top right","north east").replace("top left","north west")
    p = p.replace("bottom right","south east").replace("bottom left","south west")
    p = p.replace("centre","center")

    xmin, ymin = float(X_MIN), float(Y_MIN)
    xmax, ymax = float(X_MAX), float(Y_MAX)
    dx = xmax - xmin
    dy = ymax - ymin
    cx = xmin + dx * 0.5
    cy = ymin + dy * 0.5

    m_deg = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:deg|degrees|°)?', p)
    m_dist = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b', p)

    if m_deg:
        try:
            bearing = float(m_deg.group(1)) % 360.0
            rad = math.radians(bearing)
            if m_dist:
                r_m = float(m_dist.group(1))
                offx_map = meters_to_dx(r_m * math.sin(rad))
                offy_map = meters_to_dy(r_m * math.cos(rad))
                return np.array([cx + offx_map, cy + offy_map], dtype=float)
            else:
                r_map = 0.35 * min(dx, dy)
                offx = r_map * math.sin(rad)
                offy = r_map * math.cos(rad)
                return np.array([cx + offx, cy + offy], dtype=float)
        except Exception:
            pass

    mx = dx * 0.12
    my = dy * 0.12
    centers = {
        "center": (cx, cy),
        "north":  (cx, ymax - my),
        "south":  (cx, ymin + my),
        "east":   (xmax - mx, cy),
        "west":   (xmin + mx, cy),
        "north west": (xmin + mx, ymax - my),
        "north east": (xmax - mx, ymax - my),
        "south west": (xmin + mx, ymin + my),
        "south east": (xmax - mx, ymin + my),
        "nw": (xmin + mx, ymax - my), "ne": (xmax - mx, ymax - my),
        "sw": (xmin + mx, ymin + my), "se": (xmax - mx, ymin + my),
    }
    if p in centers:
        return np.array(centers[p], dtype=float)

    has_n = "north" in p or p == "n"
    has_s = "south" in p or p == "s"
    has_e = "east"  in p or p == "e"
    has_w = "west"  in p or p == "w"
    if has_n or has_s or has_e or has_w:
        x = cx; y = cy
        if has_w: x = xmin + mx
        if has_e: x = xmax - mx
        if has_s: y = ymin + my
        if has_n: y = ymax - my
        return np.array([x, y], dtype=float)
    if "middle" in p or "center" in p:
        return np.array(centers["center"], dtype=float)
    return None

def _recompute_safe_nodes():
    global SAFE_NODES
    if len(NODE_POS) == 0:
        SAFE_NODES = []
        return
    if not HAZARDS:
        center = NODE_POS.mean(axis=0)
        scores = np.array([map_distance_m(pt, center) for pt in NODE_POS], dtype=float)
    else:
        scores = np.empty(NODE_POS.shape[0], dtype=float)
        for i, (x, y) in enumerate(NODE_POS):
            pt = np.array([x, y], dtype=float)
            _, mn = _nearest_hazard_m(pt)
            scores[i] = mn
    order = np.argsort(-scores)
    SAFE_NODES = [NODE_IDS[i] for i in order[:N_SAFE]]

def _nearest_safe_node_from(node_id):
    safe_pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
    if not safe_pool:
        return node_id
    best_target, best_len = None, float("inf")
    for t in safe_pool:
        L = _nx_path_length(node_id, t)
        if L < best_len:
            best_len, best_target = L, t
    return best_target if best_target is not None else node_id

def _recompute_featured_safe():
    global FEATURED_SAFE
    pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
    if not pool or not HAZARDS:
        FEATURED_SAFE = set()
        return
    scored = []
    for nid in pool:
        x, y = POS_DICT[nid]
        _, d = _nearest_hazard_m(np.array([x, y], float))
        scored.append((nid, d))
    scored.sort(key=lambda t: t[1], reverse=True)
    FEATURED_SAFE = set([nid for nid, _ in scored[:max(1, min(4, len(scored)))] ])

def _is_person_expected_affected(pt_xy, t):
    if not HAZARDS: return False
    h, d_m = _nearest_hazard_m(pt_xy)
    if h is None: return False
    expected_r = float(h["r_m"]) + float(expected_growth_m.value) * float(t) + float(expected_buffer_m.value)
    return d_m <= max(0.0, expected_r)

def _spawn_people(n):
    global PEOPLE
    cyclists_target = int(round(n * float(pct_cyclists.value)/100.0))
    cycl_flags = np.array([True]*cyclists_target + [False]*(n - cyclists_target))
    rng.shuffle(cycl_flags)
    new_people = []
    for i in range(n):
        idx = rng.integers(0, NODE_POS.shape[0])
        px, py = NODE_POS[idx]
        pos2d = np.array([px, py]) + rng.uniform(-1.5, 1.5, 2)
        is_cyclist = bool(cycl_flags[i])
        base_speed = rng.normal(4.5, 0.6) if is_cyclist else rng.normal(2.0, 0.4)
        base_speed = float(np.clip(base_speed, 3.0 if is_cyclist else 1.2, 6.0 if is_cyclist else 3.5))
        panic_threshold = float(rng.uniform(0.3, 0.8))
        new_people.append({
            "pos": pos2d, "dir": np.array([0.0, 0.0]),
            "is_cyclist": is_cyclist, "base_speed": base_speed, "speed": base_speed,
            "panic": 0.0, "panic_thr": panic_threshold,
            "aware": False, "aware_delay": float(base_awareness_lag.value + rng.uniform(-2, 2)),
            "reached": False, "target_node": None, "path": [], "path_idx": 0,
            "affected_since": None,
            # timing metrics
            "evac_start_tick": None,
            "evac_end_tick": None,
            "evac_time_s": None,
        })
    PEOPLE.extend(new_people)

def _retarget_to_nearest_safe(p):
    _, s_node = _nearest_node_idx(p["pos"][0], p["pos"][1])
    t_node = _nearest_safe_node_from(s_node)
    p["target_node"] = t_node
    p["path"] = _nx_path(s_node, t_node)
    p["path_idx"] = 0

def _choose_targets_and_paths():
    _recompute_safe_nodes()
    for p in PEOPLE:
        _retarget_to_nearest_safe(p)

def _force_evacuation_mode():
    tnow = float(tick.value)
    for p in PEOPLE:
        p["aware"] = True
        p["affected_since"] = tnow
        p["reached"] = False
        _retarget_to_nearest_safe(p)

def reset_model():
    global PEOPLE
    PEOPLE = []
    _spawn_people(int(num_people.value))
    _choose_targets_and_paths()
    _recompute_featured_safe()
    tick.value += 1

def _update_panic(p):
    p["panic"] = min(1.0, p["panic"] + 0.02)
    if (not p["is_cyclist"]) and (p["panic"] > p["panic_thr"]):
        p["speed"] = p["base_speed"] * (1 + p["panic"] * 0.3)
    else:
        p["speed"] = p["base_speed"]

def _advance_hazards():
    if not HAZARDS: return
    a = math.radians(float(wind_deg.value))
    wind_m_per_tick = float(wind_speed.value)
    dx = meters_to_dx(wind_m_per_tick * math.cos(a))
    dy = meters_to_dy(wind_m_per_tick * math.sin(a))
    dr_m = float(hazard_spread.value)
    for h in HAZARDS:
        h["pos"] = h["pos"] + np.array([dx, dy], dtype=float)
        h["r_m"] = float(h["r_m"] + dr_m)
    _recompute_safe_nodes()
    _recompute_featured_safe()

def _step_once():
    _advance_hazards()
    tnow = float(tick.value)
    global_hazard = bool(HAZARDS)  # if any hazard exists, everyone evacuates

    for p in PEOPLE:
        if global_hazard:
            affected = True
            if not p.get("aware", False):
                p["aware"] = True
                p["affected_since"] = tnow
                if p.get("evac_start_tick") is None and EVAC_NOTIFICATION_TICK.value is not None:
                    p["evac_start_tick"] = int(EVAC_NOTIFICATION_TICK.value)
                _retarget_to_nearest_safe(p)
        else:
            affected = _is_person_expected_affected(p["pos"], tnow)

        if not affected:
            p["aware"] = False
            p["dir"] = np.array([0.0, 0.0])
            continue

        if (not global_hazard) and (not p["aware"]):
            if (tnow - (p["affected_since"] or tnow)) < max(0.0, p["aware_delay"]):
                p["dir"] = np.array([0.0, 0.0]); continue
            p["aware"] = True

        if p["reached"]:
            p["dir"] = np.array([0.0, 0.0]); continue

        _update_panic(p)

        path = p["path"]; k = p["path_idx"]
        if k >= len(path) - 1:
            p["reached"] = True
            p["dir"] = np.array([0.0, 0.0])
            if p.get("evac_end_tick") is None and p.get("evac_start_tick") is not None:
                p["evac_end_tick"] = int(tick.value)
                p["evac_time_s"] = float(p["evac_end_tick"] - p["evac_start_tick"])
            continue

        cur_xy = p["pos"]
        next_node = path[k+1]
        nx_xy = POS_DICT[next_node]
        next_xy = np.array([nx_xy[0], nx_xy[1]], dtype=float)

        is_panicking = p["panic"] > p["panic_thr"]
        follow_prob = 0.95 if global_hazard else (0.7 if is_panicking else 0.85)
        follow_path = rng.random() < follow_prob

        if follow_path:
            seg = next_xy - cur_xy
            dist = np.linalg.norm(seg)
            if dist <= p["speed"]:
                p["pos"] = next_xy.copy()
                p["path_idx"] = k + 1
                move = next_xy - cur_xy
            else:
                stepv = _unit(seg) * p["speed"]
                p["pos"] = cur_xy + stepv
                move = stepv
        else:
            seg = next_xy - cur_xy
            direction = _unit(seg)
            dev_mag = 0.15 if global_hazard else (0.5 if is_panicking else 0.3)
            deviation = rng.uniform(-dev_mag, dev_mag, 2)
            direction = _unit(direction + deviation)
            stepv = direction * p["speed"]
            p["pos"] = cur_xy + stepv
            _, snap_idx = KD.query(p["pos"])
            snap_xy = NODE_POS[snap_idx]
            if np.linalg.norm(p["pos"] - snap_xy) < 3.0:
                p["pos"] = snap_xy.copy()
            move = stepv

        mvn = np.linalg.norm(move)
        p["dir"] = move / (mvn + 1e-9)

        if np.linalg.norm(p["pos"] - next_xy) < 2.0 and (p["path_idx"] >= len(path) - 2):
            p["reached"] = True
            if p.get("evac_end_tick") is None and p.get("evac_start_tick") is not None:
                p["evac_end_tick"] = int(tick.value)
                p["evac_time_s"] = float(p["evac_end_tick"] - p["evac_start_tick"])

    tick.value += 1

def _path_remaining_meters(p):
    path = p["path"]; k = p["path_idx"]
    if not path or k >= len(path) - 1: return 0.0
    dist = 0.0
    nx_id = path[k+1]; nx_xy = POS_DICT[nx_id]
    dist += map_distance_m(p["pos"], np.array(nx_xy))
    for i in range(k+1, len(path)-1):
        a = path[i]; b = path[i+1]
        dist += map_distance_m(np.array(POS_DICT[a]), np.array(POS_DICT[b]))
    return dist

def agents_df():
    xs, ys, colors, sizes = [], [], [], []
    tnow = float(tick.value)
    for p in PEOPLE:
        xs.append(p["pos"][0]); ys.append(p["pos"][1])
        if p["reached"]:
            colors.append("green")
        else:
            affected = _is_person_expected_affected(p["pos"], tnow)
            if not affected and not HAZARDS:
                colors.append("steelblue")
            else:
                h, d_m = _nearest_hazard_m(p["pos"])
                r_m = _hazard_radius_at(h) if h is not None else 0.0
                in_danger = (d_m < 1.5 * r_m) if (h is not None) else False
                colors.append("purple" if (p.get("aware", False) and in_danger) else ("red"))
        sizes.append(10 if p["is_cyclist"] else 7)
    return pd.DataFrame({"x": xs, "y": ys, "color": colors, "size": sizes})

def kpi_eta_summary():
    etas = []
    for p in PEOPLE:
        if p["reached"] or not p.get("aware", False):
            continue
        remaining = _path_remaining_meters(p)
        spd = max(p["speed"], 0.1)
        etas.append(remaining / spd)
    if not etas: return "All safe or no active evacuations."
    return f"ETA to safe zone — avg: {np.mean(etas):.1f}s | P90: {np.percentile(etas,90):.1f}s | max: {np.max(etas):.1f}s"

# --------- Metrics & Effectiveness ----------
def metrics_collect():
    rows = []
    times = []
    total = len(PEOPLE)
    reached = 0
    for i, p in enumerate(PEOPLE):
        et = p.get("evac_time_s")
        rows.append({
            "agent_id": i,
            "reached": bool(p.get("reached", False)),
            "evac_time_s": None if et is None else float(et),
            "is_cyclist": bool(p.get("is_cyclist", False)),
        })
        if et is not None:
            times.append(float(et))
        if p.get("reached", False):
            reached += 1
    summary = {
        "total_agents": total,
        "reached_agents": reached,
        "pct_reached": (100.0 * reached / total) if total else 0.0,
        "avg_time_s": float(np.mean(times)) if times else None,
        "p50_time_s": float(np.percentile(times, 50)) if times else None,
        "p90_time_s": float(np.percentile(times, 90)) if times else None,
        "max_time_s": float(np.max(times)) if times else None,
        "samples": len(times),
    }
    return rows, summary

def evaluate_safe_zone_effectiveness():
    t = _totals_now()
    rows, summary = metrics_collect()
    reached_times = [r["evac_time_s"] for r in rows if r.get("reached") and r.get("evac_time_s") is not None]
    if not reached_times:
        return (
            f"**Effectiveness (Safe Zone Deployment)**\n\n"
            f"- Total people: {t['total']}\n"
            f"- Reached safe zone: {t['reached']}\n"
            f"- Aware now: {t['aware']}\n"
            f"- In envelope now: {t['evacuees']}\n\n"
            f"_No agents have reached a safe zone yet — run the sim (START/STEP) and try again._"
        )
    return (
        f"**Effectiveness (Safe Zone Deployment)**\n\n"
        f"- Total people: {t['total']}\n"
        f"- Reached safe zone: {t['reached']} "
        f"({summary['pct_reached']:.1f}% of total)\n"
        f"- Aware now: {t['aware']}\n"
        f"- In envelope now: {t['evacuees']}\n\n"
        f"- Time-to-safe (reached only): "
        f"avg {summary['avg_time_s']:.1f}s | "
        f"P50 {summary['p50_time_s']:.1f}s | "
        f"P90 {summary['p90_time_s']:.1f}s | "
        f"max {summary['max_time_s']:.1f}s "
        f"(n={summary['samples']})"
    )

# --------- ETA-based Safe Zone Optimization ----------
def _average_eta_for_safe_set(safe_nodes_set, speed_mps=1.4, sample_size=400):
    if not safe_nodes_set:
        return float("inf")
    n = min(sample_size, len(NODE_IDS))
    idxs = rng.choice(len(NODE_IDS), size=n, replace=False)
    sample_ids = [NODE_IDS[int(i)] for i in idxs]
    etas = []
    for nid in sample_ids:
        best = None; best_len = float("inf")
        for t in safe_nodes_set:
            L = _nx_path_length(nid, t)
            if L < best_len:
                best_len = L; best = t
        if best is None or not np.isfinite(best_len):
            continue
        etas.append(best_len / max(0.1, speed_mps))
    if not etas:
        return float("inf")
    return float(np.mean(etas))

def _optimize_safe_zones_by_eta(k=None, attempts=8, speed_mps=1.4, sample_size=400):
    global MANUAL_SAFE
    k = int(k or max(4, min(12, N_SAFE)))
    if not NODE_POS.size:
        return False

    if HAZARDS:
        scores = np.array([_nearest_hazard_m(NODE_POS[i])[1] for i in range(NODE_POS.shape[0])], float)
    else:
        center = NODE_POS.mean(axis=0)
        scores = np.array([map_distance_m(NODE_POS[i], center) for i in range(NODE_POS.shape[0])], float)

    order = np.argsort(-scores)
    pool_idx_full = order[:max(5*k, k+100)]

    best_set = None
    best_eta = float("inf")

    for _ in range(max(1, attempts)):
        if len(pool_idx_full) == 0:
            break
        seed = int(rng.choice(pool_idx_full))
        chosen = [seed]
        dmin = np.full(pool_idx_full.shape[0], np.inf)
        for _i in range(1, k):
            last_pt = NODE_POS[chosen[-1]]
            dmin = np.minimum(dmin, np.linalg.norm(NODE_POS[pool_idx_full] - last_pt, axis=1))
            nxt = pool_idx_full[int(np.argmax(dmin))]
            chosen.append(int(nxt))
        cand = set([NODE_IDS[i] for i in chosen])
        eta = _average_eta_for_safe_set(cand, speed_mps=speed_mps, sample_size=sample_size)
        if eta < best_eta:
            best_eta = eta
            best_set = cand

    if best_set:
        MANUAL_SAFE = set(best_set)
        _recompute_featured_safe()
        _choose_targets_and_paths()
        return True
    return False

# --------- Plume/Risk grids ----------
def _plume_concentration_grid(x_domain, y_domain):
    try:
        if not HAZARDS:
            return None
        nxg = int(max(40, min(180, int(plume_grid.value * 1.2))))  # <-- FIXED extra ')'
        nyg = int(max(30, min(150, int(plume_grid.value))))
        gx = np.linspace(x_domain[0], x_domain[1], nxg)
        gy = np.linspace(y_domain[0], y_domain[1], nyg)
        X, Y = np.meshgrid(gx, gy)

        a = math.radians(float(wind_deg.value))
        ex, ey = math.cos(a), math.sin(a)
        U = max(0.1, float(wind_speed.value))  # m/s

        Q = float(plume_Q_gs.value)
        H = float(plume_H_m.value)
        stab = plume_stab.value.upper()
        maxx = float(plume_range_m.value)

        C = np.zeros_like(X, dtype=float)
        for h in HAZARDS:
            hx, hy = float(h["pos"][0]), float(h["pos"][1])
            dx_map = X - hx; dy_map = Y - hy
            dx_m = dx_map_to_m(dx_map); dy_m = dy_map_to_m(dy_map)
            x_m = dx_m * ex + dy_m * ey
            y_m = -dx_m * ey + dy_m * ex
            mask = (x_m > 1.0) & (x_m <= maxx)
            if not np.any(mask): continue
            xm_clip = np.clip(x_m, 1.0, None)
            sy, sz = _sigmas_yz(xm_clip, stab)
            pref = Q / (2.0 * math.pi * U)
            termy = np.exp(-(y_m ** 2) / (2.0 * sy ** 2))
            termz = (np.exp(-((0.0 - H) ** 2) / (2.0 * sz ** 2))
                     + np.exp(-((0.0 + H) ** 2) / (2.0 * sz ** 2)))
            denom = sy * sz
            contrib = np.zeros_like(C)
            idx = np.where(mask)
            contrib[idx] = pref * termy[idx] * termz[idx] / denom[idx]
            C += contrib
        return gx, gy, C
    except Exception as e:
        last_error.value = f"Plume grid error: {repr(e)}"
        return None

def _risk_eta_grid(x_domain, y_domain):
    ny = int(max(30, min(180, int(risk_grid.value))))
    nx = int(max(40, min(200, int(ny * 1.2))))
    gx = np.linspace(x_domain[0], x_domain[1], nx)
    gy = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    _, idxs = KD.query(pts)
    node_ids = [NODE_IDS[i] for i in idxs]
    safe_pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
    if not safe_pool:
        return gx, gy, np.full_like(X, np.nan, dtype=float)
    speed = max(0.5, float(risk_speed_mps.value))
    memo = {}
    etas = np.empty(idxs.shape[0], dtype=float)
    for i, nid in enumerate(node_ids):
        if nid in memo:
            etas[i] = memo[nid]; continue
        tnode = _nearest_safe_node_from(nid)
        if tnode == nid:
            eta = 0.0
        else:
            dist = _nx_path_length(nid, tnode)
            eta = (dist / speed) if np.isfinite(dist) else np.nan
        memo[nid] = eta
        etas[i] = eta
    Z = etas.reshape(X.shape)
    return gx, gy, Z

def _segments_from_edges_gdf():
    seg_x, seg_y = [], []
    try:
        for geom in EDGES_GDF.geometry.values:
            if geom is None: continue
            gtype = getattr(geom, "geom_type", "")
            if gtype == "LineString":
                coords = list(geom.coords)
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]; x2, y2 = coords[i+1]
                    seg_x += [x1, x2, None]; seg_y += [y1, y2, None]
            elif gtype == "MultiLineString":
                for line in getattr(geom, "geoms", []):
                    coords = list(line.coords)
                    for i in range(len(coords) - 1):
                        x1, y1 = coords[i]; x2, y2 = coords[i+1]
                        seg_x += [x1, x2, None]; seg_y += [y1, y2, None]
        return seg_x, seg_y
    except Exception as e:
        last_error.value = f"Edge geometry error: {repr(e)}"
        return [], []

def _suggest_optimized_safe(k=None):
    global MANUAL_SAFE
    k = int(k or max(4, min(12, N_SAFE)))
    if not NODE_POS.size:
        return
    if HAZARDS:
        scores = np.array([_nearest_hazard_m(NODE_POS[i])[1] for i in range(NODE_POS.shape[0])], float)
    else:
        center = NODE_POS.mean(axis=0)
        scores = np.array([map_distance_m(NODE_POS[i], center) for i in range(NODE_POS.shape[0])], float)
    order = np.argsort(-scores)
    pool_idx = order[:max(5*k, k+50)]
    chosen = [pool_idx[0]]
    dmin = np.full(pool_idx.shape[0], np.inf)
    for _ in range(1, k):
        last_pt = NODE_POS[chosen[-1]]
        dmin = np.minimum(dmin, np.linalg.norm(NODE_POS[pool_idx] - last_pt, axis=1))
        nxt = pool_idx[int(np.argmax(dmin))]
        chosen.append(int(nxt))
    MANUAL_SAFE = set([NODE_IDS[i] for i in chosen])
    _recompute_featured_safe()

# --------- SCDF Responder Deployment Optimiser ----------
def _build_response_demand(x_domain, y_domain):
    """
    Return gx, gy, demand (higher = more need).
    Combines: plume (normalized), risk ETA (normalized), unaware density, distance-to-hazard penalty,
    distance-to-safe small boost. All terms scaled to [0,1] before combination.
    """
    # Base grids
    risk = _risk_eta_grid(x_domain, y_domain)[2] if risk_show.value else None
    plume = _plume_concentration_grid(x_domain, y_domain)
    if plume is not None:
        pgx, pgy, pZ = plume
    else:
        # build a blank plume grid aligned with risk resolution if available
        rx, ry, rZ = _risk_eta_grid(x_domain, y_domain)
        pgx, pgy, pZ = rx, ry, np.zeros_like(rZ)

    # Resample risk onto plume grid if needed
    rx, ry, rZ = _risk_eta_grid(x_domain, y_domain)
    if (rZ.shape != pZ.shape):
        # crude nearest resize to plume resolution
        pny, pnx = pZ.shape
        rny, rnx = rZ.shape
        iy = (np.linspace(0, rny - 1, pny)).astype(int)
        ix = (np.linspace(0, rnx - 1, pnx)).astype(int)
        rZr = rZ[iy][:, ix]
    else:
        rZr = rZ

    # Normalize helpers
    def _norm(a):
        a = np.array(a, dtype=float)
        m = np.nanmin(a); M = np.nanmax(a)
        if not np.isfinite(m) or not np.isfinite(M) or M - m < 1e-9:
            return np.zeros_like(a)
        out = (a - m) / (M - m)
        out[~np.isfinite(out)] = 0.0
        return out

    plume_n = _norm(pZ)
    risk_n = _norm(rZr)

    # Unaware density (agents not aware & not yet reached) onto plume grid
    px = pgx; py = pgy
    X, Y = np.meshgrid(px, py)
    unaware = np.zeros_like(X, dtype=float)
    tnow = float(tick.value)
    for p in PEOPLE:
        if p.get("reached", False): 
            continue
        aware = bool(p.get("aware", False))
        affected = _is_person_expected_affected(p["pos"], tnow)
        if (not aware) and affected:
            # deposit a small kernel at nearest pixel
            x, y = float(p["pos"][0]), float(p["pos"][1])
            ix = np.argmin(np.abs(px - x))
            iy = np.argmin(np.abs(py - y))
            if 0 <= iy < unaware.shape[0] and 0 <= ix < unaware.shape[1]:
                unaware[iy, ix] += 1.0
    unaware = _norm(unaware)

    # Distance-to-hazard (prefer not too close)
    hazard_penalty = np.zeros_like(X, dtype=float)
    if HAZARDS:
        # compute min distance to any hazard centre in meters
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pt = np.array([X[i, j], Y[i, j]])
                _, d_m = _nearest_hazard_m(pt)
                # penalty is high near hazard, low far
                hazard_penalty[i, j] = 1.0 - (1.0 / (1.0 + d_m / 50.0))
        hazard_penalty = _norm(hazard_penalty)
    else:
        hazard_penalty[:] = 0.2  # small constant if no hazards

    # Distance-to-nearest-safe (areas far from safe zones get small boost)
    safe_boost = np.zeros_like(X, dtype=float)
    pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
    if pool:
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                _, idx = KD.query([X[i, j], Y[i, j]])
                nid = NODE_IDS[idx]
                tnode = _nearest_safe_node_from(nid)
                d = _nx_path_length(nid, tnode)
                safe_boost[i, j] = d if np.isfinite(d) else 0.0
        safe_boost = _norm(safe_boost)

    # Combine (weights can be tuned)
    # Demand ~ 0.35*plume + 0.35*risk + 0.2*unaware + 0.1*safe_boost - 0.15*hazard_penalty
    demand = (0.35 * plume_n) + (0.35 * risk_n) + (0.20 * unaware) + (0.10 * safe_boost) - (0.15 * hazard_penalty)
    demand = np.maximum(demand, 0.0)
    return pgx, pgy, demand

def _suggest_responders(k=5):
    """Place k responder staging nodes near the peaks of response demand, while spreading them out."""
    global RESPONDERS
    if k <= 0:
        RESPONDERS = set(); return False
    # domain from nodes
    xmin, ymin = NODE_POS.min(axis=0)
    xmax, ymax = NODE_POS.max(axis=0)
    padx = max((xmax - xmin) * 0.03, 1.0)
    pady = max((ymax - ymin) * 0.03, 1.0)
    x_domain = [float(xmin - padx), float(xmax + padx)]
    y_domain = [float(ymin - pady), float(ymax + pady)]

    pgx, pgy, demand = _build_response_demand(x_domain, y_domain)
    if demand is None or not np.isfinite(demand).any():
        RESPONDERS = set(); return False

    # Pick top M pixels by demand, map to nearest nodes, then do farthest-point sampling
    M = int(max(50, 10 * k))
    flat = demand.ravel()
    if flat.size < M:
        M = flat.size
    top_idx = np.argpartition(-flat, M-1)[:M]
    iy, ix = np.unravel_index(top_idx, demand.shape)
    cand_pts = np.column_stack([pgx[ix], pgy[iy]])

    # snap to nearest graph nodes
    _, nidxs = KD.query(cand_pts)
    node_cands = [NODE_IDS[i] for i in nidxs]

    # remove duplicates while preserving order
    seen = set(); unique_nodes = []
    for nid in node_cands:
        if nid in seen: continue
        seen.add(nid); unique_nodes.append(nid)
    if not unique_nodes:
        RESPONDERS = set(); return False

    # farthest-point sampling on node coordinates
    chosen = [unique_nodes[0]]
    for _ in range(1, k):
        best = None; best_d = -1.0
        for nid in unique_nodes:
            if nid in chosen: continue
            # distance to nearest chosen (map-distance)
            pt = np.array(POS_DICT[nid])
            dmin = float("inf")
            for c in chosen:
                dmin = min(dmin, map_distance_m(pt, np.array(POS_DICT[c])))
            if dmin > best_d:
                best_d = dmin; best = nid
        if best is not None:
            chosen.append(best)
        else:
            break

    RESPONDERS = set(chosen)
    return True

# --------- Paths/collectors/QC ----------
def _min_hazard_distance_at_xy(xy):
    if not HAZARDS:
        return float("inf")
    _, d_m = _nearest_hazard_m(np.array([xy[0], xy[1]], dtype=float))
    return float(d_m)

def _node_xy(nid):
    x, y = POS_DICT[nid]
    return np.array([x, y], dtype=float)

def _path_length_m_along_nodes(path_nodes):
    if not path_nodes or len(path_nodes) < 2:
        return 0.0
    d = 0.0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        d += map_distance_m(_node_xy(a), _node_xy(b))
    return d

def _evacuee_current_path_nodes(p):
    path = p.get("path", [])
    k = int(p.get("path_idx", 0))
    if not path or k >= len(path) - 1:
        return path[-1:]
    return path[k:]

def _collect_paths_polylines(evac_only=True, max_agents=300):
    xs, ys = [], []
    drawn = 0
    tnow = float(tick.value)
    for p in PEOPLE:
        affected = _is_person_expected_affected(p["pos"], tnow)
        if evac_only and not affected:
            continue
        nodes = _evacuee_current_path_nodes(p)
        if len(nodes) < 2:
            continue
        for a, b in zip(nodes[:-1], nodes[1:]):
            xa, ya = POS_DICT[a]; xb, yb = POS_DICT[b]
            xs += [xa, xb, None]; ys += [ya, yb, None]
        drawn += 1
        if drawn >= max_agents:
            break
    return xs, ys

def _totals_now():
    tnow = float(tick.value)
    total = len(PEOPLE)
    evacuees_now = sum(1 for p in PEOPLE if _is_person_expected_affected(p["pos"], tnow))
    aware_now = sum(1 for p in PEOPLE if p.get("aware", False))
    reached_now = sum(1 for p in PEOPLE if p.get("reached", False))
    return dict(total=total, evacuees=evacuees_now, aware=aware_now, reached=reached_now)

def qc_run(expected_margin_m=25.0):
    summaries = []
    issues = []
    rows = []

    safe_pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
    if not safe_pool:
        issues.append("No safe zones computed or placed yet.")
    else:
        safe_dists = []
        for nid in safe_pool:
            xy = _node_xy(nid)
            dmin = _min_hazard_distance_at_xy(xy)
            safe_dists.append(dmin)
        if safe_dists:
            summaries.append(f"Safe zones: {len(safe_pool)} | min dist to hazard: {min(safe_dists):.1f} m | median: {np.median(safe_dists):.1f} m")
            suspicious = sum(1 for dmin in safe_dists if dmin < expected_margin_m)
            if suspicious > 0:
                issues.append(f"{suspicious} safe zone(s) are < {expected_margin_m:.0f} m from a hazard centre.")

    tnow = float(tick.value)
    tot = _totals_now()
    evacuees = [p for p in PEOPLE if _is_person_expected_affected(p["pos"], tnow)]
    summaries.append(f"People now: total={tot['total']} | in-envelope={tot['evacuees']} | aware={tot['aware']} | reached={tot['reached']}")

    no_path = 0
    total_len_m = []
    load_per_safe = {}
    for p in evacuees:
        nodes = _evacuee_current_path_nodes(p)
        if len(nodes) < 2:
            no_path += 1
            continue
        plen = _path_length_m_along_nodes(nodes)
        total_len_m.append(plen)
        dst = nodes[-1]
        load_per_safe[dst] = load_per_safe.get(dst, 0) + 1
        rows.append({
            "x": float(p["pos"][0]),
            "y": float(p["pos"][1]),
            "affected": True,
            "aware": bool(p.get("aware", False)),
            "path_nodes": len(nodes),
            "path_len_m": float(plen),
            "dest_safe_node": str(dst),
        })

    if no_path > 0:
        issues.append(f"{no_path} evacuee(s) currently have no valid path to a safe zone.")
    if total_len_m:
        summaries.append(f"Path length (m) — avg: {np.mean(total_len_m):.1f} | P90: {np.percentile(total_len_m,90):.1f} | max: {np.max(total_len_m):.1f}")
    if load_per_safe:
        top = sorted(load_per_safe.items(), key=lambda kv: kv[1], reverse=True)[:5]
        summaries.append("Top safe-node loads: " + ", ".join([f"{nid}:{cnt}" for nid,cnt in top]))

    still_but_marked = 0
    for p in PEOPLE:
        if not _is_person_expected_affected(p["pos"], tnow) and (np.linalg.norm(p.get("dir", np.zeros(2))) > 1e-6):
            still_but_marked += 1
        if not _is_person_expected_affected(p["pos"], tnow):
            rows.append({
                "x": float(p["pos"][0]),
                "y": float(p["pos"][1]),
                "affected": False,
                "aware": bool(p.get("aware", False)),
                "path_nodes": 0,
                "path_len_m": 0.0,
                "dest_safe_node": "",
            })
    if still_but_marked > 0:
        issues.append(f"{still_but_marked} non-affected person(s) appear to be moving (dir≠0).")

    if not issues:
        issues.append("No issues detected.")

    return summaries, issues, rows

# -------------------- UI --------------------
@sl.component
def Controls():
    # ---------- local states for ALL editable textboxes ----------
    qc_margin_s, set_qc_margin_s = sl.use_state(f"{float(qc_km_margin.value):.1f}")
    incident_where_s, set_incident_where_s = sl.use_state("")
    remove_id_s, set_remove_id_s = sl.use_state("")
    safe_phrase_s, set_safe_phrase_s = sl.use_state("")
    people_s, set_people_s = sl.use_state(str(int(num_people.value)))
    cyclists_s, set_cyclists_s = sl.use_state(str(int(pct_cyclists.value)))
    hz_radius_s, set_hz_radius_s = sl.use_state(f"{float(hazard_radius.value):.1f}")
    hz_spread_s, set_hz_spread_s = sl.use_state(f"{float(hazard_spread.value):.2f}")
    exp_growth_s, set_exp_growth_s = sl.use_state(f"{float(expected_growth_m.value):.2f}")
    exp_buffer_s, set_exp_buffer_s = sl.use_state(f"{float(expected_buffer_m.value):.1f}")
    wind_deg_s, set_wind_deg_s = sl.use_state(f"{float(wind_deg.value):.1f}")
    wind_speed_s, set_wind_speed_s = sl.use_state(f"{float(wind_speed.value):.2f}")
    base_aw_s, set_base_aw_s = sl.use_state(f"{float(base_awareness_lag.value):.1f}")
    plume_show_s, set_plume_show_s = sl.use_state("true" if plume_show.value else "false")
    plume_stab_s, set_plume_stab_s = sl.use_state(plume_stab.value)
    plume_Q_s, set_plume_Q_s = sl.use_state(f"{float(plume_Q_gs.value):.1f}")
    plume_H_s, set_plume_H_s = sl.use_state(f"{float(plume_H_m.value):.1f}")
    plume_range_s, set_plume_range_s = sl.use_state(f"{float(plume_range_m.value):.1f}")
    plume_grid_s, set_plume_grid_s = sl.use_state(str(int(plume_grid.value)))
    risk_show_s, set_risk_show_s = sl.use_state("true" if risk_show.value else "false")
    risk_grid_s, set_risk_grid_s = sl.use_state(str(int(risk_grid.value)))
    risk_speed_s, set_risk_speed_s = sl.use_state(f"{float(risk_speed_mps.value):.2f}")
    responders_s, set_responders_s = sl.use_state(str(int(N_RESPONDERS.value)))
    metrics_text, set_metrics_text = sl.use_state("")

    # ---------- sim controls ----------
    def on_step(): _step_once()
    async def loop_runner():
        while running.value:
            _step_once()
            await asyncio.sleep(0.05)
    def on_toggle():
        if not running.value:
            running.value = True
            asyncio.get_event_loop().create_task(loop_runner())
        else:
            running.value = False
    if not _runtime_loop_started.value:
        _runtime_loop_started.value = True
        asyncio.get_event_loop().create_task(_runtime_info_loop())

    # ---------- APPLY / QC handlers ----------
    def on_apply_parameters():
        prev_people = int(num_people.value)
        prev_pct_cyclists = int(pct_cyclists.value)
        try: num_people.value = max(0, int((people_s or "").strip()))
        except: pass
        try: pct_cyclists.value = max(0, min(100, int((cyclists_s or "").strip())))
        except: pass
        try: hazard_radius.value = max(0.0, float((hz_radius_s or '').strip()))
        except: pass
        try: hazard_spread.value = max(0.0, float((hz_spread_s or '').strip()))
        except: pass
        try: expected_growth_m.value = max(0.0, float((exp_growth_s or '').strip()))
        except: pass
        try: expected_buffer_m.value = max(0.0, float((exp_buffer_s or '').strip()))
        except: pass
        try: wind_deg.value = float((wind_deg_s or '').strip()) % 360.0
        except: pass
        try: wind_speed.value = max(0.0, float((wind_speed_s or '').strip()))
        except: pass
        try: base_awareness_lag.value = max(0.0, float((base_aw_s or '').strip()))
        except: pass
        try:
            s = (plume_show_s or "").strip().lower()
            plume_show.value = s in ("1","true","yes","y","on")
        except: pass
        try:
            s = (plume_stab_s or "").strip().upper()
            if s in ("A","B","C","D","E","F"):
                plume_stab.value = s
        except: pass
        try: plume_Q_gs.value = max(0.0, float((plume_Q_s or '').strip()))
        except: pass
        try: plume_H_m.value = max(0.0, float((plume_H_s or '').strip()))
        except: pass
        try: plume_range_m.value = max(0.0, float((plume_range_s or '').strip()))
        except: pass
        try: plume_grid.value = max(1, int((plume_grid_s or '').strip()))
        except: pass
        try:
            s = (risk_show_s or "").strip().lower()
            risk_show.value = s in ("1","true","yes","y","on")
        except: pass
        try: risk_grid.value = max(1, int((risk_grid_s or '').strip()))
        except: pass
        try: risk_speed_mps.value = max(0.1, float((risk_speed_s or '').strip()))
        except: pass
        try:
            N_RESPONDERS.value = max(0, int((responders_s or '').strip()))
        except:
            pass

        need_respawn = (int(num_people.value) != prev_people) or (int(pct_cyclists.value) != prev_pct_cyclists)
        if need_respawn:
            reset_model()
        else:
            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            tick.value += 1
        _notify("Parameters applied.")

    def on_qc_margin_submit():
        try:
            qc_km_margin.value = max(0.0, float((qc_margin_s or '').strip()))
            _notify("QC margin updated.")
        except:
            _notify("Enter a numeric QC margin (m).")

    # --- QC panel ---
    def on_qc_run():
        summaries, problems, rows = qc_run(expected_margin_m=float(qc_km_margin.value))
        qc_summary.value = "\n".join([f"- {s}" for s in summaries])
        qc_issues.value  = "\n".join([f"- {p}" for p in problems])
        qc_rows_cache.clear()
        qc_rows_cache.extend(rows)
        tick.value += 1

    def on_export_csv():
        if not qc_rows_cache:
            _notify("Run QC first; nothing to export.")
            return
        df = pd.DataFrame(qc_rows_cache)
        path = "/mnt/data/qc_routes.csv"
        try:
            df.to_csv(path, index=False)
            download_link_md.value = f"[Download QC CSV](sandbox:{path})"
            _notify("QC CSV exported.")
        except Exception as e:
            _notify(f"Export failed: {e}")

    # --- Hazard controls ---
    def on_reset():
        running.value = False
        reset_model()

    def on_clear_hazards():
        HAZARDS.clear()
        _recompute_safe_nodes()
        _recompute_featured_safe()
        _choose_targets_and_paths()
        tick.value += 1
        _notify("Hazards cleared.")

    def on_submit_location():
        txt = (incident_where_s or "").strip()
        pt = _point_from_phrase(txt)
        if pt is None:
            _notify("Could not parse location. Try: North, South, East, West, North West, NE, Center")
            return
        global HAZARD_ID
        HAZARDS.append({
            "id": HAZARD_ID,
            "pos": np.array([float(pt[0]), float(pt[1])], dtype=float),
            "r_m": float(max(5.0, hazard_radius.value)),
        })
        HAZARD_ID += 1

        _recompute_safe_nodes()
        _recompute_featured_safe()
        _choose_targets_and_paths()

        EVAC_NOTIFICATION_TICK.value = int(tick.value)
        for p in PEOPLE:
            p["evac_start_tick"] = int(tick.value)
            p["evac_end_tick"] = None
            p["evac_time_s"] = None

        _force_evacuation_mode()
        tick.value += 1
        _notify("Hazard added — all agents evacuating now.")

    def on_remove_by_id():
        txt = (remove_id_s or "").strip()
        if not txt.isdigit():
            _notify("Enter a numeric hazard ID."); return
        hid = int(txt)
        ok = False
        for i in range(len(HAZARDS)-1, -1, -1):
            if int(HAZARDS[i]["id"]) == hid:
                HAZARDS.pop(i); ok = True; break
        if ok:
            _recompute_safe_nodes(); _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
            _notify(f"Hazard {hid} removed.")
        else:
            _notify(f"Hazard {hid} not found.")

    def on_remove_last():
        if not HAZARDS:
            _notify("No hazards to remove."); return
        hid = int(HAZARDS[-1]["id"])
        HAZARDS.pop()
        _recompute_safe_nodes(); _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
        _notify(f"Last hazard {hid} removed.")

    # --- Metrics & Optimisers ---
    def on_eval_effectiveness():
        set_metrics_text(evaluate_safe_zone_effectiveness())

    def on_export_metrics():
        rows, summary = metrics_collect()
        df = pd.DataFrame(rows)
        path = "/mnt/data/agent_evac_times.csv"
        try:
            df.to_csv(path, index=False)
            _notify("Exported per-agent metrics to CSV.")
            set_metrics_text(
                evaluate_safe_zone_effectiveness() + f"\n[Download per-agent CSV](sandbox:{path})"
            )
        except Exception as e:
            _notify(f"Export failed: {e}")

    def on_optimize_min_eta():
        ok = _optimize_safe_zones_by_eta(
            k=N_SAFE, attempts=10, speed_mps=float(risk_speed_mps.value), sample_size=500
        )
        _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
        _notify("Optimized safe zones (min-ETA)." if ok else "Optimization failed; kept previous safe zones.")

    # --- SCDF responders ---
    def on_suggest_responders():
        try:
            k = max(0, int((responders_s or "0").strip()))
        except:
            k = int(N_RESPONDERS.value)
        ok = _suggest_responders(k=k)
        tick.value += 1
        _notify("Suggested SCDF responder staging points." if ok else "Responder suggestion failed.")

    def on_clear_responders():
        RESPONDERS.clear()
        tick.value += 1
        _notify("Cleared SCDF responder staging points.")

    # ---------- Render Controls ----------
    sl.Markdown("### Quality Check")
    with sl.Row():
        sl.InputText("Flag safe-zone if within X m of a hazard",
                     value=qc_margin_s, on_value=set_qc_margin_s,
                     continuous_update=True, placeholder="e.g., 25")
        sl.Button("Apply QC Margin", on_click=on_qc_margin_submit)
        sl.Checkbox(label="Show evac paths overlay", value=qc_show_paths)

    with sl.Row():
        sl.Button("Run QC", on_click=on_qc_run)
        sl.Button("Export QC CSV", on_click=on_export_csv)

    if qc_summary.value:
        sl.Markdown("**QC Summary**")
        sl.Markdown(qc_summary.value)
    if qc_issues.value:
        sl.Markdown("**QC Issues**")
        sl.Markdown(qc_issues.value)
    if download_link_md.value:
        sl.Markdown(download_link_md.value)
    if ui_message.value:
        sl.Markdown(f"> **{ui_message.value}**")

    sl.Markdown("### Controls")
    with sl.Row():
        sl.Button("STEP", on_click=on_step)
        sl.Button("START" if not running.value else "PAUSE", on_click=on_toggle)
        sl.Button("RESET", on_click=on_reset)
        sl.Markdown(f"**Tick:** {int(tick.value)}")

    with sl.Row():
        sl.Markdown(f"**Local time:** {time_str.value or '-'}")
        sl.Markdown(f"**Date:** {date_str.value or '-'}")
        sl.Markdown(f"**Weather:** {weather_str.value or '—'}")

    sl.Markdown("**Incidents (Hazards):** Add multiple or remove by ID")
    with sl.Row():
        sl.InputText(
            label="Where is it:",
            value=incident_where_s, on_value=set_incident_where_s,
            continuous_update=True,
            placeholder="e.g., North  |  NE  |  160°  |  North, 160 degrees  |  225°, 300m"
        )
        sl.Button("Submit", on_click=on_submit_location)
        sl.Button("Clear Hazards", on_click=on_clear_hazards)
    with sl.Row():
        sl.InputText(label="Remove Hazard ID",
                     value=remove_id_s, on_value=set_remove_id_s,
                     continuous_update=True, placeholder="e.g., 2")
        sl.Button("Remove by ID", on_click=on_remove_by_id)
        sl.Button("Remove Last", on_click=on_remove_last)

    if HAZARDS:
        sl.Markdown("**Current Hazards**")
        for h in list(HAZARDS):
            hid = int(h["id"])
            rdisp = float(h.get("r_m", 0.0))
            with sl.Row():
                sl.Markdown(f"- ID **{hid}** | radius ~ {rdisp:.1f} m")
                def _mk_remove(hid_):
                    def _inner():
                        ok = False
                        for i in range(len(HAZARDS)-1, -1, -1):
                            if int(HAZARDS[i]["id"]) == int(hid_):
                                HAZARDS.pop(i); ok = True; break
                        if ok:
                            _recompute_safe_nodes(); _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
                            _notify(f"Hazard {hid_} removed.")
                        else:
                            _notify(f"Hazard {hid_} not found.")
                    return _inner
                sl.Button("Remove", on_click=_mk_remove(hid))

    sl.Markdown("### Scenario Parameters (consolidated)")
    with sl.Column():
        with sl.Row():
            sl.InputText("People (initial)", value=people_s, on_value=set_people_s, continuous_update=True, placeholder="e.g., 200")
            sl.InputText("Cyclists (%)", value=cyclists_s, on_value=set_cyclists_s, continuous_update=True, placeholder="0-100")
        with sl.Row():
            sl.InputText("Hazard radius (new, m)", value=hz_radius_s, on_value=set_hz_radius_s, continuous_update=True, placeholder="e.g., 40.0")
            sl.InputText("Radius spread / step (m/step)", value=hz_spread_s, on_value=set_hz_spread_s, continuous_update=True, placeholder="e.g., 1.2")
        with sl.Row():
            sl.InputText("Expected growth (m/step)", value=exp_growth_s, on_value=set_exp_growth_s, continuous_update=True, placeholder="e.g., 3.0")
            sl.InputText("Expected safety buffer (m)", value=exp_buffer_s, on_value=set_exp_buffer_s, continuous_update=True, placeholder="e.g., 0.0")
        with sl.Row():
            sl.InputText("Wind (deg)", value=wind_deg_s, on_value=set_wind_deg_s, continuous_update=True, placeholder="0-360")
            sl.InputText("Wind speed (m/s)", value=wind_speed_s, on_value=set_wind_speed_s, continuous_update=True, placeholder="e.g., 1.0")
        with sl.Row():
            sl.InputText("Base awareness lag (s)", value=base_aw_s, on_value=set_base_aw_s, continuous_update=True, placeholder="e.g., 8.0")
            sl.InputText("Show plume (true/false)", value=plume_show_s, on_value=set_plume_show_s, continuous_update=True, placeholder="true/false")
        with sl.Row():
            sl.InputText("Stability (A–F)", value=plume_stab_s, on_value=set_plume_stab_s, continuous_update=True, placeholder="A..F")
            sl.InputText("Q (g/s)", value=plume_Q_s, on_value=set_plume_Q_s, continuous_update=True, placeholder="e.g., 100")
        with sl.Row():
            sl.InputText("Stack height H (m)", value=plume_H_s, on_value=set_plume_H_s, continuous_update=True, placeholder="e.g., 20")
            sl.InputText("Plume range (m)", value=plume_range_s, on_value=set_plume_range_s, continuous_update=True, placeholder="e.g., 1500")
        with sl.Row():
            sl.InputText("Heatmap resolution (rows)", value=plume_grid_s, on_value=set_plume_grid_s, continuous_update=True, placeholder="e.g., 90")
            sl.InputText("Show risk heatmap (true/false)", value=risk_show_s, on_value=set_risk_show_s, continuous_update=True, placeholder="true/false")
        with sl.Row():
            sl.InputText("Risk grid rows", value=risk_grid_s, on_value=set_risk_grid_s, continuous_update=True, placeholder="e.g., 90")
            sl.InputText("Risk walking speed (m/s)", value=risk_speed_s, on_value=set_risk_speed_s, continuous_update=True, placeholder="e.g., 1.4")
        with sl.Row():
            sl.InputText("SCDF responders (count)", value=responders_s, on_value=set_responders_s, continuous_update=True, placeholder="e.g., 5")
            sl.Button("Apply Parameters", on_click=on_apply_parameters)

    sl.Markdown("**Evacuation Metrics & Optimization**")
    with sl.Row():
        sl.Button("Evaluate Effectiveness", on_click=on_eval_effectiveness)
        sl.Button("Export Metrics CSV", on_click=on_export_metrics)
        sl.Button("Suggest (min-ETA) Safe Zones", on_click=on_optimize_min_eta)

    if metrics_text:
        sl.Markdown(metrics_text)

    sl.Markdown("**Safe Zones — manual & distance-optimize**")
    with sl.Row():
        sl.InputText(label="Add Safe Zone (phrase)",
                     value=safe_phrase_s, on_value=set_safe_phrase_s,
                     continuous_update=True, placeholder="North East")
        def on_add_safe():
            txt = (safe_phrase_s or "").strip()
            if not txt:
                _notify("Enter a phrase like 'North West', 'Center', 'SE'."); return
            pt = _point_from_phrase(txt)
            if pt is None:
                _notify("Could not parse phrase; try: North, South, East, West, NE, NW, SE, SW, Center."); return
            _, node_id = _nearest_node_idx(pt[0], pt[1])
            MANUAL_SAFE.add(node_id)
            _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
            _notify("Manual safe zone added.")
        def on_clear_safe():
            MANUAL_SAFE.clear(); _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
            _notify("Manual safe zones cleared.")
        def on_optimize_dist():
            _suggest_optimized_safe(k=N_SAFE); _recompute_featured_safe(); _choose_targets_and_paths(); tick.value += 1
            _notify("Optimized safe zones (distance-based) suggested.")
        sl.Button("Add Safe Zone Here", on_click=on_add_safe)
        sl.Button("Clear Safe Zones", on_click=on_clear_safe)
        sl.Button("Suggest (optimize) Safe Zones", on_click=on_optimize_dist)

    sl.Markdown("**SCDF Responder Deployment**")
    with sl.Row():
        sl.Button("Suggest SCDF Responder Staging", on_click=on_suggest_responders)
        sl.Button("Clear SCDF Responders", on_click=on_clear_responders)

    if last_error.value:
        sl.Markdown(f"**Render error:** {last_error.value}")

def park_chart():
    last_error.value = ""
    try:
        df = agents_df()
        seg_x, seg_y = _segments_from_edges_gdf()

        xmin, ymin = NODE_POS.min(axis=0)
        xmax, ymax = NODE_POS.max(axis=0)
        padx = max((xmax - xmin) * 0.03, 1.0)
        pady = max((ymax - ymin) * 0.03, 1.0)
        x_domain = [float(xmin - padx), float(xmax + padx)]
        y_domain = [float(ymin - pady), float(ymax + pady)]

        fig = go.Figure()

        plume_data = _plume_concentration_grid(x_domain, y_domain) if (plume_show.value and HAZARDS) else None
        risk_data  = _risk_eta_grid(x_domain, y_domain) if risk_show.value else None

        if plume_data is not None:
            gx, gy, conc = plume_data
            show_conc = (conc is not None) and np.isfinite(conc).any() and (float(np.nanmax(conc)) > 0)
            fig.add_trace(go.Heatmap(
                x=gx.tolist(), y=gy.tolist(), z=conc if show_conc else np.zeros((len(gy), len(gx))),
                zsmooth="best", coloraxis="coloraxis", opacity=0.55, showscale=True,
                name="Concentration (g/m^3)", visible=True if show_conc else False
            ))
        else:
            fig.add_trace(go.Heatmap(
                x=[], y=[], z=[], coloraxis="coloraxis", opacity=0.55, showscale=True,
                name="Concentration (g/m^3)", visible=False
            ))
        fig.update_layout(coloraxis=dict(colorscale="YlOrRd"))

        if risk_data is not None:
            rx, ry, rZ = risk_data
            show_risk = (rZ is not None) and np.isfinite(rZ).any()
            fig.add_trace(go.Heatmap(
                x=rx.tolist(), y=ry.tolist(), z=rZ if show_risk else np.zeros((len(ry), len(rx))),
                zsmooth="best", coloraxis="coloraxis2", opacity=0.45, showscale=True,
                name="ETA to safety (s)", visible=True if show_risk else False
            ))
        else:
            fig.add_trace(go.Heatmap(
                x=[], y=[], z=[], coloraxis="coloraxis2", opacity=0.45, showscale=True,
                name="ETA to safety (s)", visible=False
            ))
        fig.update_layout(coloraxis2=dict(colorscale="Blues"))

        fig.add_trace(go.Scatter(
            x=seg_x if seg_x else [], y=seg_y if seg_y else [], mode="lines",
            line=dict(color="rgba(80,80,80,0.5)", width=1),
            hoverinfo="skip", showlegend=False, name="paths", visible=True if (seg_x and seg_y) else False
        ))

        fig.add_trace(go.Scatter(
            x=(df["x"].tolist() if not df.empty else []),
            y=(df["y"].tolist() if not df.empty else []),
            mode="markers",
            marker=dict(
                size=(df["size"].tolist() if not df.empty else []),
                color=(df["color"].tolist() if not df.empty else []),
                line=dict(width=0.5, color="white")
            ),
            hoverinfo="skip", showlegend=False, name="agents",
            visible=False if df.empty else True
        ))

        shapes = []
        hx_list, hy_list = [], []
        for h in HAZARDS:
            cx, cy = float(h["pos"][0]), float(h["pos"][1])
            r_m = float(max(h["r_m"], 5.0))
            rx = meters_to_dx(r_m); ry = meters_to_dy(r_m)
            shapes.append(dict(
                type="circle", xref="x", yref="y",
                x0=cx - rx, y0=cy - ry, x1=cx + rx, y1=cy + ry,
                line=dict(color="red", width=2), fillcolor="rgba(255,0,0,0.15)",
            ))
            hx_list.append(cx); hy_list.append(cy)

        fig.add_trace(go.Scatter(
            x=hx_list, y=hy_list, mode="markers",
            marker=dict(symbol="x", size=11, line=dict(width=2), color="red"),
            hoverinfo="skip", showlegend=True, name="Hazard centers",
            visible=True if hx_list else False
        ))

        safe_pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
        sx_bg, sy_bg, sx_fg, sy_fg = [], [], [], []
        for nid in safe_pool:
            x, y = POS_DICT[nid]
            if nid in FEATURED_SAFE:
                sx_fg.append(x); sy_fg.append(y)
            else:
                sx_bg.append(x); sy_bg.append(y)

        fig.add_trace(go.Scatter(
            x=sx_bg, y=sy_bg, mode="markers",
            marker=dict(size=12, color="rgb(0,90,0)", symbol="circle"),
            name="Safe zones", hoverinfo="skip", showlegend=True,
            visible=True if sx_bg else False
        ))
        fig.add_trace(go.Scatter(
            x=sx_fg, y=sy_fg, mode="markers",
            marker=dict(size=14, color="rgb(0,180,0)", symbol="circle-open"),
            name="Featured safe zones", hoverinfo="skip", showlegend=True,
            visible=True if sx_fg else False
        ))

        px, py = ([], [])
        if qc_show_paths.value:
            px, py = _collect_paths_polylines(evac_only=True, max_agents=1000)
        fig.add_trace(go.Scatter(
            x=px if px else [], y=py if py else [], mode="lines",
            line=dict(width=2, color="orange"),
            hoverinfo="skip", showlegend=True, name="Evac paths",
            visible=True if (px and py) else False
        ))

        # SCDF responders (blue stars)
        rx_, ry_ = [], []
        for nid in RESPONDERS:
            x, y = POS_DICT[nid]
            rx_.append(x); ry_.append(y)
        fig.add_trace(go.Scatter(
            x=rx_, y=ry_, mode="markers",
            marker=dict(size=16, symbol="star", color="blue", line=dict(width=1, color="white")),
            name="SCDF responders", hoverinfo="skip", showlegend=True,
            visible=True if rx_ else False
        ))

        fig.update_layout(
            shapes=shapes,
            xaxis=dict(range=x_domain, showgrid=True, zeroline=False),
            yaxis=dict(range=y_domain, scaleanchor="x", scaleratio=1, showgrid=True, zeroline=False),
            width=1000, height=600,
            margin=dict(t=30, l=40, r=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            clickmode="none",
            uirevision="stay"
        )

        return fig

    except Exception as e:
        last_error.value = f"Chart error: {repr(e)}"
        return go.Figure(layout=go.Layout(
            width=1000, height=600,
            margin=dict(t=30, l=40, r=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            title="Chart failed to render — see 'Render error' text above.",
        ))

@sl.component
def Page():
    _ = tick.value
    chart = park_chart()
    with sl.Column():
        sl.Markdown("## West Coast Park — Evacuation & SCDF Responder Staging (Solara)")
        Controls()
        sl.Markdown(f"**Hazards: {len(HAZARDS)}** | **Safe zones (auto+manual): {len(set(SAFE_NODES)|set(MANUAL_SAFE))}** | **Responders: {len(RESPONDERS)}**")
        sl.Markdown(f"**{kpi_eta_summary()}**")
        sl.FigurePlotly(chart)

page = Page
reset_model()
_recompute_safe_nodes()
_recompute_featured_safe()

# ===================== NICEGUI FRONTEND (runs on localhost) =====================
# This replaces the Solara UI when you run `python wcp_solara_test.py`
# All simulation logic above is reused unchanged.

from nicegui import ui

# Make sure the model is initialised
reset_model()
_recompute_safe_nodes()
_recompute_featured_safe()

# --------- Small helpers for the NiceGUI wrapper ---------
def refresh_status(status_label):
    t = _totals_now()
    status_label.text = (
        f"Tick: {int(tick.value)} | "
        f"Total: {t['total']} | In-envelope: {t['evacuees']} | "
        f"Aware: {t['aware']} | Reached: {t['reached']}"
    )

def refresh_eta(eta_label):
    eta_label.text = kpi_eta_summary()

def redraw(plot):
    # Rebuild the Plotly figure and assign it to the NiceGUI element
    plot.figure = park_chart()
    
    # Optional: if you want to force a re-render, you can also call:
    plot.update()



# -------------------- LAYOUT --------------------
with ui.row().classes('w-full h-screen'):
    # -------- LEFT PANEL: controls, status, text --------
    with ui.column().classes('w-1/3 p-4 gap-2'):
        ui.markdown('## West Coast Park – Evacuation & SCDF (NiceGUI)')
        status_label = ui.label()
        eta_label = ui.label()

        # --- STEP / RUN / RESET ---
        def do_step():
            _step_once()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)

        def toggle_run():
            running.value = not running.value
            btn_run.text = 'Pause' if running.value else 'Run'

        def reset_all():
            running.value = False
            reset_model()
            _recompute_safe_nodes()
            _recompute_featured_safe()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            btn_run.text = 'Run'

        with ui.row().classes('w-full gap-2'):
            ui.button('STEP', on_click=do_step).classes('w-1/3')
            btn_run = ui.button('Run', on_click=toggle_run).classes('w-1/3')
            ui.button('RESET', on_click=reset_all).classes('w-1/3')

        # --- Hazard creation (same _point_from_phrase logic as Solara UI) ---
        ui.markdown('### Hazards')
        location_input = ui.input(
            'Where is the incident?',
            placeholder='e.g. North, NE, 160°, 225°, 300m from center',
        ).classes('w-full')

        def add_hazard():
            phrase = (location_input.value or '').strip()
            pt = _point_from_phrase(phrase)
            if pt is None:
                ui.notify(
                    'Could not parse location. Try: North, South, East, West, NE, NW, SE, SW, or "160°, 300m".',
                    type='warning',
                )
                return

            global HAZARD_ID
            HAZARDS.append({
                'id': HAZARD_ID,
                'pos': np.array([float(pt[0]), float(pt[1])], dtype=float),
                'r_m': float(max(5.0, hazard_radius.value)),
            })
            HAZARD_ID += 1

            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()

            # same behaviour as Solara: everybody evacuates once hazard is declared
            EVAC_NOTIFICATION_TICK.value = int(tick.value)
            for p in PEOPLE:
                p['evac_start_tick'] = int(tick.value)
                p['evac_end_tick'] = None
                p['evac_time_s'] = None
            _force_evacuation_mode()

            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify('Hazard added, agents evacuating.', type='positive')

        def clear_hazards():
            HAZARDS.clear()
            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify('All hazards cleared.', type='positive')

        def remove_last_hazard():
            if not HAZARDS:
                ui.notify('No hazards to remove.', type='warning')
                return
            hid = int(HAZARDS[-1]['id'])
            HAZARDS.pop()
            _recompute_safe_nodes()
            _recompute_featured_safe()
            _choose_targets_and_paths()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify(f'Removed last hazard (ID {hid}).', type='positive')

        with ui.row().classes('w-full gap-2'):
            ui.button('Add Hazard', on_click=add_hazard).classes('w-1/2')
            ui.button('Clear Hazards', on_click=clear_hazards).classes('w-1/2')
        ui.button('Remove Last Hazard', on_click=remove_last_hazard).classes('w-full mt-1')

        # --- Safe zone optimisation & SCDF responders (reuse existing logic) ---
        ui.markdown('### Safe Zones & SCDF')

        def optimise_safe_eta():
            ok = _optimize_safe_zones_by_eta(
                k=N_SAFE,
                attempts=10,
                speed_mps=float(risk_speed_mps.value),
                sample_size=500,
            )
            _recompute_featured_safe()
            _choose_targets_and_paths()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify(
                'Optimised safe zones for min ETA.' if ok else 'Safe-zone optimisation failed.',
                type='positive' if ok else 'warning',
            )

        def suggest_responders():
            ok = _suggest_responders(k=int(N_RESPONDERS.value))
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify(
                'Suggested SCDF responder staging points.' if ok else 'Responder suggestion failed.',
                type='positive' if ok else 'warning',
            )

        def clear_responders():
            RESPONDERS.clear()
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
            ui.notify('Cleared SCDF responders.', type='positive')

        with ui.row().classes('w-full gap-2'):
            ui.button('Optimise Safe Zones (min ETA)', on_click=optimise_safe_eta).classes('w-full')

        with ui.row().classes('w-full gap-2'):
            ui.button('Suggest SCDF Responders', on_click=suggest_responders).classes('w-1/2')
            ui.button('Clear Responders', on_click=clear_responders).classes('w-1/2')

        ui.separator()

        ui.markdown('### Status')
        ui.markdown('**Evacuation ETA summary:**')
        ui.label().bind_text_from(eta_label, 'text')  # mirror text
        ui.markdown('**Counts:**')
        ui.label().bind_text_from(status_label, 'text')

        # initial refresh once UI is created
        refresh_status(status_label)
        refresh_eta(eta_label)

    # -------- RIGHT PANEL: Plotly map --------
    with ui.column().classes('w-2/3 p-4'):
        plot = ui.plotly(park_chart()).classes('w-full h-full')


# -------------------- TIMER: continuous sim when running.value = True --------------------
def timer_tick():
    if running.value:
        _step_once()
        # We need a reference to labels & plot; easiest is to rebuild status via globals
        # (status_label & eta_label live in closure of the UI block above)
        # NiceGUI keeps closures alive, so we can safely access them here.
        # noinspection PyBroadException
        try:
            # These are captured from the UI-building scope
            refresh_status(status_label)
            refresh_eta(eta_label)
            redraw(plot)
        except Exception:
            # Fail silently if something goes wrong – simulation logic remains intact.
            pass

ui.timer(0.05, timer_tick)  # 20 FPS sim when running


# -------------------- RUN ON LOCALHOST ONLY --------------------
ui.run(title='West Coast Park Evac (NiceGUI)', host='127.0.0.1', port=8080)
