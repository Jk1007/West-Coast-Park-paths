# wcp_solara.py — West Coast Park evac (typed location → hazards) + nearest & manual safe zones + risk & density maps
# Run:  python -m solara run wcp_solara.py
import plotly.graph_objects as go
import asyncio
import math
import numpy as np
import pandas as pd
import solara as sl
import osmnx as ox
import networkx as nx
from scipy.spatial import KDTree

# -------------------- LOAD WEST COAST PARK GRAPH --------------------
G = ox.load_graphml("west_coast_park_walk.graphml")   # created by WCP.py
UG = nx.Graph()
for u, v, d in G.to_undirected(as_view=False).edges(data=True):
    w = float(d.get("length", 1.0))
    UG.add_edge(u, v, weight=w)

nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)[["x", "y"]]
NODE_POS = nodes_gdf[["x", "y"]].to_numpy()
NODE_IDS = nodes_gdf.index.to_list()
POS_DICT = {nid: (float(nodes_gdf.loc[nid, "x"]), float(nodes_gdf.loc[nid, "y"])) for nid in NODE_IDS}
KD = KDTree(NODE_POS)
EDGES_GDF = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry"]]

# ---- meters <-> degrees helpers (auto-detect) ----
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

num_people = sl.reactive(200)
pct_cyclists = sl.reactive(15)

# Physical hazard growth (actual circle that advances each step)
hazard_radius = sl.reactive(40.0)     # m (initial radius of a new hazard)
hazard_spread = sl.reactive(1.2)      # m/step (actual circle growth)

# Expected impact envelope (who should be evacuated)
expected_growth_m = sl.reactive(3.0)  # m/step (envelope growth)
expected_buffer_m = sl.reactive(0.0)  # m (static buffer on envelope)

# Wind / awareness
wind_deg = sl.reactive(45.0)          # 0=E, 90=N
wind_speed = sl.reactive(1.0)         # m/s
base_awareness_lag = sl.reactive(8.0)

# Location-input controls (replaces clicking)
location_text = sl.reactive("")

# Gaussian plume controls
plume_show = sl.reactive(True)
plume_Q_gs = sl.reactive(100.0)       # g/s
plume_H_m = sl.reactive(20.0)         # m
plume_stab = sl.reactive("D")         # A..F
plume_range_m = sl.reactive(1500.0)   # m
plume_grid = sl.reactive(90)          # rows

# QC / overlay toggles
qc_show_paths = sl.reactive(False)
qc_km_margin = sl.reactive(25.0)

# ---- RISK (time-to-safe) heatmap ----
risk_show = sl.reactive(True)
risk_grid = sl.reactive(90)            # rows
risk_speed_mps = sl.reactive(1.4)      # assumed evac walking speed for ETA

# ---- PEOPLE DENSITY map/contours (NEW) ----
density_show = sl.reactive(True)       # toggle people-density layer
density_grid = sl.reactive(90)         # rows
density_bandwidth_m = sl.reactive(18.0)  # Gaussian KDE bandwidth (meters)
density_as_contour = sl.reactive(True)   # contour vs heatmap

# storage for QC results
qc_summary = sl.reactive("")
qc_issues  = sl.reactive("")
qc_rows_cache = []

# Per-run state
PEOPLE = []
HAZARDS = []   # dicts: {"id": int, "pos": np.array([x,y]) in map units, "r_m": float}
HAZARD_ID = 0
SAFE_NODES = []    # auto-picked safe zones
N_SAFE = 12        # target count of safe nodes to mark
MANUAL_SAFE = set()    # user-placed safe nodes
FEATURED_SAFE = set()  # highlighted safest (farthest from hazards)

rng = np.random.default_rng(42)

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

def _hazard_radius_at(h): return float(h.get("r_m", 0.0))

# ---- Pasquill–Gifford sigmas ----
def _sigmas_yz(x_m, cls):
    x = max(1.0, float(x_m)); c = cls.upper()
    if c == "A":  sig_y = 0.22 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.20 * x
    elif c == "B":sig_y = 0.16 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.12 * x
    elif c == "C":sig_y = 0.11 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.08 * x * (1 + 0.0002 * x) ** (-0.5)
    elif c == "D":sig_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
    elif c == "E":sig_y = 0.06 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.03 * x * (1 + 0.0003 * x) ** (-1.0)
    else:         sig_y = 0.04 * x * (1 + 0.0001 * x) ** (-0.5); sig_z = 0.016 * x * (1 + 0.0003 * x) ** (-1.0)
    return max(sig_y, 0.5), max(sig_z, 0.5)

# -------------------- LOCATION PHRASE -> MAP POINT --------------------
def _point_from_phrase(phrase: str):
    if not phrase: return None
    p = phrase.strip().lower().replace("-", " ").replace("_", " ")
    p = p.replace("northwest","north west").replace("northeast","north east")
    p = p.replace("southwest","south west").replace("southeast","south east")
    p = p.replace("top right","north east").replace("top left","north west")
    p = p.replace("bottom right","south east").replace("bottom left","south west")
    p = p.replace("centre","center")

    xmin, ymin = float(X_MIN), float(Y_MIN)
    xmax, ymax = float(X_MAX), float(Y_MAX)
    dx = xmax - xmin; dy = ymax - ymin
    mx = dx * 0.12; my = dy * 0.12

    centers = {
        "center": (xmin + dx * 0.5, ymin + dy * 0.5),
        "north":  (xmin + dx * 0.5, ymax - my),
        "south":  (xmin + dx * 0.5, ymin + my),
        "east":   (xmax - mx,       ymin + dy * 0.5),
        "west":   (xmin + mx,       ymin + dy * 0.5),
        "north west": (xmin + mx, ymax - my),
        "north east": (xmax - mx, ymax - my),
        "south west": (xmin + mx, ymin + my),
        "south east": (xmax - mx, ymin + my),
        "nw": (xmin + mx, ymax - my), "ne": (xmax - mx, ymax - my),
        "sw": (xmin + mx, ymin + my), "se": (xmax - mx, ymin + my),
    }
    if p in centers: return np.array(centers[p], dtype=float)

    has_n = "north" in p or p == "n"
    has_s = "south" in p or p == "s"
    has_e = "east"  in p or p == "e"
    has_w = "west"  in p or p == "w"
    if has_n or has_s or has_e or has_w:
        x = xmin + dx * 0.5; y = ymin + dy * 0.5
        if has_w: x = xmin + mx
        if has_e: x = xmax - mx
        if has_s: y = ymin + my
        if has_n: y = ymax - my
        return np.array([x, y], dtype=float)

    if "middle" in p or "centre" in p or "center" in p:
        return np.array(centers["center"], dtype=float)
    return None

# -------------------- SAFE ZONE COMPUTATION --------------------
def _recompute_safe_nodes():
    """Choose N safest nodes (farthest from hazards) as 'safe zones'."""
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
    """Return safe node with smallest network distance from node_id."""
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
    """Pick a few safest (farthest from hazards) from pool to highlight."""
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
    FEATURED_SAFE = set([nid for nid, _ in scored[:max(1, min(4, len(scored)))]])

# -------------------- AFFECTED-ENVELOPE TEST --------------------
def _is_person_expected_affected(pt_xy, t):
    """Return True if person at pt_xy should evacuate under expected envelope at time t."""
    if not HAZARDS: return False
    h, d_m = _nearest_hazard_m(pt_xy)
    if h is None: return False
    expected_r = float(h["r_m"]) + float(expected_growth_m.value) * float(t) + float(expected_buffer_m.value)
    return d_m <= max(0.0, expected_r)

# -------------------- SPAWN / TARGETING / RESET --------------------
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
        })
    PEOPLE.extend(new_people)

def _retarget_to_nearest_safe(p):
    s_idx, s_node = _nearest_node_idx(p["pos"][0], p["pos"][1])
    t_node = _nearest_safe_node_from(s_node)
    p["target_node"] = t_node
    p["path"] = _nx_path(s_node, t_node)
    p["path_idx"] = 0

def _choose_targets_and_paths():
    _recompute_safe_nodes()
    for p in PEOPLE:
        _retarget_to_nearest_safe(p)

def reset_model():
    global PEOPLE
    PEOPLE = []
    _spawn_people(int(num_people.value))
    _choose_targets_and_paths()
    _recompute_featured_safe()
    tick.value += 1

# -------------------- DYNAMICS --------------------
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

    for p in PEOPLE:
        affected = _is_person_expected_affected(p["pos"], tnow)

        if not affected:
            p["aware"] = False
            p["dir"] = np.array([0.0, 0.0])
            continue

        if p["affected_since"] is None:
            p["affected_since"] = tnow
            _retarget_to_nearest_safe(p)

        if not p["aware"]:
            if (tnow - p["affected_since"]) < max(0.0, p["aware_delay"]):
                p["dir"] = np.array([0.0, 0.0])
                continue
            p["aware"] = True

        if p["reached"]:
            p["dir"] = np.array([0.0, 0.0]); continue

        _update_panic(p)

        path = p["path"]; k = p["path_idx"]
        if k >= len(path) - 1:
            p["reached"] = True; p["dir"] = np.array([0.0, 0.0]); continue

        cur_xy = p["pos"]
        next_node = path[k+1]
        nx_xy = POS_DICT[next_node]
        next_xy = np.array([nx_xy[0], nx_xy[1]], dtype=float)

        is_panicking = p["panic"] > p["panic_thr"]
        follow_prob = 0.7 if is_panicking else 0.85
        follow_path = rng.random() < follow_prob

        if follow_path:
            seg = next_xy - cur_xy; dist = np.linalg.norm(seg)
            if dist <= p["speed"]:
                p["pos"] = next_xy.copy(); p["path_idx"] = k + 1; move = next_xy - cur_xy
            else:
                stepv = _unit(seg) * p["speed"]; p["pos"] = cur_xy + stepv; move = stepv
        else:
            seg = next_xy - cur_xy
            direction = _unit(seg)
            deviation = rng.uniform(-0.3 if not is_panicking else -0.5,
                                    0.3 if not is_panicking else 0.5, 2)
            direction = _unit(direction + deviation)
            stepv = direction * p["speed"]; p["pos"] = cur_xy + stepv
            _, snap_idx = KD.query(p["pos"])
            snap_xy = NODE_POS[snap_idx]
            if np.linalg.norm(p["pos"] - snap_xy) < 3.0:
                p["pos"] = snap_xy.copy()
            move = stepv

        mvn = np.linalg.norm(move)
        p["dir"] = move / (mvn + 1e-9)

        if np.linalg.norm(p["pos"] - next_xy) < 2.0 and (p["path_idx"] >= len(path) - 2):
            p["reached"] = True

    tick.value += 1

# -------------------- KPI --------------------
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
            if not affected:
                colors.append("steelblue")  # unaffected, static
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

# -------------------- GAUSSIAN PLUME (HEATMAP) --------------------
def _plume_concentration_grid(x_domain, y_domain):
    try:
        if not plume_show.value or not HAZARDS:
            return None
        nxg = int(max(40, min(180, int(plume_grid.value * 1.2))))
        nyg = int(max(30, min(150, int(plume_grid.value))))
        gx = np.linspace(x_domain[0], x_domain[1], nxg)
        gy = np.linspace(y_domain[0], y_domain[1], nyg)
        X, Y = np.meshgrid(gx, gy)

        a = math.radians(float(wind_deg.value))
        ex, ey = math.cos(a), math.sin(a)
        U = max(0.1, float(wind_speed.value))  # m/s

        Q = float(plume_Q_gs.value)            # g/s
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
            sigy = np.ones_like(x_m); sigz = np.ones_like(x_m)
            xm_clip = np.clip(x_m, 1.0, None)
            idx = np.where(mask)
            sy, sz = _sigmas_yz(xm_clip[idx], stab)
            sigy[idx] = sy; sigz[idx] = sz
            pref = Q / (2.0 * math.pi * U)
            termy = np.exp(-(y_m ** 2) / (2.0 * sigy ** 2))
            termz = np.exp(-((0.0 - H) ** 2) / (2.0 * sigz ** 2)) + np.exp(-((0.0 + H) ** 2) / (2.0 * sigz ** 2))
            denom = sigy * sigz
            contrib = np.zeros_like(C)
            contrib[idx] = pref * termy[idx] * termz[idx] / denom[idx]
            C += contrib
        return gx, gy, C
    except Exception as e:
        last_error.value = f"Plume grid error: {repr(e)}"
        return None

# -------------------- RISK (ETA) HEATMAP --------------------
def _risk_eta_grid(x_domain, y_domain):
    if not risk_show.value:
        return None
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
            dist = _nx_path_length(nid, tnode)     # weighted by length (meters)
            eta = (dist / speed) if np.isfinite(dist) else np.nan
        memo[nid] = eta
        etas[i] = eta
    Z = etas.reshape(X.shape)
    return gx, gy, Z

# -------------------- PEOPLE DENSITY (KDE) --------------------
def _people_density_grid(x_domain, y_domain):
    if not density_show.value:
        return None
    # grid
    ny = int(max(30, min(200, int(density_grid.value))))
    nx = int(max(40, min(240, int(ny * 1.2))))
    gx = np.linspace(x_domain[0], x_domain[1], nx)
    gy = np.linspace(y_domain[0], y_domain[1], ny)
    X, Y = np.meshgrid(gx, gy)

    if not PEOPLE:
        return gx, gy, np.zeros_like(X, dtype=float)

    # Gaussian kernel per agent; convert bandwidth from meters to map units
    bw_m = max(5.0, float(density_bandwidth_m.value))
    bw_x = max(meters_to_dx(bw_m), 1e-6)
    bw_y = max(meters_to_dy(bw_m), 1e-6)
    inv2sx2 = 1.0 / (2.0 * bw_x * bw_x)
    inv2sy2 = 1.0 / (2.0 * bw_y * bw_y)

    Z = np.zeros_like(X, dtype=float)
    # light vectorization: sum of separable Gaussians over agents
    px = np.array([p["pos"][0] for p in PEOPLE], dtype=float)
    py = np.array([p["pos"][1] for p in PEOPLE], dtype=float)

    # compute contributions
    for i in range(px.shape[0]):
        dx2 = (X - px[i]) ** 2
        dy2 = (Y - py[i]) ** 2
        Z += np.exp(-(dx2 * inv2sx2 + dy2 * inv2sy2))

    # scale to [0,1] for stable color mapping
    zmax = float(np.max(Z)) if np.isfinite(Z).any() else 0.0
    if zmax > 0:
        Z = Z / zmax
    return gx, gy, Z

# -------------------- OSM GEOMETRY → PLOTLY SEGMENTS --------------------
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

# -------------------- SAFE-ZONE SUGGESTION (HEURISTIC) --------------------
def _suggest_optimized_safe(k=None):
    """
    Greedy farthest-point seeding using map-space spread among the safest pool,
    then assign MANUAL_SAFE to the chosen set. Goal: reduce mean ETA.
    """
    global MANUAL_SAFE
    k = int(k or max(4, min(12, N_SAFE)))
    if not NODE_POS.size:
        return
    # score by distance from hazards (or from center if no hazards)
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

# -------------------- CHART --------------------
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

        # 1) plume heatmap (coloraxis #1)
        plume_out = _plume_concentration_grid(x_domain, y_domain)
        if plume_out is not None:
            gx, gy, conc = plume_out
            if conc is not None and np.isfinite(conc).any() and (float(np.nanmax(conc)) > 0):
                fig.add_trace(go.Heatmap(
                    x=gx.tolist(), y=gy.tolist(), z=conc,
                    zsmooth="best", coloraxis="coloraxis", opacity=0.55, showscale=True,
                    name="Concentration (g/m^3)"
                ))
                fig.update_layout(coloraxis=dict(colorscale="YlOrRd"))

        # 1b) RISK (ETA) heatmap (coloraxis #2)
        risk_out = _risk_eta_grid(x_domain, y_domain)
        if risk_out is not None:
            rx, ry, rZ = risk_out
            if rZ is not None and np.isfinite(rZ).any():
                fig.add_trace(go.Heatmap(
                    x=rx.tolist(), y=ry.tolist(), z=rZ,
                    zsmooth="best", coloraxis="coloraxis2", opacity=0.45, showscale=True,
                    name="ETA to safety (s)"
                ))
                fig.update_layout(coloraxis2=dict(colorscale="Blues"))

        # 1c) PEOPLE DENSITY (coloraxis #3) — heatmap OR contour
        dens_out = _people_density_grid(x_domain, y_domain)
        if dens_out is not None:
            dx, dy, dZ = dens_out
            if dZ is not None and np.isfinite(dZ).any():
                if bool(density_as_contour.value):
                    fig.add_trace(go.Contour(
                        x=dx.tolist(), y=dy.tolist(), z=dZ,
                        contours=dict(showlines=False),
                        coloraxis="coloraxis3", opacity=0.55, showscale=True,
                        name="Crowd density (norm.)"
                    ))
                else:
                    fig.add_trace(go.Heatmap(
                        x=dx.tolist(), y=dy.tolist(), z=dZ,
                        zsmooth="best", coloraxis="coloraxis3", opacity=0.40, showscale=True,
                        name="Crowd density (norm.)"
                    ))
                fig.update_layout(coloraxis3=dict(colorscale="Greens"))

        # 2) park edges
        if seg_x and seg_y:
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode="lines",
                line=dict(color="rgba(80,80,80,0.5)", width=1),
                hoverinfo="skip", showlegend=False, name="paths"
            ))

        # 3) agents
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["x"].tolist(), y=df["y"].tolist(), mode="markers",
                marker=dict(size=df["size"].tolist(), color=df["color"].tolist(), line=dict(width=0.5, color="white")),
                hoverinfo="skip", showlegend=False, name="agents"
            ))

        # 4) hazard circles
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

        # 5) safe zones (dark green = pool, light green open circle = featured)
        safe_pool = list(set(SAFE_NODES) | set(MANUAL_SAFE))
        if safe_pool:
            sx_bg, sy_bg, sx_fg, sy_fg = [], [], [], []
            for nid in safe_pool:
                x, y = POS_DICT[nid]
                if nid in FEATURED_SAFE:
                    sx_fg.append(x); sy_fg.append(y)
                else:
                    sx_bg.append(x); sy_bg.append(y)
            if sx_bg:
                fig.add_trace(go.Scatter(
                    x=sx_bg, y=sy_bg, mode="markers",
                    marker=dict(size=10, color="rgb(0,90,0)", symbol="circle"),
                    name="safe_bg", hoverinfo="skip", showlegend=False
                ))
            if sx_fg:
                fig.add_trace(go.Scatter(
                    x=sx_fg, y=sy_fg, mode="markers",
                    marker=dict(size=12, color="rgb(0,180,0)", symbol="circle-open"),
                    name="safe_featured", hoverinfo="skip", showlegend=False
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

        if hx_list:
            fig.add_trace(go.Scatter(
                x=hx_list, y=hy_list, mode="markers",
                marker=dict(symbol="x", size=11, line=dict(width=2), color="red"),
                hoverinfo="skip", showlegend=False, name="hazard_centers"
            ))

        # optional evac-path overlays
        try:
            show_paths = qc_show_paths.value
            if show_paths:
                px, py = _collect_paths_polylines(evac_only=True, max_agents=1000)
                if px and py:
                    fig.add_trace(go.Scatter(
                        x=px, y=py, mode="lines",
                        line=dict(width=1, color="green"),
                        hoverinfo="skip", showlegend=False, name="evac_paths"
                    ))
        except Exception:
            pass

        return fig

    except Exception as e:
        last_error.value = f"Chart error: {repr(e)}"
        return go.Figure(layout=go.Layout(
            width=1000, height=600,
            margin=dict(t=30, l=40, r=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            title="Chart failed to render — see 'Render error' text above.",
        ))

# -------------------- QUALITY CHECK HELPERS --------------------
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

def qc_run(expected_margin_m=25.0):
    summaries = []
    issues = []
    rows = []

    # A. Safe zone checks
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

    # B. Graph reachability
    tnow = float(tick.value)
    evacuees = [p for p in PEOPLE if _is_person_expected_affected(p["pos"], tnow)]
    summaries.append(f"Evacuees detected now: {len(evacuees)} / {len(PEOPLE)} total")

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

    # D. Non-affected shouldn't move
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

    # --- QC panel ---
    sl.Markdown("### Quality Check")

    def on_qc_run():
        summaries, problems, rows = qc_run(expected_margin_m=float(qc_km_margin.value))
        qc_summary.value = "\n".join([f"- {s}" for s in summaries])
        qc_issues.value  = "\n".join([f"- {p}" for p in problems])
        qc_rows_cache.clear()
        qc_rows_cache.extend(rows)
        tick.value += 1

    def on_export_csv():
        if not qc_rows_cache:
            sl.notify("Run QC first; nothing to export.", timeout=3000); 
            return
        df = pd.DataFrame(qc_rows_cache)
        path = "/mnt/data/qc_routes.csv"
        df.to_csv(path, index=False)
        sl.Markdown(f"[Download QC CSV](sandbox:{path})")

    with sl.Row():
        sl.SliderFloat("Flag safe-zone if within X m of a hazard", min=0.0, max=200.0, value=qc_km_margin)
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

    def on_submit_location():
        txt = (location_text.value or "").strip()
        pt = _point_from_phrase(txt)
        if pt is None:
            sl.notify("Could not parse location. Try: North, South, East, West, North West, NE, Center", timeout=4000)
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
        tick.value += 1

    # --- Manual Safe Zones + Optimize ---
    safe_phrase = sl.reactive("")
    def _add_manual_safe_by_phrase(phrase: str):
        pt = _point_from_phrase(phrase)
        if pt is None: return False
        _, node_id = _nearest_node_idx(pt[0], pt[1])
        MANUAL_SAFE.add(node_id)
        return True

    def on_add_safe():
        txt = (safe_phrase.value or "").strip()
        if not txt:
            sl.notify("Enter a phrase like 'North West', 'Center', 'SE'.", timeout=3000); return
        ok = _add_manual_safe_by_phrase(txt)
        if not ok:
            sl.notify("Could not parse phrase; try: North, South, East, West, NE, NW, SE, SW, Center.", timeout=4000); return
        _recompute_featured_safe()
        _choose_targets_and_paths()
        tick.value += 1

    def on_clear_safe():
        MANUAL_SAFE.clear()
        _recompute_featured_safe()
        _choose_targets_and_paths()
        tick.value += 1

    def on_optimize():
        _suggest_optimized_safe(k=N_SAFE)
        _recompute_featured_safe()
        _choose_targets_and_paths()
        tick.value += 1

    with sl.Column():
        sl.Markdown("### Controls")
        with sl.Row():
            sl.Button("STEP", on_click=on_step)
            sl.Button("START" if not running.value else "PAUSE", on_click=on_toggle)
            sl.Button("RESET", on_click=on_reset)
            sl.Markdown(f"**Tick:** {int(tick.value)}")

        sl.Markdown("**Where is the incident?** (e.g., `North West`, `South`, `Center`, `NE`)")
        with sl.Row():
            sl.InputText(label="Where is it:", value=location_text, continuous_update=False, placeholder="North West")
            sl.Button("Submit", on_click=on_submit_location)
            sl.Button("Clear Hazards", on_click=on_clear_hazards)

        with sl.Row():
            sl.InputInt("People (initial)", value=num_people)
            sl.InputInt("Cyclists (%)", value=pct_cyclists)

        with sl.Row():
            sl.SliderFloat("Hazard radius (new, m)", min=5.0, max=200.0, value=hazard_radius)
            sl.SliderFloat("Radius spread / step (actual m/step)", min=0.0, max=10.0, value=hazard_spread)

        with sl.Row():
            sl.SliderFloat("Expected growth (m/step)", min=0.0, max=20.0, value=expected_growth_m)
            sl.SliderFloat("Expected safety buffer (m)", min=0.0, max=100.0, value=expected_buffer_m)

        with sl.Row():
            sl.SliderFloat("Wind (deg)", min=0.0, max=360.0, value=wind_deg)
            sl.SliderFloat("Wind speed (m/s)", min=0.0, max=10.0, value=wind_speed)

        with sl.Row():
            sl.SliderFloat("Base awareness lag (s)", min=0.0, max=40.0, value=base_awareness_lag)

        with sl.Row():
            sl.Checkbox(label="Show plume", value=plume_show)
            sl.Select(label="Stability (A–F)", value=plume_stab, values=["A","B","C","D","E","F"])
            sl.SliderFloat("Q (g/s)", min=1.0, max=1000.0, value=plume_Q_gs)
            sl.SliderFloat("Stack height H (m)", min=0.0, max=100.0, value=plume_H_m)
        with sl.Row():
            sl.SliderFloat("Plume range (m)", min=200.0, max=3000.0, value=plume_range_m)
            sl.InputInt("Heatmap resolution", value=plume_grid)

        # ---- Risk heatmap controls ----
        sl.Markdown("**Risk (ETA) heatmap**")
        with sl.Row():
            sl.Checkbox(label="Show risk (ETA) heatmap", value=risk_show)
            sl.InputInt("Risk grid rows", value=risk_grid)
            sl.SliderFloat("Risk walking speed (m/s)", min=0.5, max=3.0, value=risk_speed_mps)

        # ---- People density controls ----
        sl.Markdown("**People density (KDE) layer**")
        with sl.Row():
            sl.Checkbox(label="Show people density", value=density_show)
            sl.Checkbox(label="Use contours (else heatmap)", value=density_as_contour)
        with sl.Row():
            sl.InputInt("Density grid rows", value=density_grid)
            sl.SliderFloat("Density bandwidth (m)", min=5.0, max=60.0, value=density_bandwidth_m)

        # ---- Manual / optimized safe zones ----
        sl.Markdown("**Safe Zones — manual & optimize**")
        with sl.Row():
            sl.InputText(label="Add Safe Zone (phrase)", value=safe_phrase, continuous_update=False, placeholder="North East")
            sl.Button("Add Safe Zone Here", on_click=on_add_safe)
            sl.Button("Clear Safe Zones", on_click=on_clear_safe)
            sl.Button("Suggest (optimize) Safe Zones", on_click=on_optimize)

        if last_error.value:
            sl.Markdown(f"**Render error:** {last_error.value}")

@sl.component
def Page():
    _ = tick.value  # subscribe to re-render
    chart = park_chart()
    with sl.Column():
        sl.Markdown("## West Coast Park — Evacuation (Solara)")
        Controls()
        sl.Markdown(f"**Hazards: {len(HAZARDS)}** | **Safe zones (auto+manual): {len(set(SAFE_NODES)|set(MANUAL_SAFE))}**")
        sl.Markdown(f"**{kpi_eta_summary()}**")
        sl.FigurePlotly(chart)

# Solara entry point
page = Page
reset_model()
_recompute_safe_nodes()
_recompute_featured_safe()
