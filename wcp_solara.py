# wcp_solara.py — West Coast Park evac (typed location → hazards) + expected impact envelope
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
hazard_radius = sl.reactive(40.0)     # meters (initial radius of new hazard)
hazard_spread = sl.reactive(1.2)      # meters per step (actual circle growth)

# Expected impact envelope (who should be evacuated)
expected_growth_m = sl.reactive(3.0)  # meters per step (envelope growth)
expected_buffer_m = sl.reactive(0.0)  # meters (static buffer on envelope)

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

# Per-run state
PEOPLE = []
HAZARDS = []   # dicts: {"id": int, "pos": np.array([x,y]) in map units, "r_m": float}
HAZARD_ID = 0
rng = np.random.default_rng(42)

# -------------------- HELPERS --------------------
def _nx_path(u_node, v_node):
    try: return nx.shortest_path(UG, u_node, v_node, weight="weight")
    except nx.NetworkXNoPath: return [u_node, v_node]

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

# ---- Pasquill–Gifford sigmas (quick rural approximations) ----
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

# -------------------- AFFECTED-ENVELOPE TEST --------------------
def _is_person_expected_affected(pt_xy, t):
    """Return True if person at pt_xy should evacuate under expected envelope at time t."""
    if not HAZARDS: return False
    h, d_m = _nearest_hazard_m(pt_xy)
    if h is None: return False
    # envelope grows independently of the physical hazard (can be conservative)
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
        # awareness delay is only used once a person is in the affected envelope
        new_people.append({
            "pos": pos2d, "dir": np.array([0.0, 0.0]),
            "is_cyclist": is_cyclist, "base_speed": base_speed, "speed": base_speed,
            "panic": 0.0, "panic_thr": panic_threshold,
            "aware": False, "aware_delay": float(base_awareness_lag.value + rng.uniform(-2, 2)),
            "reached": False, "target_node": None, "path": [], "path_idx": 0,
        })
    PEOPLE.extend(new_people)

def _choose_targets_and_paths():
    if len(NODE_POS) == 0 or not PEOPLE: return
    # Safety score: farthest from hazards
    if HAZARDS:
        safety = np.empty(NODE_POS.shape[0], dtype=float)
        for i, (x, y) in enumerate(NODE_POS):
            pt = np.array([x, y], dtype=float)
            _, mn = _nearest_hazard_m(pt)
            safety[i] = mn
    else:
        center = NODE_POS.mean(axis=0)
        safety = np.array([map_distance_m(pt, center) for pt in NODE_POS], dtype=float)
    candidates = np.argsort(-safety)[:100]
    for p in PEOPLE:
        _, s_node = _nearest_node_idx(p["pos"][0], p["pos"][1])
        t_idx = int(candidates[rng.integers(0, len(candidates))])
        t_node = NODE_IDS[t_idx]
        p["target_node"] = t_node
        p["path"] = _nx_path(s_node, t_node)
        p["path_idx"] = 0

def reset_model():
    global PEOPLE
    PEOPLE = []
    _spawn_people(int(num_people.value))
    _choose_targets_and_paths()
    tick.value += 1

# -------------------- DYNAMICS --------------------
def _update_panic(p):
    # Panic only matters for evacuating people
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

def _step_once():
    _advance_hazards()
    tnow = float(tick.value)

    for p in PEOPLE:
        # decide if this person should be evacuated under expected envelope
        affected = _is_person_expected_affected(p["pos"], tnow)

        if not affected:
            # unaffected: do not move
            p["aware"] = False
            p["dir"] = np.array([0.0, 0.0])
            continue

        # affected path-goers: honor awareness lag *after* becoming affected
        if not p["aware"]:
            # time-based lag: start moving once tick surpasses per-person lag since first affected
            # we store the first-affected tick in the person dict the first time they’re affected
            if "affected_since" not in p:
                p["affected_since"] = tnow
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
                colors.append("steelblue")  # unaffected, do not move
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
        if p["reached"] or not p.get("aware", False):  # only evacuating folks
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
            y_m = -dx_m * ey + dy_m * ey*0 + dy_m * (1.0)  # keep sign consistent; y crosswind
            y_m = -dx_m * ey + dy_m * ex  # correct rotation
            mask = (x_m > 1.0) & (x_m <= maxx)
            if not np.any(mask): continue
            sigy = np.ones_like(x_m); sigz = np.ones_like(x_m)
            xm_clip = np.clip(x_m, 1.0, None)
            idx = np.where(mask)
            sy, sz = _sigmas_yz(xm_clip[idx], stab)
            sigy[idx] = sy; sigz[idx] = sz
            pref = Q / (2.0 * math.pi * U)
            termy = np.exp(-(y_m ** 2) / (2.0 * sigy ** 2))
            termz = np.exp(-(H ** 2) / (2.0 * sigz ** 2)) + np.exp(-(H ** 2) / (2.0 * sigz ** 2))
            denom = sigy * sigz
            contrib = np.zeros_like(C)
            contrib[idx] = pref * termy[idx] / denom[idx]
            C += contrib
        return gx, gy, C
    except Exception as e:
        last_error.value = f"Plume grid error: {repr(e)}"
        return None

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

        # 1) plume heatmap
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

        # 4) hazard shapes + epicentres
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

        return fig

    except Exception as e:
        last_error.value = f"Chart error: {repr(e)}"
        return go.Figure(layout=go.Layout(
            width=1000, height=600,
            margin=dict(t=30, l=40, r=10, b=40),
            paper_bgcolor="white", plot_bgcolor="white",
            title="Chart failed to render — see 'Render error' text above.",
        ))

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

    def on_reset():
        running.value = False
        reset_model()

    def on_clear_hazards():
        HAZARDS.clear()
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

        if last_error.value:
            sl.Markdown(f"**Render error:** {last_error.value}")

@sl.component
def Page():
    _ = tick.value  # subscribe to re-render
    chart = park_chart()
    with sl.Column():
        sl.Markdown("## West Coast Park — Evacuation (Solara)")
        Controls()
        sl.Markdown(f"**Hazards: {len(HAZARDS)}**")
        sl.Markdown(f"**{kpi_eta_summary()}**")
        sl.FigurePlotly(chart)

# Solara entry point
page = Page
reset_model()
