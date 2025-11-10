# wcp_solara.py — West Coast Park crowd evac, Solara UI (no mesa.visualization imports)
# Run with:  python -m solara run wcp_solara.py
import plotly.graph_objects as go
import asyncio
import math
import numpy as np
import pandas as pd
import altair as alt
import solara as sl
import osmnx as ox
import networkx as nx
from scipy.spatial import KDTree

# -------------------- LOAD WEST COAST PARK GRAPH --------------------
G = ox.load_graphml("west_coast_park_walk.graphml")   # created by WCP.py
# Use an undirected view for simple shortest-paths
UG = nx.Graph()
for u, v, d in G.to_undirected(as_view=False).edges(data=True):
    w = float(d.get("length", 1.0))
    UG.add_edge(u, v, weight=w)

# Node positions (projected graph from WCP.py means metres)
pos = ox.graph_to_gdfs(G, nodes=True, edges=False)[["x", "y"]]
NODE_POS = pos[["x", "y"]].to_numpy()
NODE_IDS = pos.index.to_list()
KD = KDTree(NODE_POS)

# -------------------- REACTIVE SIM STATE --------------------
tick = sl.reactive(0)
running = sl.reactive(False)

# Controls (reactive so you can tweak then press Reset)
num_people = sl.reactive(200)
pct_cyclists = sl.reactive(15)        # %
hazard_x = sl.reactive(float(NODE_POS[:,0].mean()))
hazard_y = sl.reactive(float(NODE_POS[:,1].mean()))
hazard_radius = sl.reactive(40.0)
hazard_spread = sl.reactive(1.2)
wind_deg = sl.reactive(45.0)          # 0=E, 90=N
wind_speed = sl.reactive(1.0)         # m/s
base_awareness_lag = sl.reactive(8.0)

# Per-run state (re-initialised by reset_model)
PEOPLE = []
PATHS = []
HAZARD_POS = None
HAZARD_R = None

rng = np.random.default_rng(42)

def _nx_path(u_node, v_node):
    try:
        return nx.shortest_path(UG, u_node, v_node, weight="weight")
    except nx.NetworkXNoPath:
        return [u_node, v_node]

def _nearest_node_idx(x, y):
    _, idx = KD.query([x, y])
    return idx, NODE_IDS[idx]

def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def reset_model():
    global PEOPLE, PATHS, HAZARD_POS, HAZARD_R
    PEOPLE = []
    PATHS = []

    hz = np.array([hazard_x.value, hazard_y.value], dtype=float)
    HAZARD_POS = hz.copy()
    HAZARD_R = float(hazard_radius.value)

    # spawn agents on nodes (with small jitter)
    n = int(num_people.value)
    cyclists_target = int(round(n * float(pct_cyclists.value)/100.0))
    cycl_flags = np.array([True]*cyclists_target + [False]*(n - cyclists_target))
    rng.shuffle(cycl_flags)

    for i in range(n):
        idx = rng.integers(0, NODE_POS.shape[0])
        px, py = NODE_POS[idx]
        pos2d = np.array([px, py]) + rng.uniform(-1.5, 1.5, 2)

        is_cyclist = bool(cycl_flags[i])
        base_speed = rng.normal(4.5, 0.6) if is_cyclist else rng.normal(2.0, 0.4)
        base_speed = float(np.clip(base_speed, 3.0 if is_cyclist else 1.2, 6.0 if is_cyclist else 3.5))

        panic_threshold = float(rng.uniform(0.3, 0.8))
        dist_from_hz = float(np.linalg.norm(pos2d - hz))
        aware_delay = float(base_awareness_lag.value + (dist_from_hz / 100.0) * 15.0 + rng.uniform(-2, 2))
        aware_delay = float(np.clip(aware_delay, 5.0, 35.0))

        PEOPLE.append({
            "pos": pos2d,
            "dir": np.array([0.0, 0.0]),
            "is_cyclist": is_cyclist,
            "base_speed": base_speed,
            "speed": base_speed,
            "panic": 0.0,
            "panic_thr": panic_threshold,
            "aware": False,
            "aware_delay": aware_delay,
            "reached": False,
            "target_node": None,
            "path": [],
            "path_idx": 0,
        })

    # assign targets: any perimeter-ish node (cheap heuristic) → pick farthest from hazard
    # build a shortlist by sampling nodes near the convex hull boundary via degree or distance
    dists = np.linalg.norm(NODE_POS - hz, axis=1)
    candidates = np.argsort(-dists)[:80]  # farthest 80 nodes
    # now assign each person to the closest among top-80 by shortest-path distance
    for p in PEOPLE:
        s_idx, s_node = _nearest_node_idx(p["pos"][0], p["pos"][1])
        # choose best target by euclidean as fast proxy (works well enough)
        t_idx = int(candidates[rng.integers(0, len(candidates))])
        t_node = NODE_IDS[t_idx]
        p["target_node"] = t_node
        p["path"] = _nx_path(s_node, t_node)
        p["path_idx"] = 0

    tick.value = 0

def _update_panic(p, hz_pos, hz_r):
    d = np.linalg.norm(p["pos"] - hz_pos)
    if d < hz_r * 1.5:
        p["panic"] = min(1.0, p["panic"] + (1 - d/(hz_r*1.5)) * 0.05)
    else:
        p["panic"] = max(0.0, p["panic"] - 0.01)
    if (not p["is_cyclist"]) and (p["panic"] > p["panic_thr"]):
        p["speed"] = p["base_speed"] * (1 + p["panic"] * 0.3)
    else:
        p["speed"] = p["base_speed"]

def _step_once():
    # advance hazard
    global HAZARD_POS, HAZARD_R
    a = math.radians(float(wind_deg.value))
    v = np.array([math.cos(a), math.sin(a)])
    HAZARD_POS = HAZARD_POS + v * float(wind_speed.value)
    HAZARD_R = HAZARD_R + float(hazard_spread.value)

    t = float(tick.value)
    # update each agent
    for p in PEOPLE:
        if not p["aware"] and t > p["aware_delay"]:
            p["aware"] = True

        if p["reached"]:
            p["dir"] = np.array([0.0, 0.0])
            continue

        if not p["aware"]:
            p["dir"] = np.array([0.0, 0.0])
            continue

        _update_panic(p, HAZARD_POS, HAZARD_R)

        path = p["path"]
        k = p["path_idx"]
        if k >= len(path) - 1:
            # at final node—treat as reached
            p["reached"] = True
            p["dir"] = np.array([0.0, 0.0])
            continue

        # Move toward next path node (with occasional deviation when panicking)
        cur_xy = p["pos"]
        next_node = path[k+1]
        next_xy = np.array([G.nodes[next_node]["x"], G.nodes[next_node]["y"]], dtype=float)

        is_panicking = p["panic"] > p["panic_thr"]
        follow_prob = 0.7 if is_panicking else 0.85
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
            # deviate but tend back to next node
            seg = next_xy - cur_xy
            direction = _unit(seg)
            deviation = rng.uniform(-0.3 if not is_panicking else -0.5,
                                    0.3 if not is_panicking else 0.5, 2)
            direction = _unit(direction + deviation)
            stepv = direction * p["speed"]
            p["pos"] = cur_xy + stepv
            # snap back to nearest node if very close
            _, snap_idx = KD.query(p["pos"])
            snap_xy = NODE_POS[snap_idx]
            if np.linalg.norm(p["pos"] - snap_xy) < 3.0:
                p["pos"] = snap_xy.copy()
            move = stepv

        mvn = np.linalg.norm(move)
        p["dir"] = move / (mvn + 1e-9)

        # reached target proximity?
        if np.linalg.norm(p["pos"] - next_xy) < 2.0 and (p["path_idx"] >= len(path) - 2):
            p["reached"] = True

    tick.value += 1

def agents_df():
    xs = []
    ys = []
    colors = []
    sizes = []
    for p in PEOPLE:
        xs.append(p["pos"][0])
        ys.append(p["pos"][1])
        if p["reached"]:
            colors.append("green")
        elif p["aware"]:
            # danger if inside ~1.5*radius
            colors.append("purple" if np.linalg.norm(p["pos"] - HAZARD_POS) < 1.5*HAZARD_R else "red")
        else:
            colors.append("lightgray")
        sizes.append(10 if p["is_cyclist"] else 7)
    return pd.DataFrame({"x": xs, "y": ys, "color": colors, "size": sizes})

def park_chart():
    df = agents_df()

    # --- build edge segments from OSM geometries ---
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    seg_x = []
    seg_y = []
    for geom in edges_gdf.geometry.values:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            coords = list(geom.coords)
            for i in range(len(coords) - 1):
                (x1, y1) = coords[i]
                (x2, y2) = coords[i + 1]
                seg_x += [x1, x2, None]
                seg_y += [y1, y2, None]
        else:  # MultiLineString
            for line in geom.geoms:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    (x1, y1) = coords[i]
                    (x2, y2) = coords[i + 1]
                    seg_x += [x1, x2, None]
                    seg_y += [y1, y2, None]

    # axis domains from node extents
    xmin, ymin = NODE_POS.min(axis=0)
    xmax, ymax = NODE_POS.max(axis=0)
    padx = (xmax - xmin) * 0.03
    pady = (ymax - ymin) * 0.03
    x_domain = [xmin - padx, xmax + padx]
    y_domain = [ymin - pady, ymax + pady]

    fig = go.Figure()

    # park edges
    fig.add_trace(go.Scatter(
        x=seg_x, y=seg_y, mode="lines",
        line=dict(color="rgba(80,80,80,0.5)", width=1),
        name="paths", hoverinfo="skip", showlegend=False
    ))

    # agents
    fig.add_trace(go.Scatter(
        x=df["x"], y=df["y"], mode="markers",
        marker=dict(
            size=df["size"],
            color=df["color"],
            line=dict(width=0.5, color="white"),
        ),
        hoverinfo="skip", showlegend=False
    ))

    # hazard circle
    fig.update_layout(
        shapes=[
            dict(
                type="circle",
                xref="x", yref="y",
                x0=HAZARD_POS[0] - HAZARD_R,
                y0=HAZARD_POS[1] - HAZARD_R,
                x1=HAZARD_POS[0] + HAZARD_R,
                y1=HAZARD_POS[1] + HAZARD_R,
                line=dict(color="red", width=2),
                fillcolor="rgba(255,0,0,0.15)",
            )
        ],
        xaxis=dict(range=x_domain, showgrid=True, zeroline=False),
        yaxis=dict(range=y_domain, scaleanchor="x", scaleratio=1, showgrid=True, zeroline=False),
        width=1000, height=600,
        margin=dict(t=30, l=40, r=10, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# -------------------- UI COMPONENTS --------------------
@sl.component
def Controls():
    def on_step():
        _step_once()

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

    with sl.Column():  # keep compatible with old Solara
        sl.Markdown("### Controls")
        with sl.Row():
            sl.Button("Step", on_click=on_step)
            sl.Button("Start" if not running.value else "Pause", on_click=on_toggle)
            sl.Button("Reset", on_click=on_reset)
            sl.Markdown(f"**Tick:** {int(tick.value)}")

        with sl.Row():
            sl.InputInt("People", value=num_people)
            sl.InputInt("Cyclists (%)", value=pct_cyclists)

        with sl.Row():
            sl.SliderFloat("Hazard radius", min=5.0, max=120.0, value=hazard_radius)
            sl.SliderFloat("Radius spread / step", min=0.0, max=4.0, value=hazard_spread)

        with sl.Row():
            sl.SliderFloat("Wind (deg)", min=0.0, max=360.0, value=wind_deg)
            sl.SliderFloat("Wind speed (m/s)", min=0.0, max=5.0, value=wind_speed)

        with sl.Row():
            sl.SliderFloat("Base awareness lag (s)", min=0.0, max=40.0, value=base_awareness_lag)



@sl.component
def Page():
    _ = tick.value      # subscribe to changes so the chart recomputes
    chart = park_chart()
    with sl.Column():
        sl.Markdown("## West Coast Park — Evacuation (Solara)")
        Controls()
        sl.FigurePlotly(chart)





# Solara entry point
page = Page

# Build once on import so Reset works immediately
reset_model()
