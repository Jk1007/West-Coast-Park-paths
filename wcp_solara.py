# --- CRITICAL FIX: Set Backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt

import math
import random
import os
import requests
import numpy as np
import networkx as nx
import zipfile
from nicegui import ui, app, run

# --- Library Check ---
try:
    import shapefile  # pyshp
    HAS_PYSHP = True
except ImportError:
    HAS_PYSHP = False

# --- Configuration ---
WIND_DIR_URL = "https://api-open.data.gov.sg/v2/real-time/api/wind-direction"
WIND_SPD_URL = "https://api-open.data.gov.sg/v2/real-time/api/wind-speed"
TARGET_STATIONS = ["Sentosa", "Clementi", "Banyan"] 

# File Paths
ZIP_FILE_PATH = "Footpath_Aug2025.zip"
SHAPEFILE_DIR = "Footpath_Aug2025"
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, "Footpath.shp")
GRAPH_FILES = ["west_coast_park_walk.graphml", "west_coast_park_walk_clean.graphml"]

# Simulation Constants
SIM_DT = 0.5            
AGENT_SPEED_WALK = 1.4  
AGENT_SPEED_RUN = 3.0   
PLUME_WIDTH = 150.0     
PLUME_COST_FACTOR = 8.0 

# Geo Projection (Singapore)
REF_LAT = 1.29
REF_LON = 103.77
R_EARTH = 6378137.0

def latlon_to_xy(lat, lon):
    y = (lat - REF_LAT) * (math.pi / 180.0) * R_EARTH
    x = (lon - REF_LON) * (math.pi / 180.0) * R_EARTH * math.cos(math.radians(REF_LAT))
    return x, y

# --- Helper: Auto-Unzip ---
def check_and_unzip():
    """Checks for map files and unzips if necessary."""
    # Check for GraphML first
    for g in GRAPH_FILES:
        if os.path.exists(g):
            return True, f"Found GraphML: {g}"

    if os.path.exists(SHAPEFILE_PATH):
        return True, "Shapefile found."
    
    if os.path.exists(ZIP_FILE_PATH):
        try:
            with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
            return True, f"Unzipped {ZIP_FILE_PATH}."
        except Exception as e:
            return False, f"Error unzipping: {e}"

    return False, "No Map Files Found. Please upload GraphML or Shapefile."

# --- Wind Logic (IDW Interpolation) ---

def extract_station_data(api_response, station_name):
    if not api_response: return None
    stations, readings = [], []
    
    if 'data' in api_response and isinstance(api_response['data'], dict):
        stations = api_response['data'].get('stations', [])
        r_list = api_response['data'].get('readings', [])
        if r_list: readings = r_list[0].get('data', [])
    elif 'metadata' in api_response:
        stations = api_response['metadata'].get('stations', [])
        if 'items' in api_response and api_response['items']:
            readings = api_response['items'][0].get('readings', [])
            
    target_id = None
    lat, lon = 0.0, 0.0
    for st in stations:
        if station_name.lower() in st.get('name', '').lower():
            target_id = st.get('id')
            loc = st.get('location', {})
            lat = loc.get('latitude', 0.0)
            lon = loc.get('longitude', 0.0)
            break
            
    if not target_id: return None

    for r in readings:
        if r.get('stationId') == target_id:
            val = r.get('value')
            return (val, lat, lon)
    return None

def fetch_wind_sync(map_center_xy):
    try:
        r_spd = requests.get(WIND_SPD_URL, timeout=3).json()
        r_dir = requests.get(WIND_DIR_URL, timeout=3).json()
    except Exception as e:
        print(f"Wind Fetch Error: {e}")
        return None, 0, 0

    station_data = []

    for name in TARGET_STATIONS:
        spd_data = extract_station_data(r_spd, name)
        dir_data = extract_station_data(r_dir, name)
        
        if spd_data and dir_data:
            s_knots, lat, lon = spd_data
            d_deg, _, _ = dir_data
            
            s_ms = float(s_knots) * 0.514444
            sx, sy = latlon_to_xy(lat, lon)
            
            cx, cy = map_center_xy
            dist = math.hypot(sx - cx, sy - cy)
            if dist < 1.0: dist = 1.0
            
            station_data.append({
                'speed': s_ms,
                'dir': float(d_deg),
                'weight': 1.0 / (dist ** 2) 
            })

    if not station_data: 
        return (0.0, -1.0), 0.0, 0.0

    total_weight = sum(s['weight'] for s in station_data)
    avg_speed = sum(s['speed'] * s['weight'] for s in station_data) / total_weight
    
    sum_x = 0.0
    sum_y = 0.0
    for s in station_data:
        rad = math.radians(s['dir'])
        u = math.sin(rad) 
        v = math.cos(rad) 
        sum_x += u * s['weight']
        sum_y += v * s['weight']
        
    avg_x = sum_x / total_weight
    avg_y = sum_y / total_weight
    avg_deg = math.degrees(math.atan2(avg_x, avg_y)) % 360
    
    rad = math.radians(avg_deg)
    vx = -math.sin(rad)
    vy = -math.cos(rad)
    
    norm = math.hypot(vx, vy)
    if norm > 0: vx, vy = vx/norm, vy/norm
    
    return (vx, vy), avg_speed, avg_deg

# --- Simulation Logic ---
class Simulation:
    def __init__(self):
        self.G = None
        self.pos = {}
        self.agents = []
        self.hazards = []
        self.safe_nodes = []
        self.plume_vec = (0, -1)
        self.wind_speed = 0
        self.running = False
        self.map_center = (0, 0)
        self.bounds = (0, 0, 0, 0)
        self.loaded_file = "None"
        
    def load_graph(self, log_callback=print):
        # Auto-Unzip/Check
        found, msg = check_and_unzip()
        log_callback(msg)
        
        files = os.listdir('.')
        
        # 1. Try GraphML FIRST (Reverted Priority)
        found_xml = None
        for f in GRAPH_FILES:
            if os.path.exists(f):
                found_xml = f
                break
        
        if found_xml:
            self.loaded_file = found_xml
            log_callback(f"Loading GraphML: {found_xml}")
            return self._process_networkx_graph(nx.read_graphml(found_xml), log_callback)

        # 2. Fallback to Shapefile
        if os.path.exists(SHAPEFILE_PATH):
            if not HAS_PYSHP:
                return False, "Shapefile found but 'pyshp' missing. Run: pip install pyshp"
            return self.load_shapefile_pyshp(SHAPEFILE_PATH, log_callback)

        return False, "ERROR: No map files found."

    def load_shapefile_pyshp(self, path, log_callback):
        self.loaded_file = path
        log_callback(f"Loading Shapefile: {path}")
        try:
            sf = shapefile.Reader(path)
            G = nx.Graph()
            
            for shape in sf.shapes():
                if shape.shapeType not in [3, 13, 23]: 
                    continue 
                
                parts = list(shape.parts) + [len(shape.points)]
                for i in range(len(parts) - 1):
                    start_idx = parts[i]
                    end_idx = parts[i+1]
                    segment_points = shape.points[start_idx:end_idx]
                    coords = [(round(p[0], 2), round(p[1], 2)) for p in segment_points]
                    
                    for j in range(len(coords) - 1):
                        u = coords[j]
                        v = coords[j+1]
                        dist = math.hypot(u[0]-v[0], u[1]-v[1])
                        if dist > 0:
                            G.add_edge(u, v, length=dist, weight=dist)
            
            if len(G) == 0:
                return False, "Shapefile loaded but contained no valid lines."

            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest).copy()
            
            for n in G.nodes():
                G.nodes[n]['x'] = n[0]
                G.nodes[n]['y'] = n[1]
                
            return self._process_networkx_graph(G, log_callback)
            
        except Exception as e:
            return False, f"Shapefile Error: {e}"

    def _process_networkx_graph(self, raw_G, log_callback):
        try:
            self.G = raw_G.to_undirected()
            self.pos = {}
            nodes_to_remove = []
            xs, ys = [], []

            for n, d in self.G.nodes(data=True):
                try:
                    # Handle Shapefile tuple nodes vs GraphML attributes
                    if 'x' in d:
                        lx, ly = float(d['x']), float(d['y'])
                    elif isinstance(n, tuple) and len(n) == 2:
                        lx, ly = float(n[0]), float(n[1])
                    else:
                        nodes_to_remove.append(n)
                        continue

                    # Projection check (Lat/Lon vs Meters)
                    if -180 <= lx <= 180 and -90 <= ly <= 90:
                        mx, my = latlon_to_xy(ly, lx)
                    else:
                        mx, my = lx, ly
                        
                    self.pos[n] = (mx, my)
                    xs.append(mx)
                    ys.append(my)
                except:
                    nodes_to_remove.append(n)
            
            self.G.remove_nodes_from(nodes_to_remove)
            
            if not self.pos:
                msg = "ERROR: Map loaded but NO valid nodes found."
                log_callback(msg)
                return False, msg

            self.map_center = (sum(xs)/len(xs), sum(ys)/len(ys))
            self.bounds = (min(xs), max(xs), min(ys), max(ys))
            
            for u, v, d in self.G.edges(data=True):
                if 'length' not in d:
                    p1 = np.array(self.pos[u])
                    p2 = np.array(self.pos[v])
                    d['length'] = np.linalg.norm(p1 - p2)
                else:
                    d['length'] = float(d['length'])
                d['weight'] = d['length']
            
            return True, f"SUCCESS: Loaded {len(self.pos)} nodes."
        except Exception as e:
            log_callback(f"Graph Process Error: {str(e)}")
            return False, str(e)

    def get_location_coords(self, location):
        if not self.pos: return (0, 0)
        min_x, max_x, min_y, max_y = self.bounds
        cx, cy = self.map_center
        w = max_x - min_x
        h = max_y - min_y
        mx, my = w * 0.15, h * 0.15
        
        loc = location.lower()
        if loc == 'north': return (cx, max_y - my)
        if loc == 'south': return (cx, min_y + my)
        if loc == 'east':  return (max_x - mx, cy)
        if loc == 'west':  return (min_x + mx, cy)
        return (cx, cy)

    def calculate_safe_zones(self, count=3):
        if not self.pos: return 0
        candidates = list(self.pos.keys())
        
        def get_safety_score(n):
            nx, ny = self.pos[n]
            # If no hazards, prefer edges/corners (distance from center)
            if not self.hazards:
                cx, cy = self.map_center
                return math.hypot(nx - cx, ny - cy)
            
            # Maximize distance to nearest hazard
            min_dist = float('inf')
            for hx, hy, hr in self.hazards:
                d = math.hypot(nx - hx, ny - hy)
                if d < min_dist: min_dist = d
            return min_dist

        candidates.sort(key=get_safety_score, reverse=True)
        self.safe_nodes = candidates[:count]
        self.update_weights()
        return len(self.safe_nodes)

    def update_weights(self):
        if not self.G: return
        vx, vy = self.plume_vec
        plume_dir = np.array([vx, vy])
        
        # Reset Base Weights
        for u, v, d in self.G.edges(data=True):
            d['weight'] = d.get('length', 1.0)
        
        # Add Hazard Penalties
        if self.hazards:
            for hx, hy, hr in self.hazards:
                h_pos = np.array([hx, hy])
                for u, v, d in self.G.edges(data=True):
                    p1 = np.array(self.pos[u])
                    p2 = np.array(self.pos[v])
                    mid = (p1 + p2) / 2.0
                    vec = mid - h_pos
                    proj = np.dot(vec, plume_dir)
                    
                    if proj > -hr: # Downstream/Near
                        perp_vec = vec - (proj * plume_dir)
                        dist = np.linalg.norm(perp_vec)
                        if dist < PLUME_WIDTH:
                            factor = 1.0 - (dist / PLUME_WIDTH)
                            d['weight'] *= (1.0 + (PLUME_COST_FACTOR * factor))
        
        # Reroute Agents
        for agent in self.agents:
            if not agent['finished']: self.route_agent(agent)

    def route_agent(self, agent):
        if not self.safe_nodes: return
        
        # Find closest safe node
        best, min_d = None, float('inf')
        for s in self.safe_nodes:
            try:
                d = np.linalg.norm(agent['pos'] - np.array(self.pos[s]))
                if d < min_d: min_d = d; best = s
            except: pass
        
        if best:
            try:
                agent['path'] = nx.shortest_path(self.G, agent['curr'], best, weight='weight')
                agent['path_idx'] = 0
            except: pass

    def spawn_agents(self, count=100):
        self.agents = []
        if not self.G: return
        nodes = list(self.G.nodes())
        if not nodes: return
        for _ in range(count):
            start = random.choice(nodes)
            self.agents.append({
                'curr': start,
                'pos': np.array(self.pos[start]),
                'path': [],
                'path_idx': 0,
                'finished': False,
                'speed': random.uniform(AGENT_SPEED_WALK, AGENT_SPEED_RUN)
            })
        self.update_weights()

    def step(self):
        if not self.running: return
        for agent in self.agents:
            if agent['finished'] or not agent['path']: continue
            if agent['path_idx'] >= len(agent['path']) - 1:
                agent['finished'] = True; continue
            next_node = agent['path'][agent['path_idx'] + 1]
            target = np.array(self.pos[next_node])
            vec = target - agent['pos']
            dist = np.linalg.norm(vec)
            move = agent['speed'] * SIM_DT
            if dist <= move:
                agent['pos'] = target; agent['curr'] = next_node; agent['path_idx'] += 1
            else:
                agent['pos'] += (vec / dist) * move

sim = Simulation()

@ui.page('/')
async def main_page():
    
    plt.close('all')

    async def refresh_wind():
        vec, spd, deg = await run.io_bound(fetch_wind_sync, sim.map_center)
        sim.plume_vec = vec
        sim.wind_speed = spd
        wind_label.set_text(f"Wind: {spd:.1f} m/s FROM {deg:.0f}°")
        if sim.hazards:
            sim.update_weights()
            log.push("Wind updated.")
        else:
            log.push("Wind updated.")

    def add_hazard():
        if not sim.pos: 
            log.push("Map not loaded.")
            return
        loc = location_select.value
        hx, hy = sim.get_location_coords(loc)
        sim.hazards.append((hx, hy, 20.0))
        sim.update_weights()
        log.push(f"Hazard added ({loc}).")
        plot.refresh()

    def auto_safe_zones():
        count = sim.calculate_safe_zones()
        log.push(f"Set {count} Safe Zones.")
        plot.refresh()

    def clear_hazards():
        sim.hazards = []
        sim.safe_nodes = [] 
        sim.update_weights()
        log.push("Hazards Cleared.")
        plot.refresh()

    def toggle_run():
        sim.running = not sim.running
        btn_run.text = "Pause" if sim.running else "Run"

    def reset_sim():
        sim.running = False
        sim.hazards = []
        sim.safe_nodes = []
        sim.spawn_agents(150)
        btn_run.text = "Run"
        plot.refresh()
        log.push("Reset.")

    def draw_map(plot_obj):
        try:
            with plot_obj:
                plt.clf()
                if not sim.G or not sim.pos:
                    plt.text(0.5, 0.5, "Map Data Not Loaded", ha='center')
                    return

                # Map Lines (Coherent, smooth lines)
                nx.draw_networkx_edges(
                    sim.G, 
                    sim.pos, 
                    ax=plt.gca(), 
                    edge_color='#555555', 
                    width=1.5, 
                    alpha=0.8
                )

                # Nodes (Small and subtle)
                coords = np.array(list(sim.pos.values()))
                if len(coords) > 0:
                    plt.scatter(coords[:,0], coords[:,1], s=2, c='#999999', alpha=0.6)

                # Hazards (Small red circles, no arrows)
                vx, vy = sim.plume_vec
                for hx, hy, hr in sim.hazards:
                    plt.gca().add_patch(plt.Circle((hx, hy), hr, color='red', alpha=0.5))

                # Agents
                active = [a['pos'] for a in sim.agents if not a['finished']]
                done = [a['pos'] for a in sim.agents if a['finished']]
                if active:
                    act = np.array(active)
                    plt.scatter(act[:,0], act[:,1], c='blue', s=12, zorder=3)
                if done:
                    dn = np.array(done)
                    plt.scatter(dn[:,0], dn[:,1], c='green', s=12, zorder=3)

                # Safe Zones
                if sim.safe_nodes:
                    safe_coords = [sim.pos[n] for n in sim.safe_nodes if n in sim.pos]
                    if safe_coords:
                        sf = np.array(safe_coords)
                        plt.scatter(sf[:,0], sf[:,1], c='lime', marker='P', s=100, edgecolors='black', zorder=4)

                plt.axis('equal')
                plt.axis('off')
        except Exception as e:
            log.push(f"Plot Error: {str(e)}")

    # --- UI Layout ---
    with ui.row().classes('w-full no-wrap'):
        with ui.column().classes('w-1/4 p-4 min-w-[300px] border-r'):
            ui.markdown("## Evac Sim")
            wind_label = ui.label("Wind: --")
            ui.button("Fetch Wind", on_click=refresh_wind).classes('w-full')
            
            ui.separator().classes('my-4')
            btn_run = ui.button("Run", on_click=toggle_run)
            ui.button("Reset", on_click=reset_sim)
            
            ui.separator().classes('my-2')
            
            location_select = ui.select(
                ['Center', 'North', 'South', 'East', 'West'], 
                value='Center', 
                label='Hazard Location'
            ).classes('w-full')
            
            ui.button("Add Hazard", on_click=add_hazard).classes('bg-red-500 text-white mt-2')
            ui.button("Auto Safe Zones", on_click=auto_safe_zones).classes('bg-green-600 text-white w-full mt-2')
            ui.button("Clear Hazards", on_click=clear_hazards).classes('w-full mt-4')
            
            log = ui.log(max_lines=20).classes('w-full h-64 bg-gray-100 p-2 text-xs font-mono mt-4')

        with ui.column().classes('w-3/4 p-4 h-screen'):
            plot = ui.pyplot(close=False, figsize=(10, 8))
            plot.refresh = lambda: draw_map(plot)

            def loop():
                sim.step()
                if sim.running: plot.refresh()
            ui.timer(SIM_DT, loop)

    # Init
    ok, msg = sim.load_graph(log_callback=log.push)
    log.push(msg)
    if ok: 
        sim.spawn_agents(150)
        plot.refresh()
    await refresh_wind()

ui.run(title='Evac Sim', port=8080, host='127.0.0.1')