
import xml.etree.ElementTree as ET

def inspect_graphml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    
    x_vals = []
    y_vals = []
    
    nodes_info = []

    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        x_val = None
        y_val = None
        
        for data in node.findall('g:data', ns):
            key = data.get('key')
            if key == 'd5': # x
                try: x_val = float(data.text)
                except: pass
            elif key == 'd4': # y
                try: y_val = float(data.text)
                except: pass
                
        if x_val is not None and y_val is not None:
            nodes_info.append((node_id, x_val, y_val))
            x_vals.append(x_val)
            y_vals.append(y_val)

    if not x_vals:
        print("No valid nodes found with d4/d5 attributes.")
        return

    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    
    print(f"Total Nodes: {len(nodes_info)}")
    print(f"X Range: {min_x} to {max_x}")
    print(f"Y Range: {min_y} to {max_y}")
    
    # Find nodes at bounds
    for nid, x, y in nodes_info:
        if x == min_x: print(f"Min X Node: {nid} at {x}, {y}")
        if x == max_x: print(f"Max X Node: {nid} at {x}, {y}")
        if y == min_y: print(f"Min Y Node: {nid} at {x}, {y}")
        if y == max_y: print(f"Max Y Node: {nid} at {x}, {y}")
        
    # Check for User's specific values (approximate match)
    user_vals = [361961.88, 362758.37, 143579.54, 142808.10, 143719.35]
    print("\nChecking for User Values (tolerance 1.0):")
    found = False
    for val in user_vals:
        for nid, x, y in nodes_info:
            if abs(x - val) < 1.0 or abs(y - val) < 1.0:
                print(f"Found User Value {val} at Node {nid} ({x}, {y})")
                found = True
                break
    if not found:
        print("No exact user matches found.")

if __name__ == "__main__":
    inspect_graphml('west_coast_park_walk_clean.graphml')
