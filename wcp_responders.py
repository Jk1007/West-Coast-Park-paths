
import numpy as np
from collections import Counter

# Global set to track responder locations (node IDs)
RESPONDERS = set()

def suggest_responders(hazards, people, kd_tree, node_ids, pos_dict, map_dist_func, k=None):
    """
    Place SCDF responder nodes with the rule:
    - Minimum 4 responders per hazard.
    - For each hazard:
        * 1 responder exactly at the hazard location (nearest graph node).
        * 3 responders on / near evacuation paths civilians use to reach safe zones.
    
    Args:
        hazards: list of hazard dicts
        people: list of person dicts
        kd_tree: scipy.spatial.KDTree of node positions
        node_ids: list of all graph node IDs
        pos_dict: dict mapping node_id -> (x, y)
        map_dist_func: function(p1, p2) -> distance_meters
        k: (unused logic-wise, kept for signature compatibility if needed)
    """
    global RESPONDERS

    RESPONDERS.clear()

    # No hazards or no people => nothing to place
    if not hazards or not people:
        return False

    # Keep track per hazard: (hazard_dict, hazard_node_id)
    hazard_nodes = []

    # ---------- 1) One responder at each hazard (nearest graph node) ----------
    for h in hazards:
        hx, hy = float(h["pos"][0]), float(h["pos"][1])
        # KDTree is already built on NODE_POS with NODE_IDS
        dist, idx = kd_tree.query([[hx, hy]])           # shape (1,2) -> nearest node index
        nid = node_ids[int(idx[0])]                # graph node id

        RESPONDERS.add(nid)
        hazard_nodes.append((h, nid))

    # ---------- 2) Three responders on/near evacuation paths per hazard ----------
    for h, hazard_node in hazard_nodes:
        hx, hy = float(h["pos"][0]), float(h["pos"][1])
        hz_pos = np.array([hx, hy], dtype=float)

        # Collect candidate nodes along paths that:
        # - belong to people who started near this hazard
        # - are near the hazard corridor (within some radius)
        candidate_nodes = Counter()

        for p in people:
            path = p.get("path") or []
            if not path:
                continue

            # Roughly decide if this person is "from this hazard region":
            # distance from current position to hazard
            p_pos = np.array(p["pos"], dtype=float)
            d0 = map_dist_func(p_pos, hz_pos)
            if d0 > 400.0:
                # Person is probably not in this hazard's affected region
                continue

            # Now walk their full evacuation path and collect nearby nodes
            for nid in path:
                if nid == hazard_node:
                    continue  # already using hazard_node as one responder
                
                # Retrieve node xy from pos_dict
                nxy = pos_dict[nid]
                node_xy = np.array([nxy[0], nxy[1]], dtype=float)
                
                dnode = map_dist_func(node_xy, hz_pos)
                if dnode <= 600.0:
                    # Node is along / near the corridor from this hazard
                    candidate_nodes[nid] += 1

        if not candidate_nodes:
            # No strong path info for this hazard; skip extra 3
            continue

        # Sort nodes by:
        # - most frequently used in paths (descending)
        # This tends to pick choke points / main arteries
        sorted_nodes = sorted(
            candidate_nodes.keys(),
            key=lambda n: -candidate_nodes[n]
        )

        # Pick up to 3 distinct nodes for this hazard
        added = 0
        for nid in sorted_nodes:
            if nid in RESPONDERS:
                continue
            RESPONDERS.add(nid)
            added += 1
            if added >= 3:
                break

    return bool(RESPONDERS)
