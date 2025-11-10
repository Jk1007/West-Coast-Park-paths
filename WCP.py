# WCP.py — build and save West Coast Park walking network (version-agnostic)
import osmnx as ox
import networkx as nx

ox.settings.use_cache = True

AREA = "West Coast Park, Singapore"

# 1) Download walkable network (directed MultiDiGraph)
G = ox.graph_from_place(AREA, network_type="walk")

# 2) Keep the largest connected component using NetworkX (version-proof)
#    For directed graphs, use WEAK connectivity so opposite-edge pairs are treated as connected.
largest_nodes = max(nx.weakly_connected_components(G), key=len)
G = G.subgraph(largest_nodes).copy()

# 3) Project to metres (for correct distances/speeds)
G = ox.project_graph(G)

# 4) Save for your Mesa model/server
ox.save_graphml(G, "west_coast_park_walk.graphml")
print("✅ Saved: west_coast_park_walk.graphml")
