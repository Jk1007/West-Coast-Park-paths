import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------------
# Load cleaned graphs
# ----------------------------
G_walk = ox.load_graphml("west_coast_park_walk_clean.graphml")
G_bike = ox.load_graphml("west_coast_park_bike_clean.graphml")

# ----------------------------
# Plot graphs without nodes
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Pedestrian paths in blue
ox.plot_graph(G_walk, ax=ax, node_size=0, edge_color="blue", show=False, close=False)

# Bike paths in green
ox.plot_graph(G_bike, ax=ax, node_size=0, edge_color="green", show=False, close=False)

plt.title("West Coast Park - Pedestrian (blue) & Bike (green) Paths")
plt.show()
