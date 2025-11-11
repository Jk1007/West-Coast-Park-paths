import osmnx as ox
import networkx as nx

place_name = "West Coast Park, Singapore"

G_walk = ox.graph_from_place(place_name, network_type="walk")
G_bike = ox.graph_from_place(place_name, network_type="bike")

G_walk = ox.project_graph(G_walk).to_undirected()
G_bike = ox.project_graph(G_bike).to_undirected()

G_walk = G_walk.subgraph(max(nx.connected_components(G_walk), key=len)).copy()
G_bike = G_bike.subgraph(max(nx.connected_components(G_bike), key=len)).copy()

# skip simplify_graph to avoid the GraphSimplificationError

for G in [G_walk, G_bike]:
    for u, v, d in G.edges(data=True):
        if "length" not in d:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            d["length"] = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

ox.save_graphml(G_walk, "west_coast_park_walk_clean.graphml")
ox.save_graphml(G_bike, "west_coast_park_bike_clean.graphml")

print("Cleaned graphs saved as:")
print(" - west_coast_park_walk_clean.graphml")
print(" - west_coast_park_bike_clean.graphml")
