# Students: DO NOT EDIT
import matplotlib.pyplot as plt
import networkx as nx

import matplotlib.pyplot as plt
import networkx as nx

def draw_manhattan_graph(G, path=None, title="NYC Manhattan Grid Landmark Graph (edge weight = blocks)"):
    """
    Draw Manhattan graph using node 'pos' attributes stored in graph.
    
    Parameters
    ----------
    G : networkx graph (or your Graph wrapper if nx-compatible)
    path : list optional
        Ordered list of nodes representing a path
    """

    plt.figure(figsize=(8, 6))

    # pull positions directly from graph
    raw_pos = nx.get_node_attributes(G, "pos")

    # flip x so avenues visually match Manhattan orientation
    pos_xy = {k: (-v[0], v[1]) for k, v in raw_pos.items()}

    # ---- base graph ----
    nx.draw(
        G,
        pos_xy,
        with_labels=True,
        node_size=900,
        font_size=8
    )

    nx.draw_networkx_edge_labels(
        G,
        pos_xy,
        edge_labels={(u, v): d["weight"] for u, v, d in G.edges(data=True)},
        font_size=7
    )

    # ---- path overlay (optional) ----
    if path is not None and len(path) > 0:
        path_edges = list(zip(path[:-1], path[1:]))

        # path edges
        nx.draw_networkx_edges(
            G,
            pos_xy,
            edgelist=path_edges,
            edge_color="red",
            width=3
        )

        # path nodes
        nx.draw_networkx_nodes(
            G,
            pos_xy,
            nodelist=path,
            node_color="red",
            node_size=1000
        )

        # start node
        nx.draw_networkx_nodes(
            G,
            pos_xy,
            nodelist=[path[0]],
            node_color="green",
            node_size=1200
        )

        # goal node
        nx.draw_networkx_nodes(
            G,
            pos_xy,
            nodelist=[path[-1]],
            node_color="purple",
            node_size=1200
        )

    plt.title(title)
    # plt.xlabel("Avenue (west→east)")
    # plt.ylabel("Street (south→north)")
    plt.grid(True)
    plt.show()


def manhattan_distance(location_1, location_2, graph_nodes):
    x1, y1 = graph_nodes[location_1]
    x2, y2 = graph_nodes[location_2]
    return abs(y2 - y1) + abs(x2 - x1)


import math

def haversine_distance(location_1, location_2, graph_nodes):
    """
    Calculates the great-circle distance between two points on the Earth.
    Input coordinates are expected to be (lon, lat).
    """
    lon1, lat1 = graph_nodes[location_1]
    lon2, lat2 = graph_nodes[location_2]
    
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles.
    return c * r

# Alias for compatibility if needed, or just use haversine
euclidean_distance = haversine_distance


def plot_busy_graph(G, path=None, node_size=2, edge_alpha=0.15, figsize=(10, 10), title="NYC Map"):
    pos = {n: (G.nodes[n]["pos"][0], G.nodes[n]["pos"][1]) for n in G.nodes}

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=edge_alpha)
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    
    if path:
        # Create edge list from path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='r')
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=node_size*2, node_color='r')
    
    plt.axis("equal")
    plt.axis("off")
    plt.title(title)
    plt.show()

def get_path_cost(graph, path):
    cost = 0
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        cost += graph.get_edge_data(u, v).get('weight', 1)
    return cost

def misplaced_heuristic(graph, state, goal):
    count = 0
    n = len(state)
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0 and val != goal[r][c]:
                count += 1
    return count

def sliding_tile_manhattan_heuristic(graph, state, goal):
    # Calculate goal positions on the fly or maybe we can assume standard goal?
    # But for general A*, goal is passed.
    # To avoid re-calculating goal positions every time, we could cache them if we had a place.
    # For now, just calculate them. It's O(N^2) which is small for 3x3 or 4x4.
    goal_positions = {}
    n = len(goal)
    for r in range(n):
        for c in range(n):
            goal_positions[goal[r][c]] = (r, c)
            
    dist = 0
    for r in range(n):
        for c in range(n):
            val = state[r][c]
            if val != 0:
                gr, gc = goal_positions[val]
                dist += abs(r - gr) + abs(c - gc)
    return dist

def manhattan_heuristic(graph, node, goal):
    try:
        pos1 = graph.nodes[node]['pos']
        pos2 = graph.nodes[goal]['pos']
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    except KeyError:
        return 0

def haversine_heuristic(graph, node, goal):
    # Using utils.haversine_distance, scaled to feet (1km = 3280.84ft)
    return haversine_distance(node, goal, {x: graph.nodes[x]['pos'] for x in graph.nodes}) * 3280.84

def zero_heuristic(graph, node, goal):
    return 0

def custom_heuristic(graph, node, goal):
    return graph.nodes[node].get('h', 0)
