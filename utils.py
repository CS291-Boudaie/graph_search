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
    plt.xlabel("Avenue (west→east)")
    plt.ylabel("Street (south→north)")
    plt.grid(True)
    plt.show()


def manhattan_distance(location_1, location_2, graph_nodes):
    x1, y1 = graph_nodes[location_1]
    x2, y2 = graph_nodes[location_2]
    return abs(y2 - y1) + abs(x2 - x1)
