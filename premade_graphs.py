from graph_wrapper import Graph
from utils import manhattan_distance

def basic_manhattan():
    graph_nodes = {
        "Times Square": (7, 42),
        "Grand Central": (3, 42),
        "Empire State": (5, 34),
        "Rockefeller Center": (5, 50),
        "Central Park South": (6, 59),
        "Columbus Circle": (8, 59),
        "Washington Sq Park": (5, 8),
        "Union Square": (4, 14),
        "Wall Street": (4, 0),
        "Brooklyn Bridge": (1, 1),
        "Bryant Park": (6, 42),
        "Herald Square": (6, 34),
        "Chelsea (23rd St)": (6, 23),
        "Pace": (3, 1)
    }

    manhattan_graph = Graph()
    for name, coordinates in graph_nodes.items():
        manhattan_graph.add_node(name, pos=coordinates)
    
    edges = [
        ("Times Square", "Bryant Park"),
        ("Bryant Park", "Grand Central"),
        ("Bryant Park", "Rockefeller Center"),
        ("Times Square", "Rockefeller Center"),
    
        ("Empire State", "Herald Square"),
        ("Herald Square", "Times Square"),
        ("Herald Square", "Bryant Park"),
    
        ("Rockefeller Center", "Central Park South"),
        ("Central Park South", "Columbus Circle"),
    
        ("Herald Square", "Chelsea (23rd St)"),
        ("Chelsea (23rd St)", "Union Square"),
        ("Union Square", "Washington Sq Park"),
    
        ("Union Square", "Wall Street"),
        ("Washington Sq Park", "Wall Street"),
    
        ("Pace", "Brooklyn Bridge"),
        ("Wall Street", "Pace"),
        ("Union Square", "Pace"),
    
    ]
    
    # Set edge weight = Manhattan block distance
    for u, v in edges:
        manhattan_graph.add_edge(u, v, weight=manhattan_distance(u, v, graph_nodes))

    return manhattan_graph
