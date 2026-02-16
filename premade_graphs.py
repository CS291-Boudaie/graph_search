# Students: DO NOT EDIT. If you want to make your own graphs, copy this into a new file. 
from graph_wrapper import Graph
from utils import manhattan_distance
import networkx as nx
import pickle

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
    manhattan_graph.start = "Pace" # Suggest default start?
    manhattan_graph.goal = "Times Square" # Suggest default goal?
    
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

def create_grid_graph(width, height, obstacles=None):
    g = Graph()
    g.start = (0, 0)
    g.goal = (width - 1, height - 1)
    
    # Pre-process blocked edges for O(1) lookup
    blocked_edges = set()
    if obstacles:
        for u, v in obstacles:
            blocked_edges.add((u, v))
            blocked_edges.add((v, u))

    for x in range(width):
        for y in range(height):
            node = (x, y)
            g.add_node(node, pos=(x, y))
            
    for x in range(width):
        for y in range(height):
            # scan 4-connected neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx_val = x + dx
                ny_val = y + dy
                
                # Check bounds
                if 0 <= nx_val < width and 0 <= ny_val < height:
                    u = (x, y)
                    v = (nx_val, ny_val)
                    
                    # Check if edge is blocked
                    if (u, v) in blocked_edges:
                        continue
                        
                    g.add_edge(u, v, weight=1)
    return g

def create_trap_graph():
    """
    Creates a graph with a local minimum / trap.
    S -> A is short and low cost direct, but leads to dead end.
    S -> B is longer/expensive but leads to G.
    """
    g = Graph()
    g.start = "S"
    g.goal = "G"
    g.add_node("S", pos=(0, 0))
    g.add_node("G", pos=(0, 4))
    g.add_node("A", pos=(0, 1)) 
    g.add_node("B", pos=(10, 0)) 
    g.add_node("C", pos=(10, 4)) 
    
    # S->A->DeadEnd
    g.add_edge("S", "A", weight=1)
    
    # S->B->C->G
    g.add_edge("S", "B", weight=10)
    g.add_edge("B", "C", weight=1)
    g.add_edge("C", "G", weight=1)
    
    return g

def create_inconsistent_heuristic_graph():
    """
    Creates a graph with explicit heuristic values that are inconsistent.
    h(A) overestimates or violates triangle inequality.
    
    Graph structure:
    S -> A (cost 1)
    S -> B (cost 1)
    A -> C (cost 1)
    B -> C (cost 10)
    C -> G (cost 100)
    
    Heuristics:
    h(S) = ? (Irrelevant for start, but let's say 100)
    h(A) = 100 (HIGH - makes S->A look bad if we came from S, 
                OR if we just generated it?
                f(A) = 1 + 100 = 101.
                f(B) = 1 + 0 = 1.
                Pick B.)
    h(B) = 0
    h(C) = 0
    h(G) = 0
    
    Path 1 via A: S->A->C->G. Cost 1+1+100 = 102.
    Path 2 via B: S->B->C->G. Cost 1+10+100 = 111.
    
    Expected behavior:
    S expands. A(f=101), B(f=1).
    Pick B.
    B expands to C. C(g=1+10=11, h=0, f=11).
    Pick C (f=11 < 101).
    C expands to G. G(g=111, h=0, f=111).
    Pick A (f=101).
    A expands to C.
    New path to C found: S->A->C. Cost 1+1=2.
    Old cost to C was 11.
    2 < 11. Update C!
    C is re-added to PQ (or updated).
    New f(C) = 2 + 0 = 2.
    Pick C (f=2).
    C expands to G. New path to G: 2+100=102.
    Update G.
    
    This tests:
    1. A* doesn't stop after finding the first path to G (if G wasn't popped yet, but here G was found via B path? No C was found).
    2. A* updates nodes in Open/Closed set if better path found. (Re-expansion of C).
    """
    g = Graph()
    g.start = "S"
    g.goal = "G"
    
    # Nodes with explicit heuristics
    g.add_node("S", h=0, pos=(0,0))
    g.add_node("A", h=100, pos=(1,0))
    g.add_node("B", h=0, pos=(0,1))
    g.add_node("C", h=0, pos=(1,1))
    g.add_node("G", h=0, pos=(2,1))
    
    # Edges
    g.add_edge("S", "A", weight=1)
    g.add_edge("S", "B", weight=1)
    
    g.add_edge("A", "C", weight=1)
    g.add_edge("B", "C", weight=10)
    
    g.add_edge("C", "G", weight=100)
    
    return g

# Load Data
import csv

def load_lion_graph(nodes_csv, edges_csv):
    G = Graph()
    
    # Internal ID to Label mapping
    id_to_label = {}
    
    # Coordinate Transformer
    # Defaulting to EPSG:2263 (NY Long Island) -> EPSG:4326 (Lat/Lon)
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    except ImportError:
        print("Warning: pyproj not installed. Using raw coordinates.")
        transformer = None
    
    # Load nodes
    with open(nodes_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['node_id'] 
            # safe conversion
            try:
                nid = int(node_id)
            except ValueError:
                nid = node_id
            
            label = row['label']
            id_to_label[nid] = label
            
            x = float(row['x'])
            y = float(row['y'])
            
            if transformer:
                lon, lat = transformer.transform(x, y)
                pos = (lon, lat)
            else:
                pos = (x, y)
                
            G.add_node(
                label,
                label=label,
                pos=pos # Store as (lon, lat)
            )

    # Load edges
    with open(edges_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                u_id = int(row['source_id'])
                v_id = int(row['target_id'])
            except ValueError:
                u_id = row['source_id']
                v_id = row['target_id']
                
            u_label = id_to_label[u_id]
            v_label = id_to_label[v_id]
            
            w = float(row['weight'])
            G.add_edge(u_label, v_label, weight=w)

    return G

def create_pace_graph():
    return load_lion_graph('files/nodes_pace.csv', 'files/edges_pace.csv')


import random

class SlidingTileGraph(Graph):
    """
    Implicit graph for Sliding Tile Puzzle.
    State is a tuple of tuples, e.g. ((1, 2, 3), (4, 0, 5), (6, 7, 8)) for 3x3.
    0 represents the empty tile.
    """
    def __init__(self, n):
        super().__init__()
        self.n = n

    def get_goal_state(self):
        # Generate standard goal: 1 to n^2-1 followed by 0
        # e.g. for n=3: ((1, 2, 3), (4, 5, 6), (7, 8, 0))
        values = list(range(1, self.n*self.n)) + [0]
        grid = []
        for i in range(0, self.n*self.n, self.n):
            grid.append(tuple(values[i:i+self.n]))
        return tuple(grid)
    
    def generate_random_start_state(self, n_moves=100, seed=None):
        if seed is not None:
            random.seed(seed)
            
        current = self.get_goal_state()
        for _ in range(n_moves):
            neighbors = self.neighbors(current)
            current = random.choice(neighbors)
        return current

    @staticmethod
    def render_board(state):
        n = len(state)
        # Use emojis if the max number is single digit (approx n<=3)
        use_emojis = (n * n - 1) <= 9
        
        emoji_map = {
            0: "⬜", 1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 
            4: "4️⃣", 5: "5️⃣", 6: "6️⃣", 7: "7️⃣", 
            8: "8️⃣", 9: "9️⃣"
        }

        res = []
        for row in state:
            line = []
            for val in row:
                if use_emojis:
                    line.append(emoji_map.get(val, str(val)))
                else:
                    line.append(f"{val:2}" if val != 0 else " _")
            res.append(" ".join(line))
        return "\n".join(res)

    def neighbors(self, state):
        self._record(state)
        
        # Find 0 (empty tile)
        zero_r, zero_c = -1, -1
        for r in range(self.n):
            for c in range(self.n):
                if state[r][c] == 0:
                    zero_r, zero_c = r, c
                    break
        
        # Possible moves: Up, Down, Left, Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        results = []
        
        for dr, dc in moves:
            nr, nc = zero_r + dr, zero_c + dc
            if 0 <= nr < self.n and 0 <= nc < self.n:
                # Create new state by swapping
                # mutable list of lists conversion
                new_grid = [list(row) for row in state]
                new_grid[zero_r][zero_c], new_grid[nr][nc] = new_grid[nr][nc], new_grid[zero_r][zero_c]
                # back to tuple of tuples
                results.append(tuple(tuple(row) for row in new_grid))
                
        return results
        
    def get_edge_data(self, u, v):
        # All moves cost 1
        return {'weight': 1}

def create_sliding_tile_graph(n):
    return SlidingTileGraph(n)