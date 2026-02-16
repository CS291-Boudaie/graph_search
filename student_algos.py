# Step 1: Attestation
# Comment one of the following below
# 1. I attest that I did NOT use ChatGPT or any other automated writing system for ANY portion of this assignment.
# 2. I acknowledge using ChatGPT [or some other system; name it] for the following tasks: [list them]. Some surprising benefits of using GPT include the following: [list them]. I also encountered a few unanticipated challenges: [list them].


# Bring your priority queue from assignment 0 here
# If your priority queue from assignment 0 is not working, you can look into using the
# heapq module in Python to help you implement a min heap. All the tets from assignment 0 should still pass. 


# --- THE GRAPH OBJECT ---
# The `graph` object passed to your functions is a NetworkX-like graph with some extra features.
# Your algorithm will be given a start node, a goal node, and a heuristic for A*
# 1. Neighbors: You can access neighbors of a node `u` using `graph.neighbors(node)`.
# 2. Edge Weights: You can access the weight of an edge (u, v) using `graph.get_edge_data(u, v).get('weight', 1)`.
#    - If the graph is unweighted, the default weight is 1.
# 3. Stats: You can access the stats of the graph using `graph.stats()`.

# --- HELPERS ---
# Please paste in your HinHeap and PriorityQueue from Assignment 0.  
# If you were not able to get it to work, look into using the `heapq` library to implement the
# MinHeap functions

class MinHeap:
    """
    A simple MinHeap implementation.
    You can use this to implement your PriorityQueue.
    """
    def __init__(self):
        # We'll store elements as tuples: (priority, item)
        self.data = []

    def __len__(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0

    def peek(self):
        """
        Returns the smallest element (priority, item) without removing it.
        Returns None if empty.
        """
        # TODO: Return (priority, item) but do NOT remove
        # If empty, return None (or raise an error)
        pass

    def add(self, priority, item):
        """
        Adds an item with the given priority to the heap.
        """
        # TODO: Add (priority, item) to end of list
        # Then bubble it UP into correct position
        pass

    def pop_min(self):
        """
        Removes and returns the smallest element (priority, item).
        Returns None if empty.
        """
        # TODO: Remove and return the smallest element (priority, item)
        # Steps:
        # 1) swap root with last element
        # 2) pop last element (former root)
        # 3) bubble DOWN new root
        pass

    def _bubble_up(self, idx):
        # TODO: Implement
        # Keep swapping this node with its parent while it has a smaller priority.
        # parent index = (idx - 1) // 2
        # Stop when you reach the root OR parent already has <= priority.
        pass

    def _bubble_down(self, idx):
        # Keep swapping this node downward until the heap property is restored.
        # left child = 2*idx + 1, right child = 2*idx + 2
        # Find the smaller child, then swap if current priority is bigger.
        # Stop when no children exist OR current is <= both children.
        pass


# Once you have a min heap, the priority queue is pretty straightforward. 
# Make sure you understand what it is doing

class PriorityQueue:
    """
    A Priority Queue built on top of MinHeap.
    """
    def __init__(self):
        self.heap = MinHeap()

    def is_empty(self):
        return self.heap.is_empty()

    def add(self, priority, item):
        """Add an item with a given priority (lower value = higher priority)"""
        self.heap.add(priority, item)

    def pop(self):
        """Remove and return the (priority, item) tuple with the lowest priority value"""
        return self.heap.pop_min()

    def peek(self):
        return self.heap.peek()

    def __len__(self):
        return len(self.heap)

# Each of these should return a path from start to goal.
# Implement duplicate checking for each.

def bfs(graph, start, goal):
    """
    Breadth-First Search (BFS)
    
    Args:
        graph: The graph object.
        start: The starting node.
        goal: The goal node.
        
    Returns:
        list: A list of nodes representing the path from start to goal (e.g., ['A', 'B', 'C']).
              Return None if no path is found.
              
    Tips:
        - Use a queue (python's `collections.deque` or just a list with `.pop(0)`).
        - Keep track of visited nodes to avoid infinite loops.
        - To pass the strict 1x efficiency test, check if a neighbor is the goal 
          IMMEDIATELY when you look at neighbors, not when you pop from the queue.
    """
    # TODO: implement
    # Hint: The networkx tutorial gets pretty close; the slides get you there close as well.
    return []

def dfs(graph, start, goal):
    """
    Depth-First Search (DFS)
    
    Args:
        graph: The graph object.
        start: The starting node.
        goal: The goal node.
        
    Returns:
        list: A list of nodes representing the path. Return None if no path is found.
        
    Tips:
        - This algorithm works very well with recursion, or just copy BFS but use a stack instead of a queue.
        - deque stands for double ended queue; so just popright instead of popleft
        - EFFICIENCY TIP: Just like BFS, check for the goal node immediately in the neighbor loop.
    """
    # TODO: implement
    return []

# For fun you can implement iterative deepening; but it won't be graded or checked

def dijkstra(graph, start, goal):
    """
    Dijkstra's Algorithm
    
    Args:
        graph: The graph object.
        start: The starting node.
        goal: The goal node.
        
    Returns:
        list: A list of nodes representing the path. Return None if no path is found.
        
    Tips:
        - Use the `PriorityQueue` class from assignment 0. Priority will be the current path cost.
        - To be extra efficient, think how you don't have to calculate the whole path cost every time.
        - The `visited` set for Dijkstra is slightly trickier than BFS.
          You should only add a node to visited when you POP it from the queue, not when you see it.
    """
    # TODO: Implement
    # Hint 2: If you do A* first, you can implement dijkstra with A* and a heuristic that always returns 0
    return []

def a_star(graph, start, goal, heuristic):
    """
    A* Search
    
    Args:
        graph: The graph object.
        start: The starting node.
        goal: The goal node.
        heuristic: A function `def h(graph, node, goal): return cost`.
        
    Returns:
        list: A list of nodes representing the path. Return None if no path is found.
        
    Tips:
        - Very similar to Dijkstra, but the priority is `f = g + h`.
        - To deal with nonconsistent heuristics, instead of keeping track of a "visited' set, keep track of the
            lowest g score you have ever seen for a specific node. If you find a lower g score, it is worth revisiting.
        - `g`: Cost from start to current node.
        - `h`: Estimated cost from current node to goal (use the `heuristic` function, pass graph, current node, and goal).
        - Heuristic will be given to you; don't worry about it, just call it with those functions.
    """
    # TODO: Implement
    return []