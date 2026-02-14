import networkx as nx

class TrackedGraph(nx.Graph):
    """
    NetworkX graph that tracks exploration behavior.

    Exploration = accessing neighbors via:
        g[node]
        g.neighbors(node)
    """

    def __init__(self, *args, goal=None, **kwargs):
        super().__init__(*args, **kwargs)

        # exploration stats
        self.total_explorations = 0
        self.exploration_order = []
        self.unique_explored = set()
        self.reexpanded_nodes = set()

        # goal tracking
        self.goal = goal
        self.goal_expanded = False

    # ---------- INTERNAL ----------
    def _record(self, node):
        self.total_explorations += 1
        self.exploration_order.append(node)

        if node in self.unique_explored:
            self.reexpanded_nodes.add(node)
        else:
            self.unique_explored.add(node)

        if node == self.goal:
            self.goal_expanded = True

    # ---------- OVERRIDES ----------
    def __getitem__(self, node):
        self._record(node)
        return super().__getitem__(node)

    def neighbors(self, node):
        self._record(node)
        return super().neighbors(node)

    # ---------- RESET ----------
    def reset_tracking(self):
        self.total_explorations = 0
        self.exploration_order = []
        self.unique_explored = set()
        self.reexpanded_nodes = set()
        self.goal_expanded = False

    # ---------- REPORT ----------
    def stats(self):
        return {
            "total_explorations": self.total_explorations,
            "unique_explored": len(self.unique_explored),
            "reexpansions": len(self.reexpanded_nodes),
            "goal_expanded": self.goal_expanded,
            "exploration_order": list(self.exploration_order),
        }
