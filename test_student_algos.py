# Students: Do NOT modify this file. 
# If you want to test your code, copy this file and modify it instead. 
import unittest
import premade_graphs
import utils
import signal
import os
import threading

class TimeoutException(Exception):
    pass

class test_timeout:
    """
    Cross-platform timeout context manager.

    - On Unix/macOS: uses signal.alarm (fast, reliable).
    - On Windows: runs the test body in a separate thread and fails if it exceeds time.
    """

    def __init__(self, seconds, error_message=None):
        self.seconds = seconds
        self.error_message = error_message or f"test timed out after {seconds}s."
        self._use_signals = hasattr(signal, "SIGALRM") and os.name != "nt"

        # Windows/thread fallback
        self._thread = None
        self._exc = None

    # ---------- Unix/macOS implementation ----------
    def _handle_timeout(self, signum, frame):
        raise TimeoutException(self.error_message)

    # ---------- Windows/thread implementation ----------
    def _thread_runner(self, fn):
        try:
            fn()
        except BaseException as e:
            self._exc = e

    def run(self, fn):
        """
        Use this ONLY on Windows (or when signals not available).
        Example:

        with test_timeout(1) as t:
            t.run(lambda: do_something())
        """
        self._thread = threading.Thread(target=self._thread_runner, args=(fn,), daemon=True)
        self._thread.start()
        self._thread.join(self.seconds)

        if self._thread.is_alive():
            raise TimeoutException(self.error_message)

        if self._exc is not None:
            raise self._exc

    # ---------- Context manager API ----------
    def __enter__(self):
        if self._use_signals:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._use_signals:
            signal.alarm(0)
        # don't suppress exceptions
        return False

# Try to import answers, fallback to student_algos for students
try:
    import answers as student_algos
except ImportError:
    import student_algos

# --- Heuristic Helpers ---
# --- Heuristic Helpers ---
# Removed, using utils.py directly

# --- Test Data ---
TEST_DATA = [
    {
        'name': 'Basic_Manhattan_Pace_to_Times_Square',
        'graph_loader': premade_graphs.basic_manhattan,
        'loader_code': 'premade_graphs.basic_manhattan()',
        'start': 'Pace',
        'goal': 'Times Square',
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 7, 'path_len': 5, 'cost': 45},
            'dfs': {'explored': 5, 'path_len': 5, 'cost': 45},
            'dijkstra': {'explored': 9, 'path_len': 5, 'cost': 45},
            'astar_manhattan': {'explored': 4, 'path_len': 5, 'cost': 45},
            'astar_zero': {'explored': 9, 'path_len': 5, 'cost': 45},
        }
    },
    {
        'name': 'Basic_Manhattan_Pace_to_Washington_Sq_Park',
        'graph_loader': premade_graphs.basic_manhattan,
        'loader_code': 'premade_graphs.basic_manhattan()',
        'start': 'Pace',
        'goal': 'Washington Sq Park',
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 3, 'path_len': 3, 'cost': 11},
            'dfs': {'explored': 2, 'path_len': 3, 'cost': 21},
            'dijkstra': {'explored': 3, 'path_len': 3, 'cost': 11},
            'astar_manhattan': {'explored': 2, 'path_len': 3, 'cost': 11},
            'astar_zero': {'explored': 3, 'path_len': 3, 'cost': 11},
        }
    },
    {
        'name': 'Basic_Manhattan_Columbus_Circle_to_Union_Square',
        'graph_loader': premade_graphs.basic_manhattan,
        'loader_code': 'premade_graphs.basic_manhattan()',
        'start': 'Columbus Circle',
        'goal': 'Union Square',
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 9, 'path_len': 7, 'cost': 51},
            'dfs': {'explored': 6, 'path_len': 7, 'cost': 53},
            'dijkstra': {'explored': 9, 'path_len': 7, 'cost': 51},
            'astar_manhattan': {'explored': 7, 'path_len': 7, 'cost': 51},
            'astar_zero': {'explored': 9, 'path_len': 7, 'cost': 51},
        }
    },
    {
        'name': '3x3_Grid_0_0_to_2_2',
        'graph_loader': lambda: premade_graphs.create_grid_graph(3, 3),
        'loader_code': 'premade_graphs.create_grid_graph(3, 3)',
        'start': (0, 0),
        'goal': (2, 2),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 7, 'path_len': 5, 'cost': 4},
            'dfs': {'explored': 4, 'path_len': 5, 'cost': 4},
            'dijkstra': {'explored': 8, 'path_len': 5, 'cost': 4},
            'astar_manhattan': {'explored': 7, 'path_len': 5, 'cost': 4},
            'astar_zero': {'explored': 8, 'path_len': 5, 'cost': 4},
        }
    },
    {
        'name': '5x5_Grid_0_0_to_4_4',
        'graph_loader': lambda: premade_graphs.create_grid_graph(5, 5),
        'loader_code': 'premade_graphs.create_grid_graph(5, 5)',
        'start': (0, 0),
        'goal': (4, 4),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 23, 'path_len': 9, 'cost': 8},
            'dfs': {'explored': 8, 'path_len': 9, 'cost': 8},
            'dijkstra': {'explored': 24, 'path_len': 9, 'cost': 8},
            'astar_manhattan': {'explored': 15, 'path_len': 9, 'cost': 8},
            'astar_zero': {'explored': 24, 'path_len': 9, 'cost': 8},
        }
    },
    {
        'name': '5x5_Grid_0_0_to_0_4',
        'graph_loader': lambda: premade_graphs.create_grid_graph(5, 5),
        'loader_code': 'premade_graphs.create_grid_graph(5, 5)',
        'start': (0, 0),
        'goal': (0, 4),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 10, 'path_len': 5, 'cost': 4},
            'dfs': {'explored': 4, 'path_len': 5, 'cost': 4},
            'dijkstra': {'explored': 11, 'path_len': 5, 'cost': 4},
            'astar_manhattan': {'explored': 4, 'path_len': 5, 'cost': 4},
            'astar_zero': {'explored': 12, 'path_len': 5, 'cost': 4},
        }
    },
    {
        'name': '5x5_Grid_Obstacles_Snake',
        'graph_loader': lambda: premade_graphs.create_grid_graph(5, 5, obstacles=[
            ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((1, 3), (2, 3)), 
            ((3, 4), (4, 4)), ((3, 3), (4, 3)), ((3, 2), (4, 2)), ((3, 1), (4, 1))
        ]),
        'loader_code': 'premade_graphs.create_grid_graph(5, 5, obstacles=[\n    ((1, 0), (2, 0)), ((1, 1), (2, 1)), ((1, 2), (2, 2)), ((1, 3), (2, 3)),\n    ((3, 4), (4, 4)), ((3, 3), (4, 3)), ((3, 2), (4, 2)), ((3, 1), (4, 1))\n])',
        'start': (0, 0),
        'goal': (4, 4),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 24, 'path_len': 17, 'cost': 16},
            'dfs': {'explored': 16, 'path_len': 17, 'cost': 16},
            'dijkstra': {'explored': 24, 'path_len': 17, 'cost': 16},
            'astar_manhattan': {'explored': 24, 'path_len': 17, 'cost': 16},
            'astar_zero': {'explored': 24, 'path_len': 17, 'cost': 16},
        }
    },
    {
        'name': '10x10_Grid_4_4_to_9_9',
        'graph_loader': lambda: premade_graphs.create_grid_graph(10, 10),
        'loader_code': 'premade_graphs.create_grid_graph(10, 10)',
        'start': (4, 4),
        'goal': (9, 9),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 98, 'path_len': 11, 'cost': 10},
            'dfs': {'explored': 10, 'path_len': 11, 'cost': 10},
            'dijkstra': {'explored': 99, 'path_len': 11, 'cost': 10},
            'astar_manhattan': {'explored': 23, 'path_len': 11, 'cost': 10},
            'astar_zero': {'explored': 99, 'path_len': 11, 'cost': 10},
        }
    },
    {
        'name': '10x10_Grid_Obstacles_Wall_with_hole',
        'graph_loader': lambda: premade_graphs.create_grid_graph(10, 10, obstacles=[((6, y), (7, y)) for y in range(10) if y != 5]),
        'loader_code': "premade_graphs.create_grid_graph(10, 10, obstacles=[((6, y), (7, y)) for y in range(10) if y != 5])",
        'start': (4, 4),
        'goal': (9, 9),
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 95, 'path_len': 11, 'cost': 10},
            'dfs': {'explored': 18, 'path_len': 19, 'cost': 18},
            'dijkstra': {'explored': 96, 'path_len': 11, 'cost': 10},
            'astar_manhattan': {'explored': 28, 'path_len': 11, 'cost': 10},
            'astar_zero': {'explored': 98, 'path_len': 11, 'cost': 10},
        }
    },
    {
        'name': '5x5_Grid_Severed_Unreachable',
        'graph_loader': lambda: premade_graphs.create_grid_graph(5, 5, obstacles=[((2, y), (3, y)) for y in range(5)]),
        'loader_code': "premade_graphs.create_grid_graph(5, 5, obstacles=[((2, y), (3, y)) for y in range(5)])",
        'start': (0, 0),
        'goal': (4, 4),
        'expect_failure': True,
        'algos': {
            'bfs': {'explored': 15, 'path_len': 0, 'cost': 0},
            'dfs': {'explored': 15, 'path_len': 0, 'cost': 0},
            'dijkstra': {'explored': 15, 'path_len': 0, 'cost': 0},
            'astar_manhattan': {'explored': 15, 'path_len': 0, 'cost': 0},
            'astar_zero': {'explored': 15, 'path_len': 0, 'cost': 0},
        }
    },
    {
        'name': 'Trap_Graph',
        'graph_loader': premade_graphs.create_trap_graph,
        'loader_code': 'premade_graphs.create_trap_graph()',
        'start': 'S',
        'goal': 'G',
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 4, 'path_len': 4, 'cost': 12}, 
            'dfs': {'explored': 3, 'path_len': 4, 'cost': 12},
            'dijkstra': {'explored': 4, 'path_len': 4, 'cost': 12},
            'astar_manhattan': {'explored': 4, 'path_len': 4, 'cost': 12},
            'astar_zero': {'explored': 4, 'path_len': 4, 'cost': 12},
        }
    },
    {
        'name': 'Inconsistent_Heuristic_Graph',
        'graph_loader': premade_graphs.create_inconsistent_heuristic_graph,
        'loader_code': 'premade_graphs.create_inconsistent_heuristic_graph()',
        'start': 'S',
        'goal': 'G',
        'expect_failure': False,
        'algos': {
            # Note: explored set to 5 to allow for 1 re-expansion required for optimality
            'astar_custom': {'explored': 5, 'path_len': 4, 'cost': 102}, 
            'astar_zero': {'explored': 4, 'path_len': 4, 'cost': 102},
        }
    },
    {
        'name': 'Pace_Graph_Real',
        'graph_loader': premade_graphs.create_pace_graph,
        'loader_code': 'premade_graphs.create_pace_graph()',
        'start': 'Node 143',
        'goal': 'Brooklyn Brg / Brooklyn Bridge Entrance Ramp / Rose St [65]',
        'expect_failure': False,
        'algos': {
            'bfs': {'explored': 58, 'path_len': 12, 'cost': 492.8650844762792},
            'dfs': {'explored': 204, 'path_len': 141, 'cost': 6903.0066217396425},
            'dijkstra': {'explored': 100, 'path_len': 12, 'cost': 492.8650844762792},
            'astar_haversine': {'explored': 12, 'path_len': 12, 'cost': 492.8650844762792},
            'astar_zero': {'explored': 100, 'path_len': 12, 'cost': 492.8650844762792},
        }
    },
    {
        'name': '8_Puzzle_3x3_Easy',
        'graph_loader': lambda: premade_graphs.create_sliding_tile_graph(3),
        'loader_code': 'premade_graphs.create_sliding_tile_graph(3, ...)',
        'start': ((1, 2, 3), (4, 5, 6), (7, 0, 8)),
        'goal': ((1, 2, 3), (4, 5, 6), (7, 8, 0)),
        'expect_failure': False,
        'is_sliding_tile': True,
        'algos': {
            'astar_manhattan': {'explored': 1, 'path_len': 2, 'cost': 1},
            'astar_misplaced': {'explored': 1, 'path_len': 2, 'cost': 1},
        }
    },
    {
        'name': '8_Puzzle_3x3_Medium',
        'graph_loader': lambda: premade_graphs.create_sliding_tile_graph(3),
        'loader_code': 'premade_graphs.create_sliding_tile_graph(3, ...)',
        'start': ((1, 2, 3), (4, 8, 5), (7, 0, 6)),
        'goal': ((1, 2, 3), (4, 5, 6), (7, 8, 0)),
        'expect_failure': False,
        'is_sliding_tile': True,
        'algos': {
            'astar_manhattan': {'explored': 3, 'path_len': 4, 'cost': 3},
            'astar_misplaced': {'explored': 3, 'path_len': 4, 'cost': 3},
        }
    },
    {
        'name': '15_Puzzle_4x4_Easy',
        'graph_loader': lambda: premade_graphs.create_sliding_tile_graph(4),
        'loader_code': 'premade_graphs.create_sliding_tile_graph(4, ...)',
        'start': (tuple(range(1, 5)), tuple(range(5, 9)), tuple(range(9, 13)), (13, 14, 0, 15)),
        'goal': (tuple(range(1, 5)), tuple(range(5, 9)), tuple(range(9, 13)), (13, 14, 15, 0)),
        'expect_failure': False,
        'is_sliding_tile': True,
        'algos': {
            'astar_manhattan': {'explored': 1, 'path_len': 2, 'cost': 1},
            'astar_misplaced': {'explored': 1, 'path_len': 2, 'cost': 1},
        }
    },
]

# --- Configuration ---
EFFICIENCY_TIERS = [10, 1.5, 1.1, 1]

class TestStudentAlgos(unittest.TestCase):
    pass

def run_algo(self, scenario, algo_key, timeout=2):
    """
    Helper to run an algorithm and return the result.
    Raises assertions for timeouts or critical failures.
    Returns: (graph, path, cost)
    """
    g = scenario['graph_loader']()
    g.start = scenario['start']
    g.goal = scenario['goal']
    
    if scenario.get('is_sliding_tile'):
        heuristics_map = {
            "manhattan": utils.sliding_tile_manhattan_heuristic,
            "misplaced": utils.misplaced_heuristic
        }
    else:
        heuristics_map = {
            "manhattan": utils.manhattan_heuristic,
            "zero": utils.zero_heuristic,
            "custom": utils.custom_heuristic,
            "haversine": utils.haversine_heuristic
        }

    g.reset_tracking()
    path = []
    
    try:
        with test_timeout(timeout):
            if algo_key == 'bfs':
                path = student_algos.bfs(g, g.start, g.goal)
            elif algo_key == 'dfs':
                path = student_algos.dfs(g, g.start, g.goal)
            elif algo_key == 'dijkstra':
                path = student_algos.dijkstra(g, g.start, g.goal)
            elif algo_key.startswith('astar_'):
                h_name = algo_key.replace('astar_', '')
                h_func = heuristics_map.get(h_name)
                if not h_func:
                    self.fail(f"Unknown heuristic {h_name}")
                path = student_algos.a_star(g, g.start, g.goal, h_func)
            else:
                self.fail(f"Unknown algo {algo_key}")
    except TimeoutException as e:
        self.fail(f"Timeout on {scenario['name']} - {algo_key}: {e} (Infinite loop?)")

    cost = utils.get_path_cost(g, path) if path else 0
    return g, path, cost

def make_test_correctness(scenario, algo_key, expected):
    def test(self):
        g, path, _ = run_algo(self, scenario, algo_key)
        
        if scenario['expect_failure']:
            self.assertFalse(path, f"{algo_key} found a path but expected none on {scenario['name']}")
        else:
            self.assertTrue(path, f"{algo_key} returned empty path on {scenario['name']}")
            self.assertEqual(path[0], g.start, "Path start match")
            self.assertEqual(path[-1], g.goal, "Path goal match")
            # Optional: Check validity of edges
            # for i in range(len(path)-1):
            #     self.assertTrue(g.has_edge(path[i], path[i+1]), f"Invalid edge {path[i]}->{path[i+1]}")
    return test

def make_test_cost(scenario, algo_key, expected):
    def test(self):
        if algo_key == 'dfs' or scenario['expect_failure']:
            return # Skip cost check for DFS or expected failure

        _, path, cost = run_algo(self, scenario, algo_key)
        self.assertTrue(path, "Path not found (cannot check cost)")
        self.assertAlmostEqual(cost, expected['cost'], delta=1e-7, msg=f"{algo_key} cost mismatch")
    return test

def make_test_unique_explored(scenario, algo_key, expected):
    def test(self):
        if scenario['expect_failure']:
             # If expected failure, we might still want to check unique explored count
             pass

        g, _, _ = run_algo(self, scenario, algo_key)
        self.assertLessEqual(g.stats()['unique_explored'], expected['explored'], f"{algo_key} unique explored count mismatch")
    return test

def make_test_efficiency_total(scenario, algo_key, expected, multiplier):
    def test(self):
        g, _, _ = run_algo(self, scenario, algo_key)
        
        # If 'explored' is 0, logic is trivial, skip multiplier check or handle gracefully
        limit = expected['explored'] * multiplier
        if limit < 1: limit = 1 # pathological case

        self.assertLessEqual(g.total_explorations, limit + 1, 
            f"{algo_key} total explorations ({g.total_explorations}) exceeded {multiplier}x limit ({limit}) (expected unique: {expected['explored']}). "
            "Likely inefficient re-expansion or missing visited set."
        )
    return test

# Dynamically generate test methods
for scenario in TEST_DATA:
    for algo_key, expected in scenario['algos'].items():
        # Sanitize name
        s_name = scenario['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('->', 'to').replace(',', '_')
        
        # 1. Correctness
        setattr(TestStudentAlgos, f"test_{s_name}_{algo_key}_1_Correctness", make_test_correctness(scenario, algo_key, expected))
        
        # 2. Cost
        setattr(TestStudentAlgos, f"test_{s_name}_{algo_key}_2_Cost", make_test_cost(scenario, algo_key, expected))

        # 3. Unique Explored
        setattr(TestStudentAlgos, f"test_{s_name}_{algo_key}_3_UniqueExplored", make_test_unique_explored(scenario, algo_key, expected))

        # 4. Total Explorations (Efficiency Tiers)
        for mult in EFFICIENCY_TIERS:
            # We use a nice name for the test method, replacing '.' with '_'
            mult_str = str(mult).replace('.', '_')
            test_name = f"test_{s_name}_{algo_key}_4_Efficiency_Total_{mult_str}x"
            setattr(TestStudentAlgos, test_name, make_test_efficiency_total(scenario, algo_key, expected, mult))


if __name__ == '__main__':
    unittest.main()
