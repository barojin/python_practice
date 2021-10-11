import unittest


def is_cycle_directed(n: int, edges: dict) -> bool:
    def dfs(v):
        if color[v] == 1:
            return True
        color[v] = 1
        for adj in edges.get(v, []):
            if color[v] == 2:
                continue
            if dfs(adj):
                return True
        color[v] = 2

    color = [0] * n  # 0 = white, 1 = gray, 2 = black
    # white: unvisited, gray: exploring, black: visited

    for v in range(n):
        if color[v] == 2:
            continue
        if dfs(v):
            return True
    return False


def get_all_cyclic_paths_directed(n: int, edges: dict) -> list:
    cycle_paths = []
    color = [0] * n

    # 0: unvisited, 1: exploring, 2: visited

    def get_cycle_path(start, cur, path):
        if path and start == cur:
            cycle_paths.append(path[:])
            return
        if color[cur] == 1:
            path.append(cur)
            for adj in edges.get(cur, []):
                get_cycle_path(start, adj, path)

    def dfs(cur):
        if color[cur] == 1:
            get_cycle_path(cur, cur, [])
            return
        if color[cur] == 2:
            return
        color[cur] = 1
        for adj in edges.get(cur, []):
            dfs(adj)
        color[cur] = 2

    for i in range(n):
        if not color[i]:
            dfs(i)
    return cycle_paths


def find_cycle_directed(n, edges):
    """
    ref: https://cp-algorithms.com/graph/finding-cycle.html
    We will run a series of DFS in the graph. Initially all vertices are colored white (0).
    From each unvisited (white) vertex, start the DFS, mark it gray (1) while entering and mark it black (2) on exit.
    If DFS moves to a gray vertex, then we have found a cycle
    (if the graph is undirected, the edge to parent is not considered).
    The cycle itself can be reconstructed using parent array.

    :param n: number of vertices
    :param edges: { vertex: [adjacent vertices of given key]}
    :return:
    """

    def _dfs(v):
        color[v] = 1
        for u in edges[v]:
            if color[u] == 0:
                parent[u] = v
                if _dfs(u):
                    return True
            elif color[u] == 1:
                nonlocal cycle_end, cycle_start
                cycle_end = v
                cycle_start = u
                return True
        color[v] = 2
        return False

    color = [0] * n
    parent = [-1] * n
    cycle_start = -1
    cycle_end = 0
    for v in range(n):
        if color[v] == 0 and _dfs(v):
            break
    if cycle_start == -1:
        # print("Acyclic")
        return []
    else:
        cycle = [cycle_start]
        v = cycle_end
        while v != cycle_start:
            cycle.append(v)
            v = parent[v]
        res = list(reversed(cycle))
        # print("Cyclic path:", res)
        return res


def find_cycle_undirected(n, edges):
    """
    A undirected doesn't traverse the previous vertex where already visited.
    Thus, we don't need a three status, just keep visited or not.
    :param n:
    :param edges:
    :return:
    """

    def is_cyclic_dfs(v, parent_v):
        visited[v] = True
        for u in edges.get(v, []):
            if u == parent_v:
                continue
            if visited[u]:
                nonlocal start, end
                start, end = u, v
                return True
            parent[u] = v
            if is_cyclic_dfs(u, parent[u]):
                return True
        return False

    start = -1
    end = -1
    visited = [False] * n
    parent = [-1] * n
    for v in range(n):
        if not visited[v] and is_cyclic_dfs(v, parent[v]):
            break

    if start == -1:
        # print("Acyclic")
        return []
    else:
        cycle = [start]
        v = end
        while v != start:
            cycle.append(v)
            v = parent[v]
        return cycle[::-1]


def hasDeadlock(g) -> bool:
    """
    :param g: graph, [[1,2,3], [3], ...], index = vertex, list of index = vertex's adjacent nodes
    :return:
    """

    def is_cycle(u):
        color[u] = 1
        for v in g[u]:
            if color[v] == 1 or is_cycle(v):
                return True
        color[u] = 2
        return False

    color = [0] * len(g)
    return any(is_cycle(u) for u in range(len(g)) if color[u] == 0)


class TestCycles(unittest.TestCase):
    def test_find_cycle_undirected(self):
        edges = {0: [1], 1: [0, 2], 2: [1, 0]}
        # 0 - 1 - 2 - 0
        n = 3
        self.assertEqual(find_cycle_undirected(n, edges), [1, 2, 0])
        edges = {0: [1], 1: [0, 2], 2: [1, 3, 4], 3: [2], 4: [1, 2]}
        # 0 - 1 - 2 - 3
        # 2 - 4 - 1
        n = 5
        self.assertEqual(find_cycle_undirected(n, edges), [2, 4, 1])

        edges = {0: [1], 1: [0, 2], 2: [1]}
        # 0 - 1 - 2 - 0
        n = 3
        self.assertEqual(find_cycle_undirected(n, edges), [])

    def test_find_cycle_directed(self):
        edges = {0: [1], 1: [0]}
        n = 2
        self.assertEqual(find_cycle_directed(n, edges), [1, 0])

        edges = {0: [1], 1: [2], 2: [1]}
        n = 3
        self.assertEqual(find_cycle_directed(n, edges), [2, 1])

        edges = {0: [1], 1: [2, 3], 2: [0, 4], 3: [5], 5: [3]}
        n = 6
        # print("3", get_all_cyclic_paths_undirected(n, edges))
        assert find_cycle_directed(n, edges) == [1, 2, 0]

        edges = {0: [1], 1: [2], 2: [3], 3: [1], 4: [5], 5: [4, 7]}
        n = 8
        # print("4", get_all_cyclic_paths_undirected(n, edges))
        assert find_cycle_directed(n, edges) == [2, 3, 1]

    def test_get_all_cyclic_paths_directed(self):
        # correct cases,
        # 1. cyclic cases
        _edges = {0: [1], 1: [0]}
        _n = 2
        self.assertEqual(get_all_cyclic_paths_directed(_n, _edges), [[0, 1]])

        _edges = {0: [1], 1: [2], 2: [1]}
        _n = 3
        self.assertEqual(get_all_cyclic_paths_directed(_n, _edges), [[1, 2]])

        _edges = {0: [1], 1: [2, 3], 2: [0, 4], 3: [5], 5: [3]}
        _n = 6
        # print("3", get_all_cyclic_paths_undirected(n, edges))
        assert get_all_cyclic_paths_directed(_n, _edges) == [[0, 1, 2], [3, 5]]

        _edges = {0: [1], 1: [2], 2: [3], 3: [1], 4: [5], 5: [4, 7]}
        _n = 8
        # print("4", get_all_cyclic_paths_undirected(n, edges))
        assert get_all_cyclic_paths_directed(_n, _edges) == [[1, 2, 3], [4, 5]]

        # 2. acyclic cases
        _edges = {0: [1], 1: [2, 3, 4], 2: [3], 3: [4]}
        _n = 5
        assert get_all_cyclic_paths_directed(_n, _edges) == []

    def test_is_cycle_directed(self):
        _edges = {0: [3, 4], 3: [1, 2], 1: [2], 2: [3]}
        _n = 5
        assert is_cycle_directed(_n, _edges) is True
        _edges = {0: [3, 4], 3: [1, 2]}
        _n = 5
        assert is_cycle_directed(_n, _edges) is False


if __name__ == '__main__':
    unittest.main()
