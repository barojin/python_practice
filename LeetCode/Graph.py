import bisect
import collections
import itertools

from utils import *
import heapq
from collections import defaultdict
from functools import lru_cache


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self): self.p = list(range(n))

            def union(self, x, y): self.p[self.find(x)] = self.find(y)

            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]
        uf, res, m = UF(len(s)), [], defaultdict(list)
        for x, y in pairs:
            uf.union(x, y)
        for i in range(len(s)):
            m[uf.find(i)].append(s[i])
        for comp_id in m.keys():
            m[comp_id].sort(reverse=True)
        for i in range(len(s)):
            res.append(m[uf.find(i)].pop())
        return ''.join(res)


    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        edges = defaultdict(list)
        for s, d, c in flights:
            edges[s].append((d, c))

        distances = [float('inf') for _ in range(n)]
        current_stops = [float('inf') for _ in range(n)]
        distances[src], current_stops[src] = 0, 0

        minHeap = [(0, 0, src)]  # (cost, stops, node)

        while minHeap:
            cost, stops, node = heapq.heappop(minHeap)
            if node == dst:
                return cost
            if stops > k:
                continue

            for adj, adj_cost in edges[node]:
                new_cost = cost + adj_cost
                if new_cost < distances[adj]:
                    distances[adj] = new_cost
                    heapq.heappush(minHeap, (new_cost, stops + 1, adj))
                elif stops < current_stops[adj]:
                    heapq.heappush(minHeap, (new_cost, stops + 1, adj))
                current_stops[adj] = stops
        return -1 if distances[dst] == float('inf') else distances[dst]
        return 0

    def longestCommonSubsequence(self, a: str, b: str) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = 1 + dp[i][j]
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[-1][-1]

        # @lru_cache(None)
        # def fn(i=0, j=0):
        #     if i >= len(a) or j >= len(b):
        #         return 0
        #     else:
        #         if a[i] == b[j]:
        #             return fn(i + 1, j + 1) + 1
        #         else:
        #             return max(fn(i, j + 1), fn(i + 1, j))
        #
        # return fn()

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[
        float]:

        gid_weight = {}

        def find(node_id):
            if node_id not in gid_weight:
                gid_weight[node_id] = (node_id, 1)
            group_id, node_weight = gid_weight[node_id]
            # The above statements are equivalent to the following one
            # group_id, node_weight = gid_weight.setdefault(node_id, (node_id, 1))

            if group_id != node_id:
                # found inconsistency, trigger chain update
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = \
                    (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                # merge the two groups together,
                # by attaching the dividend group to the one of divisor
                gid_weight[dividend_gid] = \
                    (divisor_gid, divisor_weight * value / dividend_weight)

        # Step 1). build the union groups
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)
        print(gid_weight)
        results = []
        # Step 2). run the evaluation, with "lazy" updates in find() function
        for (dividend, divisor) in queries:
            if dividend not in gid_weight or divisor not in gid_weight:
                # case 1). at least one variable did not appear before
                results.append(-1.0)
            else:
                dividend_gid, dividend_weight = find(dividend)
                divisor_gid, divisor_weight = find(divisor)
                if dividend_gid != divisor_gid:
                    # case 2). the variables do not belong to the same chain/group
                    results.append(-1.0)
                else:
                    # case 3). there is a chain/path between the variables
                    results.append(dividend_weight / divisor_weight)
        return results

    def calcEquation2(self, equations, values, queries):
        # Floyd-Warshall
        quot = collections.defaultdict(dict)
        for (num, den), val in zip(equations, values):
            quot[num][num] = quot[den][den] = 1.0
            quot[num][den] = val
            quot[den][num] = 1 / val
        print(quot)
        for k, i, j in itertools.permutations(quot, 3):
            print(i, j ,k)
            if k in quot[i] and j in quot[k]:
                quot[i][j] = quot[i][k] * quot[k][j]
        return [quot[num].get(den, -1.0) for num, den in queries]

    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        # prim algorithm, minimum spanning tree
        edges = defaultdict(list) # {vertice: [[cost, index]]}
        min_heap = [] # [[cost, index]]
        for i, v in enumerate(wells):
            heapq.heappush(min_heap, [v, i+1])
        for a, b, c in pipes:
            edges[a].append([c, b])
            edges[b].append([c, a])
        seen = set()
        res = 0
        while len(seen) < n:
            cost, vertex = heapq.heappop(min_heap)
            if vertex not in seen:
                seen.add(vertex)
                res += cost
            for adj_c, adj_v in edges[vertex]:
                if adj_v in seen:
                    continue
                heapq.heappush(min_heap, [adj_c, adj_v])
        return res

    def minCostToSupplyWater2(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        # unionfind, krukashal
        uf = {i: i for i in range(n + 1)}

        def find(x):
            if x != uf[x]:
                uf[x] = find(uf[x])
            return uf[x]

        w = [[c, 0, i] for i, c in enumerate(wells, 1)]
        p = [[c, i, j] for i , j, c in pipes]
        res = 0
        for c, x, y in sorted(w + p):
            x, y = find(x), find(y)
            if x != y:
                uf[find(x)] = find(y)
                res += c
                n -= 1
            if n == 0:
                return res

    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        def backtracking(origin, route):
            if len(route) == len(tickets) + 1:
                nonlocal itinerary
                itinerary = route
                return True

            for i, nextDest in enumerate(edges[origin]):
                if not seen[origin][i]:
                    seen[origin][i] = True
                    ret = backtracking(nextDest, route + [nextDest])
                    seen[origin][i] = False
                    if ret:
                        return True
            return False

        from collections import defaultdict
        edges = defaultdict(list)
        for a, b in tickets:
            bisect.insort(edges[a], b)
            # heapq.heappush(edges[a], b)
            # edges[a].append(b)

        seen = {}

        for depature, arrival_list in edges.items():
            # arrival_list.sort()
            seen[depature] = [False] * len(arrival_list)

        itinerary = []
        route = ['JFK']
        backtracking('JFK', route)
        return itinerary

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        # undirected
        # what is the condition for the critical connection?
        # 1. An edge is a critical connection, iff it is not in a cycle.

        # we should know that how to catch all cycles in the graph
        # rank of node

        # This algorithm works that all edges in cycle of subgraph are deleted
        # so only critical connections are left in conn_set (hash set)
        # How to find the edges in cycle?
        # Rank all visited nodes in ascending order e.g. first to last.
        # Do traverse nodes,
        # If the current node meets the next node who has a lower rank
        # that means it's cycle. Let this next node call a start node.
        # so we can remove edges among ancestor nodes those who have a higher rank than a start node by backtracking.
        # Period.
        def dfs(node: int, discovery_rank: int) -> int:
            # if node is visited, return its rank, this trigger the removing backtracking
            if rank[node]:
                return rank[node]
            # update the rank of node
            rank[node] = discovery_rank
            # set min_rank with float('inf') but discovery_rank + 1 is at most here
            min_rank = discovery_rank + 1
            for adj in graph[node]:
                # skip the parent node
                if rank[adj] and rank[adj] == discovery_rank - 1:
                    continue
                # Recurse on adjacent nodes
                recursive_rank = dfs(adj, discovery_rank + 1)
                if recursive_rank <= discovery_rank:
                    conn_set.remove(tuple(sorted([node, adj])))
                min_rank = min(min_rank, recursive_rank)
            return min_rank

        rank = dict.fromkeys(range(n))
        graph = defaultdict(list)
        conn_set = set()

        for u, v in connections:
            graph[u].append(v)
            graph[v].append(u)
            conn_set.add(tuple(sorted([u, v])))

        dfs(0, 0)
        return list(conn_set)

    def criticalConnections2(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        # tarzan algorithm
        edges = [[] for i in range(n)]
        times = [float('inf')] * n
        critical = []
        for a, b in connections:
            edges[a].append(b)
            edges[b].append(a)

        def dfs(current_node, discovery_time, parent_node):
            c, t, p = current_node, discovery_time, parent_node
            if times[c] < n:
                return times[c]
            times[c] = t
            for adj in edges[c]:
                if adj == p:
                    continue
                adj_min_time = dfs(adj, t + 1, c)
                if adj_min_time > t:
                    critical.append((c, adj))

                times[c] = min(times[c], adj_min_time)
            return times[c]

        dfs(n-1, 0, -1)
        return critical

#         edges = [[] for i in range(n)]
#         times = [None] * n
#         critical_connections = []

#         for a, b in connections:
#             edges[a].append(b)
#             edges[b].append(a)

#         def dfs(current_node, discovery_time):
#             if times[current_node]:
#                 return times[current_node]

#             c, d = current_node, discovery_time
#             times[c] = discovery_time

#             # go to adjacent nodes and get minimum discovery time
#             for adj in edges[c]:
#                 # times[adj]: means adj was visited so no traverse
#                 # times[adj] == d - 1: directed graph, so prevent access back to previous node
#                 if times[adj] and times[adj] == d - 1:
#                     continue

#                 adj_min_time = dfs(adj, d + 1)
#                 if adj_min_time > times[c]:
#                     critical_connections.append([c, adj])
#                 times[c] = min(times[c], adj_min_time)
#             return times[c]

#         dfs(0, 0)
#         return critical_connections
