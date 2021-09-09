from utils import *


import heapq
from collections import defaultdict

from functools import lru_cache
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        edges = defaultdict(list)
        for s, d, c in flights:
            edges[s].append((d, c))

        distances = [float('inf') for _ in range(n)]
        current_stops = [float('inf') for _ in range(n)]
        distances[src], current_stops[src] = 0, 0

        minHeap = [(0, 0, src)] # (cost, stops, node)

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
                    dp[i + 1][j + 1] = max(dp[i+1][j], dp[i][j+1])
        return dp[-1][-1]

        @lru_cache(None)
        def fn(i=0, j=0):
            if i >= len(a) or j >= len(b):
                return 0
            else:
                if a[i] == b[j]:
                    return fn(i+1, j+1) + 1
                else:
                    return max(fn(i, j+1), fn(i+1, j))
        return fn()

    

Solution().findCheapestPrice(3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 1)