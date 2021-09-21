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

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        d = {}
        res = []

        def find(i):
            if i not in d:
                d[i] = (i, 1)
            group_i, w = d[i]

            if group_i != i:
                new_group_id, group_w = find(group_i)
                d[i] = (new_group_id, w * group_w)
            return d[i]

        def union(dividend, divisor, value):
            x, x_w = find(dividend)
            y, y_w = find(divisor)
            if x != y:
                d[x] = (y, y_w * value / x_w)

        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        for (dividend, divisor) in queries:
            if dividend not in d or divisor not in d:
                res.append(-1.0)
            else:
                x, x_w = find(dividend)
                y, y_w = find(divisor)
                if x != y:
                    res.append(-1.0)
                else:
                    res.append(x_w / y_w)
        return res

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


x = [3,5,7]
print(bisect.bisect_left(x, 1))