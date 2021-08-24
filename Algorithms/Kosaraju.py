from collections import defaultdict


class Solution:

    def reverseEdges(self, edges):
        newEdges = defaultdict(list)
        for i in edges:
            for j in edges[i]:
                newEdges[j].append(i)
        return newEdges

    def scc(self, edges, n):
        def dfs(node, stack):
            visited[node] = True
            for adj in edges[node]:
                if not visited[adj]:
                    dfs(adj, stack)
            stack.append(node)

        visited = [False] * n
        st = []
        for i in range(n):
            if not visited[i]:
                dfs(i, st)

        edges = self.reverseEdges(edges)
        visited = [False] * n
        res = []
        while st:
            node = st.pop()
            temp = []
            if not visited[node]:
                dfs(node, temp)
                res.append(temp)
        return res


g = defaultdict(list)
g[0].append(1)
g[1].append(2)
g[2].append(3)
g[2].append(4)
g[3].append(0)
g[4].append(5)
g[5].append(6)
g[6].append(4)
g[6].append(7)
n = 8
print(Solution().scc(g, 8))

