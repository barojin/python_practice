
class UnionFind:
    def __init__(self, n):
        """ rank on the depth of the trees """
        self.parent = list(range(n))
        self.size = [1] * n

    def make_set(self, v):
        self.parent[v] = v
        self.size[v] = 1

    def find_set(self, v):
        if v == self.parent[v]:
            return v
        return self.find_set(self.parent[v])

    def union_sets(self, a, b):
        size = self.size
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            self.parent[a] = b
            size[b] += size[a]


