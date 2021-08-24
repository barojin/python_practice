from itertools import combinations
from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        nCk, comination when pick k items in n items
        return all possible combinations of k numbers out of the range [1, n].
        :param n: total number
        :param k: number you want to select
        :return: all possible combinations of k numbers out of the range [1, n].
        """
        def f(i, path):
            if len(path) == k:
                res.append(path.copy())
            for j in range(i, n+1):
                path.append(j)
                f(j + 1, path)
                path.pop()
        res = []
        f(1, [])
        return res
