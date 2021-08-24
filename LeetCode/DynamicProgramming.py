from utils import *

class Solution:
    def longestCommonSubsequence(self, a: str, b: str) -> int:

        @lru_cache(None)
        def fn(p1, p2):
            if p1 == len(a) or p2 == len(b):
                return 0

            opt1 = fn(p1 + 1, p2)

            first_occurence = b.find(a[p1], p2)
            opt2 = 0
            if first_occurence != -1:
                opt2 = 1 + fn(p1 + 1, first_occurence + 1)

            return max(opt1, opt2)

        return fn(0, 0)

a = "abcde"
b = "ace"
x = Solution().longestCommonSubsequence(a, b)
print(x)