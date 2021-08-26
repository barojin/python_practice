from utils import *

class Solution:
    def longestCommonSubsequence(self, a: str, b: str) -> int:
        dp = [0] * (len(b) + 1)
        for i in range(len(a)):
            prev = 0
            for j in range(len(b)):
                adig = prev
                prev = dp[j + 1]
                dp[j + 1] = adig + 1 if a[i] == b[j] else max(dp[j], dp[j + 1])
        return dp[-1]


a = "abcde"
b = "ace"
x = Solution().longestCommonSubsequence(a, b)
print(x)