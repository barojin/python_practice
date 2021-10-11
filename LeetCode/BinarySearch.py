import bisect

from utils import *

class Solution:
    def minAbsoluteSumDiff(self, A: List[int], B: List[int]) -> int:
        absdiffsum, absdiff = 0, []
        mi = float('inf')
        n = len(A)
        for i in range(n):
            diff = abs(A[i] - B[i])
            absdiffsum += diff
            absdiff.append(diff)
        A.sort()
        for i in range(len(B)):
            diff = absdiff[i]
            idx = bisect.bisect_left(A, B[i])
            if idx > 0:
                mi = min(mi, absdiffsum - diff + abs(B[i] - A[idx-1]))
            if idx < n:
                mi = min(mi, absdiffsum - diff + abs(B[i] - A[idx]))
        return mi % (10**9 + 7)

    def minAbsDifference(self, nums: List[int], goal: int) -> int:
        sets1, sets2 = set([0]), set([0])
        res, n = abs(goal), len(nums)

        for num in nums[:n // 2]:
            sets1 |= {num + prev for prev in sets1}
        for num in nums[n // 2:]:
            sets2 |= {num + prev for prev in sets2}

        sets1, sets2 = sorted(sets1), sorted(sets2)
        n1, n2 = len(sets1), len(sets2)

        l, r = 0, n2 - 1
        while l < n1 and r >= 0:
            res = min(res, abs(sets1[l] + sets2[r] - goal))
            if res == 0:
                return 0
            if sets1[l] + sets2[r] > goal:
                r -= 1
            elif sets1[l] + sets2[r] < goal:
                l += 1
        return res

    def minAbsDifference2(self, nums: List[int], goal: int) -> int:
        def getAllSubset(nums):
            ans = {0}
            for x in nums:
                ans |= {x + y for y in ans}
            return ans

        s1 = getAllSubset(nums[0:len(nums) // 2])
        s2 = getAllSubset(nums[len(nums) // 2:])
        s2 = sorted(s2)
        result = 2 * (10 ** 9)
        for s in s1:
            index = bisect.bisect(s2, goal - s)
            if index > 0:
                result = min(result, abs(goal - s - s2[index - 1]))
            if index < len(s2):
                result = min(result, abs(goal - s - s2[index]))

        return result


a = [5,-7,3,5]
b = 6
# Solution().minAbsDifference2(a, b)

a = [1,2,3,4,5]
# 1, 2, 3, 4, 5
# 1 2 / 2 3 / 3 4/ 4 5
# 1,2,3 / 2 3 4 / 3 4 5
# 1 2 3 4 / 2 3 4 5
# 1 2 3 4 5

out = [[]]
nums = [1,2,3]
for num in nums:
    out += [cur + [num] for cur in out]