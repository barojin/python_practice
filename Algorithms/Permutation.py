from typing import List
from collections import Counter


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def f(i=0):
            if i == len(nums):
                res.append(nums.copy())
            for j in range(i, len(nums)):
                nums[i], nums[j] = nums[j], nums[i]
                f(i + 1)
                nums[i], nums[j] = nums[j], nums[i]
        res = []
        f()
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def fn():
            if len(p) == len(nums):
                res.append(p[:])
            else:
                for n in cnt.keys():
                    if cnt[n] == 0:
                        continue
                    p.append(n)
                    cnt[n] -= 1
                    fn()
                    p.pop()
                    cnt[n] += 1
        res = []
        p = []
        cnt = Counter(nums)
        fn()
        return res
