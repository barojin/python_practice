from collections import defaultdict


class Solution:
    def subarraySum(self, nums, k):
        prefix_sums = {0: 1}
        acc_sum = 0
        cnt = 0
        for n in nums:
            acc_sum += n
            diff = acc_sum - k
            cnt += prefix_sums.get(diff, 0)
            prefix_sums[acc_sum] = prefix_sums.get(acc_sum, 0) + 1
        return cnt

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}  # acc_sum : index
        for i, v in enumerate(nums):
            if v in d:
                return [d[v], i]
            d[target - v] = i

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        ...

    # https://leetcode.com/problems/subarray-product-less-than-k/
    # https://leetcode.com/problems/find-pivot-index/
    # https://leetcode.com/problems/subarray-sums-divisible-by-k/
    # https://leetcode.com/problems/minimum-operations-to-reduce-x-to-zero/


a = [1,-1,1,1,1,1]
k = 3
print(Solution().subarraySum(a, k))