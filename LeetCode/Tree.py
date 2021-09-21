from utils import *


class PhoneDirectory(object):

    def __init__(self, max_number):
        self.tree = [True] * 2 * max_number
        self.max_number = max_number

    def get(self):
        if not self.tree[1]:
            return -1
        i = 1
        while i < len(self.tree) // 2:
            if 2 * i < len(self.tree) and self.tree[2 * i]:
                i = 2 * i
            if 2 * i + 1 < len(self.tree) and self.tree[2 * i + 1]:
                i = 2 * i + 1
        res = i - self.max_number

        self.tree[i] = False
        i //= 2
        while i > 0:
            self.tree[i] = self.tree[2 * i] or self.tree[2 * i + 1]
            i //= 2
        return res

    def check(self, number):
        return -1 < number < self.max_number and self.tree[number + self.max_number]

    def release(self, number):
        i = self.max_number + number
        while i > 0:
            self.tree[i] = True
            i //= 2

class NumArray:
    # https://www.youtube.com/watch?v=CWDQJGaN1gY&t=220s
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.BITree = [0] + nums
        for i in range(1, len(nums)):
            j = i + (i & -i)
            if j < len(nums) + 1:
                self.BITree[j] += self.BITree[i]

    def preSum(self, idx):
        idx += 1
        res = 0
        while idx:
            res += self.BITree[idx]
            idx -= idx & -idx
        return res

    def update(self, idx: int, val: int) -> None:
        diff = val - self.nums[idx]
        self.nums[idx] = val
        idx += 1
        while idx < len(self.BITree):
            self.BITree[idx] += diff
            idx += idx & -idx

    def sumRange(self, left: int, right: int) -> int:
        return self.preSum(right) - self.preSum(left - 1)


x = 9
print("{0:b}".format(x))
print("{0:b}".format(x))
# print(NumArray([1,4,-2,5]))
