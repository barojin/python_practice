def binary_search_rightmost(self, a, target):
    import math
    l, r = 0, len(a)
    while l < r:
        m = math.floor(l + (r - l) // 2)
        if a[m] > target:
            r = m
        else:
            l = m + 1
    return r - 1