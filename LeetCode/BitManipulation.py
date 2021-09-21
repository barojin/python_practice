import sys


def is_power_two(x: int):
    # 0100 = 4, 0011 = 3,  0100 & 0011 = 0000
    return not x & (x - 1)


def count_one(x):
    cnt = 0
    while x:
        x = x & (x - 1)
        cnt += 1
    return cnt


assert count_one(sys.maxsize) == '{0:b}'.format(sys.maxsize).count('1')
assert count_one(0) == 0
assert count_one(3) == 2
assert count_one(4) == 1