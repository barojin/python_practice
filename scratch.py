class Solution:
    def isValidSudoku(self, grid) -> bool:
        N = 9
        # Use binary number to check previous occurrence
        rows = [0] * N
        cols = [0] * N
        boxes = [0] * N

        for r in range(N):
            for c in range(N):
                # Check if the position is filled with number
                if grid[r][c] == ".":
                    continue

                pos = int(grid[r][c]) - 1

                # Check the row
                if rows[r] & (1 << pos):
                    return False
                rows[r] |= (1 << pos)

                # Check the column
                if cols[c] & (1 << pos):
                    return False
                cols[c] |= (1 << pos)

                # Check the box
                idx = (r // 3) * 3 + c // 3
                if boxes[idx] & (1 << pos):
                    return False
                boxes[idx] |= (1 << pos)

        return True

from collections import defaultdict
d = defaultdict(set)
d[1].update({1,2,3})
d[2].update({4,5,6})
res = set()
for s in d.values():
    res |= s

print(res)

path = {(1,2), (5,6)}
print(path | {(9,8)})