from utils import *


class Robot: # This is the robot's control interface. You should not implement it, or speculate about its implementation
    def move(self):
        """
        Returns true if the cell in front is open and robot moves into the cell.
        Returns false if the cell in front is blocked and robot stays in the current cell.
        :rtype bool
        """
        pass

    def turnLeft(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """
        pass

    def turnRight(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """
        pass

    def clean(self):
        """
        Clean the current cell.
        :rtype void
        """
        pass


class Solution:
    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """

        def btk(cell, d):
            """
            cell = (i, j) = current robot's coordinate
            d = direction = 0 is up, 1 right, 2 down, 3 left

            using the backtracking, clean, mark visited, move robot, come back to starting point
            """
            visited.add(cell)
            robot.clean()
            for i in range(len(directions)):
                nd = (i + d) % len(directions)
                ncell = (directions[nd][0] + cell[0], directions[nd][1] + cell[1])
                if ncell not in visited and robot.move():
                    btk(ncell, nd)
                    robot.turnRight()
                    robot.turnRight()
                    robot.move()
                    robot.turnRight()
                    robot.turnRight()
                robot.turnRight()
            return None

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        visited = set()
        btk((0, 0), 0)

    def solveSudoku(self, board: List[List[str]]) -> None:

        def returnMinItem(possible_values):
            return min(possible_values.items(), key=lambda x: len(x[1]))[0]
            i, j = next(iter(possible_values))
            min_value_len = len(possible_values[i, j])
            for k, v in possible_values.items():
                if len(v) == 1:
                    return k
                if len(v) < min_value_len:
                    (i, j), min_value_len = k, len(v)
            return i, j

        def placeNextDigit(board, possible_values):
            i, j = returnMinItem(possible_values)
            numbers = possible_values.pop((i, j))

            for n in numbers:
                board[i][j] = n
                if not possible_values:
                    return

                discarded = []

                for (i2, j2), v in possible_values.items():
                    if (i == i2 or j == j2 or (i // 3, j // 3) == (i2 // 3, j2 // 3)) and n in v:
                        if len(v) == 1:
                            for v in discarded:
                                v.add(n)
                            possible_values[i, j] = numbers
                            return
                        v.discard(n)
                        discarded.append(v)

                placeNextDigit(board, possible_values)

                if not possible_values:
                    return

                for v in discarded:
                    v.add(n)

            possible_values[i, j] = numbers

        possible_values = {(i, j): {"1", "2", "3", "4", "5", "6", "7", "8", "9"} \
                                   - {board[i][k] for k in range(9)} \
                                   - {board[k][j] for k in range(9)} \
                                   - {board[3 * (i // 3) + di][3 * (j // 3) + dj]
                                      for di in range(3) for dj in range(3)}
                           for i in range(9) for j in range(9)
                           if board[i][j] == '.'}

        i, j = returnMinItem(possible_values)
        while possible_values and len(possible_values[i, j]) == 1:
            for n in possible_values.pop((i, j)):
                board[i][j] = n
                for (i2, j2), v in possible_values.items():
                    if (i == i2 or j == j2 or (i // 3, j // 3) == (i2 // 3, j2 // 3)) and n in v:
                        v.discard(n)
            if possible_values:
                i, j = returnMinItem(possible_values)
        if possible_values:
            placeNextDigit(board, possible_values)

    def generateParenthesis(self, n: int) -> List[str]:
        def fn(a=[]):
            if len(a) == 2*n:
                if isValid(a):
                    res.append(''.join(a))
                else:
                    a.append('(')
                    fn(a)
                    a.pop()
                    a.append(')')
                    fn(a)
                    a.pop()

        def isValid(a):
            bal = 0
            for c in a:
                if c == '(':
                    bal += 1
                else:
                    bal -= 1
                if bal < 0:
                    return False
            return bal == 0

        res = []
        fn()
        return res


print(Solution().solveSudoku([["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]))
