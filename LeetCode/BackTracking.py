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
        """
        Do not return anything, modify board in-place instead.
        """
        N = 9
        row = [[True] * len(board[0]) for _ in range(len(board))]
        col = [[True] * len(board[0]) for _ in range(len(board))]
        box = [[True] * len(board[0]) for _ in range(len(board))]
        subN = 9 // 3
        get_idx = lambda x, y: x // subN * subN + y // subN
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '.':
                    continue
                k = get_idx(i, j)
                n = int(board[i][j]) - 1
                row[i][n] = False
                col[j][n] = False
                box[k][n] = False

        def btk(r, c):
            if c == len(board[0]):
                r += 1
                c = 0
            if r == len(board):
                return True

            if board[r][c] != '.':
                return btk(r, c + 1)

            k = get_idx(r, c)
            for n in range(N):
                if row[r][n] and col[c][n] and box[k][n]:
                    board[r][c] = str(n + 1)
                    row[r][n] = col[c][n] = box[k][n] = False
                    if btk(r, c + 1):
                        return True
                    board[r][c] = '.'
                    row[r][n] = col[c][n] = box[k][n] = True
            return False

        btk(0, 0)

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
