from utils import *


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def fn(node, low=float('-inf'), high=float('inf')):
            if not node:
                return True

            if node.val <= low or node.val >= high:
                return False

            return fn(node.right, node.val, high) and fn(node.left, low, node.val)
        return fn(root)
