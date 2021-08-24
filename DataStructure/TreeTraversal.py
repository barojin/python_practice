from LeetCode.utils import *


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:


    def postorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(node):
            if node:
                dfs(node.left)
                dfs(node.right)
                res.append(node.val)

        res = []
        dfs(root)
        return res

    def preorderTraversal_morris(self, root: TreeNode) -> List[int]:
        node = root
        res = []
        while node:
            if not node.left:
                res.append(node.val)
                node = node.right
            else:
                predecessor = node.left
                while predecessor.right and predecessor.right is not node:
                    predecessor = predecessor.right

                if not predecessor.right:
                    res.append(node.val)
                    predecessor.right = node
                    node = node.left
                else:
                    predecessor.right = None
                    node = node.right
        return res



    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def preorder(node):
            if node:
                res.append(node.val)
                preorder(node.left)
                preorder(node.right)

        res = []
        preorder(root)
        return res

    def preorderTraversal_iter(self, root: TreeNode) -> List[int]:
        st = [root]
        res = []

        while st:
            node = st.pop()
            if node:
                res.append(node.val)
                st.append(node.right)
                st.append(node.left)
        return res

    def inorderTraversal_iter(self, root: TreeNode) -> List[int]:
        res =[]
        st = []
        p = root
        while st or p:
            while p:
                st.append(p)
                p = p.left
            p = st.pop()
            res.append(p.val)
            p = p.right
        return res

    def postorderTraversal_iter(self, root: TreeNode) -> List[int]:
        res = []
        st = [root]
        while st:
            node = st.pop()
            if node:
                res.append(node.val)
                st.append(node.left)
                st.append(node.right)
        return reversed(res)