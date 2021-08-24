from utils import *

class Solution:

    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # Iterative
        head = ListNode()
        ptr = head
        while l1 and l2:
            if l1.val <= l2.val:
                ptr.next = l1
                l1 = l1.next
            else:
                ptr.next = l2
                l2 = l2.next
            ptr = ptr.next
        ptr.next = l1 or l2
        return head.next

        # Recursive
        if not l1:
            return l2
        elif not l2:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

    def myPow(self, x: float, n: int) -> float:
        # Iterative
        if n < 0:
            x = 1 / x
            n = -n
        acc = 1
        while n:
            if n & 1:
                acc *= x
            x *= x
            n >>= 1
        return acc

        # Tail Recursion
        def fn(x, n, acc):
            if n == 0:
                return acc
            elif n % 2 == 1:
                return fn(x, n - 1, acc * x)
            elif n % 2 == 0:
                return fn(x * x, n // 2, acc)

        if n < 0:
            return fn(1 / x, -n, 1)
        return fn(x, n, 1)

    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def rec(i, j):
            if i < j:
                s[i], s[j] = s[j], s[i]
                rec(i + 1, j - 1)

        rec(0, len(s) - 1)

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Recursive
        if not head or not head.next:
            return head

        a = head
        b = head.next
        a.next = self.swapPairs(head.next.next)
        b.next = a
        return b

        # Iterative
        dummy = ListNode()
        dummy.next = head
        prev = dummy
        while head and head.next:
            a = head
            b = head.next

            prev.next = b
            a.next = b.next
            b.next = a

            prev = a
            head = a.next
        return dummy.next

    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        nxt = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return nxt

        prev = None
        while head:
            nxt = head.next
            head.next = prev
            prev = head
            head = nxt
        return prev

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val == val:
            return root
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return self.searchBST(root.right, val)

    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1

        dp = [0] * (n + 1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    def maxDepth(self, root: Optional[TreeNode], depth=0) -> int:
        # BFS
        if not root:
            return 0
        depth = 0
        dq = deque([root])
        while dq:
            depth += 1
            size = len(dq)
            for _ in range(size):
                node = dq.popleft()
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
        return depth

        # Iterative DFS
        max_depth = 0
        st = [] if not root else [(root, depth + 1)]
        while st:
            node, depth = st.pop()
            max_depth = max(max_depth, depth)
            if node.left:
                st.append((node.left, depth + 1))
            if node.right:
                st.append((node.right, depth + 1))
        return max_depth

        # Recursive DFS
        if not root:
            return depth
        res = max(self.maxDepth(root.left, depth + 1), self.maxDepth(root.right, depth + 1))
        return res

    def kthGrammar(self, n: int, k: int) -> int:
        pass

    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:

        def node(val, left, right):
            node = TreeNode(val)
            node.left = left
            node.right = right
            return node

        def fn2(first, last):
            if first > last:
                return [None]

            res = []
            for root in range(first, last + 1):
                for left in fn(first, root - 1):
                    for right in fn(root + 1, last):
                        res.append(node(root, left, right))
            return res

        def fn(first, last):
            return [node(root, left, right)
                    for root in range(first, last + 1)
                    for left in fn(first, root - 1)
                    for right in fn(root + 1, last)] or [None]

        return fn2(1, n)

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # recursive
        """
        def inorder(node):
            if node:
                inorder(node.left)
                res.append(node.val)
                inorder(node.right)

        res = []
        inorder(root)
        return res
        """
        st = []
        res = []
        cur = root
        while st or cur:
            while cur:
                st.append(cur)
                cur = cur.left
            cur = st.pop()
            res.append(cur.val)
            cur = cur.right

        return res

print(Solution().generateTrees(3))