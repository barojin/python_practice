import collections
from collections import OrderedDict


class LRUCache(OrderedDict):
    def __init__(self, capacity) -> None:
        self.capacity = capacity
        super().__init__()

    def get(self, k: int) -> int:
        if k not in self:
            return -1
        self.move_to_end(k)
        return self.__getitem__(k)

    def put(self, k, v):
        if k in self:
            self.move_to_end(k)
        super().__setitem__(k, v)
        if len(self) > self.capacity:
            del self[next(iter(self))]


class DoublyLinkedNode:
    def __init__(self, key=0, value=0, prev=None, nxt=None):
        self.key = key
        self.value = value
        self.prev = prev
        self.next = nxt


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.head, self.tail = DoublyLinkedNode(), DoublyLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_node(self, node: DoublyLinkedNode):
        p = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail

    def _remove_node(self, node: DoublyLinkedNode):
        node.prev.next, node.next.prev = node.next, node.prev

    def get(self, key):
        if key in self.cache:
            # fetch the node from the cache
            node = self.cache[key]
            # update the LRU
            self._remove_node(node)
            self._add_node(node)
            return node.value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        # took out the node from the list since it is used
        if key in self.cache:
            self._remove_node(self.cache[key])

        # put new node into the list
        node = DoublyLinkedNode(key, value)
        self._add_node(node)
        # store the addr of the node of the list
        self.cache[key] = node

        # if cache is exceeded, evict the least recently used key
        if len(self.cache) > self.capacity:
            first_node = self.head.next
            self._remove_node(first_node)
            del self.cache[first_node.key]


from typing import List


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


from collections import OrderedDict
from collections import defaultdict


class LFUCache:

    def __init__(self, capacity: int):
        self.cap = capacity
        self.cacheFreq = defaultdict(OrderedDict)
        self.cacheKey = {}
        self.leastFreq = 1

    def _evict(self):
        key, _ = self.cacheFreq[self.leastFreq].popitem(last=False)
        del self.cacheKey[key]

    def _update(self, key, new_v=None):
        # get the current freq and value
        freq, value = self.cacheKey[key]['freq'], self.cacheKey[key]['value']
        # remove the old one
        del self.cacheFreq[freq][key]

        if not self.cacheFreq[self.leastFreq]:
            self.leastFreq += 1
        # update the cache with freq + 1
        self.cacheKey[key] = {'freq': freq + 1, 'value': new_v or value}
        # append the cacheFreq[freq + 1] = new_vlaue or value
        self.cacheFreq[freq + 1][key] = new_v or value

    def get(self, key: int) -> int:
        if key in self.cacheKey:
            self._update(key)
            return self.cacheKey[key]['value']
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cacheKey:
            self._update(key, value)
        else:
            freq = 1
            self.cacheKey[key] = {'freq': freq, 'value': value}
            self.cacheFreq[freq][key] = value
            if len(self.cacheKey) > self.cap:
                self._evict()
            self.leastFreq = 1


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return
        self.dp = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]
        dp = self.dp
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                dp[r + 1][c + 1] = dp[r + 1][c] + dp[r][c + 1] + matrix[r][c] - dp[r][c]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        dp = self.dp
        return dp[row2 + 1][col2 + 1] - dp[row1][col2 + 1] - dp[row2 + 1][col1] + dp[row1][col1]


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

class Solution:
    def numRescueBoats(self, people, limit):
        people.sort(reverse=True)
        i, j = 0, len(people) - 1
        while i <= j:
            if people[i] + people[j] <= limit: j -= 1
            i += 1
        return i

    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        # tarzan algorithm
        graph = [[] for i in range(n)]

        for a, b in connections:
            graph[a].append(b)
            graph[b].append(a)

        lows = [n] * n
        critical = []

        def dfs(node, time, parent):
            if lows[node] == n:
                lows[node] = time
                for neighbor in graph[node]:
                    if neighbor != parent:
                        discovery_t = dfs(neighbor, time + 1, node)
                        if discovery_t > time:
                            critical.append((node, neighbor))

                        lows[node] = min(lows[node], discovery_t)
            return lows[node]

        dfs(0, 0, -1)
        return critical

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        from collections import deque
        q = deque()
        res = []
        for i in range(k):
            # make the queue continue to contain the largest number's index on index 0, the left most index
            # if there is a smaller number than comming new number, pop
            while q and nums[q[-1]] < nums[i]:
                q.pop()

            # put the index in sliding queue
            q.append(i)
        # then the sliding queue is consists of indices and the left most index indicates the largest number and others stored in the queue in comming order
        res.append(nums[q[0]])

        for i in range(k, len(nums)):
            # update the sliding queue, when i = i + 1, the number on the left most index of the nums is popped.
            if q[0] == i - k:
                q.popleft()

            # make the queue bigger head
            while q and nums[q[-1]] < nums[i]:
                q.pop()
            q.append(i)
            res.append(nums[q[0]])
        return res

    def maximumAverageSubtree(self, root: TreeNode) -> float:
        self.max_value = 0

        def dfs(node):
            if not node: return [0, 0.0]
            n1, s1 = dfs(node.left)
            n2, s2 = dfs(node.right)
            n = n1 + n2 + 1  # number of nodes
            s = s1 + s2 + root.val  # summ of values of ndoes
            self.max_value = max(self.res, s / n)
            return [n, s]

        dfs(root)
        return self.max_value

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        from functools import lru_cache
        job_cnt = len(jobDifficulty)
        if job_cnt < d: return -1

        @lru_cache(None)
        def min_score(last_score=0, cur_idx=0, div_left=d - 1):
            if div_left == 0:
                return max([last_score] + jobDifficulty[cur_idx:])
            cur_score = max(last_score, jobDifficulty[cur_idx])

            if job_cnt - cur_idx == div_left + 1:
                return cur_score + sum(jobDifficulty[cur_idx + 1:])
            join_score = min_score(cur_score, cur_idx + 1, div_left)
            div_score = cur_score + min_score(0, cur_idx + 1, div_left - 1)
            return min(join_score, div_score)

        return min_score()

    def romanToInt(self, s: str) -> int:
        if not s:
            return 0
        D = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = D[s[0]]
        N = len(s)
        for i in range(1, N):
            res += D[s[i]]
            if D[s[i]] > D[s[i - 1]]:
                res -= D[s[i - 1]] * 2
        return res

    def cutOffTree(self, forest: List[List[int]]) -> int:
        # Add sentinels (a border of zeros) so we don't need index-checks later on.
        forest.append([0] * len(forest[0]))
        for row in forest:
            row.append(0)

        # Find the trees.
        trees = [(height, i, j)
                 for i, row in enumerate(forest)
                 for j, height in enumerate(row)
                 if height > 1]

        # Can we reach every tree? If not, return -1 right away.
        queue = [(0, 0)]
        reached = set()
        for i, j in queue:
            if (i, j) not in reached and forest[i][j]:
                reached.add((i, j))
                queue += (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)
        if not all((i, j) in reached for (_, i, j) in trees):
            return -1

        # Distance from (i, j) to (I, J).
        def distance(i, j, I, J):
            now, soon = [(i, j)], []
            expanded = set()
            manhattan = abs(i - I) + abs(j - J)
            detours = 0
            while True:
                if not now:
                    now, soon = soon, []
                    detours += 1
                i, j = now.pop()
                if (i, j) == (I, J):
                    return manhattan + 2 * detours
                if (i, j) not in expanded:
                    expanded.add((i, j))
                    for i, j, closer in (i + 1, j, i < I), (i - 1, j, i > I), (i, j + 1, j < J), (i, j - 1, j > J):
                        if forest[i][j]:
                            (now if closer else soon).append((i, j))

        # Sum the distances from one tree to the next (sorted by height).
        trees.sort()
        return sum(distance(i, j, I, J) for (_, i, j), (_, I, J) in zip([(0, 0, 0)] + trees, trees))

    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        from collections import defaultdict
        from collections import deque
        from functools import lru_cache

        graph = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                graph[word[:i] + "*" + word[i + 1:]].append(word)
        pred = defaultdict(list)
        queue = deque()
        queue.append((beginWord, 0))
        visited = set()
        visited.add(beginWord)
        while queue:
            curr_word, curr_level = queue.popleft()
            for i in range(len(curr_word)):
                for word in graph[curr_word[:i] + "*" + curr_word[i + 1:]]:
                    if not pred[word] or pred[word][0][1] == curr_level + 1:
                        pred[word].append((curr_word, curr_level + 1))
                    if word not in visited:
                        queue.append((word, curr_level + 1))
                        visited.add(word)

        @lru_cache
        def dfs(word):
            if word == beginWord:
                return [[word]]
            else:
                ans = []
                for next_word, level in pred[word]:
                    for sol in dfs(next_word):
                        ans.append(sol + [word])
                return ans

        return dfs(endWord)

    def maxPathSum(self, root: TreeNode) -> int:
        def get_maxgain(node):
            if not node:
                return 0

            gain_left = max(get_maxgain(node.left), 0)
            gain_right = max(get_maxgain(node.right), 0)

            cur_maxgain = node.val + gain_left + gain_right
            self.maxgain = max(self.maxgain, cur_maxgain)

            return node.val + max(gain_left, gain_right)

        self.maxgain = root.val
        get_maxgain(root)
        return self.maxgain

    def numberToWords(self, num: int) -> str:
        def one(num):
            d = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return d.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)

        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)

        def two(num):
            if not num:
                return ''
            elif num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)

        def three(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return one(hundred) + ' Hundred ' + two(rest)
            elif not hundred and rest:
                return two(rest)
            elif hundred and not rest:
                return one(hundred) + ' Hundred'

        if not num:
            return 'Zero'

        billion_unit = 1000000000
        million_unit = 1000000
        thousand_unit = 1000
        billion = num // billion_unit
        million = (num - billion * billion_unit) // million_unit
        thousand = (num - billion * billion_unit - million * million_unit) // thousand_unit
        rest = num - billion * billion_unit - million * million_unit - thousand * thousand_unit

        result = ''
        if billion:
            result = three(billion) + ' Billion'
        if million:
            if result: result += ' '
            result += three(million) + ' Million'
        if thousand:
            if result: result += ' '
            result += three(thousand) + ' Thousand'
        if rest:
            if result: result += ' '
            result += three(rest)
        return result

    def lengthOfLongestSubstring(self, s: str) -> int:
        last_index_dict = {}  # dict stores the last index of the character
        res = 0
        # the first index of the substring which contains non repeating characters
        idx_first_curSubstring = 0
        for i, c in enumerate(s):
            if c in last_index_dict:
                idx_first_curSubstring = max(idx_first_curSubstring, last_index_dict[c])

            res = max(res, i - idx_first_curSubstring + 1)
            last_index_dict[c] = i + 1
        return res

    def reverseParentheses(self, s):
        opened = []
        pair = {}
        for i, c in enumerate(s):
            if c == '(':
                opened.append(i)
            if c == ')':
                j = opened.pop()
                pair[i], pair[j] = j, i
        res = []
        i, d = 0, 1
        while i < len(s):
            if s[i] in '()':
                i = pair[i]
                d = -d
            else:
                res.append(s[i])
            i += d
        return ''.join(res)

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.res = 0

        def dfs(node):
            if node:
                l = dfs(node.left)
                r = dfs(node.right)
                self.res = max(self.res, l + r)
                return max(l, r) + 1
            else:
                return 0

        dfs(root)
        return self.res

    def canFinish(self, n, prerequisites):
        graph = [[] for _ in range(n)]  # relationshiop pre = [nex, nex ...]
        indegrees = [0] * n  # indegree of each course
        for nx, pr in prerequisites:
            graph[pr].append(nx)
            indegrees[nx] += 1

        bfs = []  # get the starting courses
        for i in range(n):
            if indegrees[i] == 0:
                bfs.append(i)

        for pre in bfs:
            # pre is the course which doesn't hold a precourse anymore
            for nex in graph[pre]:  # pre course done, take the nex course
                indegrees[nex] -= 1
                if indegrees[nex] == 0:
                    bfs.append(nex)
        return len(bfs) == n  # if there is a cyclic, not added into bfs so False

    def toLowerCase(self, str: str) -> str:
        # res = []
        # for c in str:
        #     ordinal = ord(c)
        #     print(c)
        #     if 65 <= ordinal <= 90:
        #         res.append(chr(ordinal + 32))
        #     else:
        #         res.append(c)
        # return ''.join(res)

        # is_upper = lambda x : 'A' <= x <= 'Z'
        # to_lower = lambda x : chr(ord(x) | 32)
        # return ''.join([to_lower(x) if is_upper(x) else x for x in str])

        upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lower = "abcdefghijklmnopqrstuvwxyz"

        h = dict(zip(upper, lower))
        return ''.join([h[x] if x in h else x for x in str])

    def uniqueMorseRepresentations(self, words) -> int:
        # v = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        # k = 'abcdefghijklmnopqrstuvwxyz'
        # D = dict(zip(k, v))
        # completedSet = set()
        # for w in words:
        #     mc = []
        #     for c in w:
        #         mc.append(D[c])
        #     completedSet.add(''.join(mc))
        # return len(completedSet)

        MORSE = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.",
                 "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        return len({''.join(MORSE[ord(c) - ord('a')] for c in word) for word in words})

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        ln = len(nums)
        ans = [0] * ln

        ans[0] = 1
        for i in range(1, ln):
            ans[i] = nums[i - 1] * ans[i - 1]

        R = 1
        for i in reversed(range(ln)):
            ans[i] = ans[i] * R
            R *= nums[i]

        return ans

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def cut(i, j):
            if isConnected[i][j]:
                isConnected[j][j] = 0
                for k in range(j + 1, N):
                    cut(j, k)

        res = 0
        N = len(isConnected)
        for i in range(N):
            if isConnected[i][i] == 1:
                res += 1
                for j in range(i + 1, N):
                    cut(i, j)
        return res

    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        import collections, heapq
        # count = collections.Counter(words)
        # candidates = sorted(count.keys(), key=lambda w: (-count[w], w))
        # return candidates[:k]

        count = collections.defaultdict(int)
        for w in words:
            count[w] -= 1

        h = []
        for word, ncnt in count.items():
            heapq.heappush(h, (ncnt, word))
        return [heapq.heappop(h)[1] for _ in range(k)]

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        # at least, one word after its identifier
        # sort letter-logs
        # no nothing on digit-logs

        def cmpfunc(data: list):
            idf, log = data.split(' ', 1)
            if log[0].isalpha():
                return (0, log, idf)
            else:
                return (1, None, None)
            # self is the tuple comparison
            # >>> (1,3) > (1, 4)
            # False
            # >>> (1, 4) < (2,2)
            # True
            # >>> (1, 4, 1) < (2, 1)
            # True

        return sorted(logs, key=cmpfunc)

    def mostCommonWord(self, P: str, B: List[str]) -> str:
        import re
        import collections

        ban_set = set(B)
        tp = re.findall(r'\w+', P.lower())
        return collections.Counter(w for w in tp if w not in ban_set).most_common(1)[0][0]

    def trap(self, H: List[int]) -> int:
        l, r = 0, len(H) - 1
        left_max, right_max = 0, 0
        res = 0
        while l < r:
            if H[l] < H[r]:
                left_max = max(H[l], left_max)
                res += left_max - H[l]
                l += 1
            else:
                right_max = max(H[r], right_max)
                res += right_max - H[r]
                r -= 1
        return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        d = dict()
        for num in nums:
            if num not in d:
                d[num] = 1
            else:
                d[num] += 1
        result = []
        if 0 in d and d[0] > 2:
            result.append([0, 0, 0])

        pos = [num for num in d if num > 0]
        neg = [num for num in d if num < 0]

        for p in pos:
            for n in neg:
                other = -(p + n)
                if other in d:
                    if other == p and d[p] > 1:
                        result.append([n, p, p])
                    elif other == n and d[n] > 1:
                        result.append([n, n, p])
                    elif n < other < p:
                        result.append([n, other, p])
        return result

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        diff = float('inf')

        for i in range(len(nums)):
            l, r = i + 1, len(nums) - 1
            while l < r:
                summ = nums[i] + nums[l] + nums[r]
                cur_diff = target - summ
                if abs(cur_diff) < abs(diff):
                    diff = cur_diff
                if summ < target:
                    l += 1
                else:
                    r -= 1

            if diff == 0:
                break
        return target - diff

    def rotate(self, m: List[List[int]]) -> None:
        m[:] = map(list, zip(*m[::-1]))

    def productExceptSelf(self, d: List[int]) -> List[int]:
        N = len(d)
        c = [1] * N

        for i in reversed(range(N - 1)):
            c[i] = c[i + 1] * d[i + 1]

        r = 1
        for i in range(N):
            c[i] = c[i] * r
            r = r * d[i]
        return c

    def missingNumber(self, nums: List[int]) -> int:
        # xor
        # missing = len(nums)
        # for i in range(len(nums)):
        #     missing ^= i ^ nums[i]
        # return missing

        # xor with reduce
        from functools import reduce
        import operator
        return reduce(operator.xor, nums + list(range(len(nums) + 1)))

        # Gauss formula
        # N = len(nums)
        # gauss_sum = (N * (N + 1))//2
        # summ = sum(nums)
        # return gauss_sum - summ

    def firstUniqChar(self, s: str) -> int:
        from collections import Counter
        ctr = Counter(s)

        for i, c in enumerate(s):
            if ctr[c] == 1:
                return i
        return -1

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ptr = ListNode()
        carry = 0

        while l1 or l2 or carry:
            val1 = val2 = 0
            if l1:
                val1 = l1.val
                l1 = l1.next
            if l2:
                val2 = l2.val
                l2 = l2.next

            sm = val1 + val2 + carry
            carry, sm = divmod(sm, 10)
            ptr.next = ListNode(sm)
            ptr = ptr.next

        return head.next

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
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

        # head = ListNode()
        # ptr = head

        # while l1 and l2:
        #     if l1.val <= l2.val:
        #         ptr.next = l1
        #         l1 = l1.next
        #     else:
        #         ptr.next = l2
        #         l2 = l2.next
        #     ptr = ptr.next
        #
        # # ptr.next = l1 if l1 else l2
        # ptr.next = l1 or l2
        #
        # return head.next

    def reverseKGroup(self, head, k):
        ### recursive way
        # i, node = 0, head
        #
        # # pass if head has more than k nodes, else return head
        # while node:
        #     i += 1
        #     node = node.next
        # if k <= 1 or i < k:
        #     return head
        #
        # pre, cur = None, head
        # for _ in range(k):
        #     next_node = cur.next
        #     cur.next = pre
        #     pre = cur
        #     cur = next_node
        #     # cur.next, prev, cur = prev, cur, cur.next
        # head.next = self.reverseKGroup(cur, k)
        # return pre

        ### iterative way
        def reverseLinkedList(head, k):
            new_head, ptr = None, head
            for _ in range(k):
                next_node = ptr.next
                ptr.next = new_head
                new_head = ptr
                ptr = next_node
            return new_head

        ptr = head
        ktail = None
        new_head = None

        while ptr:
            count = 0
            ptr = head

            while count < k and ptr:
                ptr = ptr.next
                count += 1

            if count == k:
                revHead = reverseLinkedList(head, k)
                if not new_head:
                    new_head = revHead
                if ktail:
                    ktail.next = revHead
                ktail = head
                head = ptr
        if ktail:
            ktail.next = head

        return new_head if new_head else head

    def isValidBST(self, root, left=float('-inf'), right=float('inf')):
        return not root or left < root.val < right and \
               self.isValidBST(root.left, left, root.val) and \
               self.isValidBST(root.right, root.val, right)

    def minWindow(self, s: str, t: str) -> str:
        from collections import Counter
        dt = Counter(t)
        i = 0
        l, r = 0, len(s)
        len_t = len(t)
        res = ""

        for j, c in enumerate(s):
            if dt[c] > 0:  # len_t decreased only c in t
                len_t -= 1
            dt[c] -= 1

            while len_t == 0:  # if len_t != 0, move j to right to contain c in t
                dt[s[i]] += 1
                if dt[s[i]] > 0:
                    len_t += 1
                    if j - i < r - l:
                        r, l = j, i
                        res = s[l:r + 1]
                i += 1
                if i > j:  # to avoid the infinite loop
                    break
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        D = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        return []

    def generateParenthesis(self, n: int) -> List[str]:

        def inner(string: str, num_open: int, num_close: int):
            if num_open == 0:
                return [string + ")" * num_close]

            if num_open == num_close:
                return inner(string + "(", num_open - 1, num_close)

            return inner(string + "(", num_open - 1, num_close) + inner(string + ")", num_open, num_close - 1)

        return inner("", n, n)

    def exist(self, board: List[List[str]], word: str) -> bool:
        R = len(board)
        C = len(board[0])

        def backtrack(i, j, suffix):
            if len(suffix) == 0:
                return True
            if i < 0 or i == R or j < 0 or j == C or \
                    board[i][j] != suffix[0]:
                return False

            res = False

            board[i][j] = '#'
            for x, y in [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j]]:
                res = backtrack(x, y, suffix[1:])
                if res:
                    break
            board[i][j] = suffix[0]

            return res

        for i in range(R):
            for j in range(C):
                if backtrack(i, j, word):
                    return True
        return False

    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        WORD_KEY = '$'

        trie = {}
        for word in words:
            node = trie
            for letter in word:
                # retrieve the next node; If not found, create a empty node.
                node = node.setdefault(letter, {})
            # mark the existence of a word in trie node
            node[WORD_KEY] = word

        rowNum = len(board)
        colNum = len(board[0])

        matchedWords = []

        def backtracking(row, col, parent):

            letter = board[row][col]
            currNode = parent[letter]

            # check if we find a match of word
            word_match = currNode.pop(WORD_KEY, False)
            if word_match:
                # also we removed the matched word to avoid duplicates,
                #   as well as avoiding using set() for results.
                matchedWords.append(word_match)

            # Before the EXPLORATION, mark the cell as visited
            board[row][col] = '#'

            # Explore the neighbors in 4 directions, i.e. up, right, down, left
            for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                newRow, newCol = row + rowOffset, col + colOffset
                if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                    continue
                if not board[newRow][newCol] in currNode:
                    continue
                backtracking(newRow, newCol, currNode)

            # End of EXPLORATION, we restore the cell
            board[row][col] = letter

            # Optimization: incrementally remove the matched leaf node in Trie.
            if not currNode:
                parent.pop(letter)

        for row in range(rowNum):
            for col in range(colNum):
                # starting from each of the cells
                if board[row][col] in trie:
                    backtracking(row, col, trie)

        return matchedWords

    def search(self, nums: List[int], target: int) -> int:
        def binarySearch(l, r):
            while l <= r:
                m = l + (r - l) // 2
                if nums[m] == target:
                    return m
                elif nums[m] < target:
                    l = m + 1
                else:
                    r = m - 1
            return -1

        # begins #
        if not nums: return -1

        l, r = 0, len(nums) - 1
        while l < r:  # find the index of smallest number which is a pivot
            m = l + (r - l) // 2
            if nums[m] > nums[r]:
                l = m + 1
            else:
                r = m
        pivot = l
        if nums[pivot] <= target <= nums[-1]:
            return binarySearch(pivot, len(nums) - 1)
        else:
            return binarySearch(0, pivot)

        # l, r = 0, len(nums) - 1
        #
        # while l <= r:
        #     m = l + (r - l) // 2
        #     if nums[m] == target:
        #         return m
        #     elif nums[l] <= nums[m]:
        #         if nums[l] <= target < nums[m]:
        #             r = m - 1
        #         else:
        #             l = m + 1
        #     elif nums[m] < nums[l]:
        #         if nums[m] < target <= nums[r]:
        #             l = m + 1
        #         else:
        #             r = m - 1
        # return -1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) < 2:
            return intervals

    def findMedianSortedArrays(self, nums1, nums2):
        a, b = sorted((nums1, nums2), key=len)  # len(a) < len(b)
        total = len(a) + len(b)
        half = (total + 1) // 2
        l, r = 0, len(a) - 1
        while True:
            m = l + (r - l) // 2
            i = half - m - 2
            al = a[m] if m >= 0 else float('-inf')
            ar = a[m + 1] if m + 1 < len(a) else float('inf')
            bl = b[i] if i >= 0 else float('-inf')
            br = b[i + 1] if i + 1 < len(b) else float('inf')
            if al > br:
                r = m - 1
            elif bl > ar:
                l = m + 1
            else:
                if total % 2:  # odd
                    return float(max(al, bl))
                return (max(al, bl) + min(ar, br)) / 2  # even

        print("Error")

    def findKthLargest(self, nums: List[int], k: int) -> int:
        import random

        def partition(l, r, p):
            pivot = nums[p]
            nums[p], nums[r] = nums[r], nums[p]
            s = l
            for i in range(l, r):
                if nums[i] < pivot:
                    nums[s], nums[i] = nums[i], nums[s]
                    s += 1
            nums[r], nums[s] = nums[s], nums[r]
            return s

        def select(l, r):
            if l == r:
                return nums[l]
            p = random.randint(l, r)
            p = partition(l, r, p)
            pos = len(nums) - k
            if pos == p:
                return nums[pos]
            elif pos < p:
                return select(l, p - 1)
            else:
                return select(p + 1, r)

        return select(0, len(nums) - 1)

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        res = 0

        return res

    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""

        if len(s) == 1 or s == s[::-1]:
            return s

        max_len, start = 1, 0
        for i in range(1, len(s)):
            even = s[i - max_len: i + 1]
            odd = s[i - max_len - 1: i + 1]
            if i - max_len - 1 >= 0 and odd == odd[::-1]:
                start = i - max_len - 1
                max_len += 2
                continue
            if i - max_len >= 0 and even == even[::-1]:
                start = i - max_len
                max_len += 1
                continue
        return s[start:start + max_len]

    def maxSubArray(self, nums: List[int]) -> int:
        if not nums: return 0
        maxsum = cursum = nums[0]

        for i in range(1, len(nums)):
            if nums[i] > cursum:
                cursum = nums[i]
            else:
                cursum += nums[i]
            if cursum > maxsum:
                maxsum = cursum
        return maxsum

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        wordset = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in wordset:
                    dp[i] = True
                    break
        return dp[len(s)]

    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        # without sorting,
        C = defaultdict(int)
        freqD = defaultdict(int)
        max_freq = float('-inf')

        for num in arr:
            C[num] += 1

        for freq in C.values():
            freqD[freq] += 1
            max_freq = max(max_freq, freq)

        rem = len(C)
        for f in range(1, max_freq + 1):
            if f <= k:
                rem -= min(freqD[f], k // f)
                k -= freqD[f]
            else:
                return rem
        return rem

    def minFallingPathSum(self, arr: List[List[int]]) -> int:
        import heapq
        if not arr or not arr[0]:
            return 0
        n = len(arr)
        m = len(arr[0])

        for i in range(1, n):
            q = heapq.nsmallest(2, arr[i - 1])
            for j in range(m):
                arr[i][j] += q[0] if q[0] != arr[i - 1][j] else q[1]
        return min(arr[-1])

    def maxVowels(self, s: str, k: int) -> int:
        length = 0
        vowels = {'a', 'e', 'i', 'o', 'u'}
        res = 0
        for i in range(len(s)):
            if s[i] in vowels:
                length += 1
            else:
                length = 0
            res = max(res, length)

        return min(res, k)

    def numIslands(self, grid: List[List[str]]) -> int:
        from collections import deque
        def dfs(x, y):
            q = deque()
            grid[x][y] = '0'
            q.append((x, y))
            while q:
                x, y = q.popleft()
                for di, dj in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if 0 <= di < n and 0 <= dj < m and grid[di][dj] == '1':
                        grid[di][dj] = '0'
                        q.append((di, dj))

        n = len(grid)
        m = len(grid[0])
        res = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    res += 1
                    dfs(i, j)
        return res

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        def explore(q, seen: dict, others_seen: dict):
            word, lvl = q.popleft()
            for i in range(wn):
                sw = word[:i] + '*' + word[i + 1:]
                for w in words_dict[sw]:
                    if w in others_seen:
                        return lvl + others_seen[w]
                    if w not in seen:
                        seen[w] = lvl + 1
                        q.append((w, lvl + 1))
            return None

        # main
        wn = len(beginWord)
        words_dict = defaultdict(list)
        for word in wordList:
            for i in range(wn):
                words_dict[word[:i] + '*' + word[i + 1:]].append(word)
        from collections import deque
        bq = deque([(beginWord, 1)])
        eq = deque([(endWord, 1)])
        b_seen = {beginWord: 1}
        e_seen = {endWord: 1}
        while bq and eq:
            res = explore(bq, b_seen, e_seen)
            if res:
                return res
            res = explore(eq, e_seen, b_seen)
            if res:
                return res
        return 0

    def minKnightMoves(self, x: int, y: int) -> int:
        from functools import lru_cache
        @lru_cache(None)
        def dp(i, j):
            if i + j == 0:
                return 0
            if i + j == 2:
                return 2
            return min(dp(i - 2, j - 1), dp(i - 1, j - 2)) + 1

        return dp(x, y)

    def maxProfit(self, prices) -> int:
        profit = 0  # profit in the first transaction
        min_price = prices[0]  # minimum price at first transaction and second tran
        max_mid_profit = float('-inf')
        profit2 = 0  # profit in the second transaction
        for price in prices:
            min_price = min(min_price, price)
            profit = max(price - min_price, profit)
            # Assume that sell the stock now at minimum p and sell it at maximum p,
            # we can get the maximum profit with two transactions.
            # Thus, store mid_profit is max(profit - price[i] after selling in all cases)
            max_mid_profit = max(max_mid_profit, profit - price)
            profit2 = max(profit2, max_mid_profit + price)
        return profit2

    # solution = https://leetcode.com/problems/house-robber/discuss/156523/From-good-to-great.-How-to-approach-most-of-DP-problems.
    def rob(self, a: List[int]) -> int:
        ### 1. recursive relation
        # f(0) = a[0] = max(f(-2) + a[0], f(-1))
        # f(1) = max(f(-1) + a[1], f(0))
        # f(2) = max(f(0) + a[2], f(1))
        # f(i) = max(f(i-2) + a[i], f(i-1))

        ### 2. top-down
        # def f(i):
        #     # if i == -2 or i == -1: return 0 => if i < 0: return 0
        #     if i < 0:
        #         return 0
        #     return max(f(i-2) + a[i], f(i-1))
        # return f(len(a)-1)

        ### 3. top-down with memo
        # n = len(a)
        # memo = [-1] * (n + 1)
        # def f(i):
        #     if i < 0:
        #         return 0
        #     if memo[i] > -1:
        #         return memo[i]
        #     memo[i] = max(f(i-2) + a[i], f(i-1))
        #     return memo[i]
        # return f(n-1)

        ### 4. bottom up, iterative
        # n = len(a)
        # if n > 1:
        #     a[1] = max(a[0], a[1])
        # for i in range(2, n):
        #     a[i] = max(a[i-2] + a[i], a[i-1])
        # return a[-1]

        ### 5. bottom up with constant memory
        prev2 = 0  # a[i-2]
        prev1 = 0  # a[i-1]
        for i in range(len(a)):
            prev1, prev2 = max(prev2 + a[i], prev1), prev1
        return prev1

# https://leetcode.com/problems/maximal-square/discuss/600149/Python-Thinking-Process-Diagrams-DP-Approach
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        if not matrix[0]:
            return 0
        n = len(matrix)
        m = len(matrix[0])
        dp = [0] * (m + 1)
        side = prev = 0
        for i in range(n):
            for j in range(m):
                temp = dp[j + 1]
                if matrix[i][j] == '1':
                    dp[j + 1] = min(prev, dp[j], dp[j + 1]) + 1
                    side = max(side, dp[j + 1])
                else:
                    dp[j + 1] = 0
                prev = temp
        return side ** 2
    #         n = len(matrix)Ò
    #         if n == 0:
    #             return 0
    #         m = len(matrix[0])
    #         if m == 0:
    #             return 0

    #         dp = [[0] * (m + 1) for _ in range(n + 1)]
    #         side = 0
    #         for i in range(n):
    #             for j  in range(m):
    #                 if matrix[i][j] == '1':
    #                     dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
    #                     side = max(side, dp[i+1][j+1])
    #         return side * side

    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        import bisect
        # sort it with end time in ascending order
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
        # make the dp, [end time, accumulative profit]
        dp = [[0, 0]]
        for s, e, p in jobs:
            # using the bisect, find the idx which ends before current start
            i = bisect.bisect(dp, [s+1]) - 1
            # if the new combo non-overlapping profit > current maximum accumulative profit, append it to dp with current end time
            new_profit = dp[i][1] + p
            last_profit = dp[-1][1]
            if new_profit > last_profit:
                dp.append([e, new_profit])
        return dp[-1][1]

    def findShortestSubArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        from collections import Counter
        # I can keep the number, count, first idx and last idx for it,
        # return (last idx - first idx + 1) which has the most count
        if not nums:
            return 0
        counts = Counter(nums)
        freq = max(counts.values())
        if freq == 1: return 1
        elements = [k for k, v in counts.items() if v == freq]
        if len(elements) == 1:
            e = elements[0]
            last_index = len(nums) - nums[::-1].index(e) - 1
            return last_index - nums.index(e) + 1
        else:
            answer = len(nums)
            for e in elements:
                last_index = len(nums) - nums[::-1].index(e) - 1
                temp = last_index - nums.index(e) + 1
                answer = min(temp, answer)
            return answer

    def maxProduct(self, nums: List[int]) -> int:
        res = 0
        max_n = nums[0]
        min_n = nums[0]
        for i in range(1, len(nums)):
            a = nums[i] * max_n
            b = nums[i] * min_n
            temp = max(nums[i], a, b)
            min_n = min(nums[i], a, b)
            max_n = temp
            res = max(res, max_n)
        return res

    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0 # amount = 0, coin # = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                # x = the current amount
                # dp[x] = the fewest number of coins to make up the amount x
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

    def change(self, m: int, coins: List[int]) -> int:
        dp = [0] * (m + 1)
        dp[0] = 1
        for coin in coins:
            for x in range(coin, m + 1):
                dp[x] = dp[x] + dp[x - coin]
        return dp[m]

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        a = sorted(intervals, key=lambda x: x[0])
        res = [a[0]]
        for i in range(1, len(a)):
            # no overlapping
            if res[-1][1] < a[i][0]:
                res.append(a[i])
            else:
                res[-1][1] = max(res[-1][1], a[i][1])
        return res

    def isRobotBounded(self, instructions: str) -> bool:
        i = j = 0
        direction = 0
        # up right down left
        move = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        move_length = len(move)

        for c in instructions:
            if c == 'G':
                i, j = move[direction][0] + i, move[direction][1] + j
            elif c == 'L':
                direction -= 1
                if direction == -1:
                    direction = 3
            elif c == 'R':
                direction = (direction + 1) % move_length
        return True if i == 0 and j == 0 else False

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        cur = []
        num_letters = 0
        for w in words:
            number_of_space = len(cur)
            fit = num_letters + len(w) + number_of_space
            if fit > maxWidth:
                if len(cur) == 1:
                    res.append(cur[0] + ' ' * (maxWidth - num_letters))
                else:
                    num_spaces = maxWidth - num_letters
                    space_between_words, num_extra_spaces = divmod(num_spaces, len(cur) - 1)
                    for i in range(num_extra_spaces):
                        cur[i] += ' '
                    res.append((' ' * space_between_words).join(cur))

                cur = []
                num_letters = 0

            cur += [w]
            num_letters += len(w)

        res.append(' '.join(cur) + ' ' * (maxWidth - num_letters - len(cur) + 1))
        return res

    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        from functools import lru_cache
        @lru_cache(None)
        def f(i):
            if i >= len(days):
                return 0
            j = i
            res = float('inf')
            for k, extra in enumerate([1, 7, 30]):
                while j < len(days) and days[j] < days[i] + extra:
                    j += 1
                res = min(res, f(j) + costs[k])
            return res
        return f(0)

    def minimumOneBitOperations(self, n: int) -> int:
        def op1(num):
            return num ^ 1

        def op2(num, i):
            b = bin(num)
            if b[i-1] == 1 and b[i-2] == 0:
                b[i]
        a = bin(n)

    def duplicateZeros(self, arr):
        k = arr.count(0)
        n = len(arr)
        for i in range(n - 1, -1, -1):
            if i + k < n:
                arr[i + k] = arr[i]
            if arr[i] == 0:
                k -= 1
                if i + k < n:
                    arr[i + k] = 0

        return None


    def checkIfExist(self, arr: List[int]) -> bool:
        double = set(arr)

        for a in arr:
            if a * 2 in double:
                return True
        return False

    def subarraysWithKDistinct(self, A, K):
        return self.atMostK(A, K) - self.atMostK(A, K - 1)

    def atMostK(self, A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            if count[A[j]] == 0:
                K -= 1
            count[A[j]] += 1

            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0:
                    K += 1
                i += 1

            res += j - i + 1
        return res

    def findMaxLength(self, nums: List[int]) -> int:
        cnt = 0
        max_length = 0
        table = {0: 0}
        for i, num in enumerate(nums, 1):
            if num == 0:
                cnt -= 1
            else:
                cnt += 1

            if cnt in table:
                max_length = max(max_length, i - table[cnt])
            else:
                table[cnt] = i
        return max_length

    def sortedSquares(self, nums: List[int]) -> List[int]:
        i, j = 0, len(nums) - 1
        for idx, n in enumerate(nums):
            nums[idx] = n * n
        res = collections.deque()
        while i < j:
            if nums[i] > nums[j]:
                res.appendleft(nums[i])
                i += 1
            else:
                res.appendleft(nums[j])
                j -= 1
        return list(res)

    def smallestSubsequence(self, s: str) -> str:
        last = {c: i for i, c in enumerate(s)}
        st = []
        for i, c in enumerate(s):
            if c in st:
                continue
            while st and st[-1] > c and i < last[st[-1]]:
                st.pop()
            st.append(c)
        return ''.join(st)

    def triangleNumber(self, nums):
        nums.sort()
        cnt = 0
        for k in range(reversed(len(nums) - 2)):
            i = 0
            j = k - 1
            while i < j:
                if nums[i] + nums[j] > nums[k]:
                    cnt += j - i
                    j -= 1
                else:
                    i += 1
        return cnt

s = 'bcabc'
x = Solution().smallestSubsequence(s)
print(x)