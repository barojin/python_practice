class Node:
    def __init__(self, val):
        self.val = val
        self.next = None


class SinglyLinkedList:
    def __init__(self):
        self.head = Node(0)
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        ptr = self.head.next
        for _ in range(index):
            ptr = ptr.next
        return ptr.val

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return None
        self.size += 1
        ptr = self.head
        for _ in range(index):
            ptr = ptr.next
        node = Node(val)
        node.next = ptr.next
        ptr.next = node

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return None
        self.size -= 1
        ptr = self.head
        for _ in range(index):
            ptr = ptr.next
        ptr.next = ptr.next.next
