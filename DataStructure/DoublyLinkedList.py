class ListNode:

    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None


class DoublyLinkedList:

    def __init__(self):
        self.size = 0
        self.head = ListNode(0)
        self.tail = ListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def getPointerByIndex(self, index: int) -> ListNode:
        if index < 0 or index > self.size:
            return None
        if index + 1 < self.size - index:
            p = self.head.next
            for _ in range(index):
                p = p.next
        else:
            p = self.tail
            for _ in range(self.size - index):
                p = p.prev
        return p

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        return self.getPointerByIndex(index).val

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return
        p = self.getPointerByIndex(index)
        node = ListNode(val)
        node.prev = p.prev
        node.next = p
        p.prev.next = node
        p.prev = node
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return
        p = self.getPointerByIndex(index)
        p.prev.next = p.next
        p.next.prev = p.prev
        self.size -= 1
