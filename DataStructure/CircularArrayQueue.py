class CircularArrayQueue(object):

    def __init__(self, capacity: int):
        self.arr = [0] * capacity
        self.front = 0
        self.rear = 0
        self.length = 0
        self.cap = capacity

    def enqueue(self, element: int) -> None:
        if self.size() == self.cap:
            return

        self.arr[self.front] = element
        self.front = (self.front + 1) % self.cap
        self.length += 1

    def dequeue(self) -> int:
        if self.size() == 0:
            return -1

        element = self.arr[self.rear]
        self.rear = (self.rear + 1) % self.cap
        self.length -= 1

        return element

    def size(self) -> int:
        return self.length
