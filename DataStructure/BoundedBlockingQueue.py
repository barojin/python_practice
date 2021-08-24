import threading


class BoundedBlockingQueue(object):

    def __init__(self, capacity: int):
        self.arr = [0] * capacity
        self.front = 0
        self.rear = 0
        self.length = 0
        self.cap = capacity
        self.thcond = threading.Condition()

    def enqueue(self, element: int) -> None:
        with self.thcond:
            while self.size() == self.cap:
                self.thcond.wait(10)
            self.arr[self.front] = element
            self.front = (self.front + 1) % self.cap
            self.length += 1

            self.thcond.notify()

    def dequeue(self) -> int:
        with self.thcond:
            while self.size() == 0:
                self.thcond.wait(10)

            element = self.arr[self.rear]
            self.rear = (self.rear + 1) % self.cap
            self.length -= 1

            self.thcond.notify()
            return element

    def size(self) -> int:
        return self.length
