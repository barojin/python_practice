import time

n = 1000
a = [1] * n
start = time.time()
b = a.copy()
first = time.time() - start
print("1. ", first)

start = time.time()
c = a[:]
second = time.time() - start
print("2. ", second)
#asd
if first > second:
    print("Second is faster")
else:
    print("First is faster")

a = [1,2,3]
a.sort(reverse=True)
print(a)