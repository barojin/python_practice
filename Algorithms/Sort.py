class SortCollections:

    def insert_sort(self, arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr

    def selection_sort(self, arr, size):
        n = len(arr)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if arr[j] < arr[min_idx]:
                    min_idx = i
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

    def merge_sort_top_down(self, nums):
        if len(nums) <= 1:
            return nums

        pivot = len(nums) // 2
        left_list = self.merge_sort_top_down(nums[0:pivot])
        right_list = self.merge_sort_top_down(nums[pivot:])
        return self._merge_top_down(left_list, right_list)

    def _merge_top_down(self, L, R):
        i = j = 0
        res = []
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                res.append(L[i])
                i += 1
            else:
                res.append(R[j])
                j += 1
        res.extend(L[i:])
        res.extend(R[j:])
        return res

    def quick_sort(self, arr):
        def partition(l, r):
            pivot = arr[r]
            i = l
            for j in range(l, r):
                if arr[j] < pivot:
                    arr[i], arr[j] = arr[j], arr[i]
                    i += 1
            arr[i], arr[r] = arr[r], arr[i]
            return i

        def fn(l, r):
            if l < r:
                pivot = partition(l, r)
                fn(l, pivot - 1)
                fn(pivot + 1, r)

        n = len(arr)
        fn(0, n - 1)
        return arr

x = SortCollections().quick_sort([5,4,3,8,5,4,3,1,2,3,4,8,7,6,5,4])
print(x)