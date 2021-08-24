class Solution:
    # @param A : integer
    # @param B : integer
    # @return an integer
    def gcd(self, A, B):
        if B == 0:
            return A
        return self.gcd(B, A % B)

    """ Recursive function to return gcd of a and b."""
    def gcd_euclidean(self, a, b):
        # Everything divides 0
        if a == 0:
            return b
        if b == 0:
            return a

        # base case
        if a == b:
            return a

        # a is greater
        if a > b:
            return self.gcd_euclidean(a - b, b)
        return self.gcd_euclidean(a, b - a)
