import math

sqrt2 = math.sqrt(2)

def richardson_archimedes(n):
    A = [[] for i in range(n)]
    A[0] = [2]
    fourn = 4

    for i in range(1, n):
        A[i] = [0 for j in range(i + 1)]
        A_im0 = A[i-1][0]
        A[i][0] = sqrt2 * A_im0 / math.sqrt(1 + math.sqrt(1 - A_im0 * A_im0 / fourn))

        for j in range(1, i + 1):
            A[i][j] = (4**j * A[i][j-1] - A[i-1][j-1]) / (4**j - 1)

        fourn *= 4

    return A

def main():
    A = richardson_archimedes(10)
    for A_i in A:
        print(A_i)

main()
