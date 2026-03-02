import sympy as sp

# Declare symbolic dimensions
m, k, n = 3, 3, 3

# Create indexed symbols for A[i, j], B[j, k]
A = sp.IndexedBase("A")
B = sp.IndexedBase("B")
C = sp.IndexedBase("C")
i, j, l = sp.symbols("i j l", integer=True)

# Compute symbolic sums of all rows of A to form additional row
row_sum = [sum(A[i, j] for i in range(m)) for j in range(k)]  # 1 x k row vector

# Compute symbolic sums of all columns of B to form additional column
col_sum = [
    sum(B[i, j] for j in range(n)) for i in range(k)
]  # k x 1 column vector

# Append row_sum to A
A_aug = sp.Matrix(m + 1, k, lambda i, j: row_sum[j] if i == m else A[i, j])

# Append col_sum to B
B_aug = sp.Matrix(k, n + 1, lambda i, j: col_sum[i] if j == n else B[i, j])


# Display the augmented matrix A
print("Augmented Matrix A:")
sp.pprint(A_aug)

# Display the augmented matrix B
print("Augmented Matrix B:")
sp.pprint(B_aug)

# Compute augmented matrix multiplication C_aug = A_aug * B_aug
C_aug = A_aug * B_aug

print("Augmented Matrix C:")
sp.pprint(C_aug)

C_matrix = sp.Matrix(
    m,
    n,
    lambda i, j: sp.simplify(sp.Sum(A[i, l] * B[l, j], (l, 0, k - 1)).doit()),
)
# Display the symbolic matrix
print("Symbolic GEMM Result:")
sp.pprint(C_matrix)
