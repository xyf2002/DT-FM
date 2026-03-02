import time
from math import ceil, floor, sqrt

import numpy as np
import pandas as pd


def chebyshev_generator_matrix(n, k):
    """Generate n x k generator matrix using Chebyshev polynomials evaluated over [-1, 1]."""
    x = np.linspace(-1, 1, n, dtype=np.float32)
    G = np.zeros((n, k), dtype=np.float32)
    G[:, 0] = 1
    if k > 1:
        G[:, 1] = x
        for j in range(2, k):
            G[:, j] = 2 * x * G[:, j - 1] - G[:, j - 2]
    return G


def encode_blocks_chebyshev(blocks, n):
    """Chebyshev-based encoding of matrix blocks."""
    k = len(blocks)
    block_shape = blocks[0].shape
    G = chebyshev_generator_matrix(n, k)
    flat_blocks = np.stack([b.reshape(-1) for b in blocks])
    coded_matrix = G @ flat_blocks
    coded_blocks = [coded_matrix[i].reshape(block_shape) for i in range(n)]
    return coded_blocks, G


def gaussian_generator_matrix(n, k, seed=42):
    np.random.seed(seed)
    return np.random.randn(n, k)


def encode_blocks_gaussian(blocks, n, seed=42):
    k = len(blocks)
    shape = blocks[0].shape
    G = gaussian_generator_matrix(n, k, seed)
    print(f"Gaussian matrix shape: {G.shape}")
    flat_blocks = np.stack([b.reshape(-1) for b in blocks])
    coded_matrix = G @ flat_blocks  # shape: n x D
    print(
        f"Coded matrix shape: {coded_matrix.shape}, bytes: {coded_matrix.nbytes / (1024**2):.2f} MB"
    )
    coded_blocks = [coded_matrix[i].reshape(shape) for i in range(n)]
    return coded_blocks, G


def vandermonde_matrix(n, k):
    return np.vander(np.arange(1, n + 1), k, increasing=True)


def encode_blocks(blocks, n):
    k = len(blocks)
    shape = blocks[0].shape
    G = vandermonde_matrix(n, k)
    print(f"Vandermonde matrix shape: {G.shape}")
    flat_blocks = np.stack([b.reshape(-1) for b in blocks])
    coded_matrix = G @ flat_blocks
    return [coded_matrix[i].reshape(shape) for i in range(n)], G


def decode_blocks_by_global_ids(coded_blocks, G, k, global_ids, survivor_ids):
    shape = coded_blocks[0].shape
    local_indices = [np.where(survivor_ids == gid)[0][0] for gid in global_ids]
    flat_blocks = np.stack([coded_blocks[i].reshape(-1) for i in local_indices])
    G_sub = G[global_ids, :]
    G_inv = np.linalg.inv(G_sub)
    decoded = G_inv @ flat_blocks
    return [decoded[i].reshape(shape) for i in range(k)]


def assign_tasks_irregular(devices, total_tasks):
    """Assigns a varying number of tasks to each device (at least 1 per device)."""
    tasks = (
        np.random.multinomial(total_tasks - devices, [1 / devices] * devices)
        + 1
    )
    return tasks


# Config
p = 1007  # total devices
k1, k2 = 64, 64
t = 256
n1, n2 = k1 + t, k2 + t

# Irregular task assignment (row x col blocks)
device_count = p
task_assignments = assign_tasks_irregular(device_count, n1 * n2)
print(f"Task assignments: {task_assignments}")

# Matrix setup
m, d, n = 128, 4096, 4096 * 4
# np.random.seed(42)
A = np.random.randn(m, d)
B = np.random.randn(d, n)

print(f"Matrix A shape: {A.shape}, bytes: {A.nbytes / (1024**2):.2f} MB")
print(f"Matrix B shape: {B.shape}, bytes: {B.nbytes / (1024**2):.2f} MB")

A_blocks = np.array_split(A, k1, axis=0)
B_blocks = np.array_split(B, k2, axis=1)

start = time.perf_counter()
A_coded, GA = encode_blocks_gaussian(A_blocks, n1)
B_coded, GB = encode_blocks_gaussian(B_blocks, n2)
end = time.perf_counter()
print(f"Encoding time: {end - start:.4f} seconds")

# difference between coded and original in bytes


# print matrix condition numbers
A_cond = np.linalg.cond(GA)
B_cond = np.linalg.cond(GB)
print(f"Condition number of A: {A_cond:.2e}")
print(f"Condition number of B: {B_cond:.2e}")

# Construct task-to-device mapping
task_map = []
task_id = 0
for device_id, count in enumerate(task_assignments):
    for _ in range(count):
        row = task_id // n2
        col = task_id % n2
        task_map.append((device_id, row, col))
        task_id += 1

# Build the full C grid and mask some blocks randomly
C_grid = [[A_coded[i] @ B_coded[j] for j in range(n2)] for i in range(n1)]

# np.random.seed(None)  # Reset random seed for reproducibility
# Simulate partial failures: randomly remove some device results
device_failure_rate = 0.3
num_failures = int(device_count * device_failure_rate)
failed_devices = np.random.choice(device_count, num_failures, replace=False)
C_masked = [[None for _ in range(n2)] for _ in range(n1)]
for device_id, i, j in task_map:
    if device_id not in failed_devices:
        C_masked[i][j] = C_grid[i][j]
print(f"Failed devices: {failed_devices}")
# print(f"Masked C grid: {C_masked}")

start = time.perf_counter()
# Decode rows
decoded_C_rows = []
for i in range(n1):
    row_blocks = [b for b in C_masked[i] if b is not None]
    survivor_j = np.array([j for j in range(n2) if C_masked[i][j] is not None])
    if len(row_blocks) >= k2:
        decoded_row = decode_blocks_by_global_ids(
            row_blocks, GB, k2, survivor_j[:k2], survivor_j
        )
        decoded_C_rows.append(decoded_row)
    else:
        decoded_C_rows.append(None)  # mark as undecodable
        # decoded_C_rows.append([np.zeros_like(C_grid[0][0]) for _ in range(k2)])
        # assert False, "Not enough blocks to decode row {}. Only {} blocks available.".format(i, len(row_blocks))

# Decode columns
decoded_C_matrix = []
for j in range(k2):
    col_blocks = [
        decoded_C_rows[i][j] for i in range(n1) if decoded_C_rows[i] is not None
    ]
    survivor_i = np.array(
        [i for i in range(n1) if decoded_C_rows[i] is not None]
    )
    decoded_col = decode_blocks_by_global_ids(
        col_blocks, GA, k1, survivor_i[:k1], survivor_i
    )
    decoded_C_matrix.append(decoded_col)

end = time.perf_counter()
print(f"Decoding time: {end - start:.4f} seconds")

# Assemble final result
C_final_blocks = [list(row) for row in zip(*decoded_C_matrix)]
C_final = np.block(C_final_blocks)
C_true = A @ B
is_correct = np.allclose(C_true, C_final, atol=1e-6)
max_diff = np.max(np.abs(C_true - C_final))
max_percent_diff = max_diff / np.max(np.abs(C_true)) * 100
print("Max diff: {:.6f}".format(max_diff))
print("Max percent diff: {:.6f}%".format(max_percent_diff))

# print(C_final)
# print(C_true)

# Check if the final result is correct
assert is_correct, (
    "The decoded matrix does not match the original matrix. Max diff: {}".format(
        max_diff
    )
)
print("The decoded matrix matches the original matrix.")
