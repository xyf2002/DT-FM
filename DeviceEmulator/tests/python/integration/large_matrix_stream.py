import logging

import numpy as np
import torch

# Set up logging
log_file = "matrix_comparison.log"
logging.basicConfig(
    filename=log_file, level=logging.INFO, format="%(asctime)s %(message)s"
)


def compare_matrices(mat1, mat2, epsilon=1e-6):
    return np.allclose(mat1.cpu().numpy(), mat2, atol=epsilon)


size = 2000

A = torch.ones((size, size), dtype=torch.float32)
B = torch.ones((size, size), dtype=torch.float32) * 2

expected_value = 2 * size
C_expected = np.full((size, size), expected_value, dtype=np.float32)

n_iterations = 100
discrepancies = 0

for i in range(n_iterations):
    logging.info(f"Iteration {i + 1}")
    C_intercepted = torch.matmul(A, B)

    if not compare_matrices(C_intercepted, C_expected):
        logging.info("Discrepancy found in matrix multiplication results")
        discrepancies += 1

        diff = np.abs(C_intercepted.cpu().numpy() - C_expected)
        logging.info(f"Max difference: {np.max(diff)}")
        logging.info(f"Mean difference: {np.mean(diff)}")
        logging.info(f"First 3x3 of the difference matrix:\n{diff[:3, :3]}")
    else:
        logging.info("Matrix multiplication results match")

    logging.info("First 3x3 values of A:")
    logging.info(f"\n{A[:3, :3]}")
    logging.info("First 3x3 values of B:")
    logging.info(f"\n{B[:3, :3]}")
    logging.info("Expected result (C_expected) first 3x3:")
    logging.info(f"\n{C_expected[:3, :3]}")
    logging.info("Intercepted result (C_intercepted) first 3x3:")
    logging.info(f"\n{C_intercepted[:3, :3]}")
    logging.info("\n")

logging.info(
    f"Number of discrepancies found: {discrepancies} out of {n_iterations} iterations"
)
