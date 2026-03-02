import hashlib
import math

import numpy as np
import pandas as pd

# Define matrix dimensions
m, k, n = 128, 128, 128

# Generate random matrices A and B
np.random.seed(0)
A = np.random.randn(m, k)
B = np.random.randn(k, n)

# Simulate a poisoned result (e.g., attacker changes a single entry in AB)
AB_true = A @ B
AB_poisoned = AB_true.copy()
AB_poisoned[1, 1] += 5  # simulate a subtle poisoning attack

print("True Matrix A@B:")
print(np.sum(AB_true))  # sum over rows

# Verifier's side: compute a = sum over rows of A, b = sum over columns of B
a = A.sum(axis=0)  # shape: (k,)
b = B.sum(axis=1)  # shape: (k,)

# Use the identity: sum(AB) == dot(a, b)
expected_sum = np.dot(a, b)
true_sum = AB_true.sum()
poisoned_sum = AB_poisoned.sum()


# Simple encryption-like hash (simulated commitment to expected value)
def hash_commitment(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


commitment = hash_commitment(expected_sum)

# Verifier recomputes from observed AB (either true or poisoned)
observed_hash_true = hash_commitment(true_sum)
observed_hash_poisoned = hash_commitment(poisoned_sum)

# Compare hashes to detect poisoning
results = pd.DataFrame(
    {
        "Scenario": ["Correct Result", "Poisoned Result"],
        "Observed Sum": [true_sum, poisoned_sum],
        "Matches Commitment": [
            math.isclose(true_sum, expected_sum),
            math.isclose(poisoned_sum, expected_sum),
        ],
    }
)

print("Verification Results:")
print(results)
