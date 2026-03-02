import time

import numpy as np

# Try importing TenSEAL, if not available, raise a warning (it requires a real Python environment with TenSEAL installed)
try:
    import tenseal as ts

    tenseal_available = True
except ImportError:
    tenseal_available = False

# Create sample matrices for GEMM
np.random.seed(42)
m, k, n = 32, 32, 128
A = np.random.randn(m, k).astype(np.float32)
B = np.random.randn(k, n).astype(np.float32)

# Perform plaintext GEMM
start_plain = time.time()
C_plain = A @ B
end_plain = time.time()
plain_time = end_plain - start_plain
plain_size_bytes = A.nbytes + B.nbytes + C_plain.nbytes

# Prepare encrypted GEMM if TenSEAL is available
if tenseal_available:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2**40
    context.generate_galois_keys()

    # Encrypt each row of A
    encrypted_A_rows = [ts.ckks_vector(context, row.tolist()) for row in A]

    # GEMM: multiply each encrypted row with B
    start_enc = time.time()
    encrypted_result_rows = []
    for enc_row in encrypted_A_rows:
        result_row = []
        for col in B.T:
            dot = enc_row.dot(col.tolist())
            result_row.append(dot)
        encrypted_result_rows.append(result_row)
    end_enc = time.time()
    enc_time = end_enc - start_enc

    # Decrypt and reconstruct result
    C_encrypted = np.array(
        [[val.decrypt()[0] for val in row] for row in encrypted_result_rows]
    )

    # Size estimate of ciphertext
    enc_size_bytes = sum([len(r.serialize()) for r in encrypted_A_rows])
else:
    C_encrypted = None
    enc_time = None
    enc_size_bytes = None

# compare the numerical difference
if tenseal_available:
    diff = np.abs(C_plain - C_encrypted)
    max_diff = np.max(diff)
    print(f"Max difference between plaintext and encrypted GEMM: {max_diff}")
else:
    print("TenSEAL not available, skipping encrypted GEMM comparison.")

import pandas as pd

data = {
    "Mode": ["Plaintext", "Encrypted (CKKS)"],
    "Time (s)": [plain_time, enc_time if enc_time else "N/A"],
    "Encrypted Size (KB)": [
        f"{plain_size_bytes / 1024:.2f} KB",
        f"{enc_size_bytes / 1024:.2f} KB" if enc_size_bytes else "N/A",
    ],
}
df = pd.DataFrame(data)
# tools.display_dataframe_to_user(name="GEMM Timing and Size Comparison", dataframe=df)
print(df)
