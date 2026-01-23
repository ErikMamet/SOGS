import numpy as np
from scipy.stats import norm

###############################################################################
# 1) Gaussian Equal-Probability Quantization (16-bit)
###############################################################################

def gaussian_int16_quantize(r):
    r_data = r.reshape(-1)

    # Defensive: ensure input is non-empty
    if r_data.size == 0:
        raise ValueError("gaussian_int16_quantize: input array 'r' is empty; cannot quantize an empty tensor")

    # Gaussian fit
    mu = r_data.mean()
    sigma = r_data.std(ddof=1)

    # Standard normal transform
    z = (r_data - mu) / sigma

    # Uniform on (0,1)
    u = norm.cdf(z)
    u = np.clip(u, 1e-12, 1 - 1e-12)

    # Map to int16 index [0,65535]
    q = np.floor(u * 65536).astype(np.uint16)

    return q.reshape(r.shape), mu, sigma


def gaussian_int16_dequantize(q, mu, sigma):
    # Midpoint probability reconstruction
    u = (q.astype(np.float32) + 0.5) / 65536
    u = np.clip(u, 1e-12, 1 - 1e-12)

    z = norm.ppf(u)
    return mu + sigma * z


###############################################################################
# 2) Uniform Quantization (16-bit)
###############################################################################

def uniform_quantize(r, num_bits): #num bits has to be under 16 (10-12 or 16)
    print("size of r in uniform quantize: ", r.size)
    r_min = r.min()
    r_max = r.max()
    scale = 2**num_bits / (r_max - r_min)

    if num_bits > 8:
        q = np.round((r - r_min) * scale).clip(0, 2**num_bits-1).astype(np.uint16)
    if num_bits == 8:
        q = np.round((r - r_min) * scale).clip(0, 2**num_bits-1).astype(np.uint8)
    return q, r_min, r_max

def uniform_dequantize(q, r_min, r_max, num_bits):
    scale = (r_max - r_min) / 2**num_bits
    return q.astype(np.float32) * scale + r_min


def custom_quantize(r, q_type="uniform", num_bits=12):
    if q_type == "gaussian":
        raise NotImplementedError(f"Gaussian quantization is not implemented.")
    elif q_type == "uniform":
        if num_bits <= 16:
            return uniform_quantize(r, num_bits=num_bits)
        else:
            raise NotImplementedError(f"Uniform quantization with {num_bits} bits>16 not implemented.")
    else:
        raise ValueError(f"Unknown quantization type: {q_type}")

def custom_dequantize(q, r_min, r_max, q_type="uniform", num_bits=16):
    if q_type == "gaussian":
        NotImplementedError(f"Gaussian dequantization is not implemented, only 16 bits is implemented")
    elif q_type == "uniform":
        if num_bits <= 16:
            return uniform_dequantize(q, r_min, r_max, num_bits=num_bits)
        else:
            raise NotImplementedError(f"Uniform dequantization with {num_bits} bits not implemented.")
    else:
        raise ValueError(f"Unknown dequantization type: {q_type}")
###############################################################################
# 3) Quantization Error Measurement
###############################################################################

def quantization_mse(original, reconstructed):
    """Mean squared error"""
    return np.mean((original - reconstructed) ** 2)


def compare_quantization_schemes(r):
    # Gaussian scheme
    q_g, mu, sigma = gaussian_int16_quantize(r)
    r_g = gaussian_int16_dequantize(q_g, mu, sigma)
    mse_gaussian = quantization_mse(r, r_g)

    # Uniform scheme
    q_u, r_min, r_max = custom_quantize(r, q_type="uniform", num_bits=16)
    r_u = custom_quantize(q_u, r_min, r_max)
    mse_uniform = quantization_mse(r, r_u)

    print("=== Quantization MSE Comparison ===")
    print(f"Gaussian Quantization MSE: {mse_gaussian:.8f}")
    print(f"Uniform Quantization MSE:  {mse_uniform:.8f}")
    print("-----------------------------------")
    print("Winner: Gaussian" if mse_gaussian < mse_uniform else "Winner: Uniform")

    return {
        "gaussian_mse": mse_gaussian,
        "uniform_mse": mse_uniform,
        "gaussian_params": (mu, sigma),
        "uniform_params": (r_min, r_max),
        "q_gaussian": q_g,
        "q_uniform": q_u,
        "r_gaussian": r_g,
        "r_uniform": r_u
    }
