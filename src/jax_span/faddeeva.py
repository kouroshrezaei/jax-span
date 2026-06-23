import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import numpy as np

# Compute Weideman's coefficients a_k for given N.
N = 24
M = 2 * N
M2 = 2 * M
k = np.arange(-M + 1, M)  # length M2
L = np.sqrt(N / np.sqrt(2))  # Optimal scaling parameter

theta = k * np.pi / M
t = L * np.tan(theta / 2)

f = np.exp(-(t**2)) * (L**2 + t**2)
f = np.concatenate(([0.0], f))  # prepend 0 as in MATLAB

# FFT to get coefficients
a_full = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
a = a_full[1 : N + 1][::-1]


def faddeeva_w(z: ArrayLike) -> Array:
    """
    Compute w(z) using Weideman's rational approximation.
    Assumes Im(z) >= 0 (or handle analytic continuation separately).
    
    w(z) = exp(-z²)·erfc(-iz), the scaled complementary error function / plasma dispersion function,
    order N=24, max abs error ≈ 7e-8 vs scipy.special.wofz near the real axis at complex64 precision (note that's the float32 floor, not the method's limit),
    Weideman, J.A.C. (1994), SIAM J. Numer. Anal. 31(5), 1497–1518.
    """
    Z = (L + 1j * z) / (L - 1j * z)
    p = jnp.polyval(a, Z)
    return 2 * p / (L - 1j * z) ** 2 + (1 / jnp.sqrt(jnp.pi)) / (L - 1j * z)
