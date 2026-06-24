import numpy as np
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

"""Faddeeva function w(z) via Weideman's rational approximation."""
# Order of the rational approximation. N=24 yields max abs error ~7e-8 vs
# scipy.special.wofz near the real axis (the float32 floor, not the method's limit).
_N = 24
_L = np.sqrt(_N / np.sqrt(2))  # optimal scaling parameter (Weideman 1994)


def _weideman_coeffs(n: int, scale: float) -> np.ndarray:
    """Compute Weideman's coefficients a_k for the order-n approximation.

    Args:
        n: Order of the approximation.
        scale: Optimal scaling parameter L.

    Returns:
        Array of n coefficients, ordered for use with ``jnp.polyval``.
    """
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m)  # length m2

    theta = k * np.pi / m
    t = scale * np.tan(theta / 2)

    f = np.exp(-(t**2)) * (scale**2 + t**2)
    f = np.concatenate(([0.0], f))  # prepend 0 as in MATLAB reference

    a_full = np.real(np.fft.fft(np.fft.fftshift(f))) / m2
    return a_full[1 : n + 1][::-1]


_A = _weideman_coeffs(_N, _L)


def faddeeva_w(z: ArrayLike) -> Array:
    """Compute the Faddeeva function w(z) via Weideman's rational approximation.

    w(z) = exp(-z²)·erfc(-iz) is the scaled complementary error function (also
    the plasma dispersion function). Assumes Im(z) >= 0; handle analytic
    continuation separately otherwise.

    Reference:
        Weideman, J.A.C. (1994), SIAM J. Numer. Anal. 31(5), 1497-1518.

    Args:
        z: Complex argument(s).

    Returns:
        w(z), evaluated elementwise.
    """
    denom = _L - 1j * z
    zz = (_L + 1j * z) / denom
    p = jnp.polyval(_A, zz)
    return 2 * p / denom**2 + (1 / jnp.sqrt(jnp.pi)) / denom