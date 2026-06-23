import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def gaussian(
    x: ArrayLike, amp: ArrayLike, center: ArrayLike, sigma: ArrayLike
) -> Array:
    """Height-parameterized Gaussian. Peak value == amp at x == center.

    Argument order (x, amp, center, sigma) is the family contract:
    lorentzian and voigt must match it, and spectrum()'s param
    stacking (axis=-1) must unpack in this order.
    """
    return amp * jnp.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentzian(
    x: ArrayLike, amp: ArrayLike, center: ArrayLike, sigma: ArrayLike
) -> Array:
    """Height-parameterized Lorentzian. Peak value == amp at x == center.

    Argument order (x, amp, center, sigma) is the family contract:
    """
    return amp * ((sigma**2) / ((x - center) ** 2 + sigma**2))
