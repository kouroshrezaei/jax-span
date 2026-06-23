import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
import jax

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

def spectrum(params: ArrayLike, x: ArrayLike) -> Array:    
    """Sum of N peaks over x, where params is stacked (N,3) array of per-peak
    (amp, center, sigma).

    returns shape (len(x),)
    """
    amp    = params[:, 0]   # all amps,    shape (N,)
    center = params[:, 1]   # all centers, shape (N,)
    sigma  = params[:, 2]   # all sigmas,  shape (N,)
    
    batch_peaks = jax.vmap(gaussian, in_axes=(None, 0, 0, 0))
    return jnp.sum(batch_peaks(x,  amp, center, sigma), axis=0)