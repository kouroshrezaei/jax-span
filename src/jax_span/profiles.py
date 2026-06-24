"""Peak-shape primitives and spectrum synthesis for spectral fitting."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from jax_span.faddeeva import faddeeva_w



def gaussian(
    x: ArrayLike, amp: ArrayLike, center: ArrayLike, sigma: ArrayLike
) -> Array:
    """Height-parameterized Gaussian; peak value equals ``amp`` at ``x == center``.

    The argument order (x, amp, center, sigma) is the family contract:
    ``lorentzian`` and ``voigt`` must match it, and ``spectrum``'s param
    stacking (axis=-1) unpacks in this order.

    Args:
        x: Points at which to evaluate.
        amp: Peak height.
        center: Peak location.
        sigma: Gaussian width.

    Returns:
        Gaussian evaluated at ``x``.
    """
    return amp * jnp.exp(-((x - center) ** 2) / (2 * sigma**2))


def lorentzian(
    x: ArrayLike, amp: ArrayLike, center: ArrayLike, sigma: ArrayLike
) -> Array:
    """Height-parameterized Lorentzian; peak value equals ``amp`` at ``x == center``.

    Follows the same argument-order contract as ``gaussian``.

    Args:
        x: Points at which to evaluate.
        amp: Peak height.
        center: Peak location.
        sigma: Half-width at half-maximum.

    Returns:
        Lorentzian evaluated at ``x``.
    """
    return amp * (sigma**2) / ((x - center) ** 2 + sigma**2)


def voigt(
    x: ArrayLike,
    amp: ArrayLike,
    center: ArrayLike,
    sigma: ArrayLike,
    gamma: ArrayLike,
) -> Array:
    """Height-parameterized Voigt profile; peak value equals ``amp`` at ``x == center``.

    The Voigt profile is the convolution of a Gaussian (width ``sigma``) and a
    Lorentzian (width ``gamma``), evaluated via the Faddeeva function and
    normalized so the peak height equals ``amp``.
    Note:
        Based on w(z) = exp(-z²)·erfc(-iz), valid for Im(z) >= 0 (always true
        here since gamma, sigma > 0). Runs complex64 through the Faddeeva
        approximation, so accuracy is ~1e-3 relative to peak height, not
        machine precision.
    
    Args:
        x: Points at which to evaluate.
        amp: Peak height.
        center: Peak location.
        sigma: Gaussian width.
        gamma: Lorentzian half-width.

    Returns:
        Voigt profile evaluated at ``x``.
    """
    sigma_sqrt2 = sigma * jnp.sqrt(2.0)
    z = ((x - center) + 1j * gamma) / (sigma_sqrt2)
    peak = jnp.real(faddeeva_w(1j * gamma / (sigma_sqrt2)))
    return (amp / peak) * jnp.real(faddeeva_w(z))


def spectrum(params: ArrayLike, x: ArrayLike) -> Array:     #TODO docstring should change
    """Sum of N Gaussian peaks evaluated over ``x``.

    Args:
        params: Stacked ``(N, 3)`` array of per-peak ``(amp, center, sigma)``.
        x: Points at which to evaluate the spectrum.

    Returns:
        Summed spectrum, shape ``(len(x),)``.
    """
    amp, center, sigma = params[:, 0], params[:, 1], params[:, 2]
    batch_peaks = jax.vmap(gaussian, in_axes=(None, 0, 0, 0))
    return jnp.sum(batch_peaks(x, amp, center, sigma), axis=0)