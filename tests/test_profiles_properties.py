from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from jax_span.profiles import gaussian, lorentzian, spectrum, voigt
import numpy as np
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("profile", [gaussian, lorentzian])
@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(1, 500),
        elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    ),
    amp=st.floats(0.0001, 100, allow_nan=False),
    center=st.floats(-100, 100, allow_nan=False),
    sigma=st.floats(0.0001, 100, allow_nan=False),
)
# def test_gaussian_prop(x, amp, center, sigma):
#     g = gaussian(x, amp, center, sigma)
#     assert bool(jnp.all(g <= amp))
#     assert bool(jnp.all(g >= 0))

# @given(
#     x = arrays(
#     dtype=np.float64,
#     shape=st.integers(1, 500),
#     elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
#     ),
#     amp=st.floats(0.0001, 100, allow_nan=False),
#     center=st.floats(-100, 100, allow_nan=False),
#     sigma=st.floats(0.0001, 100, allow_nan=False),
# )
# def test_lorentzian_prop(x, amp, center, sigma):
#     g = lorentzian(x, amp, center, sigma)
#     assert bool(jnp.all(g >= 0))
#     assert bool(jnp.all(g <= amp))
def test_profile_bounds(profile, x, amp, center, sigma):
    peak = profile(x, amp, center, sigma)
    assert bool(jnp.all(peak >= 0))
    assert bool(
        jnp.all(peak <= amp + 1e-5 * jnp.abs(amp))
    )  # The comparison float32_result <= float64_amp fails
@pytest.mark.parametrize("profile", [gaussian, lorentzian])

@given(
    d=arrays(dtype=np.float64, shape=st.integers(1, 500),
             elements=st.floats(0.0, 100, allow_nan=False, allow_infinity=False)),
    amp=st.floats(0.0001, 100, allow_nan=False),
    center=st.floats(-100, 100, allow_nan=False),
    sigma=st.floats(0.0001, 100, allow_nan=False),
)
def test_profile_symmetry(profile, d, amp, center, sigma):
    left = profile(center - d, amp, center, sigma)
    right = profile(center + d, amp, center, sigma)
    assert bool(jnp.allclose(left, right))

@given(
    n=st.integers(1, 100),
    x=arrays(dtype=np.float64, shape=st.integers(1, 500),
             elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)),
    data=st.data(),   # lets you draw dependent strategies inside the test
)
def test_spectrum_length(n, x, data):
    amps    = data.draw(arrays(np.float64, n, elements=st.floats(0.0001, 100, allow_nan=False)))
    centers = data.draw(arrays(np.float64, n, elements=st.floats(-100, 100, allow_nan=False)))
    sigmas  = data.draw(arrays(np.float64, n, elements=st.floats(0.0001, 100, allow_nan=False)))
    params  = jnp.stack([amps, centers, sigmas], axis=-1)   # (n, 3)
    peak = spectrum(params, jnp.asarray(x))
    assert peak.shape == x.shape