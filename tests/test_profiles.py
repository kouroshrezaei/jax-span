from jax_span.profiles import gaussian, lorentzian, spectrum
import jax.numpy as jnp

x = jnp.linspace(-2.0,6.0,350)
amplitude = 5.0
center = 0.2
sigma = 1.5

def test_gaussian_apex():
    peak = gaussian(jnp.array(center), amp=amplitude, center=center, sigma=sigma)
    assert peak.item() == amplitude
    # assert isinstance(peak, Array) # this will be checked via typing and mypy!
    
def test_gaussian_shape():
    peak = gaussian(x, amp=amplitude, center=center, sigma=sigma)
    assert peak.shape == x.shape
    
def test_lorentzian_hwhm():
    x_hwhm = jnp.array(center + sigma)
    peak = lorentzian(x_hwhm, amp=amplitude, center=center, sigma=sigma)
    assert peak.item() == amplitude / 2
    assert peak.shape == x_hwhm.shape
       
def test_lorentzian_apex():
    peak = lorentzian(jnp.array(center), amp=amplitude, center=center, sigma=sigma)
    assert peak.item() == amplitude
    
def test_spectrum_sum_of_two_gaussians():
    par_1 = jnp.array([5.0,3.2,1.5])
    par_2 = jnp.array([5.0,4,1.5])
    params = jnp.stack([par_1, par_2], axis=0)
    peak1 = gaussian(x, par_1[0], par_1[1], par_1[2])
    peak2 = gaussian(x, par_2[0], par_2[1], par_2[2])
    manual = peak1 + peak2

    spec = spectrum(params, x)
    
    assert spec.shape == x.shape
    assert bool(jnp.allclose(spec, manual))