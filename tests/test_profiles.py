from jax_span.profiles import gaussian, lorentzian
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
    