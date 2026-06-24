from jax_span.faddeeva import faddeeva_w
from scipy.special import wofz
import numpy as np
import jax.numpy as jnp
import jax

def test_faddeeva_matches_wofz():
    re = jnp.linspace(-4.0, 4.0, 50)
    im = jnp.linspace(0.01, 2.0, 50)  
    RE, IM = jnp.meshgrid(re, im)
    z = jax.lax.complex(RE, IM)        
    oracle = wofz(np.asarray(z, dtype=np.complex128))
    approx = faddeeva_w(z)
    max_err = jnp.max(jnp.abs(approx - oracle))
    assert max_err < 1e-6