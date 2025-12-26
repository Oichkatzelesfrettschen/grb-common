Constants Module
================

.. automodule:: grb_common.constants
   :members:
   :undoc-members:
   :show-inheritance:

Physical Constants
------------------

All constants are provided in CGS units (cm, g, s) and are derived from
astropy.constants for maximum precision.

Fundamental Constants
~~~~~~~~~~~~~~~~~~~~~

==================== ================= ================
Constant             Symbol            Value (CGS)
==================== ================= ================
Speed of light       ``C_LIGHT``       2.998e10 cm/s
Planck constant      ``H_PLANCK``      6.626e-27 erg s
Boltzmann constant   ``K_BOLTZMANN``   1.381e-16 erg/K
==================== ================= ================

Particle Properties
~~~~~~~~~~~~~~~~~~~

==================== ================= ================
Constant             Symbol            Value (CGS)
==================== ================= ================
Electron mass        ``M_ELECTRON``    9.109e-28 g
Proton mass          ``M_PROTON``      1.673e-24 g
Electron charge      ``Q_ELECTRON``    4.803e-10 esu
Thomson cross-section ``SIGMA_T``      6.652e-25 cm^2
==================== ================= ================

Astrophysical Constants
~~~~~~~~~~~~~~~~~~~~~~~

==================== ================= ================
Constant             Symbol            Value (CGS)
==================== ================= ================
Solar mass           ``M_SUN``         1.989e33 g
Solar luminosity     ``L_SUN``         3.828e33 erg/s
Parsec               ``PC``            3.086e18 cm
Megaparsec           ``MPC``           3.086e24 cm
Astronomical unit    ``AU``            1.496e13 cm
==================== ================= ================

Usage Example
-------------

.. code-block:: python

    from grb_common.constants import C_LIGHT, M_ELECTRON, SIGMA_T

    # Synchrotron critical frequency
    def nu_synchrotron(gamma_e, B):
        return (3 / (4 * np.pi)) * gamma_e**2 * Q_ELECTRON * B / (M_ELECTRON * C_LIGHT)

    # Compton scattering cross section ratio
    x = h_nu / (M_ELECTRON * C_LIGHT**2)
    sigma_klein_nishina = SIGMA_T * (3/4) * ((1+x)/x**3) * (...)
