"""
Physical constants for GRB astrophysics in CGS units.

All constants are derived from astropy.constants and converted to CGS.
This ensures consistency with the standard astrophysics literature.

Constants are organized into categories:
    - Fundamental: c, h, hbar, k_B, G, sigma_SB
    - Particle masses: m_e, m_p, m_n
    - Electromagnetic: e, sigma_T, alpha_fs
    - Solar/astronomical: M_sun, R_sun, L_sun, AU, pc, Mpc
    - GRB-specific: Derived quantities for synchrotron emission

Usage:
    from grb_common.constants import C_LIGHT, M_ELECTRON, SIGMA_T

    # Lorentz factor from velocity
    beta = v / C_LIGHT
    gamma = 1 / np.sqrt(1 - beta**2)
"""

from astropy import constants as const
from astropy import units as u

# Speed of light
C_LIGHT: float = const.c.cgs.value  # cm/s
C_LIGHT_CGS: float = C_LIGHT
C_LIGHT_KMS: float = const.c.to(u.km / u.s).value

# Planck constant
H_PLANCK: float = const.h.cgs.value  # erg s
HBAR: float = const.hbar.cgs.value  # erg s

# Boltzmann constant
K_BOLTZMANN: float = const.k_B.cgs.value  # erg/K

# Gravitational constant
G_NEWTON: float = const.G.cgs.value  # cm^3 g^-1 s^-2

# Stefan-Boltzmann constant
SIGMA_SB: float = const.sigma_sb.cgs.value  # erg cm^-2 s^-1 K^-4

# Radiation constant
A_RAD: float = 4 * SIGMA_SB / C_LIGHT  # erg cm^-3 K^-4

# Elementary charge
E_CHARGE: float = const.e.gauss.value  # statcoulomb (esu)

# Electron mass
M_ELECTRON: float = const.m_e.cgs.value  # g

# Proton mass
M_PROTON: float = const.m_p.cgs.value  # g

# Neutron mass
M_NEUTRON: float = const.m_n.cgs.value  # g

# Atomic mass unit
M_AMU: float = const.u.cgs.value  # g

# Thomson cross-section
SIGMA_T: float = const.sigma_T.cgs.value  # cm^2

# Fine structure constant
ALPHA_FS: float = const.alpha.value  # dimensionless

# Solar mass
M_SUN: float = const.M_sun.cgs.value  # g

# Solar radius
R_SUN: float = const.R_sun.cgs.value  # cm

# Solar luminosity
L_SUN: float = const.L_sun.cgs.value  # erg/s

# Astronomical unit
AU: float = const.au.cgs.value  # cm

# Parsec
PC: float = const.pc.cgs.value  # cm

# Megaparsec
MPC: float = (const.pc * 1e6).cgs.value  # cm

# Electron rest energy
M_E_C2: float = (const.m_e * const.c**2).cgs.value  # erg

# Proton rest energy
M_P_C2: float = (const.m_p * const.c**2).cgs.value  # erg

# Classical electron radius
R_ELECTRON: float = (const.e.gauss**2 / (const.m_e * const.c**2)).cgs.value  # cm

# Year in seconds
YEAR: float = (1 * u.yr).to(u.s).value  # s

# Day in seconds
DAY: float = (1 * u.day).to(u.s).value  # s

# Hour in seconds
HOUR: float = 3600.0  # s

__all__ = [
    "C_LIGHT",
    "C_LIGHT_CGS",
    "C_LIGHT_KMS",
    "H_PLANCK",
    "HBAR",
    "K_BOLTZMANN",
    "G_NEWTON",
    "SIGMA_SB",
    "A_RAD",
    "E_CHARGE",
    "M_ELECTRON",
    "M_PROTON",
    "M_NEUTRON",
    "M_AMU",
    "SIGMA_T",
    "ALPHA_FS",
    "M_SUN",
    "R_SUN",
    "L_SUN",
    "AU",
    "PC",
    "MPC",
    "M_E_C2",
    "M_P_C2",
    "R_ELECTRON",
    "YEAR",
    "DAY",
    "HOUR",
]
