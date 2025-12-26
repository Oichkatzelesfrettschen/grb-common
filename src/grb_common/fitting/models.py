import numpy as np
from typing import Union

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    
    # Dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def norris_pulse(t, start_time, amplitude, tau1, tau2):
    """
    Norris et al. (1996) pulse model.
    I(t) = A * exp(-tau1/(t-t_s) - (t-t_s)/tau2)
    
    Args:
        t (array): Time array.
        start_time (float): Start time (t_s).
        amplitude (float): Amplitude (A).
        tau1 (float): Rise time parameter.
        tau2 (float): Decay time parameter.
    """
    # Avoid division by zero and invalid times
    t_shifted = t - start_time
    
    # Result array
    intensity = np.zeros_like(t)
    
    # Mask for t > t_s
    mask = t_shifted > 0
    
    if np.any(mask):
        t_valid = t_shifted[mask]
        val = amplitude * np.exp(-tau1/t_valid - t_valid/tau2)
        
        # Numba doesn't like boolean indexing assignment easily with mixed types?
        # Actually it handles it fine for arrays.
        # But to be safe and fast:
        
        # Manual loop for safety in JIT? 
        # No, vectorized math is fine in Numba if shapes match.
        
        # We need to assign back.
        # Let's do a loop, it's often faster/safer in Numba than fancy indexing
        for i in range(len(t)):
            if t[i] > start_time:
                dt = t[i] - start_time
                intensity[i] = amplitude * np.exp(-tau1/dt - dt/tau2)
                
    return intensity

@jit(nopython=True)
def fred_pulse(t, start_time, amplitude, tau, asymmetry):
    """
    Fast Rise Exponential Decay (FRED) model (Kobayashi 1997-like).
    Simple broken power law or exponential.
    
    Kocevski (2003) parametrization:
    F(t) = F_m * (t/t_m)^r * (d/(d+r) + r/(d+r)*(t/t_m)^(r+1))^-1
    This is complex.
    
    Simple FRED:
    I(t) = A * exp(-(t-t_0)/tau) for t > t_0
    """
    intensity = np.zeros_like(t)
    for i in range(len(t)):
        if t[i] > start_time:
            intensity[i] = amplitude * np.exp(-(t[i]-start_time)/tau)
    return intensity
