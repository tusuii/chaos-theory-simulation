import numpy as np


def rk4_system(deriv_fn, state, t, dt, **kwargs):
    """Generic 4th-order Runge-Kutta step"""
    s = np.array(state, dtype=float)
    k1 = np.array(deriv_fn(s,           t,        **kwargs))
    k2 = np.array(deriv_fn(s + dt/2*k1, t + dt/2, **kwargs))
    k3 = np.array(deriv_fn(s + dt/2*k2, t + dt/2, **kwargs))
    k4 = np.array(deriv_fn(s + dt*k3,   t + dt,   **kwargs))
    return s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
