import numpy as np
from scipy.integrate import solve_ivp

G = 9.81


def ode(t, y, L=1.0, b=0.0):
    """
     y = [θ (rad), θ̇'(rad/s)]
    """
    theta, omega = y
    return [omega, -(G / L) * np.sin(theta) - b * omega]


def simulate(theta0_deg, omega0=0.0, L=1.0, b=0.0, t_max=30.0, dt=0.01):
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(
        fun=lambda t, y: ode(t, y, L=L, b=b),
        t_span=(0, t_max),
        y0=[np.radians(theta0_deg), omega0],
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y[0], sol.y[1]


def natural_frequency(L=1.0):
    return (1 / (2 * np.pi)) * np.sqrt(G / L)