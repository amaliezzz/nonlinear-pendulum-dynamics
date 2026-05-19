import numpy as np
from scipy.integrate import solve_ivp

G = 9.81


def ode(t, y, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    """
    Double pendulum as a first-order system derived from the Euler-Lagrange equations.
    y = [θ₁, θ̇₁, θ₂, θ̇₂]
    """
    theta1, omega1, theta2, omega2 = y
    delta = theta2 - theta1
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    M = m1 + m2

    denom1 = M * L1 - m2 * L1 * cos_d ** 2
    denom2 = (L2 / L1) * denom1

    domega1 = (
        m2 * L1 * omega1 ** 2 * sin_d * cos_d
        + m2 * G * np.sin(theta2) * cos_d
        + m2 * L2 * omega2 ** 2 * sin_d
        - M * G * np.sin(theta1)
    ) / denom1

    domega2 = (
        -m2 * L2 * omega2 ** 2 * sin_d * cos_d
        + M * G * np.sin(theta1) * cos_d
        - M * L1 * omega1 ** 2 * sin_d
        - M * G * np.sin(theta2)
    ) / denom2

    return [omega1, domega1, omega2, domega2]


def simulate(
    theta1_deg,
    theta2_deg,
    omega1_0=0.0,
    omega2_0=0.0,
    L1=1.0,
    L2=1.0,
    m1=1.0,
    m2=1.0,
    t_max=30.0,
    dt=0.01,
):
    y0 = [
        np.radians(theta1_deg),
        omega1_0,
        np.radians(theta2_deg),
        omega2_0,
    ]
    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(
        fun=lambda t, y: ode(t, y, L1, L2, m1, m2),
        t_span=(0, t_max),
        y0=y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]


def total_energy(theta1, omega1, theta2, omega2, L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    """
    Total mechanical energy of the double pendulum.

    Returns
    E : total energy array (J)
    """
    T = (
        0.5 * m1 * (L1 * omega1) ** 2
        + 0.5 * m2 * (
            (L1 * omega1) ** 2
            + (L2 * omega2) ** 2
            + 2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
        )
    )
    V = (
        -(m1 + m2) * G * L1 * np.cos(theta1)
        - m2 * G * L2 * np.cos(theta2)
    )
    return T + V
