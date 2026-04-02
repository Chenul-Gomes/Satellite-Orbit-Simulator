"""
Orbital mechanics and physics calculations.
"""

import math

from config import G, M, SCALE


def orbital_radius(
    semi_major_axis: float,
    eccentricity: float,
    true_anomaly: float,
) -> float:
    """
    Calculate the orbital radius at a given true anomaly using the
    standard conic section equation for an ellipse.

    Args:
        semi_major_axis: Semi-major axis of the orbit (km).
        eccentricity: Orbital eccentricity (0 = circular, <1 = elliptical).
        true_anomaly: Current true anomaly angle (radians).

    Returns:
        Orbital radius (km) at the given true anomaly.
    """
    p = semi_major_axis * (1 - eccentricity ** 2)
    return p / (1 + eccentricity * math.cos(true_anomaly))


def orbital_speed(radius: float, semi_major_axis: float) -> float:
    """
    Calculate orbital speed using the vis-viva equation.

    Args:
        radius: Current distance from the central body (km).
        semi_major_axis: Semi-major axis of the orbit (km).

    Returns:
        Orbital speed (km/s).
    """
    return math.sqrt(G * M * (2 / radius - 1 / semi_major_axis))


def step_true_anomaly(
    true_anomaly: float,
    semi_major_axis: float,
    radius: float,
    step: float = 0.01,
) -> float:
    """
    Advance the true anomaly by one simulation step.

    Uses a simplified Kepler scaling so the satellite moves faster near
    periapsis and slower near apoapsis (conserving angular momentum roughly).

    Args:
        true_anomaly: Current true anomaly (radians).
        semi_major_axis: Semi-major axis of the orbit (km).
        radius: Current orbital radius (km).
        step: Base angular step size.

    Returns:
        Updated true anomaly (radians), wrapped to [0, 2pi).
    """
    delta = step * (semi_major_axis / radius) ** 2
    new_anomaly = true_anomaly + delta
    return new_anomaly % (2 * math.pi)


def calculate_initial_zoom(semi_major_axis: float, eccentricity: float) -> float:
    """
    Compute a sensible initial camera zoom based on the orbit's apoapsis.

    Args:
        semi_major_axis: Semi-major axis (km).
        eccentricity: Orbital eccentricity.

    Returns:
        Negative Z translation value for OpenGL (zoomed out to fit orbit).
    """
    apoapsis = semi_major_axis * (1 + eccentricity)
    return -apoapsis * SCALE * 2.5