"""
Entry point for the Satellite Orbit Simulator.
Run this file to start the program.
"""

import math

from simulation import run


def get_manual_parameters() -> tuple:
    """
    Prompt the user to enter orbital parameters manually via the terminal.

    Returns:
        A tuple of (eccentricity, semi_major_axis_km, inclination_rad,
        ascending_node_rad, argument_of_periapsis_rad).
    """
    eccentricity = float(input("Eccentricity (as a decimal, e.g. 0.01): "))
    semi_major_axis = float(input("Semi-Major Axis (in km, e.g. 7000): "))
    inclination = math.radians(float(input("Inclination (degrees): ")))
    ascending_node = math.radians(float(input("Longitude of the Ascending Node (degrees): ")))
    periapsis = math.radians(float(input("Argument of Periapsis (degrees): ")))
    return eccentricity, semi_major_axis, inclination, ascending_node, periapsis


def main() -> None:
    """
    Main entry point. Prompts the user for orbital parameters and launches
    the simulation.
    """
    print("\n=== Satellite Orbit Simulator ===")
    parameters = get_manual_parameters()
    run(*parameters)


if __name__ == "__main__":
    main()