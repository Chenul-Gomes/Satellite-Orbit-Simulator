"""
Manages the main simulation loop and user input (mouse rotation, zoom).
"""

import pygame
from OpenGL.GLU import GLU_FILL, gluNewQuadric, gluQuadricDrawStyle

from physics import calculate_initial_zoom, orbital_radius, step_true_anomaly
from renderer import draw_scene, init_display


def run(
    eccentricity: float,
    semi_major_axis: float,
    inclination: float,
    ascending_node: float,
    argument_of_periapsis: float,
) -> None:
    """
    Initialize and run the satellite orbit simulation.

    Handles the pygame event loop, camera controls (mouse drag to rotate,
    arrow keys to zoom), and advances the satellite's true anomaly each frame.

    Args:
        eccentricity: Orbital eccentricity (0 = circular).
        semi_major_axis: Semi-major axis in km.
        inclination: Inclination in radians.
        ascending_node: Longitude of the ascending node in radians.
        argument_of_periapsis: Argument of periapsis in radians.
    """
    zoom = calculate_initial_zoom(semi_major_axis, eccentricity)
    init_display(zoom)

    # Create quadrics once — reusing each frame is more efficient
    earth_quad = gluNewQuadric()
    satellite_quad = gluNewQuadric()
    gluQuadricDrawStyle(earth_quad, GLU_FILL)
    gluQuadricDrawStyle(satellite_quad, GLU_FILL)

    angle_x, angle_y = 0.0, 0.0
    true_anomaly = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Left mouse button held
                    angle_x += event.rel[1]
                    angle_y += event.rel[0]
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    zoom -= zoom / 10  # Zoom in
                elif event.key == pygame.K_DOWN:
                    zoom += zoom / 10  # Zoom out

        r = orbital_radius(semi_major_axis, eccentricity, true_anomaly)
        true_anomaly = step_true_anomaly(true_anomaly, semi_major_axis, r)

        draw_scene(
            semi_major_axis,
            eccentricity,
            true_anomaly,
            zoom,
            angle_x,
            angle_y,
            earth_quad,
            satellite_quad,
        )

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()