"""
Handles all OpenGL and pygame rendering for the simulation.
"""

import math

import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    GL_LINE_STRIP,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_TRUE,
    GL_VERSION,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glDepthFunc,
    glDepthMask,
    glDisable,
    glEnable,
    glEnd,
    glGetString,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTranslatef,
    glVertex3f,
)
from OpenGL.GLU import GLU_FILL, gluNewQuadric, gluPerspective, gluQuadricDrawStyle, gluSphere
from pygame.locals import DOUBLEBUF, OPENGL

from config import D_EARTH, D_MOON, DISPLAY_HEIGHT, DISPLAY_WIDTH, SCALE
from physics import orbital_radius


def init_display(zoom: float) -> None:
    """
    Initialize the pygame window and OpenGL context.

    Args:
        zoom: Initial Z-axis translation (camera distance).
    """
    pygame.init()
    pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Satellite Orbit Simulator")
    _init_gl(zoom)


def _init_gl(zoom: float) -> None:
    """
    Configure the OpenGL projection and modelview matrices.

    Args:
        zoom: Initial Z-axis translation (negative = further away).
    """
    print("OpenGL version:", glGetString(GL_VERSION))

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, DISPLAY_WIDTH / DISPLAY_HEIGHT, 0.001, 1e9)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, zoom)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)


def draw_scene(
    semi_major_axis: float,
    eccentricity: float,
    true_anomaly: float,
    zoom: float,
    angle_x: float,
    angle_y: float,
    earth_quad,
    satellite_quad,
) -> None:
    """
    Render the entire scene: Earth, satellite, and orbital path.

    Args:
        semi_major_axis: Semi-major axis of the orbit (km).
        eccentricity: Orbital eccentricity.
        true_anomaly: Current true anomaly (radians).
        zoom: Current zoom level (camera distance).
        angle_x: Rotation angle around X-axis (degrees).
        angle_y: Rotation angle around Y-axis (degrees).
        earth_quad: GLU quadric object for Earth.
        satellite_quad: GLU quadric object for the satellite.
    """
    glLoadIdentity()
    glTranslatef(0.0, 0.0, zoom)
    glRotatef(angle_x, 1, 0, 0)
    glRotatef(angle_y, 0, 1, 0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    _draw_orbit(semi_major_axis, eccentricity)
    _draw_satellite(semi_major_axis, eccentricity, true_anomaly, satellite_quad)
    _draw_earth(earth_quad)


def _draw_orbit(semi_major_axis: float, eccentricity: float) -> None:
    """
    Draw the orbital path as a grey line loop.

    Args:
        semi_major_axis: Semi-major axis (km).
        eccentricity: Orbital eccentricity.
    """
    glLineWidth(1.0)
    glBegin(GL_LINE_STRIP)
    glColor3f(0.3, 0.3, 0.3)
    for theta in np.linspace(0, 2 * math.pi, 360):
        r = orbital_radius(semi_major_axis, eccentricity, theta) * SCALE
        glVertex3f(r * math.cos(theta), r * math.sin(theta), 0)
    glEnd()


def _draw_earth(quadric) -> None:
    """
    Draw Earth at the origin as a blue sphere.

    Args:
        quadric: A GLU quadric object used for rendering.
    """
    glPushMatrix()
    glColor3f(0.0, 0.5, 2.0)
    glTranslatef(0, 0, 0)
    gluSphere(quadric, (D_EARTH / 2) * SCALE, 32, 32)
    glPopMatrix()


def _draw_satellite(
    semi_major_axis: float,
    eccentricity: float,
    true_anomaly: float,
    quadric,
) -> None:
    """
    Draw the satellite as a red sphere at its current orbital position.

    Args:
        semi_major_axis: Semi-major axis (km).
        eccentricity: Orbital eccentricity.
        true_anomaly: Current true anomaly (radians).
        quadric: A GLU quadric object used for rendering.
    """
    r = orbital_radius(semi_major_axis, eccentricity, true_anomaly) * SCALE
    x = r * math.cos(true_anomaly)
    y = r * math.sin(true_anomaly)
    glPushMatrix()
    glColor3f(1.0, 0.0, 0.0)
    glTranslatef(x, y, 0)
    gluSphere(quadric, (D_MOON / 2) * SCALE, 32, 32)
    glPopMatrix()