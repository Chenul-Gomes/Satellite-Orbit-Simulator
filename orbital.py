import os
import re
import math
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pygame.init()
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int((info.current_w/2)-550)}, {int(info.current_h/2-300)}"

#Global constants
G = 6.67430e-11  # Gravitational constant
M = 1.989e+30    # Mass of the sun, adjust if necessary
D_earth = 12800 #km 
D_moon = 3500 #km

def main():
    choice = input("Please type 'a' for manual input and 'b' for AI input: ").upper()
    if choice == 'A':
        eccentricity = float(input("Eccentricity (as a decimal): "))
        semiMajorAxis = float(input("Semi-Major Axis (in km): "))
        inclination = math.radians(float(input("Inclination (in degrees): ")))
        ascendingNode = math.radians(float(input("Longitude of the Ascending Node (in degrees): ")))
        periapsis = math.radians(float(input("Argument of Periapsis (in degrees): ")))
        runSimulation(eccentricity, semiMajorAxis, inclination, ascendingNode, periapsis)
    elif choice == 'B':
        satelliteName = input('Enter name of satellite: ')
        prompt = f"Provide the 'MOST RECENT' and 'KNOWN' orbital parameters (eccentricity, semi-major axis in km, inclination in degrees, longitude of the ascending node in degrees, and argument of periapsis in degrees) for the satellite named {satelliteName}."
        parameters = ai_search(prompt)
        if parameters:
            print("Retrieved Orbital Parameters:", parameters)
            runSimulation(*parameters)
        else:
            print("AI failed to retrieve valid data.")
    else:
        print("Invalid choice. Please enter 'a' or 'b'.")

def ai_search(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        text_response = response.choices[0].message.content.strip()
        print("AI Response:", text_response)

        parameters = {
            'eccentricity': re.search(r"Eccentricity:\s*([0-9.]+)", text_response),
            'semi-major axis': re.search(r"Semi-major axis:\s*~?([\d,]*\.?\d+)\s*km", text_response),
            'inclination': re.search(r"Inclination:\s*~?([0-9.]+)", text_response),
            'longitude of the ascending node': re.search(r"Longitude of the ascending node:\s*~?([0-9.]+)", text_response),
            'argument of periapsis': re.search(r"Argument of periapsis:\s*~?([0-9.]+)", text_response)
        }

        if all(param and param.group(1) for param in parameters.values()):
            return (
                float(parameters['eccentricity'].group(1)),
                float(parameters['semi-major axis'].group(1).replace(',', '')),
                math.radians(float(parameters['inclination'].group(1))),
                math.radians(float(parameters['longitude of the ascending node'].group(1))),
                math.radians(float(parameters['argument of periapsis'].group(1)))
            )
        else:
            missing_params = [key for key, value in parameters.items() if not value]
            print(f"Missing or malformed parameters: {missing_params}")
            return None
    except Exception as e:
        print("Error during AI search:", e)
        return None

def calculate_initial_zoom(a, e):
    apoapsis = a * (1 + e)
    zoom = -apoapsis * 1.5
    return zoom


def runSimulation(e, a, i, omegaS, omegaL):
    initial_zoom = calculate_initial_zoom(a, e)
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    initialize_gl(initial_zoom)
    main_loop(e, a, i, omegaS, omegaL, initial_zoom)

def initialize_gl(zoom):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (800 / 600), 0.1, 1e7)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, zoom)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glColor3f(1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)

def main_loop(e, a, i, omegaS, omegaL, zoom):
    earth = gluNewQuadric()
    gluQuadricDrawStyle(earth, GLU_FILL)
    satelite = gluNewQuadric()
    gluQuadricDrawStyle(satelite, GLU_FILL)
    angle_x, angle_y = 0, 0
    v = 0  
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Left mouse button
                    angle_x, angle_y = angle_x + event.rel[1], angle_y + event.rel[0]
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    zoom -= (zoom/10)
                elif event.key == pygame.K_DOWN:
                    zoom += (zoom/10)

        # Calculate distance r based on current true anomaly v
        r = draw_satelite(a, e, v, satelite)
        orbital_speed = math.sqrt(G * M * (2 / r - 1 / a))  # Using vis-viva equation
        delta_t = 0.05  # Time step in seconds
        delta_v = orbital_speed * delta_t / r  # Change in true anomaly

        #v += delta_v 
        v += 0.01*(a/r)**2
        if v >= 2 * math.pi:
            v = 0

        glLoadIdentity()
        glTranslatef(0.0, 0.0, zoom)
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_orbit(a, e)
        draw_satelite(a, e, v, satelite)
        draw_earth(earth)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

def draw_orbit(a, e):
    glLineWidth(1.0)
    glBegin(GL_LINE_STRIP)
    glColor3f(0.3, 0.3, 0.3)
    for angle in np.linspace(0, 2 * math.pi, 360):
        r = (a * (1 - e**2)) / (1 + e * math.cos(angle))
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        glVertex3f(x, y, 0)
    glEnd()

def draw_earth(star):
    glPushMatrix()
    glColor3f(0, 0.5, 2)
    glTranslatef(0, 0, 0)
    gluSphere(star, (D_earth/2), 32, 32)
    glPopMatrix()

def draw_satelite(a, e, v, planet):
    r = (a * (1 - e**2)) / (1 + e * math.cos(v))
    x = r * math.cos(v)
    y = r * math.sin(v)
    glPushMatrix()
    glColor3f(1, 0, 0)
    glTranslatef(x, y, 0)
    gluSphere(planet, (D_moon/2), 32, 32)
    glPopMatrix()
    return r

if __name__ == "__main__":
    main()