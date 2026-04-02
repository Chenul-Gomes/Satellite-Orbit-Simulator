# 🛰️ Satellite Orbit Simulator

A 3D interactive simulator that lets you visualize how satellites orbit the Earth. Enter a satellite's orbital parameters and watch it move through space in real time — rotate the view, zoom in and out, and explore orbits from any angle.

---

## What Does It Do?

This program opens a 3D window showing Earth at the center with a satellite orbiting around it. You provide the orbital parameters (either your own or looked up online), and the simulator does the rest — showing the orbital path and animating the satellite's movement, speeding up near Earth and slowing down at its furthest point, just like real physics.

---

## Requirements

Before running the simulator, make sure you have the following installed:

- [Python 3.10 or higher](https://www.python.org/downloads/)
- The following Python packages (installed in the next step):
  - `pygame` — handles the window and user input
  - `PyOpenGL` — handles the 3D rendering
  - `numpy` — handles the math behind the orbit

---

## Setup & Installation

**1. Download or clone the project**

If you have Git installed:
```bash
git clone https://github.com/Chenul-Gomes/Satellite-Orbit-Simulator.git
cd Satellite-Orbit-Simulator
```
Or just download the ZIP from GitHub and extract it.

**2. Create a virtual environment** *(recommended — keeps your system clean)*
```bash
python -m venv .venv
```

**3. Activate the virtual environment**

On Windows:
```bash
.venv\Scripts\activate
```
On macOS/Linux:
```bash
source .venv/bin/activate
```

**4. Install the required packages**
```bash
pip install -r requirements.txt
```

**5. Run the simulator**
```bash
python main.py
```

---

## How to Use

When you run the program, it will ask you to enter five orbital parameters in the terminal. Once entered, the simulation window will open automatically.

### Controls

| Input | Action |
|---|---|
| Left mouse button + drag | Rotate the view |
| ↑ Arrow key | Zoom in |
| ↓ Arrow key | Zoom out |
| Close window | Quit |

---

## Orbital Parameters Explained

When you launch the simulator, you'll be asked for these five values. Here's what they mean in plain English:

| Parameter | What It Means | Example (ISS) |
|---|---|---|
| **Eccentricity** | How circular the orbit is. `0` = perfect circle, closer to `1` = more stretched out ellipse | `0.0006` |
| **Semi-Major Axis** | The average distance from the center of Earth to the satellite, in km | `6786` |
| **Inclination** | The tilt of the orbit relative to the equator, in degrees. `0°` = orbits above the equator, `90°` = passes over the poles | `51.6` |
| **Longitude of Ascending Node** | Describes the rotation of the orbital plane around Earth's axis, in degrees | `336.2` |
| **Argument of Periapsis** | Describes the orientation of the ellipse within the orbital plane, in degrees | `245.2` |

### Example Values to Try

**International Space Station (ISS)**
```
Eccentricity:                  0.0006215
Semi-Major Axis:               6786.5
Inclination:                   51.6344
Longitude of Ascending Node:   336.2407
Argument of Periapsis:         245.2164
```

**The Moon**
```
Eccentricity:                  0.0549
Semi-Major Axis:               384400
Inclination:                   5.145
Longitude of Ascending Node:   0
Argument of Periapsis:         0
```

> 💡 You can find orbital parameters for any satellite on [Heavens Above](https://www.heavens-above.com) or [NASA's orbital data](https://science.nasa.gov).

---

## Project Structure

```
Satellite-Orbit-Simulator/
│
├── main.py           # Start here — handles user input and launches the simulation
├── simulation.py     # Runs the main loop and camera controls
├── renderer.py       # Draws everything you see (Earth, satellite, orbit path)
├── physics.py        # The orbital mechanics calculations
├── config.py         # Settings and constants (sizes, scale, display)
│
├── requirements.txt  # Python packages needed to run the project
└── README.md         # You're reading it!
```

---

## Built With

- [Python](https://www.python.org/)
- [pygame](https://www.pygame.org/)
- [PyOpenGL](https://pyopengl.sourceforge.net/)
- [NumPy](https://numpy.org/)