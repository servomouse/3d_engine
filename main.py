import tkinter as tk
import random
import json
import sys
import time
import math

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
WORLD_LIMITS = [
    [0, WINDOW_WIDTH],
    [0, WINDOW_HEIGHT]
]
VERTICAL_AXIS = 1
ATOM_RADIUS = 10
POINT_MASS = 1  # 1 gramm


# Returns atom id
def draw_atom(canvas, coords, radius, color):
    print(f"Drawing an atom at {coords}")
    return canvas.create_oval(
        coords[0] - radius, WINDOW_HEIGHT - (coords[1] - radius),  # Top-left corner
        coords[0] + radius, WINDOW_HEIGHT - (coords[1] + radius),  # Bottom-right corner
        fill=color, outline="darkblue"
    )


# Returns line id
def draw_line(canvas, coords1, coords2):
    print(f"Drawing a line at {coords1}, {coords2}, length = {get_distance(coords1, coords2)}")
    return canvas.create_line(
        coords1[0], WINDOW_HEIGHT - coords1[1], # Start
        coords2[0], WINDOW_HEIGHT - coords2[1], # End
        fill="blue", width=3
    )


def get_random_coords(radius=ATOM_RADIUS):
    return [
        random.randint(radius, WINDOW_WIDTH - radius),
        random.randint(radius, WINDOW_HEIGHT - radius)
    ]


def add_gravity(atoms):
    for atom in atoms:
        atom["force"][VERTICAL_AXIS] += -0.98 * atom["mass"]


def vector_len(vec):
    s = 0
    for axis in vec:
        s += axis**2
    return math.sqrt(s)


def calculate_force_vectors(point1, point2, force_value):
    direction_vector = [p2 - p1 for p1, p2 in zip(point1, point2)]

    distance = sum(comp ** 2 for comp in direction_vector) ** 0.5

    if distance == 0:   # Handle zero distance case
        raise ValueError("The points coincide; cannot calculate force.")

    unit_vector = [comp / distance for comp in direction_vector]    # Normalize the direction vector

    force_on_p1 = [force_value * comp for comp in unit_vector]
    force_on_p2 = [-f for f in force_on_p1]

    return force_on_p1, force_on_p2


def get_distance(point1, point2):
    return math.sqrt(sum((p2 - p1)**2 for p1, p2 in zip(point1, point2)))


def calculate_forces(atoms, links):
    for link in links:
        point1 = atoms[link["atoms"][0]]["coords"]
        point2 = atoms[link["atoms"][1]]["coords"]
        link_len = get_distance(point1, point2)
        force = (link_len - link["length"]) * link["stiffness"]
        force_p1, force_p2 = calculate_force_vectors(point1, point2, force)
        # print(f"{force_p1 = }, {force_p2 = }")
        for i in range(len(WORLD_LIMITS)):
            atoms[link["atoms"][0]]["force"][i] += force_p1[i]
            atoms[link["atoms"][1]]["force"][i] += force_p2[i]


def update_coords(atoms):
    add_gravity(atoms)
    accelerations = []
    for i in range(len(atoms)):
        accelerations.append([])
        for axis in range(len(WORLD_LIMITS)):
            accelerations[-1].append(atoms[i]["force"][axis] / atoms[i]["mass"])
            atoms[i]["force"][axis] = 0 # Clear force vecxtor after calculating accelerations
    
    for i in range(len(atoms)):
        for axis in range(len(WORLD_LIMITS)):
            atoms[i]["speed"][axis] += accelerations[i][axis]

    for i in range(len(atoms)):
        for axis in range(len(WORLD_LIMITS)):
            atoms[i]["coords"][axis] += atoms[i]["speed"][axis]
            if atoms[i]["coords"][axis] < (WORLD_LIMITS[axis][0] + atoms[i]["radius"]):
                atoms[i]["coords"][axis] = WORLD_LIMITS[axis][0]
                atoms[i]["speed"][axis] = 0
            elif atoms[i]["coords"][axis] > (WORLD_LIMITS[axis][1] - atoms[i]["radius"]):
                atoms[i]["coords"][axis] = WORLD_LIMITS[axis][1]
                atoms[i]["speed"][axis] = 0

    # for atom in atoms:
    #     atom["coords"] = get_random_coords(atom["radius"])


def update_speeds(atoms):
    for atom in atoms:
        for axis in range(len(atom["speed"])):
            atom["speed"][axis] *= 0.9


# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [500, 535],
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [465, 500],
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [535, 500],
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#FF0000"
    },
]

links = [
    {
        "atoms": [0, 1],
        "length": 70,
        "stiffness": 1,
        "id": None
    },
    {
        "atoms": [1, 2],
        "length": 70,
        "stiffness": 1,
        "id": None
    },
    {
        "atoms": [2, 0],
        "length": 70,
        "stiffness": 1,
        "id": None
    },
]

first_run = True

def update_world():
    global first_run

    if not first_run:
        update_speeds(atoms)
        calculate_forces(atoms, links)
        update_coords(atoms)
    else:
        first_run = False

    for atom in atoms:
        if atom["id"] is not None:
            canvas.delete(atom["id"])
        atom["id"] = draw_atom(canvas, atom["coords"], atom["radius"], atom["color"])
    
    for link in links:
        if link["id"] is not None:
            canvas.delete(link["id"])
        link["id"] = draw_line(canvas, atoms[link["atoms"][0]]["coords"], atoms[link["atoms"][1]]["coords"])
    # time.sleep(10)
    # sys.exit()


    root.after(40, update_world)

update_world()
root.mainloop()