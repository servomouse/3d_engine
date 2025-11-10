import tkinter as tk
import random
import json
import sys

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


def get_random_coords(radius=ATOM_RADIUS):
    return [
        random.randint(radius, WINDOW_WIDTH - radius),
        random.randint(radius, WINDOW_HEIGHT - radius)
    ]


def add_gravity(atoms):
    for atom in atoms:
        atom["force"][VERTICAL_AXIS] += -9.8 * atom["mass"]


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
        "coords": get_random_coords(),
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#0000FF"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": get_random_coords(),
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": get_random_coords(),
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#FF0000"
    },
]

first_run = True

def update_world():
    global first_run

    if not first_run:
        update_coords(atoms)
    else:
        first_run = False

    for atom in atoms:
        if atom["id"] is not None:
            canvas.delete(atom["id"])
        atom["id"] = draw_atom(canvas, atom["coords"], atom["radius"], atom["color"])

    root.after(40, update_world)

update_world()
root.mainloop()