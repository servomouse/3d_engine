import tkinter as tk
import random
import json
import sys
import time
import math
import copy
import vector_math as vec
from pid import pid_force

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
WORLD_LIMITS = [
    [0, WINDOW_WIDTH],
    [0, WINDOW_HEIGHT]
]
NUM_DIMENSIONS = len(WORLD_LIMITS)
DRAG_COEFFICIENT = 0.1
G_CONST = -0.98
VERTICAL_AXIS = 1
ATOM_RADIUS = 10
POINT_MASS = 1  # 1 gramm
MAX_RELATIVE_SPEED = 100
DAMPING_FACTOR = 5
STIFFNESS = 0.75


# Returns atom id
def draw_atom(canvas, coords, radius, color):
    # print(f"Drawing an atom at {coords}")
    return canvas.create_oval(
        coords[0] - radius, WINDOW_HEIGHT - (coords[1] - radius),  # Top-left corner
        coords[0] + radius, WINDOW_HEIGHT - (coords[1] + radius),  # Bottom-right corner
        fill=color, outline="darkblue"
    )


# Returns line id
def draw_line(canvas, coords1, coords2):
    # print(f"Drawing a line at {coords1}, {coords2}, length = {get_distance(coords1, coords2)}")
    return canvas.create_line(
        coords1[0], WINDOW_HEIGHT - coords1[1], # Start
        coords2[0], WINDOW_HEIGHT - coords2[1], # End
        fill="blue", width=3
    )


def different_signs(val1, val2):
    if (val1 >= 0 and val2 >= 0) or (val1 < 0 and val2 < 0):
        return False
    return True


def get_target_coords(atom0, atom1, link):
    coords1 = atom0["temp_coords"]
    m1 = atom0["mass"]
    coords2 = atom1["temp_coords"]
    m2 = atom1["mass"]
    target_distance = link["length"]

    if len(coords1) != len(coords2):
        raise ValueError("Both points must have the same dimensionality")

    # Vector from p1 to p2
    diff = [c2 - c1 for c1, c2 in zip(coords1, coords2)]
    cur_dist = math.sqrt(sum(d * d for d in diff))

    # No movement needed if already at target or points coincide
    if cur_dist == 0 or math.isclose(cur_dist, target_distance, rel_tol=1e-9):
        return list(coords1), list(coords2)

    # Desired change in distance
    delta = target_distance - cur_dist

    # Unit direction vector
    unit = [d / cur_dist for d in diff]

    # Inverse‑mass weighting (heavier → smaller move)
    w1 = 1.0 / m1 if m1 != 0 else 0.0
    w2 = 1.0 / m2 if m2 != 0 else 0.0
    total_w = w1 + w2
    if total_w == 0:    # both masses zero → split equally
        w1 = w2 = 0.5
        total_w = 1.0

    # Portion of the distance change each point should take
    move1 = (w1 / total_w) * delta
    move2 = (w2 / total_w) * delta

    # Apply the moves along the line connecting the points
    new1 = [c1 - u * move1 for c1, u in zip(coords1, unit)]
    new2 = [c2 + u * move2 for c2, u in zip(coords2, unit)]

    return new1, new2


def centroid(points):
    """
    Compute the geometric centre (centroid) of a collection of points
    in *n*‑dimensional space.

    Parameters
    ----------
    points : list of coordinate lists
        Each inner list contains the coordinates of one point. All points
        must have the same dimensionality and the list must contain at least
        one point.

    Returns
    -------
    list of float
        Coordinates of the centroid.
    """
    if not points:
        raise ValueError("The points list cannot be empty")

    dim = len(points[0])
    for p in points:
        if len(p) != dim:
            raise ValueError("All points must have the same number of dimensions")

    # Sum each dimension
    sums = [0.0] * dim
    for p in points:
        for i, coord in enumerate(p):
            sums[i] += coord

    n = len(points)
    return [s / n for s in sums]


def update_coords_custom(atoms, links, timestep=1):
    for atom in atoms:
        atom["new_coords"] = []
        # atom["prev_coords"] = [atom["coords"][i] for i in range(NUM_DIMENSIONS)]
        atom["delta"] = [0 for i in range(NUM_DIMENSIONS)]
        for axis in range(NUM_DIMENSIONS):
            atom["temp_coords"][axis] = atom["coords"][axis] + atom["speed"][axis] * timestep
            if axis == VERTICAL_AXIS:
                atom["temp_coords"][axis] += (G_CONST * timestep**2) / 2
    # Process links:
    for link in links:
        a0 = atoms[link["atoms"][0]]
        a1 = atoms[link["atoms"][1]]
        new_coords0, new_coords1 = get_target_coords(a0, a1, link)
        a0["new_coords"].append(lerp_coords(a0["temp_coords"], new_coords0, link["stiffness"], timestep))
        a1["new_coords"].append(lerp_coords(a1["temp_coords"], new_coords1, link["stiffness"], timestep))
    for atom in atoms:
        atom["new_coords"] = centroid(atom["new_coords"])
        atom["delta"] = [atom["new_coords"][i] - atom["temp_coords"][i] for i in range(NUM_DIMENSIONS)]
    # Calculate new positions:
    for atom in atoms:
        for axis in range(NUM_DIMENSIONS):
            atom["new_coords"][axis] = atom["temp_coords"][axis] + (atom["speed"][axis] + atom["delta"][axis]) * timestep
            atom["new_speed"][axis] = atom["speed"][axis] + atom["delta"][axis] * 0.9
            if axis == VERTICAL_AXIS:
                atom["new_coords"][axis] += (G_CONST * timestep**2) / 2
                atom["new_speed"][axis] += G_CONST * timestep
    # Process collisions:
    # TODO: ImplementMe!
    # Update coords:
    for atom in atoms:
        for axis in range(NUM_DIMENSIONS):
            atom["coords"][axis] = atom["new_coords"][axis]
            atom["speed"][axis] = atom["new_speed"][axis]


def lerp_coords(current, target, percent, timestep=1):
    """
    Linear interpolation between two points in *n*‑dimensional space.

    Parameters
    ----------
    current : list of float
        Starting coordinates (point A).
    target : list of float
        Destination coordinates (point B).
    percent : float
        Fraction of the path to travel, expressed as a decimal
        (e.g. 0.85 for 85%). Values outside [0,1] are allowed and will
        extrapolate beyond the segment.
    timestep: float
        Use timestep to calculate the fractional movement

    Returns
    -------
    list of float
        New coordinates located `percent` of the way from `current` to `target`.
    """
    if percent < 0 or percent > 1:
        raise Exception("Error: percentage should be in the [0, 1] range!")
    
    percent *= timestep

    return [c + (t - c) * percent for c, t in zip(current, target)]


atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [450, 500],
        "new_coords": [],
        "temp_coords": [0 for _ in range(NUM_DIMENSIONS)],
        "prev_coords": [450, 500],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [0, 0],
        "new_speed": [0, 0],
        "prev_speed": [-20, 0],
        "invert_speed": [0 for _ in range(NUM_DIMENSIONS)],
        "mass": POINT_MASS,
        "inv_mass": 1/POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [500, 550],
        "new_coords": [],
        "temp_coords": [0 for _ in range(NUM_DIMENSIONS)],
        "prev_coords": [500, 550],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [0, 0],
        "new_speed": [0, 0],
        "prev_speed": [20, 0],
        "invert_speed": [0 for _ in range(NUM_DIMENSIONS)],
        "mass": POINT_MASS,
        "inv_mass": 1/POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [550, 500],
        "new_coords": [],
        "temp_coords": [0 for _ in range(NUM_DIMENSIONS)],
        "prev_coords": [550, 500],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [0, 0],
        "new_speed": [0, 0],
        "prev_speed": [0, 8],
        "invert_speed": [0 for _ in range(NUM_DIMENSIONS)],
        "mass": POINT_MASS,
        "inv_mass": 1/POINT_MASS,
        "color": "#FF0000"
    },
]

links = [
    {
        "atoms": [0, 1],
        "length": 70,
        "stiffness": STIFFNESS,
        "damping": DAMPING_FACTOR,
        "id": None,
        "integral_error": [None, None],
        "previous_error": [None, None]
    },
    {
        "atoms": [1, 2],
        "length": 70,
        "stiffness": STIFFNESS,
        "damping": DAMPING_FACTOR,
        "id": None,
        "integral_error": [None, None],
        "previous_error": [None, None]
    },
    {
        "atoms": [2, 0],
        "length": 70,
        "stiffness": STIFFNESS,
        "damping": DAMPING_FACTOR,
        "id": None,
        "integral_error": [None, None],
        "previous_error": [None, None]
    },
]

# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

first_run = True
counter = 0

def update_world():
    global first_run, counter

    if not first_run:
        update_coords_custom(atoms, links)
        # update_coords(atoms, links)
        # update_coords_new(atoms, links)
        counter += 1
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
    print(f"\r{counter = }", end="")
    time.sleep(0.1)
    # if counter == 5:
    #     sys.exit()


    root.after(40, update_world)

update_world()
root.mainloop()
