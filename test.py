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
USE_GRAVITY = True


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


def get_target_coords(coords1, m1, coords2, m2, link):
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


def get_temp_coords(init_coords, init_velocities, num_atoms, timestep):
    # Find next positions based on the current velocity and gravity
    temp_coords = []
    for idx in range(num_atoms):
        temp_v = vec.add(init_coords[idx], vec.scale(init_velocities[idx], timestep))
        if USE_GRAVITY:
            temp_v[VERTICAL_AXIS] += (G_CONST * timestep**2) / 2
        temp_coords.append(temp_v)
    return temp_coords


def process_links(step1_coords, masses, links, timestep):
    num_atoms = len(masses)
    fin_coords = copy.deepcopy(step1_coords)
    for _ in range(10):
        temp_coords = [[] for _ in range(num_atoms)]
        for link in links:
            a0_idx = link["atoms"][0]
            a1_idx = link["atoms"][1]
            m0 = masses[a0_idx]
            m1 = masses[a1_idx]
            new_coords0, new_coords1 = get_target_coords(fin_coords[a0_idx], m0, fin_coords[a1_idx], m1, link)
            temp_coords[a0_idx].append(new_coords0)
            temp_coords[a1_idx].append(new_coords1)
        for idx in range(num_atoms):
            if len(temp_coords[idx]) > 0:   # Skip if atom doesn't have any link
                fin_coords[idx] = centroid(temp_coords[idx])

    deltas = []
    for idx in range(num_atoms):
        d = vec.scale(vec.sub(fin_coords[idx], step1_coords[idx]), timestep)
        deltas.append(d)
        fin_coords[idx] = vec.add(step1_coords[idx], d)
    return fin_coords, deltas


def update_velocities(init_velocities, deltas, num_atoms, timestep):
    new_velocities = []
    for idx in range(num_atoms):
        new_velocities.append(vec.add(init_velocities[idx], vec.scale(deltas[idx], 0.9)))
        if USE_GRAVITY:
            new_velocities[-1][VERTICAL_AXIS] += G_CONST * timestep
    return new_velocities


def get_time_substep(atoms):
    # Calculates the time required for the fastest atom to cross it's radius
    v_max = 0
    r = 0
    for a in atoms:
        v = vec.magnitude(a["speed"]) / a["radius"]
        if v > v_max:
            v_max = v
            r = a["radius"]
    if v_max == 0:
        return 1
    return r / v_max
    

def get_first_collision():
    pass

def update_coords(atoms, links, timestep=1):
    num_atoms = len(atoms)
    init_velocities = [atoms[i]["speed"] for i in range(num_atoms)]
    init_coords = [atoms[i]["coords"] for i in range(num_atoms)]
    masses = [atoms[i]["mass"] for i in range(num_atoms)]
    passed_time = 0
    while passed_time < timestep:
        ts = get_time_substep(atoms)
        last_iter = False
        if (passed_time + ts) > timestep:
            ts = timestep - passed_time
            last_iter = True
        step1_coords = get_temp_coords(init_coords, init_velocities, num_atoms, ts)
        step2_coords, deltas = process_links(step1_coords, masses, links, ts)
        new_velocities = update_velocities(init_velocities, deltas, num_atoms, ts)

        # Process collisions:
        # TODO: ImplementMe!

        # Update coords:
        for idx in range(num_atoms):
            atoms[idx]["coords"] = step2_coords[idx]
            atoms[idx]["speed"] = new_velocities[idx]

        passed_time += ts
        if last_iter:   # Avoid float errors
            break


atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [450, 500],
        "force": [0, 0],
        "speed": [5, 0],
        "mass": POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [500, 550],
        "force": [0, 0],
        "speed": [8, 0],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [550, 500],
        "force": [0, 0],
        "speed": [5, 0],
        "mass": POINT_MASS,
        "color": "#FF0000"
    },
]

links = [
    {
        "atoms": [0, 1],
        "length": 70,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [1, 2],
        "length": 70,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [2, 0],
        "length": 70,
        "stiffness": STIFFNESS,
        "id": None
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
        update_coords(atoms, links)
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
    # time.sleep(0.1)
    # if counter == 5:
    #     sys.exit()
    root.after(40, update_world)

update_world()
root.mainloop()
