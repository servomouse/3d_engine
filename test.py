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


def coords_within_box(coords, velocity, radius):
    # Set velocity to None to not update it
    for axis in range(NUM_DIMENSIONS):
        if coords[axis] <= (WORLD_LIMITS[axis][0] + radius):
            coords[axis] = WORLD_LIMITS[axis][0] + radius
            if velocity and velocity[axis] < 0:
                velocity[axis] *= -1
        elif coords[axis] >= (WORLD_LIMITS[axis][1] - radius):
            coords[axis] = WORLD_LIMITS[axis][1] - radius
            if velocity and velocity[axis] > 0:
                velocity[axis] *= -1


def get_temp_coords(radii, init_coords, init_velocities, num_atoms, timestep):
    # Find next positions based on the current velocity and gravity
    temp_coords = []
    for idx in range(num_atoms):
        coords_within_box(init_coords[idx], init_velocities[idx], radii[idx]* 1.01)
    for idx in range(num_atoms):
        temp_v = vec.add(init_coords[idx], vec.scale(init_velocities[idx], timestep))
        if USE_GRAVITY:
            temp_v[VERTICAL_AXIS] += (G_CONST * timestep**2) / 2
        temp_coords.append(temp_v)
    return temp_coords


def process_links(radii, step1_coords, masses, links, timestep):
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
                coords_within_box(fin_coords[idx], None, radii[idx]* 1.01)

    deltas = []
    for idx in range(num_atoms):
        d = vec.scale(vec.sub(fin_coords[idx], step1_coords[idx]), timestep)
        deltas.append(d)
        fin_coords[idx] = vec.add(step1_coords[idx], d)
        vec.trim_coords(fin_coords[idx])
    return fin_coords, deltas


def update_velocities(init_velocities, deltas, num_atoms, timestep):
    new_velocities = []
    for idx in range(num_atoms):
        new_velocities.append(vec.add(init_velocities[idx], vec.scale(deltas[idx], 0.9)))
        if USE_GRAVITY:
            new_velocities[-1][VERTICAL_AXIS] += G_CONST * timestep
    return new_velocities


def get_time_substep(velocities, radii, num_atoms):
    # Calculates the time required for the fastest atom to cross it's radius
    v_max = 0
    r = 0
    for i in range(num_atoms):
        v = vec.magnitude(velocities[i]) / radii[i]
        if v > v_max:
            v_max = v
            r = radii[i]

    if v_max == 0:
        return 1
    return r / v_max


def wall_collision_time(initial_pos, velocity, acceleration, target_value):
    """
    Calculates the smallest positive time 't' when a point hits a target value 
    along a specific dimension.
    
    Args:
        initial_pos (float): Initial coordinate
        velocity (float): Initial velocity along the axis
        acceleration (float): Constant acceleration
        target_value (float): The coordinate value of the "wall"
        
    Returns:
        float: The earliest positive time t, or None if it never hits.
    """
    x0 = initial_pos
    v0 = velocity
    a = acceleration
    
    # Coefficients for the quadratic equation: At^2 + Bt + C = 0
    A = 0.5 * a
    B = v0
    C = x0 - target_value

    # Case 1: No acceleration (Linear equation: Bt + C = 0)
    if A == 0:
        if B == 0:
            print("Stationary point")
            return None # Stationary and not at target
        t = -C / B
        if t >= 0:
            return t
        else:
            print("Negative time")
            return None

    # Case 2: Quadratic equation (At^2 + Bt + C = 0)
    # Calculate discriminant: D = B^2 - 4AC
    discriminant = B**2 - 4 * A * C
    
    if discriminant < 0:
        print("discriminant < 0")
        return None # No real solution (point turns back before hitting)
    
    sqrt_d = math.sqrt(discriminant)
    t1 = (-B + sqrt_d) / (2 * A)
    t2 = (-B - sqrt_d) / (2 * A)
    
    # We want the smallest positive time
    solutions = [t for t in [t1, t2] if t >= 0]
    
    if solutions:
        return min(solutions)
    else:
        print(f"no solutions, {initial_pos = }, {velocity = }, {acceleration = }, {target_value = }")
        return None


def get_edge_collision(radii, init_coords, new_coords, init_velocities, new_velocities, timestep):
    """ Return value: 
    {
        "c_type": "wall",
        "c_time": timestep,
        "c_axis": axis,
        "atom": -1,
    } or None
    """
    num_atoms = len(init_coords)
    min_t = timestep
    idx = -1
    axis = 0
    for i in range(num_atoms):
        for d in range(NUM_DIMENSIONS):
            s0 = WORLD_LIMITS[d][0] + radii[i]
            s1 = WORLD_LIMITS[d][1] - radii[i]
            if new_coords[i][d] <= s0:
                v0 = init_velocities[i][d]
                v1 = new_velocities[i][d]
                dv = v1 - v0
                a = dv / timestep
                t = wall_collision_time(init_coords[i][d], v0, a, s0)
                if t is None:
                    print(f"{i = }, {d = }, {new_coords[i][d] = }")
                if t < min_t:
                    min_t = t
                    idx = i
                    axis = d
            elif new_coords[i][d] >= s1:
                v0 = init_velocities[i][d]
                v1 = new_velocities[i][d]
                dv = v1 - v0
                a = dv / timestep
                t = wall_collision_time(init_coords[i][d], v0, a, s1)
                if t is None:
                    print(f"{i = }, {d = }, {new_coords[i][d] = }")
                if t < min_t:
                    min_t = t
                    idx = i
                    axis = d
    if idx > -1:
        return {
        "c_type": "wall",
        "c_time": min_t,
        "c_axis": axis,
        "atom": idx,
    }
    return None


def get_atom_collision(radii, init_coords, new_coords, init_velocities, new_velocities, timestep):
    """ Return value: 
    {
        "c_type": "atom",
        "c_time": timestep,
        "atoms": [-1, -1]
    } or None
    """
    return None


def get_first_collision(radii, init_coords, new_coords, init_velocities, new_velocities, timestep):
    ts1 = get_edge_collision(radii, init_coords, new_coords, init_velocities, new_velocities, timestep)
    ts2 = get_atom_collision(radii, init_coords, new_coords, init_velocities, new_velocities, timestep)
    if ts1 and ts2:
        if ts1["c_time"] < ts2["c_time"]:
            return ts1
        return ts2
    if ts1:
        return ts1
    return None


def process_collision(c, coords, step2_coords, velocities, new_velocities, ts):
    pass


def update_coords(atoms, links, timestep=1):
    num_atoms = len(atoms)
    velocities = [atoms[i]["speed"] for i in range(num_atoms)]
    coords = [atoms[i]["coords"] for i in range(num_atoms)]
    masses = [atoms[i]["mass"] for i in range(num_atoms)]
    radii = [atoms[i]["radius"] for i in range(num_atoms)]
    passed_time = 0
    while passed_time < timestep:
        ts = get_time_substep(velocities, radii, num_atoms)
        last_iter = False
        if (passed_time + ts) > timestep:
            ts = timestep - passed_time
            last_iter = True

        step1_coords = get_temp_coords(radii, coords, velocities, num_atoms, ts)
        step2_coords, deltas = process_links(radii, step1_coords, masses, links, ts)
        new_velocities = update_velocities(velocities, deltas, num_atoms, ts)

        c = get_first_collision(radii, coords, step2_coords, velocities, new_velocities, ts)
        if c:
            uts = c["c_time"]
            if (passed_time + uts) > timestep:
                uts = timestep - passed_time
            step1_coords = get_temp_coords(radii, coords, velocities, num_atoms, uts)
            step2_coords, deltas = process_links(radii, step1_coords, masses, links, uts)
            new_velocities = update_velocities(velocities, deltas, num_atoms, uts)
            if c["c_type"] == "wall":
                idx = c["atom"]
                new_velocities[idx][c["c_axis"]] *= -0.9   # Wall collision losses
                coords_within_box(step2_coords[idx], new_velocities[idx], radii[idx]* 1.01)
                print(f"{idx = }, {step2_coords[idx] = }")
            elif c["c_type"] == "atom":
                pass    # TODO: ImplementMe!
            else:
                raise Exception(f"Error: Unknown collision type: {c['type']}")
            
            passed_time += uts
        else:
            passed_time += ts

        # Update coords:
        for idx in range(num_atoms):
            for d in range(NUM_DIMENSIONS):
                coords[idx][d]     = step2_coords[idx][d]
                velocities[idx][d] = new_velocities[idx][d]

        if last_iter:   # Avoid float errors
            break

    # Update coords:
    for idx in range(num_atoms):
        atoms[idx]["coords"] = coords[idx]
        atoms[idx]["speed"] = velocities[idx]


atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [450, 500],
        "force": [0, 0],
        "speed": [0, 3],
        "mass": POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [500, 550],
        "force": [0, 0],
        "speed": [0, 0],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [550, 500],
        "force": [0, 0],
        "speed": [0, -3],
        "mass": POINT_MASS,
        "color": "#FF0000"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [300, 300],
        "force": [0, 0],
        "speed": [5, 15],
        "mass": POINT_MASS,
        "color": "#FFFF00"
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
    time.sleep(1/60)
    # if counter == 5:
    #     sys.exit()
    root.after(40, update_world)

update_world()
root.mainloop()
