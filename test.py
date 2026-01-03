import tkinter as tk
import random
import json
import sys
import time
import math
import copy
import vector_math as vec
from pid import pid_force
from objects import add_wheel

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
WORLD_DEPTH = 600
WORLD_LIMITS = [
    [0, WINDOW_WIDTH],
    [0, WINDOW_HEIGHT],
    [0, WORLD_DEPTH]
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
USE_GRAVITY = False


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


def invert_velocity(v, c_axis, factor):
    for axis in range(NUM_DIMENSIONS):
        if axis == c_axis:
            v[axis] *= -factor
        # else:
        #     v[axis] *= factor/20    # Desrease velocity along other axes to imitate friction


def coords_within_box(coords, velocity, radius, velocity_conservation=1):
    # Set velocity to None to not update it
    collision_found =False
    for axis in range(NUM_DIMENSIONS):
        if coords[axis] <= (WORLD_LIMITS[axis][0] + radius):
            coords[axis] = WORLD_LIMITS[axis][0] + radius
            if velocity and velocity[axis] < 0:
                invert_velocity(velocity, axis, velocity_conservation)
            collision_found = True
        elif coords[axis] >= (WORLD_LIMITS[axis][1] - radius):
            coords[axis] = WORLD_LIMITS[axis][1] - radius
            if velocity and velocity[axis] > 0:
                invert_velocity(velocity, axis, velocity_conservation)
            collision_found = True
    return collision_found


def get_temp_coords(radii, init_coords, init_velocities, num_atoms, timestep):
    # Find next positions based on the current velocity and gravity
    temp_coords = []
    collided_atoms = []
    for idx in range(num_atoms):
        if coords_within_box(init_coords[idx], init_velocities[idx], radii[idx]* 1.01, 0.9):
            collided_atoms.append(idx)
    for idx in range(num_atoms):
        temp_v = vec.add(init_coords[idx], vec.scale(init_velocities[idx], timestep))
        if USE_GRAVITY:
            if idx not in collided_atoms:
                # Accelerate half way, decelerate half way -> no total acceleration
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
    v_max *= 2  # Double to get half a radius
    return r / v_max


def resolve_collision(m1, r1, p1, v1, m2, r2, p2, v2, elasticity=0.9):
    """
    Calculates post-collision velocities and positions for two spheres.
    
    Parameters:
    m1, m2 : float - Masses
    r1, r2 : float - Radii
    p1, p2 : array - Position vectors
    v1, v2 : array - Velocity vectors
    elasticity : float - Coefficient of restitution (1.0 = perfectly elastic)
    """# 1. Directional Vector (p2 - p1)
    delta_p = [p2[i] - p1[i] for i in range(len(p1))]
    distance = vec.magnitude(delta_p)
    
    if distance == 0: 
        return p1, v1, p2, v2

    # 2. Normal Unit Vector
    normal = [x / distance for x in delta_p]
    
    # --- STEP 1: STATIC RESOLUTION (Separation) ---
    overlap = (r1 + r2) - distance
    if overlap > 0:
        total_m = m1 + m2
        # Move p1 back and p2 forward based on mass ratio
        move_dist1 = overlap * (m2 / total_m)
        move_dist2 = overlap * (m1 / total_m)
        
        p1 = [p1[i] - normal[i] * move_dist1 for i in range(len(p1))]
        p2 = [p2[i] + normal[i] * move_dist2 for i in range(len(p2))]

    # --- STEP 2: DYNAMIC RESOLUTION (Impulse) ---
    # Relative velocity (v1 - v2)
    rel_v = [v1[i] - v2[i] for i in range(len(v1))]
    
    # Normal velocity (scalar projection)
    vel_along_normal = vec.dot(rel_v, normal)
    
    # If they are already moving apart, don't bounce
    if vel_along_normal < 0:
        return p1, v1, p2, v2
    
    # Impulse scalar calculation
    # Formula: j = -(1 + e) * (v_rel . n) / (1/m1 + 1/m2)
    impulse_mag = (-(1 + elasticity) * vel_along_normal) / (1/m1 + 1/m2)
    
    # Final velocities
    v1_final = [v1[i] + (normal[i] * impulse_mag) / m1 for i in range(len(v1))]
    v2_final = [v2[i] - (normal[i] * impulse_mag) / m2 for i in range(len(v2))]
    
    return p1, v1_final, p2, v2_final


def wall_collisions(radii, coords, velocities, num_atoms):
    collision_found = False
    for idx in range(num_atoms):
        if coords_within_box(coords[idx], velocities[idx], radii[idx]* 1.01, 0.9):
            collision_found = True
    return collision_found


def atom_collisions(radii, masses, coords, velocities, num_atoms):
    collision_found = False
    for idx0 in range(num_atoms-1):
        for idx1 in range(idx0+1, num_atoms):
            d = vec.distance(coords[idx0], coords[idx1])
            rsum = radii[idx0] + radii[idx1]
            if d < rsum:
                # print(f"Atom collision detected! {d = }, {rsum = }, {coords[idx0] = }, {coords[idx1] = }")
                c0, v0, c1, v1 = resolve_collision(
                    masses[idx0], radii[idx0], coords[idx0], velocities[idx0],
                    masses[idx1], radii[idx1], coords[idx1], velocities[idx1])
                coords[idx0] = c0
                velocities[idx0] = v0
                coords[idx1] = c1
                velocities[idx1] = v1
                collision_found = True
    return collision_found


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

        c_counter = num_atoms * 2
        while c_counter > 0:
            wc = wall_collisions(radii, step2_coords, new_velocities, num_atoms)
            ac = atom_collisions(radii, masses, step2_coords, new_velocities, num_atoms)
            if (not wc) and (not ac):
                break
            c_counter -= 1

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


def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"


atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [450, 500, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [0, 3, random.random()],
        "mass": POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [500, 550, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [0, 0, random.random()],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [550, 500, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [0, -3, random.random()],
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
    {
        "atoms": [10, 11],
        "length": 100,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [11, 12],
        "length": 100,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [12, 13],
        "length": 100,
        "stiffness": STIFFNESS,
        "id": None
    },
]


add_wheel(atoms, links)


# Add random atoms
for i in range(20):
    atoms.append({
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [random.randint(WORLD_LIMITS[a][0]+50, WORLD_LIMITS[a][1]-50) for a in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [random.randint(-100, 100) for a in range(NUM_DIMENSIONS)],
        "mass": POINT_MASS,
        "color": random_color()
    })

# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

first_run = True
counter = 0

scale_coeff = -0.5/WORLD_DEPTH

def get_coords(coords):
    if len(coords) == 2:
        return coords   # Skip for 2D
    coeffs = [
        (WORLD_LIMITS[0][1] / 2),
        (WORLD_LIMITS[1][1] / 2)
    ]
    x = coords[0] / coeffs[0] - 1
    y = coords[1] / coeffs[1] - 1
    z = coords[2]
    x = x * (z*scale_coeff + 1) + 1
    y = y * (z*scale_coeff + 1) + 1
    return [
        x * coeffs[0],
        y * coeffs[1]
    ]


def scale_radius(r, z):
    return r * (z*scale_coeff + 1)


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
        atom["2d_coords"] = get_coords(atom["coords"])
        if NUM_DIMENSIONS > 2:
            atom["id"] = draw_atom(canvas, atom["2d_coords"], scale_radius(atom["radius"], atom["coords"][2]), atom["color"])
        else:
            atom["id"] = draw_atom(canvas, atom["2d_coords"], scale_radius(atom["radius"], 1), atom["color"])
    
    for link in links:
        if link["id"] is not None:
            canvas.delete(link["id"])
        a0 = link["atoms"][0]
        a1 = link["atoms"][1]
        link["id"] = draw_line(canvas, atoms[a0]["2d_coords"], atoms[a1]["2d_coords"])
    print(f"\r{counter = }", end="")
    time.sleep(1/60)
    if counter == 200:
        links[0]["length"] = 35
    if counter == 300:
        links[0]["length"] = 70
    # if counter == 5:
    #     sys.exit()
    root.after(40, update_world)

if NUM_DIMENSIONS > 2:
    x0y0z1 = get_coords([WORLD_LIMITS[0][0], WORLD_LIMITS[1][0], WORLD_LIMITS[2][1]])
    x0y1z1 = get_coords([WORLD_LIMITS[0][0], WORLD_LIMITS[1][1], WORLD_LIMITS[2][1]])
    x1y0z1 = get_coords([WORLD_LIMITS[0][1], WORLD_LIMITS[1][0], WORLD_LIMITS[2][1]])
    x1y1z1 = get_coords([WORLD_LIMITS[0][1], WORLD_LIMITS[1][1], WORLD_LIMITS[2][1]])
    draw_line(canvas, x0y0z1, x0y1z1)
    draw_line(canvas, x0y0z1, x1y0z1)
    draw_line(canvas, x1y1z1, x1y0z1)
    draw_line(canvas, x1y1z1, x0y1z1)

    draw_line(canvas, [WORLD_LIMITS[0][0], WORLD_LIMITS[1][0]], x0y0z1)
    draw_line(canvas, [WORLD_LIMITS[0][0], WORLD_LIMITS[1][1]], x0y1z1)
    draw_line(canvas, [WORLD_LIMITS[0][1], WORLD_LIMITS[1][0]], x1y0z1)
    draw_line(canvas, [WORLD_LIMITS[0][1], WORLD_LIMITS[1][1]], x1y1z1)

update_world()
root.mainloop()
