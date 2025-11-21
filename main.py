import tkinter as tk
import random
import json
import sys
import time
import math
import copy
import vector_math as vec

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
        atom["force"][VERTICAL_AXIS] += G_CONST * atom["mass"]


def atom_add_gravity(atom):
    atom["force"][VERTICAL_AXIS] += G_CONST * atom["mass"]


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


def calculate_forces(atoms, links, atoms_to_process=[]):
    for link in links:
        if len(atoms_to_process) > 0:
            if (link["atoms"][0] not in atoms_to_process) and (link["atoms"][0] not in atoms_to_process):
                continue
        point1 = atoms[link["atoms"][0]]["coords"]
        point2 = atoms[link["atoms"][1]]["coords"]
        link_len = get_distance(point1, point2)
        force = (link_len - link["length"]) * link["stiffness"]
        force_p1, force_p2 = calculate_force_vectors(point1, point2, force)
        # print(f"{force_p1 = }, {force_p2 = }")
        for i in range(NUM_DIMENSIONS):
            atoms[link["atoms"][0]]["force"][i] += force_p1[i]
            atoms[link["atoms"][1]]["force"][i] += force_p2[i]


def apply_drag(v):
    if v < 0:
        return v + (DRAG_COEFFICIENT * v**2)
    return v - (DRAG_COEFFICIENT * v**2)


def get_atom_accelerations(atom):
    acceleration = []
    for axis in range(NUM_DIMENSIONS):
        f = atom["force"][axis]
        if axis == VERTICAL_AXIS:
            f += G_CONST
        acceleration.append(f / atom["mass"])
        atom["force"][axis] = 0 # Clear force vector after calculating accelerations
    return acceleration


def get_accelerations(atoms):
    accelerations = []
    for i in range(len(atoms)):
        accelerations.append(get_atom_accelerations(atoms[i]))
        # for axis in range(NUM_DIMENSIONS):
        #     f = atoms[i]["force"][axis]
        #     if axis == VERTICAL_AXIS:
        #         f += G_CONST
        #     accelerations[-1].append(f / atoms[i]["mass"])
        #     atoms[i]["force"][axis] = 0 # Clear force vector after calculating accelerations
    return accelerations


def sum_vectors(v1, v2):
    result = [0 for _ in range(len(v1))]
    for axis in range(len(v1)):
        result[axis] = v1[axis] + v2[axis]
    return result


def get_timestep(atoms, accelerations):
    max_speed = 0
    fastest_idx = -1
    for i in range(len(atoms)):
        v = vec.vector_len(sum_vectors(atoms[i]["speed"], accelerations[i]))
        if v > max_speed:
            max_speed = v
            fastest_idx = i
    num_substeps = math.ceil(max_speed / atoms[fastest_idx]["radius"])
    return 1 / num_substeps if num_substeps > 0 else 1


def get_new_position(init_position, velocity, acceleration, timestep):
    new_position = [init_position[axis] for axis in range(NUM_DIMENSIONS)]
    for axis in range(NUM_DIMENSIONS):
        v = velocity[axis]
        a = acceleration[axis]
        t = timestep
        np = new_position[axis] + v * t + (a * t**2) / 2
        if axis == VERTICAL_AXIS and np < 0:
            np = 0
        new_position[axis] = round(np, 9)
    return new_position


def detect_collisions(atoms, accelerations, new_positions, timestep):
    collisions = []
    for i in range(len(atoms)-1):
        r1 = atoms[i]["radius"]
        for j in range(i+1, len(atoms)):
            r2 = atoms[j]["radius"]
            d = get_distance(new_positions[i], new_positions[j])
            if d <= r1 + r2:
                t = collision_time(atoms[i], atoms[j], accelerations[i], accelerations[j], timestep)
                # print(f"Collision detected between atoms {i} and {j} at time +{t}")
                collisions.append({
                    "atoms": [i, j],
                    "time": t
                })
    return collisions


def get_first_atoms_bump(atoms, accelerations, new_positions, timestep):
    collision = {
        "type": "atom to atom",
        "atoms": [],
        "time": timestep
    }
    collision_found = False
    for i in range(len(atoms)-1):
        r1 = atoms[i]["radius"]
        for j in range(i+1, len(atoms)):
            r2 = atoms[j]["radius"]
            d = get_distance(new_positions[i], new_positions[j])
            if d <= r1 + r2:
                t = collision_time(atoms[i], atoms[j], accelerations[i], accelerations[j], timestep)
                # print(f"Collision detected between atoms {i} and {j} at time +{t}")
                if t < collision["time"]:
                    collision["atoms"] = [i, j]
                    collision["time"] = t
                    collision_found = True
    if collision_found:
        return collision
    return None


def get_first_wall_collision_time(initial_position, velocity, acceleration, target_coordinate):
    a = 0.5 * acceleration
    b = velocity
    c = initial_position - target_coordinate

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise Exception("No real solutions!")

    sqrt_discriminant = discriminant**0.5
    t1 = (-b + sqrt_discriminant) / (2 * a) if a != 0 else -c / b
    t2 = (-b - sqrt_discriminant) / (2 * a) if a != 0 else -c / b

    positive_solutions = []
    if t1 >= 0:
        positive_solutions.append(t1)
    if t2 >= 0 and t2 != t1:
        positive_solutions.append(t2)

    # print(f"{initial_position = }, {velocity = }, {acceleration = }, {target_coordinate = }, {positive_solutions = }")

    if len(positive_solutions) == 1:
        return positive_solutions[0]
    elif len(positive_solutions) == 2:
        return min(*positive_solutions)
    # print(f"{initial_position = }, {velocity = }, {acceleration = }, {target_coordinate = }")
    # raise Exception("No collision time was found!")
    return None


def get_first_wall_collision(atoms, accelerations, new_positions, timestep):
    collision = {
        "type": "atom to wall",
        "atom": [],
        "time": timestep,
        "axis": None
    }
    collision_found = False
    for i in range(len(atoms)):
        for axis in range(NUM_DIMENSIONS):
            t = timestep
            if new_positions[i][axis] <= WORLD_LIMITS[axis][0]:
                t = get_first_wall_collision_time(atoms[i]["coords"][axis], atoms[i]["speed"][axis], accelerations[i][axis], WORLD_LIMITS[axis][0])
                if t == 0:
                    # print(f"Flipping atom speed vector, {atoms[i]['speed'][axis] = }")
                    atoms[i]["speed"][axis] *= -1   # If it's on the floor, flip the velocity vector
                    t = timestep
            elif new_positions[i][axis] >= WORLD_LIMITS[axis][1]:
                t = get_first_wall_collision_time(atoms[i]["coords"][axis], atoms[i]["speed"][axis], accelerations[i][axis], WORLD_LIMITS[axis][1])
            else:
                continue
            if t is None:
                continue
            try:
                if t < collision["time"]:
                    collision["time"] = t
                    collision["atom"] = i
                    collision["axis"] = axis
                    collision_found = True
            except TypeError:
                raise Exception(f"{t = }, {collision['time'] = }")

    if collision_found:
        return collision
    return None

def get_first_collision(atoms, accelerations, new_positions, ts):
    atoms_bump = get_first_atoms_bump(atoms, accelerations, new_positions, ts)
    wall_bump = get_first_wall_collision(atoms, accelerations, new_positions, ts)
    if atoms_bump and wall_bump:
        if atoms_bump["time"] < wall_bump["time"]:
            return atoms_bump
    if wall_bump:
        return wall_bump
    if atoms_bump:
        return atoms_bump
    return None


def update_velocities(atoms, accelerations, ts):
    for i in range(len(atoms)):
        for axis in range(NUM_DIMENSIONS):
            atoms[i]["speed"][axis] += accelerations[i][axis] * ts


def update_coords(atoms, links, n):
    # print(f"Step {n}")
    time_passed = 0
    num_steps = 0
    while time_passed < 1:
        calculate_forces(atoms, links)
        accelerations = get_accelerations(atoms)
        ts = get_timestep(atoms, accelerations)
        if ts == 0:
            raise Exception("Timestep == 0!")
        if ts + time_passed > 1:
            ts = 1 - time_passed    # Trim timestep to end at 1
        new_positions = [get_new_position(atoms[i]["coords"], atoms[i]["speed"], accelerations[i], ts) for i in range(len(atoms))]
        c = get_first_collision(atoms, accelerations, new_positions, ts)    # Find the first collision within the timestep if any
        if c:
            # print(f"Collision detected at {c['time'] = }")
            new_positions = [get_new_position(atoms[i]["coords"], atoms[i]["speed"], accelerations[i], c["time"]) for i in range(len(atoms))]
            update_velocities(atoms, accelerations, c["time"])
            if c["type"] == "atom to atom":
                i = c["atoms"][0]
                j = c["atoms"][1]
                atoms[i]["speed"], atoms[j]["speed"] = collide_atoms(atoms[i]["speed"], atoms[j]["speed"], atoms[i]["coords"], atoms[j]["coords"])
            elif c["type"] == "atom to wall":
                i = c["atom"]
                axis = c["axis"]
                atoms[i]["speed"][axis] *= -0.9
            time_passed += c["time"]
        else:
            # print("No collisions")
            update_velocities(atoms, accelerations, ts)
        # print(f"{atoms = }")
        # for i in range(len(atoms)):
        #     print(f"{atoms[i]['coords'] = }, {new_positions[i] = }")
        # Update positions
        for i in range(len(atoms)):
            for axis in range(NUM_DIMENSIONS):
                atoms[i]["coords"][axis] = new_positions[i][axis]
        
        num_steps += 1
        time_passed += ts
        if time_passed == 1:
            # print(f"Num steps: {num_steps}")
            break

    # while True:
    #     calculate_forces(atoms, links)
    #     # add_gravity(atoms)

    #     accelerations = get_accelerations(atoms)

    #     ts = get_timestep(atoms, accelerations)
    #     if ts == 0:
    #         break   # Nothing to change
    #     if time_passed + ts > 1:
    #         ts = 1 - time_passed

    #     new_positions = [get_new_position(atoms[i]["coords"], atoms[i]["speed"], accelerations[i], ts) for i in range(len(atoms))]
    #     collisions = detect_collisions(atoms, accelerations, new_positions, ts)
    #     exclude_new_speed_calc = []
    #     for c in collisions:
    #         i = c["atoms"][0]
    #         j = c["atoms"][1]
    #         atoms[i]["coords"] = position_at_time(atoms[i]["coords"], atoms[i]["speed"], accelerations[i], c["time"])
    #         atoms[j]["coords"] = position_at_time(atoms[j]["coords"], atoms[j]["speed"], accelerations[j], c["time"])
    #         atoms[i]["speed"], atoms[j]["speed"] = collide_atoms(
    #             speed_at_time(atoms[i]["speed"], accelerations[i], c["time"]),
    #             speed_at_time(atoms[j]["speed"], accelerations[j], c["time"]),
    #             atoms[i]["coords"],
    #             atoms[j]["coords"]
    #         )
    #         remaining_time = ts - c["time"]
    #         calculate_forces(atoms, links, [i, j])
    #         a1 = get_atom_accelerations(atoms[i])
    #         a2 = get_atom_accelerations(atoms[j])
    #         new_positions[i] = get_new_position(atoms[i]["coords"], atoms[i]["speed"], a1, remaining_time)
    #         new_positions[j] = get_new_position(atoms[j]["coords"], atoms[j]["speed"], a2, remaining_time)
    #         atoms[i]["speed"] = speed_at_time(atoms[i]["speed"], a1, remaining_time)
    #         atoms[j]["speed"] = speed_at_time(atoms[j]["speed"], a2, remaining_time)
    #         exclude_new_speed_calc.extend([i, j])


    #     # Update velocities
    #     for i in range(len(atoms)):
    #         if i in exclude_new_speed_calc:
    #             continue
    #         for axis in range(NUM_DIMENSIONS):
    #             atoms[i]["speed"][axis] += accelerations[i][axis] * ts

    #     # Update positions
    #     for i in range(len(atoms)):
    #         for axis in range(NUM_DIMENSIONS):
    #             atoms[i]["coords"][axis] = new_positions[i][axis]
    #             if atoms[i]["coords"][axis] < (WORLD_LIMITS[axis][0] + atoms[i]["radius"]):
    #                 atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][0] + atoms[i]["radius"]) #  + random.random()
    #                 atoms[i]["speed"][axis] = 0
    #             elif atoms[i]["coords"][axis] > (WORLD_LIMITS[axis][1] - atoms[i]["radius"]):
    #                 atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][1] - atoms[i]["radius"]) #  - random.random()
    #                 atoms[i]["speed"][axis] = 0

    #     num_steps += 1
    #     time_passed += ts
    #     if time_passed == 1:
    #         # print(f"Num steps: {num_steps}")
    #         break

    # time_step = 1

    # calculate_forces(atoms, links)

    # accelerations = get_accelerations(atoms)

    # # Update velocities
    # for i in range(len(atoms)):
    #     for axis in range(NUM_DIMENSIONS):
    #         atoms[i]["speed"][axis] += accelerations[i][axis] * time_step
    #         if axis == VERTICAL_AXIS:
    #             atoms[i]["speed"][axis] +=  G_CONST * time_step

    # # Update positions
    # for i in range(len(atoms)):
    #     for axis in range(NUM_DIMENSIONS):
    #         atoms[i]["coords"][axis] += atoms[i]["speed"][axis] * time_step
    #         if atoms[i]["coords"][axis] < (WORLD_LIMITS[axis][0] + atoms[i]["radius"]):
    #             atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][0] + atoms[i]["radius"]) #  + random.random()
    #             atoms[i]["speed"][axis] = 0
    #         elif atoms[i]["coords"][axis] > (WORLD_LIMITS[axis][1] - atoms[i]["radius"]):
    #             atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][1] - atoms[i]["radius"]) #  - random.random()
    #             atoms[i]["speed"][axis] = 0


def collide_with_wall(atom, axis):
    atom["speed"][axis] *= -1


def collide_atoms(vA, vB, xA, xB):
    """
    Compute post-collision velocities of two equal-mass balls in n dimensions.
    
    Parameters:
        vA (list): velocity vector of ball A
        vB (list): velocity vector of ball B
        xA (list): position vector of ball A at collision
        xB (list): position vector of ball B at collision
    
    Returns:
        (list, list): post-collision velocity vectors (vAnew, vBnew)
    """
    # Collision normal (unit vector along line of centers)
    n = vec.sub(xB, xA)
    n_len = vec.norm(n)
    n = [ni / n_len for ni in n]
    
    # Relative velocity along n
    rel_vel = vec.dot(vec.sub(vA, vB), n) * 0.9
    
    # Update velocities
    vAnew = vec.sub(vA, vec.scale(n, rel_vel))
    vBnew = vec.add(vB, vec.scale(n, rel_vel))
    
    return vAnew, vBnew


def position_at_time(p, v, a, t):
    # p + v*t + 0.5*a*t^2
    return vec.add(vec.add(p, vec.scale(v, t)), vec.scale(a, 0.5*t*t))


def speed_at_time(v0, a, t):
    return vec.add(v0, vec.scale(a, t))


def collision_time(atom1, atom2, acceleration1, acceleration2, dt, tolerance=1e-8, max_iter=100):
    p1, v1, a1, r1 = atom1["coords"], atom1["speed"], acceleration1, atom1["radius"]
    p2, v2, a2, r2 = atom2["coords"], atom2["speed"], acceleration2, atom2["radius"]

    R = r1 + r2

    def f(t):
        d = vec.sub(position_at_time(p1, v1, a1, t), position_at_time(p2, v2, a2, t))
        return vec.dot(d, d) - R*R

    # Check if collision happens within [0, dt]
    if f(0) <= 0: 
        return 0.0
    if f(dt) > 0:
        return None  # no collision in this step

    # Bisection
    lo, hi = 0.0, dt
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tolerance:
            return mid
    return 0.5*(lo+hi)


# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

# atoms = [
#     {'id': None, 'radius': 10, 'coords': [160.744899958, 0.0], 'force': [0, 0], 'speed': [-9.261153255066688, 0.00035659582159219854], 'mass': 1, 'color': '#0000FF'},
#     {'id': None, 'radius': 10, 'coords': [952.810849173, 64.247182173], 'force': [0, 0], 'speed': [11.001459040236467, -7.185995012697021], 'mass': 1, 'color': '#00FF00'},
#     {'id': None, 'radius': 10, 'coords': [891.040322309, 45.849991594], 'force': [0, 0], 'speed': [-5.247168537588703, -2.518588637578341], 'mass': 1, 'color': '#FF0000'}]

atoms = [
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [800, 500],
        "force": [0, 0],
        "speed": [-20, 0],
        "mass": POINT_MASS,
        "color": "#0000FF",
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [200, 500],
        "force": [0, 0],
        "speed": [20, 0],
        "mass": POINT_MASS,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": ATOM_RADIUS,
        "coords": [600, 500],
        "force": [0, 0],
        "speed": [0, 8],
        "mass": POINT_MASS,
        "color": "#FF0000"
    },
]

links = [
    # {
    #     "atoms": [0, 1],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None
    # },
    # {
    #     "atoms": [1, 2],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None
    # },
    # {
    #     "atoms": [2, 0],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None
    # },
]

first_run = True
n_steps = 0

def update_world():
    global first_run, n_steps

    if not first_run:
        # update_speeds(atoms)
        # calculate_forces(atoms, links)
        update_coords(atoms, links, n_steps)
        n_steps += 1
    else:
        first_run = False
    # print(f"{atoms = }")

    for atom in atoms:
        if atom["id"] is not None:
            canvas.delete(atom["id"])
        atom["id"] = draw_atom(canvas, atom["coords"], atom["radius"], atom["color"])
    
    for link in links:
        if link["id"] is not None:
            canvas.delete(link["id"])
        link["id"] = draw_line(canvas, atoms[link["atoms"][0]]["coords"], atoms[link["atoms"][1]]["coords"])
    # time.sleep(10)
    # if n_steps > 1:
    #     sys.exit()


    root.after(40, update_world)

update_world()
root.mainloop()