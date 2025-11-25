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


def get_relative_velocities(velocity_a, velocity_b):
    """
    Calculate the speed at which two points move toward each other.

    Parameters:
    velocity_a (list): Velocity vector for point A.
    velocity_b (list): Velocity vector for point B.

    Returns:
    (float, float): Two values indicating how fast point A is moving towards B and
                    how fast point B is moving towards A.
    """
    # Calculate the direction from A to B
    direction_ab = [b - a for a, b in zip(velocity_a, velocity_b)]
    
    # Calculate the magnitude of this direction vector
    distance_magnitude = vec.magnitude(direction_ab)
    
    # Calculate the relative velocity vector of A towards B
    if distance_magnitude != 0:
        speed_a_towards_b = vec.dot(velocity_a, direction_ab) / distance_magnitude
        relative_velocity_a = [(speed_a_towards_b * d / distance_magnitude) for d in direction_ab]
    else:
        relative_velocity_a = [0 for _ in velocity_a]  # No movement if points are the same

    # Calculate the relative velocity vector of B towards A
    if distance_magnitude != 0:
        speed_b_towards_a = vec.dot(velocity_b, [-d for d in direction_ab]) / distance_magnitude
        relative_velocity_b = [(-speed_b_towards_a * d / distance_magnitude) for d in direction_ab]
    else:
        relative_velocity_b = [0 for _ in velocity_b]  # No movement if points are the same

    return relative_velocity_a, relative_velocity_b


def find_link_target_points(p0, p1, target_length):
    link_length = get_distance(p0, p1)
    coeff = target_length/link_length
    middle_point = [(x0 + x1) / 2 for x0, x1 in zip(p0, p1)]
    target_point_0 = [middle_point[i] + coeff * (p0[i] - middle_point[i]) for i in range(NUM_DIMENSIONS)]
    target_point_1 = [middle_point[i] + coeff * (p1[i] - middle_point[i]) for i in range(NUM_DIMENSIONS)]
    return target_point_0, target_point_1


def calculate_forces(atoms, links, atoms_to_process=[]):
    for link in links:
        if len(atoms_to_process) > 0:
            if (link["atoms"][0] not in atoms_to_process) and (link["atoms"][0] not in atoms_to_process):
                continue
        point0 = atoms[link["atoms"][0]]["coords"]
        point1 = atoms[link["atoms"][1]]["coords"]
        # v0 = atoms[link["atoms"][0]]["speed"]
        # v1 = atoms[link["atoms"][1]]["speed"]
        target_point_0, target_point_1 = find_link_target_points(point0, point1, link["length"])
        error = get_distance(point0, point1) - link["length"]
        p0_res = pid_force(point0, target_point_0, error, link["stiffness"], link["integral_error"][0], link["previous_error"][0])
        p1_res = pid_force(point1, target_point_1, error, link["stiffness"], link["integral_error"][1], link["previous_error"][1])

        link["integral_error"][0] = p0_res["integral_error"]
        link["previous_error"][0] = p0_res["previous_error"]
        # force_p0 = p0_res["force"]

        link["integral_error"][1] = p1_res["integral_error"]
        link["previous_error"][1] = p1_res["previous_error"]
        # force_p1 = p1_res["force"]

        for i in range(NUM_DIMENSIONS):
            atoms[link["atoms"][0]]["force"][i] += p0_res["force"][i]
            atoms[link["atoms"][1]]["force"][i] += p1_res["force"][i]

        # vel1 = atoms[link["atoms"][0]]['speed']
        # vel2 = atoms[link["atoms"][1]]['speed']
        # link_vector = vec.sub(point0, point1)
        # # unit_link_vector = vec.normalize(link_vector)
        # # relative_velocity = vec.sub(vel2, vel1)
        # # relative_velocity_along_link = vec.dot(relative_velocity, unit_link_vector)
        # damping_coefficient = -0.1  # Set it to 1.0 for this example
        # # damping_ratio = min(1, abs(relative_velocity_along_link) / MAX_RELATIVE_SPEED)
        # # damping_force_on_atom1 = [-damping_coefficient * damping_ratio * relative_velocity_along_link * unit_link_vector[i] for i in range(NUM_DIMENSIONS)]
        # # damping_force_on_atom2 = [damping_coefficient * damping_ratio * relative_velocity_along_link * unit_link_vector[i] for i in range(NUM_DIMENSIONS)]
        # # vel1_proj = vec.vector_projection(link_vector, vel1)
        # # vel2_proj = vec.vector_projection(link_vector, vel2)
        # rel_vel1, rel_vel2 = get_relative_velocities(vel1, vel2)

        # # for axis in range(NUM_DIMENSIONS):
        # #     atoms[link["atoms"][0]]['speed'][axis] -= 0.9 * vel1_proj[axis]
        # #     atoms[link["atoms"][1]]['speed'][axis] -= 0.9 * vel2_proj[axis]

        # link_len = get_distance(point0, point1)
        # force = (link_len - link["length"]) * link["stiffness"] * 0.2
        # force_p1, force_p2 = calculate_force_vectors(point0, point1, force)
        # total_force_on_atom1 = [force_p1[i] + rel_vel1[i] * damping_coefficient for i in range(NUM_DIMENSIONS)]
        # total_force_on_atom2 = [force_p2[i] + rel_vel2[i] * damping_coefficient for i in range(NUM_DIMENSIONS)]
        # for i in range(NUM_DIMENSIONS):
        #     # atoms[link["atoms"][0]]["force"][i] += force_p1[i]
        #     # atoms[link["atoms"][1]]["force"][i] += force_p2[i]
        #     atoms[link["atoms"][0]]["force"][i] += total_force_on_atom1[i]
        #     atoms[link["atoms"][1]]["force"][i] += total_force_on_atom2[i]


# def apply_drag(v):
#     if v < 0:
#         return v + (DRAG_COEFFICIENT * v**2)
#     return v - (DRAG_COEFFICIENT * v**2)


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

    if len(positive_solutions) == 1:
        return positive_solutions[0]
    elif len(positive_solutions) == 2:
        return min(*positive_solutions)
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


def update_coords(atoms, links):
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
            update_velocities(atoms, accelerations, ts)
        for i in range(len(atoms)):
            for axis in range(NUM_DIMENSIONS):
                atoms[i]["coords"][axis] = new_positions[i][axis]
        
        num_steps += 1
        time_passed += ts
        if time_passed == 1:
            break


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
        d = get_distance(get_new_position(p1, v1, a1, t), get_new_position(p2, v2, a2, t))
        # d = vec.sub(position_at_time(p1, v1, a1, t), position_at_time(p2, v2, a2, t))
        # return vec.dot(d, d) - R*R
        return d - R

    # Check if collision happens within [0, dt]
    if f(0) <= 0: 
        return 0.0
    if f(dt) > 0:
        print(f"{atom1 = }\n{atom2 = }\n, {acceleration1 = }\n{acceleration2 = }\n{dt = }")
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
        "coords": [450, 500],
        "prev_coords": [450, 500],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [-20, 0],
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
        "prev_coords": [500, 550],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [20, 0],
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
        "prev_coords": [550, 500],
        "path": [0 for _ in range(NUM_DIMENSIONS)],
        "acceleration": [G_CONST if i == VERTICAL_AXIS else 0 for i in range(NUM_DIMENSIONS)],
        "force": [0, 0],
        "speed": [0, 8],
        "prev_speed": [0, 8],
        "invert_speed": [0 for _ in range(NUM_DIMENSIONS)],
        "mass": POINT_MASS,
        "inv_mass": 1/POINT_MASS,
        "color": "#FF0000"
    },
]

links = [
    # {
    #     "atoms": [0, 1],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None,
    #     "integral_error": [None, None],
    #     "previous_error": [None, None]
    # },
    # {
    #     "atoms": [1, 2],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None,
    #     "integral_error": [None, None],
    #     "previous_error": [None, None]
    # },
    # {
    #     "atoms": [2, 0],
    #     "length": 70,
    #     "stiffness": 10,
    #     "id": None,
    #     "integral_error": [None, None],
    #     "previous_error": [None, None]
    # },
]

def time_to_fall(height, init_velocity):
    g = -G_CONST  # acceleration due to gravity in m/s^2

    a = 0.5 * g
    b = init_velocity
    c = -(height - WORLD_LIMITS[VERTICAL_AXIS][0])

    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # No real solution, object never reaches the ground
    
    multiplier = 1 if init_velocity >= 0 else -1

    # Calculate two possible times
    t1 = multiplier * (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = multiplier * (-b - math.sqrt(discriminant)) / (2 * a)
    # print(f"{height = }, {init_velocity = }, {t1 = }, {t2 = }")

    # We only want the positive time
    return max(t1, t2)


def atom_process_accelerations(atom, ts):
    for axis in range(NUM_DIMENSIONS):
        atom["prev_coords"][axis] = atom["coords"][axis]
        # if axis == VERTICAL_AXIS:
        if axis == VERTICAL_AXIS:
            new_pos = atom["coords"][axis] + atom["speed"][axis] * ts + 0.5 * atom["acceleration"][axis] * ts**2
            if new_pos <= WORLD_LIMITS[axis][0]:
                t1 = time_to_fall(atom["coords"][axis], atom["speed"][axis])
                if t1 >= ts:
                    raise Exception(f"{t1 = }, {ts = }")
                t2 = ts - t1
                v_temp = atom["speed"][axis] + atom["acceleration"][axis] * t1  # Time when reaches 0
                v_temp *= -0.9  # Invert and decrease
                v1 = v_temp + atom["acceleration"][axis] * t2
                new_pos = WORLD_LIMITS[axis][0] + v_temp * t2 + 0.5 * atom["acceleration"][axis] * t2**2
                if new_pos < 0: # Handle multiple small bumps case
                    new_pos = 0
                    v1 = 0
            else:
                v1 = atom["speed"][axis] + atom["acceleration"][axis] * ts
            t = abs(-atom["speed"][axis]/G_CONST)
            if atom["speed"][axis] > 0 and v1 < 0:
                apogee = atom["coords"][axis] + atom["speed"][axis] * ts + 0.5 * atom["acceleration"][axis] * t**2
                print(f"Apogee: {apogee}")
            atom["coords"][axis] = round(new_pos, 6)
        else:
            atom["coords"][axis] += atom["speed"][axis] * ts + 0.5 * atom["acceleration"][axis] * ts**2
            v1 = atom["speed"][axis] + atom["acceleration"][axis] * ts
        atom["path"][axis] = v1 * ts


def update_coords_new(atoms, links, ts=1):
    for atom in atoms:
        atom_process_accelerations(atom, ts)

    for atom in atoms:
        for axis in range(NUM_DIMENSIONS):
            if atom["coords"][axis] < WORLD_LIMITS[axis][0]:
                print(f"Hit 0-th wall, {atom['coords'][axis] = }")
                d = (WORLD_LIMITS[axis][0] - atom["coords"][axis])
                atom["path"][axis] = (atom["path"][axis] + d * 0.2) * -1
                atom["coords"][axis] = d * 0.8
                # atom["invert_speed"][axis] = 1
                # print(f"Updated {atom['path'] = }, {atom['coords'] = }, {d = }")
            elif atom["coords"][axis] > WORLD_LIMITS[axis][1]:
                d = atom["coords"][axis] - WORLD_LIMITS[axis][1]
                atom["path"][axis] = (atom["path"][axis] - d * 0.2) * -1
                atom["coords"][axis] = WORLD_LIMITS[axis][1] - d * 0.8
                # atom["invert_speed"][axis] = 1
                # print("Hit 1-th wall")
                # print(f"Updated {atom['path'] = }, {atom['coords'] = }, {d = }")
        # print(f"Updated {atom['path'] = }, {atom['coords'] = }")
        # sys.exit()

    for atom in atoms:
        for axis in range(NUM_DIMENSIONS):
            atom["speed"][axis] = atom["path"][axis] / ts
            if atom["invert_speed"][axis]:
                atom["speed"][axis] *= -1
                atom["invert_speed"][axis] = 0


# atom['path'] =         [20.0, -25.480000000000388], atom['prev_coords'] = [1000.0, 231.49999999999417]
# Updated atom['path'] = [20.0, -25.480000000000388], atom['coords'] =      [1020.0, 206.0199999999938], d = 980.0

first_run = True
counter = 0

def update_world():
    global first_run, counter

    if not first_run:
        update_coords_new(atoms, links)
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
    # time.sleep(1)
    # if counter == 27:
    #     sys.exit()


    root.after(40, update_world)

update_world()
root.mainloop()