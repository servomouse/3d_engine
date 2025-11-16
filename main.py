import tkinter as tk
import random
import json
import sys
import time
import math
import copy

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
        for i in range(NUM_DIMENSIONS):
            atoms[link["atoms"][0]]["force"][i] += force_p1[i]
            atoms[link["atoms"][1]]["force"][i] += force_p2[i]


def apply_drag(v):
    if v < 0:
        return v + (DRAG_COEFFICIENT * v**2)
    return v - (DRAG_COEFFICIENT * v**2)


def get_accelerations(atoms):
    accelerations = []
    for i in range(len(atoms)):
        accelerations.append([])
        for axis in range(NUM_DIMENSIONS):
            f = atoms[i]["force"][axis]
            if axis == VERTICAL_AXIS:
                f += G_CONST
            accelerations[-1].append(f / atoms[i]["mass"])
            atoms[i]["force"][axis] = 0 # Clear force vector after calculating accelerations
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
        v = vector_len(sum_vectors(atoms[i]["speed"], accelerations[i]))
        if v > max_speed:
            max_speed = v
            fastest_idx = i
    num_substeps = math.ceil(max_speed / atoms[fastest_idx]["radius"])
    return 1 / num_substeps if num_substeps > 0 else 0


def get_new_position(init_position, velocity, acceleration, timestep):
    new_position = [init_position[axis] for axis in range(NUM_DIMENSIONS)]
    for axis in range(NUM_DIMENSIONS):
        v = velocity[axis]
        a = acceleration[axis]
        t = timestep
        new_position[axis] += v * t + (a * t**2) / 2
    return new_position


def update_coords(atoms, links):
    time_passed = 0
    num_steps = 0
    while True:
        calculate_forces(atoms, links)
        add_gravity(atoms)

        accelerations = get_accelerations(atoms)

        ts = get_timestep(atoms, accelerations)
        if ts == 0:
            break   # Nothing to change
        if time_passed + ts > 1:
            ts = 1 - time_passed

        new_positions = [get_new_position(atoms[i]["coords"], atoms[i]["speed"], accelerations[i], ts) for i in range(len(atoms))]
        

        # Update velocities
        for i in range(len(atoms)):
            for axis in range(NUM_DIMENSIONS):
                atoms[i]["speed"][axis] += accelerations[i][axis] * ts
                # if axis == VERTICAL_AXIS:
                #     atoms[i]["speed"][axis] +=  G_CONST * ts

        # Update positions
        for i in range(len(atoms)):
            for axis in range(NUM_DIMENSIONS):
                atoms[i]["coords"][axis] = new_positions[i][axis]
                if atoms[i]["coords"][axis] < (WORLD_LIMITS[axis][0] + atoms[i]["radius"]):
                    atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][0] + atoms[i]["radius"]) #  + random.random()
                    atoms[i]["speed"][axis] = 0
                elif atoms[i]["coords"][axis] > (WORLD_LIMITS[axis][1] - atoms[i]["radius"]):
                    atoms[i]["coords"][axis] = (WORLD_LIMITS[axis][1] - atoms[i]["radius"]) #  - random.random()
                    atoms[i]["speed"][axis] = 0

        num_steps += 1
        time_passed += ts
        if time_passed == 1:
            # print(f"Num steps: {num_steps}")
            break

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
        "coords": [455, 500],
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

def update_world():
    global first_run

    if not first_run:
        # update_speeds(atoms)
        # calculate_forces(atoms, links)
        update_coords(atoms, links)
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