ATOM_RADIUS = 10
POINT_MASS = 1  # 1 gramm
STIFFNESS = 0.75

wheel_atoms = [
    {
        "id": None, # 1
        "radius": ATOM_RADIUS,
        "coords": [475, 340, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 2
        "radius": ATOM_RADIUS,
        "coords": [525, 340, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 3
        "radius": ATOM_RADIUS,
        "coords": [500, 300, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 4
        "radius": ATOM_RADIUS,
        "coords": [525, 260, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 5
        "radius": ATOM_RADIUS,
        "coords": [475, 260, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 6
        "radius": ATOM_RADIUS,
        "coords": [550, 300, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 0, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
    {
        "id": None, # 7
        "radius": ATOM_RADIUS,
        "coords": [450, 300, 100],
        "2d_coords": [],
        "force": [0, 0],
        "speed": [-2, 50, 0],
        "mass": POINT_MASS,
        "color": "#105010"
    },
]

wheel_links = [
    {
        "atoms": [0, 1],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [1, 2],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [0, 2],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [0, 6],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [1, 5],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [2, 3],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [3, 4],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [2, 4],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [3, 5],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [6, 4],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [6, 2],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
    {
        "atoms": [2, 5],
        "length": 50,
        "stiffness": STIFFNESS,
        "id": None
    },
]

def add_wheel(atoms, links):
    inc = len(atoms)
    for link in wheel_links:
        link["atoms"][0] += inc
        link["atoms"][1] += inc
    atoms.extend(wheel_atoms)
    links.extend(wheel_links)