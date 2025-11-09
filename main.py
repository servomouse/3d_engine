import tkinter as tk
import random

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480
CIRCLE_RADIUS = 10


# Returns circle id
def draw_circle(canvas, coords, radius, color):
    return canvas.create_oval(
        coords[0] - radius, WINDOW_HEIGHT - (coords[1] - radius),  # Top-left corner
        coords[0] + radius, WINDOW_HEIGHT - (coords[1] + radius),  # Bottom-right corner
        fill=color, outline="darkblue"
    )


def get_random_coords(radius=CIRCLE_RADIUS):
    return [
        random.randint(radius, WINDOW_WIDTH - radius),
        random.randint(radius, WINDOW_HEIGHT - radius)
    ]


# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

circles = [
    {
        "id": None,
        "radius": CIRCLE_RADIUS,
        "coords": get_random_coords(),
        "force": [0, -1],
        "speed": [0, 0],
        "mass": 1,
        "color": "#0000FF"
    },
    {
        "id": None,
        "radius": CIRCLE_RADIUS,
        "coords": get_random_coords(),
        "force": [],
        "speed": [],
        "mass": 1,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": CIRCLE_RADIUS,
        "coords": get_random_coords(),
        "force": [],
        "speed": [],
        "mass": 1,
        "color": "#FF0000"
    },
]

def update_world():
    for circle in circles:
        if circle["id"] is not None:
            circle["coords"] = get_random_coords(circle["radius"])

    for circle in circles:
        if circle["id"] is not None:
            canvas.delete(circle["id"])
        
        # circle["coords"] = get_random_coords(circle["radius"])
        circle["id"] = draw_circle(canvas, circle["coords"], circle["radius"], circle["color"])

    root.after(1000, update_world)

# Start the animation
update_world()

# Start the Tkinter event loop
root.mainloop()