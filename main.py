import tkinter as tk
import random

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480

# Create the main window
root = tk.Tk()
root.title("Moving Circle")

# Create a canvas
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
canvas.pack()

circles = [
    {
        "id": None,
        "radius": 10,
        "coords": [],
        "force": [],
        "speed": [],
        "mass": 1,
        "color": "#0000FF"
    },
    {
        "id": None,
        "radius": 10,
        "coords": [],
        "force": [],
        "speed": [],
        "mass": 1,
        "color": "#00FF00"
    },
    {
        "id": None,
        "radius": 10,
        "coords": [],
        "force": [],
        "speed": [],
        "mass": 1,
        "color": "#FF0000"
    },
]

def update_world():
    for circle in circles:
        
        # Delete the previous circle if it exists
        if circle["id"] is not None:
            canvas.delete(circle["id"])
        
        # Generate random coordinates within canvas boundaries
        x = random.randint(circle["radius"], WINDOW_WIDTH - circle["radius"])
        y = random.randint(circle["radius"], WINDOW_HEIGHT - circle["radius"])
        
        # Draw the new circle
        circle["id"] = canvas.create_oval(
            x - circle["radius"], y - circle["radius"],  # Top-left corner
            x + circle["radius"], y + circle["radius"],  # Bottom-right corner
            fill=circle["color"], outline="darkblue"
        )
    
    # Schedule the next move after 1000ms (1 second)
    root.after(1000, update_world)

# Start the animation
update_world()

# Start the Tkinter event loop
root.mainloop()