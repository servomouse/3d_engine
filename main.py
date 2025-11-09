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

# Circle properties
radius = 10
circle_id = None  # Will store the canvas ID of the circle

def move_circle():
    global circle_id
    
    # Delete the previous circle if it exists
    if circle_id is not None:
        canvas.delete(circle_id)
    
    # Generate random coordinates within canvas boundaries
    x = random.randint(radius, WINDOW_WIDTH - radius)
    y = random.randint(radius, WINDOW_HEIGHT - radius)
    
    # Draw the new circle
    circle_id = canvas.create_oval(
        x - radius, y - radius,  # Top-left corner
        x + radius, y + radius,  # Bottom-right corner
        fill="blue", outline="darkblue"
    )
    
    # Schedule the next move after 1000ms (1 second)
    root.after(1000, move_circle)

# Start the animation
move_circle()

# Start the Tkinter event loop
root.mainloop()