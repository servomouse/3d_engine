import tkinter as tk
import random

SPHERE_RADIUS = 5


class PhyEngine:
    def __init__(self):
        pass

    def update_with_collisions(self, points, box, max_distance):
        for item in points:
            pass
    
    def calc_next_position(self, point, initial_speed, acceleration, t=1):
        next_position = [point[d] + (initial_speed[d] * t + (acceleration[d] * t * t) / 2) for d in range(len(point))]
        return next_position

    def calculate_next_positions(self, points, box, t=1):
        new_points = []
        max_distance = 0

        for item in points:
            force_vector = item["force_vector"]
            current_speed = item["speed"]
            current_coords = item["coords"]
            mass = item["mass"]
            point_id = item["point_id"]

            # Calculate acceleration (a = F/m)
            acceleration = [force / mass for force in force_vector]

            # Update speed (v = v + a)
            new_speed = [current_speed[i] + (acceleration[i] * t) for i in range(len(current_coords))]

            next_position = self.calc_next_position(current_coords, current_speed, acceleration, t)
            # next_position = [current_coords[i] + new_speed[i] for i in range(len(current_coords))]
            reflected = False
            for d in range(len(next_position)):
                if next_position[d] < box[d][0]:
                    next_position[d] = box[d][0] + ((box[d][0] - next_position[d]) * 0.8)
                    new_speed[d] *= -1
                    reflected = True
                if next_position[d] > box[d][1]:
                    next_position[d] = box[d][1] - ((next_position[d] - box[d][1]) * 0.8)
                    new_speed[d] *= -1
                    reflected = True
            if reflected:
                for d in range(len(new_speed)):
                    new_speed[d] *= 0.8

            # Store the new position and speed for the next iteration
            new_points.append({
                "force_vector": [force_vector[i] for i in range(len(force_vector))],
                "coords": next_position,
                "mass": mass,
                "speed": new_speed,
                "point_id": point_id
            })

        return new_points

class MyApplication(tk.Frame):
    def __init__(self, root):
        self.pe = PhyEngine()

        self.num_particles = 2
        self.canvas_width = 600
        self.sidebar_width = 200
        self.canvas_height = 600

        self.g = [0, -0.9]

        super().__init__(root)
        self.root = root
        self.root.title("Main window")
        self.root.geometry(f"{self.canvas_width+self.sidebar_width}x{self.canvas_height}")
        self.points = [
            {
                "force_vector": self.g,  # x, y, z
                "coords": [100 + random.randint(0, 500), 500 + random.randint(0, 90)],
                "mass": 1,
                "speed": [3 - random.randint(0, 6), 0],
                "point_id": 1
            } for _ in range(self.num_particles)
        ]
        
        # Create a sidebar with naughty buttons
        sidebar_frame = tk.Frame(root, width=self.sidebar_width, bg='black')
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        button1 = tk.Button(sidebar_frame, text="Button 1", command=lambda: print("Button 1 was pressed"), bg='red')
        button1.pack()
        
        button2 = tk.Button(sidebar_frame, text="Button 2", command=lambda: print("Button 2 was pressed"), bg='blue')
        button2.pack()
        
        # Create a canvas for our erotic art
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)
        self.custom_main_loop()
    
    def draw_circle(self, x, y, color):
        if x > self.canvas_width or y > self.canvas_height:
            return
        y = self.canvas_height - y
        self.canvas.create_oval(x-SPHERE_RADIUS, y-SPHERE_RADIUS, x+SPHERE_RADIUS, y+SPHERE_RADIUS, fill=color)

    def draw_line(self, x1, y1, x2, y2, color):
        if x1 > self.canvas_width or y1 > self.canvas_height:
            return
        if x2 > self.canvas_width or y2 > self.canvas_height:
            return
        y1 = self.canvas_height - y1
        y2 = self.canvas_height - y2
        self.canvas.create_line(x1, y1, x2, y2, width=10, fill=color)
    
    def custom_main_loop(self):
        # Clear the canvas
        self.canvas.delete('all')

        # x = random.randint(0, self.canvas_width)
        # y = random.randint(0, self.canvas_height)
        for p in self.points:
            x = p["coords"][0]
            y = p["coords"][1]
            print(f"Draw a circle at {x = }, {y = }, speed = {p['speed'][1]}")
            self.draw_circle(x, y, "red")
        
        self.points = self.pe.calculate_next_positions(self.points, [[0, 600], [0, 600]])
        # for pt in self.points:
        #     if pt['coords'][1] < 0:
        #         pt['coords'][1] = -1 * pt['coords'][1]
        #         pt['speed'][1] = -1 * pt['speed'][1]
        #         for d in range(len(pt['speed'])):
        #             pt['speed'][d] *= 0.8

        # Wait for a second before the next circle
        self.root.after(20, self.custom_main_loop)


    def run(self):
        self.pack()
        # self.draw_line(10, 10, 30, 30, "blue")
        # self.draw_circle(10, 10, "red")
        # self.draw_circle(30, 30, "red")
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApplication(root)
    app.run()
