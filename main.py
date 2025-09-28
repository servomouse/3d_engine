import tkinter as tk
import random
import math

SPHERE_RADIUS = 5


class PhyEngine:
    def __init__(self):
        pass

    def get_distance(self, point1, point2):
        c_sum = 0
        for d in range(len(point1)):
            c_sum += (point1[d] - point2[d])**2
        return math.sqrt(c_sum)

    def update_with_collisions(self, points, box, max_distance):
        # points: [[current_coords, next_position, new_speed]...]
        new_points = []
        num_steps = math.ceil(max_distance / SPHERE_RADIUS)
        for p in points:
            reflected = False
            new_position = [p[1][i] for i in range(len(p[1]))]
            new_speed = [p[2][i] for i in range(len(p[2]))]
            for d in range(len(new_position)):
                if new_position[d] < box[d][0]:
                    new_position[d] = box[d][0] + ((box[d][0] - new_position[d]) * 0.8)
                    new_speed[d] *= -1
                    reflected = True
                if new_position[d] > box[d][1]:
                    new_position[d] = box[d][1] - ((new_position[d] - box[d][1]) * 0.8)
                    new_speed[d] *= -1
                    reflected = True
            if reflected:
                for d in range(len(new_speed)):
                    new_speed[d] *= 0.8
            new_points.append([new_position, new_speed])
        return new_points
    
    def calc_next_position(self, point, initial_speed, acceleration, t=1):
        next_position = [point[d] + (initial_speed[d] * t + (acceleration[d] * t * t) / 2) for d in range(len(point))]
        return next_position

    def calculate_next_positions(self, curr_points, box, t=1):
        new_points = []
        max_distance = 0
        points = []

        for item in curr_points:
            force_vector = item["force_vector"]
            current_speed = item["speed"]
            current_coords = item["coords"]
            mass = item["mass"]
            point_id = item["point_id"]

            acceleration = [force / mass for force in force_vector]

            new_speed = [current_speed[i] + (acceleration[i] * t) for i in range(len(current_coords))]

            next_position = self.calc_next_position(current_coords, current_speed, acceleration, t)
            # next_position = [current_coords[i] + new_speed[i] for i in range(len(current_coords))]
            points.append([current_coords, next_position, new_speed])

            distance = self.get_distance(current_coords, next_position)
            if distance > max_distance:
                max_distance = distance

        next_step = self.update_with_collisions(points, box, max_distance)
        for i in range(len(curr_points)):
            force_vector = curr_points[i]["force_vector"]
            current_speed = curr_points[i]["speed"]
            current_coords = curr_points[i]["coords"]
            mass = curr_points[i]["mass"]
            point_id = curr_points[i]["point_id"]

            next_position = next_step[i][0]
            new_speed = next_step[i][1]

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
