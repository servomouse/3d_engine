import tkinter as tk
import random
import math

SPHERE_RADIUS = 5


class PhyEngine:
    def __init__(self):
        pass

    def dot(self, u, v):
        return sum(ui * vi for ui, vi in zip(u, v))

    def magnitude(self, v):
        return math.sqrt(self.dot(v, v))

    def scalar_multiply(self, scalar, v):
        return [scalar * vi for vi in v]

    def vector_add(self, u, v):
        return [ui + vi for ui, vi in zip(u, v)]

    def vector_subtract(self, u, v):
        return [ui - vi for ui, vi in zip(u, v)]

    def time_and_velocity_at_point(self, r0, v0, a, r_target, tol=1e-6):
        """
        Parameters:
            r0 - initial coordinates
            v0 - initial speed
            a - acceleration
            r_target - target coordinates
        Returns:
            time when r_target is reached
            velocity at r_target
        """
        # Displacement vector
        d = self.vector_subtract(r_target, r0)

        # Direction of motion (unit vector)
        d_mag = self.magnitude(d)
        if d_mag == 0:
            raise ValueError("Target point is the same as initial point.")
        direction = [di / d_mag for di in d]

        # Project vectors onto direction
        s = self.dot(d, direction)
        v_proj = self.dot(v0, direction)
        a_proj = self.dot(a, direction)

        # Solve quadratic: 0.5 * a_proj * t^2 + v_proj * t - s = 0
        A = 0.5 * a_proj
        B = v_proj
        C = -s

        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            raise ValueError("No real solution: target point not on trajectory.")

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-B + sqrt_disc) / (2*A) if A != 0 else -C / B
        t2 = (-B - sqrt_disc) / (2*A) if A != 0 else None

        # Choose the smallest positive time
        valid_times = [t for t in [t1, t2] if t is not None and t >= tol]
        if not valid_times:
            raise ValueError("No valid time found for reaching the target point.")

        t = min(valid_times)
        v = self.vector_add(v0, self.scalar_multiply(t, a))

        return t, v

    def get_distance(self, point1, point2):
        c_sum = 0
        for d in range(len(point1)):
            c_sum += (point1[d] - point2[d])**2
        return math.sqrt(c_sum)
    
    def get_coords_and_velocity_at_time(self, r, v, a, t=1):
        """
        Calculate the final position and velocity of a point given initial position, velocity, acceleration, and time.

        Parameters:
            r - initial_position (list): Initial coordinates [x, y, z].
            v - velocity (list): Velocity vector [vx, vy, vz].
            a - acceleration (list): Acceleration vector [ax, ay, az].
            t - time (float): Time duration.

        Returns:
            dictionary: 
                "coords" - final coordinates [x', y', z']
                "velocity" - final velocity [vx', vy', vz'] after the given time
                "time" - time when the point reaches the target coordinates
        """
        final_position = [r[i] + v[i] * t + 0.5 * a[i] * t ** 2 for i in range(3)]
        final_velocity = [v[i] + a[i] * t for i in range(3)]
        return {
                "coords": final_position,
                "velocity": final_velocity,
                "time": t
            }
    
    def get_intersection_point(self, seg_start, seg_end, boundary_d, boundary_val):
        """
        Calculates an intersection point between a line segment and a boundary
        """
        deltas = [seg_end[i] - seg_start[i] for i in range(len(seg_start))]
        coords = []
        for d in range(len(seg_start)):
            if d == boundary_d:
                coords.append(boundary_val)
            else:
                val = deltas[d] * (boundary_val - seg_start[boundary_d])
                val /= deltas[boundary_d]
                val -= seg_start[d]
                coords.append(val)
        return coords
    
    def get_init_trajectory(self, point, velocity, acceleration, t=1):
        return [
            {
                "coords": [i for i in point],
                "velocity": [i for i in velocity],
                "time": 0
            },
            self.get_coords_and_velocity_at_time(point, velocity, acceleration, t)
        ]

    def update_trajectories_with_boundaries(self, trajectories, box):
        for t in trajectories:
            for i in range(len(t)-1):
                seg_start = t[i]
                seg_end = t[i+1]
                intersections = [0 for _ in range(len(box))]
                intersects = False
                for d in len(box):
                    if seg_end[d] < box[d][0]:
                        intersections[d] = self.get_intersection_point(seg_start, seg_end, box[d][0])
                        intersects = True
                    elif seg_end[d] > box[d][1]:
                        intersections[d] = self.get_intersection_point(seg_start, seg_end, box[d][1])
                        intersects = True


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
        trajectories = []

        for item in curr_points:
            force_vector = item["force_vector"]
            current_speed = item["speed"]
            current_coords = item["coords"]
            mass = item["mass"]
            point_id = item["point_id"]

            acceleration = [force / mass for force in force_vector]

            new_speed = [current_speed[i] + (acceleration[i] * t) for i in range(len(current_coords))]

            trajectories.append(self.get_init_trajectory(current_coords, current_speed, acceleration, t))

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
