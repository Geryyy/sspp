import numpy as np

class CubicPath:
    def __init__(self):
        """Initialize an empty CubicPath object."""
        self.coefficients = None  # Coefficients for x(t) and y(t) if path is 2D/3D
    
    def plan(self, start, via, end):
        """
        Plans a cubic spline path through the start, via, and end points.
        Args:
            start (array-like): The starting point of the path (e.g., [x0, y0] or [x0, y0, z0]).
            via (array-like): The intermediate point the path should pass through.
            end (array-like): The ending point of the path.
        """
        start = np.array(start)
        via = np.array(via)
        end = np.array(end)
        
        # Dimensionality of the problem (2D, 3D, etc.)
        dimensions = start.shape[0]
                
        self.a = 2*(end + 3*start - 4*via)
        self.b = 4*(via-start-self.a/8)
        self.c = 0
        self.d = start

        return True
    
    
    def evaluate(self, u):
        """
        Evaluates the cubic path at parameter u (0 <= u <= 1).
        Args:
            u (float): The parameter along the path (0 <= u <= 1)
        Returns:
            position (np.array): The position on the path at parameter u.
        """
        u = np.clip(u, 0, 1)  # Ensure u stays within [0, 1]
        
        position = self.a * u**3 + self.b * u**2 + self.c * u + self.d
            
        return position
    
    
    def evaluate_with_derivatives(self, u):
        """
        Evaluates the cubic path at parameter u (0 <= u <= 1).
        Args:
            u (float): The parameter along the path (0 <= u <= 1)
        Returns:
            position (np.array): The position on the path at parameter u.
        """
        u = np.clip(u, 0, 1)  # Ensure u stays within [0, 1]

        position = self.a * u**3 + self.b * u**2 + self.c * u + self.d
        velocity = 3 * self.a * u**2 + 2 * self.b * u + self.c
        acceleration = 6 * self.a * u + 2 * self.b
        
        return position, velocity, acceleration
    




def main():
    import matplotlib.pyplot as plt
    import numpy as np

    # Initialize the cubic path
    path = CubicPath()

    # Plan the path between three points (start, via, end) in 2D
    start = [0, 0]      # (x0, y0)
    via = [0.5, 1]      # (x1, y1) intermediate point
    end = [1, 0]        # (x2, y2)
    path.plan(start, via, end)

    # Evaluate the path for different u values
    u_values = np.linspace(0, 1, 100)
    positions = np.array([path.evaluate(u) for u in u_values])

    # Plot the path
    plt.plot(positions[:, 0], positions[:, 1], label="Path", color="blue")
    plt.scatter([start[0], via[0], end[0]], [start[1], via[1], end[1]], color="red", label="Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Cubic Spline Path")
    plt.legend()
    plt.grid()
    plt.show()


def main_3D():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
    import numpy as np

    # Initialize the cubic path
    path = CubicPath()

    # Plan a 3D path between start, via, and end points
    start = [0, 0, 0]       # (x0, y0, z0)
    via = [1, 2, 3]         # (x1, y1, z1) intermediate point
    end = [2, 0, 1]         # (x2, y2, z2)
    path.plan(start, via, end)

    # Evaluate the path, velocity, and acceleration for different u values
    u_values = np.linspace(0, 1, 100)
    positions = np.array([path.evaluate(u) for u in u_values])
    velocities, accelerations = [], []

    for u in u_values:
        pos, vel, acc = path.evaluate_with_derivatives(u)
        velocities.append(vel)
        accelerations.append(acc)

    velocities = np.array(velocities)
    accelerations = np.array(accelerations)

    # Plot the 3D path
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Path", color="blue")
    ax.scatter([start[0], via[0], end[0]], [start[1], via[1], end[1]], [start[2], via[2], end[2]], color="red", label="Control Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Cubic Spline Path")
    ax.legend()
    ax.grid()

    # Plot the velocities and accelerations over u
    ax2 = fig.add_subplot(222)
    ax2.plot(u_values, velocities[:, 0], label="vx", color='r')
    ax2.plot(u_values, velocities[:, 1], label="vy", color='g')
    ax2.plot(u_values, velocities[:, 2], label="vz", color='b')
    ax2.set_xlabel("u")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Velocity Components")
    ax2.legend()
    ax2.grid()

    ax3 = fig.add_subplot(224)
    ax3.plot(u_values, accelerations[:, 0], label="ax", color='r')
    ax3.plot(u_values, accelerations[:, 1], label="ay", color='g')
    ax3.plot(u_values, accelerations[:, 2], label="az", color='b')
    ax3.set_xlabel("u")
    ax3.set_ylabel("Acceleration")
    ax3.set_title("Acceleration Components")
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

    main_3D()