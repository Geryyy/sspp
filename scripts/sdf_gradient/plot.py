import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import BSplines as bs  # Your B-spline module

def plot_spline_with_gradients(knot_pts, ctrl_pts, gradient_list, k=2, 
                             num_samples=100, arrow_scale=1.0, 
                             ax=None, show_control_points=True, 
                             show_spline=True, show_gradients=True):
    """
    Plots a B-spline with control points and gradient arrows.
    
    Parameters:
    - knot_pts: Knot vector for the B-spline
    - ctrl_pts: Control points array (Nx3)
    - gradient_list: List of gradient vectors for each control point (Nx3)
    - k: B-spline degree (default: 2)
    - num_samples: Number of points to sample along the spline
    - arrow_scale: Scale factor for gradient arrows
    - ax: Matplotlib 3D axis (creates new if None)
    - show_control_points: Whether to show control points
    - show_spline: Whether to show the spline curve
    - show_gradients: Whether to show gradient arrows
    
    Returns:
    - fig, ax: Matplotlib figure and axis objects
    """
    
    if ax is None:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    
    # Sample points along the spline
    if show_spline:
        u_samples = np.linspace(0, 1, num_samples)
        spline_points = np.array([bs.bspline(u, knot_pts, ctrl_pts, k) for u in u_samples])
        
        # Plot the spline curve
        ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], 
                'b-', linewidth=3, label='B-spline trajectory', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(*spline_points[0], color='green', s=100, label='Start', marker='o')
        ax.scatter(*spline_points[-1], color='red', s=100, label='End', marker='s')
    
    # Plot control points
    if show_control_points:
        ax.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2], 
                  color='orange', s=80, label='Control Points', marker='^', alpha=0.9)
        
        # Connect control points with dashed lines
        ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2], 
                'orange', linestyle='--', linewidth=1, alpha=0.5, label='Control Polygon')
        
        # Add control point labels
        for i, pt in enumerate(ctrl_pts):
            ax.text(pt[0], pt[1], pt[2] + 0.05, f'CP{i}', fontsize=10, ha='center')
    
    # Plot gradient arrows
    if show_gradients and gradient_list is not None:
        for i, (ctrl_pt, grad) in enumerate(zip(ctrl_pts, gradient_list)):
            if np.linalg.norm(grad) > 1e-8:  # Only plot non-zero gradients
                # Scale the gradient arrow
                grad /= np.linalg.norm(grad)  # Normalize
                arrow_end = ctrl_pt + arrow_scale * grad
                
                # Plot gradient arrow
                ax.quiver(ctrl_pt[0], ctrl_pt[1], ctrl_pt[2],
                         arrow_scale * grad[0], arrow_scale * grad[1], arrow_scale * grad[2],
                         color='red', alpha=0.8, linewidth=2,
                         label='Collision Gradients' if i == 0 else "")
                
                # Add gradient magnitude as text
                grad_magnitude = np.linalg.norm(grad)
                ax.text(arrow_end[0], arrow_end[1], arrow_end[2], 
                       f'{grad_magnitude:.3f}', fontsize=8, color='red')
    
    # Set axis properties
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    # Get the range of each axis
    all_points = ctrl_pts
    if show_spline:
        all_points = np.vstack([ctrl_pts, spline_points])
    
    max_range = np.array([all_points[:, i].max() - all_points[:, i].min() for i in range(3)]).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.title('B-spline Trajectory with Collision Gradients', fontsize=14, pad=20)
    plt.tight_layout()
    
    return fig, ax


# Example usage function that you can call from your main code:
def visualize_trajectory_optimization(knot_pts, ctrl_pts, gradient_list, k=2):
    """
    Convenience function to visualize your trajectory optimization.
    Call this after computing your collision gradients.
    """
    
    fig, ax = plot_spline_with_gradients(knot_pts, ctrl_pts, gradient_list, k=k, 
                                       arrow_scale=0.1, num_samples=50)
    
    # Add some styling
    ax.view_init(elev=20, azim=45)  # Set a nice viewing angle
    
    # Print gradient information
    print("\nGradient Information:")
    print("-" * 40)
    for i, grad in enumerate(gradient_list):
        grad_norm = np.linalg.norm(grad)
        print(f"Control Point {i}: ||grad|| = {grad_norm:.6f}")
        if grad_norm > 1e-8:
            grad_unit = grad / grad_norm
            print(f"  Direction: [{grad_unit[0]:.3f}, {grad_unit[1]:.3f}, {grad_unit[2]:.3f}]")
    
    plt.show()
    return fig, ax


def plot_triad(transformations, labels, ax=None, scale=0.1, block=False):
    """
    Plots triads (x, y, z axes) for a list of homogeneous transformations with labels.
    
    Parameters:
    - transformations: List of homogeneous transformation matrices (4x4 numpy arrays).
    - labels: List of labels corresponding to the transformations.
    - ax: (optional) Matplotlib 3D Axes object. Creates a new one if not provided.
    - scale: Scale of the triad axes (default 0.1).
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot each transformation
    for i, (T, label) in enumerate(zip(transformations, labels)):
        # Origin of the frame
        origin = T[:3, 3]
        # Extract rotation matrix (upper 3x3 block)
        R = T[:3, :3]
        
        # Define axes vectors
        x_axis = origin + scale * R[:, 0]  # X-axis (red)
        y_axis = origin + scale * R[:, 1]  # Y-axis (green)
        z_axis = origin + scale * R[:, 2]  # Z-axis (blue)
        
        # Plot axes
        ax.quiver(*origin, *(x_axis - origin), color='r', label='X' if i == 0 else "", linewidth=1)
        ax.quiver(*origin, *(y_axis - origin), color='g', label='Y' if i == 0 else "", linewidth=1)
        ax.quiver(*origin, *(z_axis - origin), color='b', label='Z' if i == 0 else "", linewidth=1)
        
        # Add label at the origin
        ax.text(*origin, label, color='k', fontsize=10)
    
    # Set axis limits and labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot
    ax.legend()

    if block:
        plt.show(block=block)
        return None
    else:
        plt.show(block=False)
        return fig, ax
