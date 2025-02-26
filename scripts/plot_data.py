import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Read CSV file
data = pd.read_csv("/home/geraldebmer/repos/robocrane/sspp/cmake-build-debug/tsp.csv")

# Extract columns
x = data.iloc[:, 0].values    # First column is X (used for color)
y1 = data.iloc[:, 1].values   # Second column -> X-axis in 3D plot
y2 = data.iloc[:, 2].values   # Third column  -> Y-axis in 3D plot
y3 = data.iloc[:, 3].values   # Fourth column -> Z-axis in 3D plot

# Normalize x for color mapping
norm = mcolors.Normalize(vmin=min(x), vmax=max(x))
cmap = cm.viridis  # Choose colormap
colors = cmap(norm(x))  # Get color for each point

# Create 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot 3D line with gradient color
for i in range(len(x) - 1):
    ax.plot(y1[i:i+2], y2[i:i+2], y3[i:i+2], color=colors[i], linewidth=2)

# Labels & Title
ax.set_xlabel("Y1 (X in 3D)")
ax.set_ylabel("Y2 (Y in 3D)")
ax.set_zlabel("Y3 (Z in 3D)")
ax.set_title("3D Line Plot with X-Color Gradient")

# Add color bar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("X Value (Color Gradient)")

plt.show()
