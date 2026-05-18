import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  # This gives the LaTeX look to math text
    "axes.edgecolor": "black", # Black bounding box
    "axes.linewidth": 0.8
})


########################## 
### Stem Plot Template ###
##########################


# Dummy data
x = np.arange(0, 10.5, 0.5)
y = 0.1 * x

# Create the figure with a wide aspect ratio
fig, ax = plt.subplots(figsize=(8, 3))

# Plot the stem graph
# linefmt: stem line style, markerfmt: marker style, basefmt: baseline style
markerline, stemlines, baseline = ax.stem(
    x, y, 
    linefmt="#224DB2",  # Color of the stem lines
    markerfmt='D',      # Placeholder, we overwrite it below
    basefmt='k-'        # Black solid baseline
)

# Force the markers to be solid red circles to match the line color
plt.setp(markerline, marker='o', color="#224DB2", markersize=6, clip_on=False, zorder=10)

# Format axes limits and ticks
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.0) # Matches the bounding box exactly
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 1.1, 0.2))

# Add the light grid lines
ax.grid(True, linestyle='-', linewidth=0.5, color='#D3D3D3')

# Add labels using raw strings (r'') for LaTeX math mode
ax.set_xlabel(r'$t/T$', fontsize=12)
ax.set_ylabel(r'$\mathcal{P}_h[n] / \max\{\mathcal{P}_h[n]\}$', fontsize=12)

# Ensure everything fits nicely
plt.tight_layout()

# Display the plot
plt.show()