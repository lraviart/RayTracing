import numpy as np
import matplotlib.pyplot as plt



radar = np.array([5, 0])
target = np.array([10, 40])
wall = np.array([[0, -5], [0, 50]])

def find_reflection_point(radar, target):
    x1, y1 = radar
    x2, y2 = target

    theta = np.arctan2(y2-y1, x1+x2)
    y = x1*np.tan(theta)
    return np.array([0, y])
    
reflection_point = find_reflection_point(radar, target)


# First-order reflection
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(radar[0], radar[1], color='black', label='Radar')
ax.scatter(target[0], target[1], color='tab:blue', label='Target', marker='s')
ax.plot(wall[:, 0], wall[:, 1], color='black', label='Wall')

arrow_style = dict(arrowstyle='->', color='tab:green', linestyle='-', linewidth=1.5)

ax.annotate('', xy=reflection_point, xytext=radar, arrowprops=arrow_style)
ax.annotate('', xy=target, xytext=reflection_point, arrowprops=arrow_style)
ax.annotate('', xy=radar, xytext=target, arrowprops=arrow_style)

plt.xlim(-15, 15)
plt.ylim(-10, 60)
plt.legend()
plt.show()


# Second-order reflection
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(radar[0], radar[1], color='black', label='Radar')
ax.scatter(target[0], target[1], color='tab:blue', label='Target', marker='s')
ax.plot(wall[:, 0], wall[:, 1], color='black', label='Wall')


arrow_style = dict(arrowstyle='<->', color='tab:red', linestyle='-', linewidth=1.5)

ax.annotate('', xy=reflection_point, xytext=radar, arrowprops=arrow_style)
ax.annotate('', xy=target, xytext=reflection_point, arrowprops=arrow_style)

plt.xlim(-15, 15)
plt.ylim(-10, 60)
plt.legend()
plt.show()