
import numpy as np
import matplotlib.pyplot as plt

def z_function(x, y):
    return np.sin(x)*y

def calculate_gradient(x,y):
    return np.cos(x)*y, np.sin(x)

x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)

X,Y = np.meshgrid(x,y)

Z = z_function(X,Y)

ax = plt.subplot(projection='3d')

ax.plot_surface(X,Y,Z, cmap="viridis")
plt.show()

current_pos = (0.3, 0.5, z_function(0.3, 0.5))

learning_rate = 0.01
for i in range(1000):
    x_derivative, y_derivative = calculate_gradient(current_pos[0], current_pos[1])
    x_new = current_pos[0]-learning_rate*x_derivative
    y_new = current_pos[1]-learning_rate*y_derivative
    current_pos = (x_new, y_new, z_function(x_new, y_new))

print(current_pos)
