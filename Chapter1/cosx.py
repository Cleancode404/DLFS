import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 8, 0.1)
y1 = np.cos(x)
y2 = np.sin(x)
plt.plot(x, y1, label = "cos")
plt.plot(x, y2, label = "sin")
plt.xlabel("x")
plt.ylabel("y")
plt.title("cos(x) vs. sin(x)")
plt.legend()
plt.show()
