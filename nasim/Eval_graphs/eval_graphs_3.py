import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
Q1  = [41, 41, 42, 43, 42, 38, 46, 41, 45, 43, 38, 41, 36, 34, 34, 37, 44, 47, 33, 31, 37, 41, 36, 40]
Q2  = [27, 27, 27, 28, 27, 28, 23, 23, 20, 18, 20, 18, 24, 24, 20, 17, 20, 17, 24, 24, 20, 17, 20, 17]
Q3  = [9,   9,  7, 17,  7, 17, 15, 15, 14, 11, 14, 11, 10, 11, 23, 17, 14, 21, 10, 11, 23, 17, 14, 21]
x = [i for i in range(24)]

"""
ax = plt.subplot(1,1,1)
ax.bar(x , Q1 , width= 0.2 ,color = "black" )
ax.bar(x ,Q2, width= 0.2, color = "blue" )
ax.bar(x  ,Q3, width= 0.2, color = "red" )
ax.xlabel("X-Werte")
ax.ylabel("Y-Werte")
ax.show()
"""

X_axis = np.arange(len(x))

plt.bar(X_axis - 0.2, Q1, 0.2, label='Topology A', color ="black")
plt.bar(X_axis, Q2, 0.2, label='Topology B', color ="red")
plt.bar(X_axis + 0.2, Q3, 0.2, label='Topology C', color ="blue")
plt.xticks(X_axis, x)
plt.xlabel("p_environment number")
plt.ylabel("successfully solved ")
plt.title("Solved p_environment per Topology")
plt.legend()
plt.show()