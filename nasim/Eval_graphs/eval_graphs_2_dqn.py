import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd
# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.interpolate
import pandas as pd

# helpfunction to smooth

"""
def smooth(x,y):
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(x.min(), x.max(), 300)
    spl = scipy.interpolate.make_interp_spline(x, y, k=3)
    smooth_v= spl(xnew)
    return [xnew,smooth_v]
"""

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(1, 3)


figure.set_size_inches(18.5, 4, forward=True)

data1 = pd.read_csv("./data_2/long_best_dqn_1.csv")
data1_s = data1.ewm(alpha=(1 - 0.85)).mean()
data1 = data1.ewm(alpha=(1 - 0)).mean()

data2 = pd.read_csv("./data_2/long_best_dqn_2.csv")
data2_s = data2.ewm(alpha=(1 - 0.85)).mean()
data2 = data2.ewm(alpha=(1 - 0)).mean()

data3 = pd.read_csv("./data_2/long_best_dqn_3.csv")
data3_s = data3.ewm(alpha=(1 - 0.85)).mean()
data3 = data3.ewm(alpha=(1 - 0)).mean()


axis[0].plot(data1_s['Step'], data1_s['Value'], color='b' , label="0.001;64;100000;0.001;10000;0.9;[128, 128];1000")
axis[0].plot(data1['Step'], data1['Value'], color='b' ,alpha=0.2)
axis[0].set_title("Topology A")
axis[0].legend(loc="lower center")

axis[1].plot(data2_s['Step'], data2_s['Value'], color='b' , label="0.5;64;10000;0.3;500;0.9;[64, 64];1000")
axis[1].plot(data2['Step'], data2['Value'], color='b' ,alpha=0.2)
axis[1].set_title("Topology B")
axis[1].legend(loc="lower center")

axis[2].plot(data3_s['Step'], data3_s['Value'], color='b' , label="0.001;64;10000;0.001;10000;0.9;[128, 128];1000")
axis[2].plot(data3['Step'], data3['Value'], color='b' ,alpha=0.2)
axis[2].set_title("Topology C")
axis[2].legend(loc="lower center")

figure.text(0.5, 0.02, 'steps', ha='center', va='center')
figure.text(0.09, 0.5, 'reward', ha='center', va='center', rotation='vertical')

plt.show()