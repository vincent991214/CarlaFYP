# https://medium.com/%E8%B3%87%E6%96%99%E9%9A%A8%E7%AD%86/python101-%E5%BB%A3%E6%9D%B1%E8%A9%B1python%E5%85%A5%E9%96%80-simple-linear-regression-ce98e62ff1fa
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('Chris_Samples/250ms_0ms_2.txt', sep=" ")
routine = pd.read_csv('WaypointInScenarioIII.txt', sep=" ")
data.dropna()

# print(data.head(n=10))
f = open('Inclinations.txt', "w")

# Making Inclination data
Inclination = []
for i in range(len(routine) - 1):
    x_diff = routine.Waypoint_x[i + 1] - routine.Waypoint_x[i]
    y_diff = routine.Waypoint_y[i + 1] - routine.Waypoint_y[i]
    if x_diff == 0:
        angle, slope = angle, slope
    else:
        slope = y_diff / x_diff
        angle = math.degrees(np.arctan(slope))
        if x_diff < 0:
            angle += 180
        elif x_diff > 0:
            angle += 360
    Inclination.append(angle)
    f.write(str(routine.Waypoint_x[i]) + " " + str(routine.Waypoint_y[i]))
    f.write(" " + str(routine.Waypoint_x[i + 1]) + " " + str(routine.Waypoint_y[i + 1]) + " " + str(slope) + " " + str(
        angle) + "\n")
# Inclination.append(Inclination[-1])
# print(Inclination)


# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(4)
figure.suptitle("(Command Delay:Display Delay)\n100ms:100ms\nParticipant A", fontsize=13, fontweight="bold")

# delta_mean_list = [np.mean(data.DeltaD)] * len(data.Time)
# delta_mean = round(delta_mean_list[0], 3)
personal_mean = [0.362] * len(data.Time)
print(max(data.DeltaD))
larger_elements = [element for element in data.DeltaD if element > 0.362]
number_of_elements = len(larger_elements)
# print(data.Time[0])
number_series = data.DeltaD
windows = number_series.rolling(50)
ma = windows.mean()
data['EWA'] = data['DeltaD'].ewm(span=50, adjust=False).mean()
axis[0].plot(data.Time, data.DeltaD, '.', c='r')
axis[0].plot(data.Time, personal_mean, '--', c='black', label="Personal Average")
axis[0].plot(data.Time, ma, c='c', label="MA")
axis[0].plot(data.Time, data['EWA'], c='orange', label="EMA")
axis[0].legend(["delta △", "# of samples over PA(0.362m) = " + str(number_of_elements), "50 MA", "50 EMA"])
axis[0].set_title("Delta Distance (△)", fontsize=10)

speed_mean = [np.mean(data.Speed)] * len(data.Time)
axis[1].plot(data.Time, data.Speed, c='g')
axis[1].plot(data.Time, speed_mean, '--', c='black')
axis[1].legend(["mean = " + str(round(speed_mean[0], 2))])
axis[1].set_title("Speed", fontsize=10)

axis[2].plot(data.Time, np.cumsum(data.DeltaD), c='b')
axis[2].set_title("Cumulative Error", fontsize=10)
axis[2].legend(["Cumulative Delta △ (Total = %im)"%(sum(data.DeltaD))])

axis[3].plot(data.Time, Inclination, '.', c='c')
axis[3].set_title("Inclination of Road ", fontsize=10)
axis[3].legend(["Inclination θ"])

plt.setp(axis[0], ylabel='Delta Distance (m)')
plt.setp(axis[1], ylabel='Speed (km/h)')
plt.setp(axis[2], ylabel='Delta Distance (m)')
plt.setp(axis[3], ylabel='Angle (degree)')

plt.xlabel("time (s)")

# set size of each subplot
plt.subplots_adjust(left=0.125,
                    bottom=0.05,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.35)
f.close
plt.show()
