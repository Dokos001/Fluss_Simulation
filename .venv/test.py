import math
from matplotlib import animation
from matplotlib.collections import PolyCollection
import numpy as np
import matplotlib.pyplot as plt


frames = 160
interval = 30
vol = []
values = []

R = float(1.0)
deltaZ = float(0.5)
v0 = float(1.0)
sensorPos = float(1.5)
tmax = float(30.0)

r1 = float(0.0)
r2 = float(0.0)
t = np.linspace(0+1/frames, tmax+1/frames, frames)
tstart = t[0]


filled = 0

def reg1():
    return math.pi * math.pow(R, 2) * (1-(math.pow(r2, 2)/math.pow(R, 2)))

def reg2(t):
    return (math.pi * math.pow(R, 2) * deltaZ)/(v0*t)

def reg3():
    return math.pi * math.pow(R, 2) * (math.pow(r1, 2)/math.pow(R, 2))

def reg4und0():
    return float(0.0)

# Define the parabola function
def parabola_P1(x,t_par):
    return -(((v0*t_par)*np.square(x))/math.pow(R,2))+(v0*t_par+deltaZ/2)
    #return -(((v0*t_par+deltaZ/2)*np.square(x))/math.pow(R,2))-(deltaZ/(2*R))*x+(v0*t_par+deltaZ/2)
def getR1(y,t_par):
    #return math.sqrt(math.pow(R,2)-((y*math.pow(R,2))/(v0*t_par+deltaZ/2)))
    return math.sqrt(((v0*t_par+deltaZ/2)*math.pow(R,2)-(y*math.pow(R,2)))/(v0*t_par))

def parabola_P2(x,t_par):
    return -(((v0*t_par)*np.square(x))/math.pow(R,2))+(v0*t_par-deltaZ/2)
    #return -(((v0*t_par-deltaZ/2)*np.square(x))/math.pow(R,2))+(deltaZ/(2*R))*x+(v0*t_par-deltaZ/2)
def getR2(y,t_par):
    #return math.sqrt(math.pow(R,2)-((y*math.pow(R,2))/(v0*t_par-deltaZ/2)))
    return math.sqrt(((v0*t_par-deltaZ/2)*math.pow(R,2)-(y*math.pow(R,2)))/(v0*t_par))

# Define the x values within a reasonable range
x_values = np.linspace(-R , R, 100)
y1_values = parabola_P1(x_values,tstart)
y2_values = parabola_P2(x_values,tstart)

# Plot the parabola
# plt.axhline(0, color='black', linewidth=0.5, ls='--')  # x-axis
# plt.axvline(0, color='black', linewidth=0.5, ls='--')  # y-axis

# plt.axvline(-R, color='red', linewidth=1, ls='--')  # restriction line
# plt.axvline( R, color='red', linewidth=1, ls='--')  # restriction line

# plt.xlim(-R - 1, R + 1)
# plt.ylim(v0*t - 2, v0*t + 2)
# plt.fill_between(x_values, y1_values, y2_values, where=(y1_values > y2_values), color='red', alpha=0.5)
# plt.title("Parabeln")
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.legend()
# plt.show()
figGraph, axGraph = plt.subplots()
x_valuesGraph = t
line = axGraph.plot(0,0, color='red', alpha=0.5)
axGraph.set_xlim(0, tmax)#frames/interval)
axGraph.set_ylim(-0.05, math.pi * math.pow(R, 2)/2)
axGraph.set_title("Volume over Time")
axGraph.set_xlabel('t')
axGraph.set_ylabel('volume')
axGraph.grid()



fig, ax = plt.subplots()




ax.set_xlim(-R - 1, R + 1)
ax.set_ylim(v0 - 2, v0 + 2)

ax.axvline(-R, color='red', linewidth=1, ls='--')  # restriction line
ax.axvline( R, color='red', linewidth=1, ls='--')  # restriction line
para1 = ax.plot(x_values, y1_values, label=f'y1 vorlaufende Parabel')
para2 = ax.plot(x_values, y2_values, label=f'y2 nachlaufende Parabel')
pointR1 = ax.plot(getR1(deltaZ/2,tstart), sensorPos, 'bo', label=f'R1')
pointR2 = ax.plot(getR2(-deltaZ/2,tstart), sensorPos, 'ro', label=f'R2')
sensor = ax.axhline(sensorPos, color='black', linewidth=1, ls='--', label= "Sensor Position")  # x-axis

#ax.fill_between(x_values, y1_values, y2_values, where=(y1_values > y2_values), color='red', alpha=0.5)
ax.set_title("Parabeln")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
ax.legend()


def update(frame):

    global filled
    global r1
    global r2
    # for each frame, update the data stored on each artist.
    ts = t[frame]
    y1_values = parabola_P1(x_values,ts)
    y2_values = parabola_P2(x_values,ts)
    
    if (v0 * ts + deltaZ / 2) >= sensorPos:
        r1 = getR1(sensorPos, ts)
        pointR1[0].set_xdata([r1])
    else:
        # If not reached
        #r1 = float(0.0)
        pointR1[0].set_xdata([0])

    if (v0 * ts - deltaZ / 2) >= sensorPos:
        r2 = getR2(sensorPos, ts)
        pointR2[0].set_xdata([r2])
    else:
        # If not reached
        #r2 = float(0.0)
        pointR2[0].set_xdata([0])


    # update the line plot:
    para1[0].set_ydata(y1_values)
    para2[0].set_ydata(y2_values)

    if filled != 0:
        filled.remove()

    filled = ax.fill_between(x_values, y1_values, y2_values, where=(y1_values > y2_values), color='red', alpha=0.5)

    if  frame == frames-1:
        r1 = float(0.0)
        r2 = float(0.0)


    return (para1, para2, pointR1, pointR2)


def update2(frame):

    global r1
    global r2
    global vol
    global values
    values.append(frame)
    #Graph-Drawning
    x = x_valuesGraph[:len(values)]
    ts = t[frame]

    if r1>0 and r2==0:
        vol.append(reg3())
    elif r1>r2 and r2!=0:
        vol.append(reg2(ts))
    elif r1==R and r2<R:
        vol.append(reg1())
    else:
        vol.append(reg4und0())

    if  frame == frames-1:
        x = []
        values = []
        with open('your_file.txt', 'w') as f:
            for val in vol:
                f.write(f"{val}\n")
        vol = []
        values = []
        r1 = float(0.0)
        r2 = float(0.0)
    #print("r1: "+str(r1))
    #print("r2: "+str(r2))
    
    line[0].set_xdata(x)
    line[0].set_ydata(vol)

    return (line)

ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval)
ani2 = animation.FuncAnimation(fig=figGraph, func=update2, frames=frames, interval=interval)
plt.show()
ani.save('parabola.gif', writer='Pillow', fps=30)
ani2.save('volume.gif', writer='Pillow', fps=30)
    

     