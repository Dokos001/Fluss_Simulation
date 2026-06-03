import math
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt


frames = 280
interval = 10
vol = []
vol2 = []
values = []

R = float(1.0)
deltaZ = float(0.5)
v0 = float(2.0)
sensorPos = float(1.5)
sensorPos2 = float(3.0)
tmax = float(10.0)

r1 = float(0.0)
r2 = float(0.0)
r3 = float(0.0)
r4 = float(0.0)
t = np.linspace(0+1/frames, tmax+1/frames, frames)
tstart = t[0]


filled = 0

def reg1(r2_t = r2):
    return math.pi * math.pow(R, 2) * (1-(math.pow(r2_t, 2)/math.pow(R, 2)))

def reg2(t, r2_t = r2, r1_t = r1):
    #return (math.pi * math.pow(R, 2) * ((math.pow(r2_t,2)/math.pow(R,2))-(math.pow(r1_t,2)/math.pow(R,2))))
    return (math.pi * math.pow(R, 2) * deltaZ)/(v0*t)

def reg3(r1_t = r1):
    return math.pi * math.pow(R, 2) * (math.pow(r1_t, 2)/math.pow(R, 2))

def reg4und0():
    return float(0.0)


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

x_values = np.linspace(-R , R, 100)
y1_values = parabola_P1(x_values,tstart)
y2_values = parabola_P2(x_values,tstart)

figGraph, (axGraph1, axGraph2) = plt.subplots(2,1 )
x_valuesGraph = t


axGraph1.set_xlim(-R - 1, R + 1)
axGraph1.set_ylim(v0 - 2, v0 + 2)

axGraph1.axvline(-R, color='red', linewidth=1, ls='--')  # restriction line
axGraph1.axvline( R, color='red', linewidth=1, ls='--')  # restriction line
para1 = axGraph1.plot(x_values, y1_values, label=f'y1 vorlaufende Parabel')
para2 = axGraph1.plot(x_values, y2_values, label=f'y2 nachlaufende Parabel')
pointR1 = axGraph1.plot(getR1(deltaZ/2,tstart), sensorPos, 'bo', label=f'R1')
pointR2 = axGraph1.plot(getR2(-deltaZ/2,tstart), sensorPos, 'ro', label=f'R2')
pointR3 = axGraph1.plot(getR1(deltaZ/2,tstart), sensorPos2, 'go', label=f'R3')
pointR4 = axGraph1.plot(getR2(-deltaZ/2,tstart), sensorPos2, 'mo', label=f'R4')
sensor = axGraph1.axhline(sensorPos, color='black', linewidth=1, ls='--', label= "Sensor Position")  # x-axis
sensor2 = axGraph1.axhline(sensorPos2, color='green', linewidth=1, ls='--', label= "Sensor2 Position")  # x-axis
#ax.fill_between(x_values, y1_values, y2_values, where=(y1_values > y2_values), color='red', alpha=0.5)
axGraph1.set_title("Parabeln")
axGraph1.set_xlabel('x')
axGraph1.set_ylabel('y')
axGraph1.grid()
axGraph1.legend()


line = axGraph2.plot(0,0, color='red', alpha=0.5)
line2= axGraph2.plot(0,0, color='blue', alpha=0.5)
axGraph2.set_xlim(0, tmax)#frames/interval)
axGraph2.set_ylim(-0.05, math.pi * math.pow(R, 2)/2)
axGraph2.set_title("Volumen über der Zeit")
axGraph2.set_xlabel('t in s')
axGraph2.set_ylabel('Volumen')
axGraph2.grid()

def update(frame):


    # Parabola-Drawning
    global filled
    global r1
    global r2
    global r3
    global r4

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

    if (v0 * ts + deltaZ / 2) >= sensorPos2:
        r3 = getR1(sensorPos2, ts)
        pointR3[0].set_xdata([r3])
    else:
        # If not reached
        #r3 = float(0.0)
        pointR3[0].set_xdata([0])

    if (v0 * ts - deltaZ / 2) >= sensorPos:
        r2 = getR2(sensorPos, ts)
        pointR2[0].set_xdata([r2])
    else:
        # If not reached
        #r2 = float(0.0)
        pointR2[0].set_xdata([0])

    if (v0 * ts - deltaZ / 2) >= sensorPos2:
        r4 = getR2(sensorPos2, ts)
        pointR4[0].set_xdata([r4])
    else:
        # If not reached
        #r4 = float(0.0)
        pointR4[0].set_xdata([0])


    # update the line plot:
    para1[0].set_ydata(y1_values)
    para2[0].set_ydata(y2_values)

    if filled != 0:
        filled.remove()

    filled = axGraph1.fill_between(x_values, y1_values, y2_values, where=(y1_values > y2_values), color='red', alpha=0.5)

    #VolumenBerechnung
    global vol
    global vol2
    global values
    values.append(frame)
    #Graph-Drawning
    x = x_valuesGraph[:len(values)]
    if len(x) > frames:
        x = x[-frames:]
    ts = t[frame]

    if len(vol) < frames:

        if r1>0 and r2==0:
            vol.append(reg3(r1))
        elif r1>r2 and r2!=0:
            vol.append(reg2(ts))
        elif r1==R and r2<R:
            vol.append(reg1(r2))
        else:
            vol.append(reg4und0())

    if len(vol2) < frames:

        if r3>0 and r4==0:
            vol2.append(reg3(r3))
        elif r3>r4 and r4!=0:
            vol2.append(reg2(ts, r2_t=r4, r1_t=r3))
        elif r3==R and r4<R:
            vol2.append(reg1(r4))
        else:
            vol2.append(reg4und0())

    if  frame == frames-1:
        x = []
        values = []
        with open('your_file.txt', 'w') as f:
            for val in vol:
                f.write(f"{val}\n")
            for val in vol2:
                f.write(f"{val}\n")
        vol = []
        vol2 = []
        values = []
        r1 = float(0.0)
        r2 = float(0.0)
        r3 = float(0.0)
        r4 = float(0.0)
    #print("r1: "+str(r1))
    #print("r2: "+str(r2))
    
    line[0].set_xdata(x)
    line[0].set_ydata(vol)

    line2[0].set_xdata(x)
    line2[0].set_ydata(vol2)

    return (para1, para2, pointR1, pointR2, pointR3, pointR4, line, line2)

ani = animation.FuncAnimation(fig=figGraph, func=update, frames=frames, interval=interval)
plt.show()
ani.save('parabolaAndVolume.gif', writer='Pillow', fps=30)