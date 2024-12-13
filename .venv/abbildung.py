import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
# Erstellen Sie das Layout
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(6, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 1, 1, 1], hspace=1, wspace=0.2)

# Daten-Dummy-Plot
ax1 = fig.add_subplot(gs[5, 1])
ax1.plot(range(100), [i % 5 for i in range(100)])


# Erstellen der Subplots

# 1. Subplot für die 1D-Convolution (mit zufälligen Daten als Platzhalter)
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(np.random.randn(100), label="1D Convolution")
ax1.set_title('1D Convolution (4,1)')

# 2. Subplot für Dropout
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(np.random.randn(100), label="Dropout", color='orange')
ax2.set_title('Dropout (0.2)')

# 3. Subplot für Dilated 1D-Convolution
ax3 = fig.add_subplot(gs[2, 1])
ax3.plot(np.random.randn(100), label="Dilated 1D Convolution")
ax3.set_title('Dilated 1D Convolution (128, 7)')

# 4. Subplot für Dense Layer
ax4 = fig.add_subplot(gs[3, 1])
ax4.plot(np.random.randn(100), label="Dense Layer", color='green')
ax4.set_title('Dense (128)')

# 5. Subplot für Max Pooling
ax5 = fig.add_subplot(gs[4, 1])
ax5.plot(np.random.randn(100), label="Max Pooling", color='red')
ax5.set_title('Max Pool (2, 2)')

# 6. Subplot für das zweite 1D-Convolution
ax6 = fig.add_subplot(gs[5, 1])
ax6.plot(np.random.randn(100), label="Second 1D Convolution", color='blue')
ax6.set_title('1D Convolution (i, 3)')

# Fügen Sie die Beschriftungen und Titel hinzu
fig.suptitle('SleepPPG-Net Architecture', fontsize=16)

plt.show()