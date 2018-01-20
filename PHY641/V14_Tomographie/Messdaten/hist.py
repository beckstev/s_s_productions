import numpy as np
import matplotlib.pyplot as plt

y = np.genfromtxt('Originaldaten/Blei/plot.txt', unpack = True)
x = np.linspace(0, 512, 512)
#plt.plot(x[30:80], y[30:80], 'rx')
plt.bar(x[20:100], y[20: 100], width = 1, color = 'b', edgecolor='k', align = 'center', label = '$\gamma$-Spektrum')
plt.bar(x[54:60], y[54: 60], width = 1, color = 'r', edgecolor='k', align = 'center', label = 'Integrationsbereich')
plt.legend()
plt.xlim(30, 100)
plt.savefig('hist.pdf')
#plt.show()
