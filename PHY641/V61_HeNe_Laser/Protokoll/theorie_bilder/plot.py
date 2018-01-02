import numpy as np
import matplotlib.pyplot as plt

if plt.rcParams["text.usetex"] is False:
    plt.rcParams["text.usetex"] = True

if plt.rcParams["text.latex.unicode"] is False:
    plt.rcParams["text.latex.unicode"] = True

if "siunitx" not in plt.rcParams["text.latex.preamble"]:
    plt.rcParams["text.latex.preamble"].append(r"\usepackage{siunitx}")

def g_1_g_2(L, r_1, r_2):
    return (1 - L/r_1) * (1 - L/r_2)
def L_nullstellen(r_1, r_2, a):
    return np.array([(r_1 + r_2)/2 - np.sqrt((r_1 + r_2)**2/4 + r_1 * r_2 *(a - 1)), (r_1 + r_2)/2 + np.sqrt((r_1 + r_2)**2/4 + r_1 * r_2 *(a - 1))])

L = np.linspace(0, 3)

r_2 = 1.4
plt.barh(0.5 , 3, 1, 0, color="green", alpha=0.2, edgecolor="gray", label="Stabiler Bereich")
r_1 = 1.0
plt.plot(L, g_1_g_2(L, r_1, r_2), 'b-', label = '$r_{1} = \SI{1000}{\milli\meter}$', linewidth = 1)
#plt.plot(L_nullstellen(r_1, r_2, 0), g_1_g_2(L_nullstellen(r_1, r_2, 0), r_1, r_2), 'kx')
#plt.plot(L_nullstellen(r_1, r_2, 1), g_1_g_2(L_nullstellen(r_1, r_2, 1), r_1, r_2), 'kx')


r_1 = 1e10
plt.plot(L, g_1_g_2(L, r_1, r_2), 'r-', label = 'Flacher Spiegel $(r_{1} = \infty)$', linewidth = 1)
#plt.plot(L_nullstellen(r_1, r_2, 0), g_1_g_2(L_nullstellen(r_1, r_2, 0), r_1, r_2), 'kx')
#plt.plot(L_nullstellen(r_1, r_2, 1), g_1_g_2(L_nullstellen(r_1, r_2, 1), r_1, r_2), 'kx')
plt.xlim(0, 3)
plt.legend()
plt.xlabel('$L/\si{\meter}$')
plt.ylabel('$g_1 g_2$')
plt.grid()
plt.savefig('g_1_g_2.pdf')
