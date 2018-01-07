import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
from scipy.stats import sem
import latex as l
#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS
a = [1.1, 2.2, 3.3]
b = [0.1, 0.2, 0.3]
c = [3, 4, 5]
d = [6, 7, 8] #EXAMPLE ARRAYS

f = unp.uarray(a, b) #EXAMPLE UARRAY

l.Latexdocument('latex_example.tex').tabular(
data = [c, d, f], #Data incl. unpuarray
header = ['\Delta Q / \giga\elementarycharge', 'T_1 / \micro\second', 'T_1 / \micro\second'],
places = [1, 1, (1.1, 1.1)],
caption = 'testcaption.',
label = 'testlabel')
##############################################

#r.makeresults()

def E_00(d,I_0,d_0, omega):
    return I_0*np.exp( -2*( (d-d_0)/omega )**2 )

def E_10(d, I_01, d_01, omega_1, I_02, d_02, omega_2):
    return I_01 * np.exp( -2 * ( (d-d_01) / omega_1 )**2 )+ I_02 * np.exp( -2 * ( ( d-d_02) / omega_2 )**2 )

def polarisation( phi, I_0, phi_0):
    return I_0*( np.sin(phi-phi_0) )**2

def lamb(spaltbreite, n, abstand_haupt, abstand_neben):
    bib={}
    bib['theta']=unp.arctan(abstand_neben/abstand_haupt)
    bib['lambda']=spaltbreite/n*unp.sin(bib['theta'])
    return bib

def quadrat(x,a,b,c):
    return a*x**2+b*x+c

def g(x,m,b):
    return m*x+b

def g_1g_2(L,r_1,r_2): # Resonatorlänge L, Krümmungsradius r
    return (1- L/r_1)*(1-L/r_2)

r_ebene=1e15
r_k1= 100
r_k2=140





###################
# Auswertung T_00 #
###################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Grundmode T_00', '\n', '-----------------------------------------------------------\n\n\n')
r_00, I_00=np.genfromtxt('T_00.txt',unpack=True)
params_I_00, cov_I_00 = curve_fit(E_00,r_00,I_00 )
error_I_00= np.sqrt(np.diag(cov_I_00))


amplitude_I_00 = ufloat(params_I_00[0],error_I_00[0])
verschiebung_I_00 = ufloat( params_I_00[1], error_I_00[1])
omega_I_00 = ufloat( params_I_00[2], error_I_00[2])

print('Amplitude: ', amplitude_I_00,'\n')
print('Verschiebung ', verschiebung_I_00,'\n')
print('Frequenz ', omega_I_00, '\n')

# Tabelle erstellen

l.Latexdocument('./table/T_00.tex').tabular(
data = [r_00[0:7], I_00[0:7], r_00[7:14], I_00[7:14], r_00[14:22], I_00[14: 22]], #Data incl. unpuarray
header = ['r / \milli\meter ', 'I_p / \micro\\ampere', 'r / \milli\meter ', 'I_p / \micro\\ampere', 'r / \milli\meter ', 'I_p / \micro\\ampere'],
places = [2, 2, 2, 2, 2, 2],
caption = 'Messwerte der $T_{00}$ Mode.',
label = 'T_00')


# Plotten
d_00=np.linspace(r_00[0]-1,r_00[-1]+1,10000)

plt.plot(r_00, I_00, '.', label='Messwerte')
plt.plot(d_00, E_00(d_00, params_I_00[0], params_I_00[1], params_I_00[2]), label='Fit')
plt.legend()
plt.xlabel(r'$r \, / \, mm $')
plt.ylabel(r'$ I_{\mathrm{p}}\, / \, \mu A$')
plt.grid()
plt.savefig('./plots/T_00.pdf')
#plt.show()




###################
# Auswertung T_10 #
###################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Grundmode T_10', '\n', '-----------------------------------------------------------\n\n\n')

r_10, I_10=np.genfromtxt('T_10_2.txt',unpack=True)



I_max_left=max(I_10[1:10])
I_max_right=max(I_10[10:-1])
null=np.where(I_10<0.04)[0][0]
r_left_max=r_10[np.where(I_10==I_max_left)[0][0]]
r_right_max=r_10[np.where(I_10==I_max_right)[0][0]]

omega_guess_left=8*sem( I_10[1:13])
omega_guess_right=8*sem( I_10[13:-1])
print(omega_guess_left)
print('Verwendete Startwerte für den T_10 Fit: \n')
print('I_1 ',I_max_left)
print('d_1 ', r_left_max)

print('\n I_2 ',I_max_right)
print('d_2 ', r_right_max)



params_I_10, cov_I_10 = curve_fit(E_10,r_10,I_10,p0=[I_max_left, r_left_max, 1, I_max_right, r_right_max,1])
error_I_10= np.sqrt(np.diag(cov_I_10))

amplitude_I_01 = ufloat(params_I_10[0],error_I_10[0])
verschiebung_d_01 = ufloat( params_I_10[1], error_I_10[1])
omega_1 = ufloat( params_I_10[2], error_I_10[2])

amplitude_I_02 = ufloat(params_I_10[3],error_I_10[3])
verschiebung_d_02 = ufloat( params_I_10[4], error_I_10[4])
omega_2 = ufloat( params_I_10[5], error_I_10[5])


print('Amplitude 1: ', amplitude_I_01,'\n')
print('Verschiebung 1: ', verschiebung_d_01,'\n')
print('Frequenz 1: ', omega_1, '\n')

print('Amplitude 2: ', amplitude_I_02,'\n')
print('Verschiebung 2: ', verschiebung_d_02,'\n')
print('Frequenz 2: ', omega_2, '\n')

# Tabelle erstellen
l.Latexdocument('./table/T_10.tex').tabular(
data = [r_10[0:13], I_10[0:13],r_10[13:], I_10[13:]], #Data incl. unpuarray
header = ['r / \milli\meter ', 'I_p / \micro\\ampere','r / \milli\meter ', 'I_p / \micro\\ampere'],
places = [2, 2, 2, 2],
caption = 'Messwerte der $T_{10}$  Mode.',
label = 'T_10')


# Plotten
d_10=np.linspace(r_10[0]-1,r_10[-1]+1,10000)
plt.clf()
plt.plot(r_10, I_10, '.', label='Messwerte')
plt.plot(d_10, E_10( d_10,params_I_10[0], params_I_10[1], params_I_10[2], params_I_10[3], params_I_10[4], params_I_10[5]), label='Fit')
plt.legend()
plt.xlabel(r'$r \, / \, mm $')
plt.ylabel(r'$ I_{\mathrm{p}}\, / \, \mu A$')
plt.grid()
#plt.show()
plt.savefig('./plots/T_10.pdf')




###########################
# Auswertung Polarisation #
###########################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Polarisation', '\n', '-----------------------------------------------------------\n\n\n')

winkel, I_pola= np.genfromtxt('polarisation.txt', unpack=True)
winkel_deg=winkel
winkel=np.deg2rad(winkel)
print(max(I_pola))

params_pola, cov_pola = curve_fit(polarisation,winkel,I_pola)
error_pola= np.sqrt(np.diag(cov_pola))

amplitude_pola=ufloat( params_pola[0], error_pola[0])
phase_pola= ufloat( params_pola[1], error_pola[1])

print(' Amplitude Pola: ', amplitude_pola,'\n')
print(' Phase: ', phase_pola, ufloat(np.rad2deg(noms(phase_pola)), np.rad2deg(stds(phase_pola))) , '\n')

# Erstellen der Tabelle
# Expand list to a length of 39
winkel_list=list(winkel)
winkel_list.append(0)
winkel_list.append(0)

winkel_deg_list=list(winkel_deg)
winkel_deg_list.append(0)
winkel_deg_list.append(0)

I_pola_list=list(I_pola)
I_pola_list.append(0)
I_pola_list.append(0)


l.Latexdocument('./table/polar.tex').tabular(
data = [winkel_deg_list[0:13], winkel_list[0:13], I_pola_list[0:13], winkel_deg_list[13:26], winkel_list[13:26], I_pola_list[13:26], winkel_deg_list[26:], winkel_list[26:], I_pola_list[26:]], #Data incl. unpuarray
header = ['\\varphi / \degree ','\\varphi / \\radian ', 'I_p / \milli\\ampere', '\\varphi / \degree ','\\varphi / \\radian ', 'I_p / \milli\\ampere', '\\varphi / \degree ','\\varphi / \\radian ', 'I_p / \milli\\ampere'],
places = [0,2, 2, 0,2, 2, 0,2, 2],
caption = 'Aufgenommene Werte bei der Polarisationsmessungs.',
label = 'pola')

# Plot

phi= np.linspace(winkel[0], winkel[-1],1000)
plt.clf()
plt.plot( winkel, I_pola, '.', label='Messdaten')
plt.plot( phi, polarisation(phi, params_pola[0], params_pola[1]), label='Fit')
plt.xlabel(r'$\varphi \, / \, \mathrm{rad}$')
plt.ylabel( r'$ I_{\mathrm{p}} \, / \, mA$')
plt.legend()
plt.grid()
plt.xticks([0,0.25*np.pi,0.5*np.pi,0.75*np.pi,np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi, 2*np.pi],['0','$\\frac{1}{4}\,\\pi$', '$\\frac{1}{2}\,\\pi$','$\\frac{3}{4}\,\\pi$' ,'$\\pi$','$\\frac{5}{4}\,\\pi$','$\\frac{3}{2}\, \\pi$','$\\frac{7}{4}\, \\pi$', '$2\, \\pi$'])
plt.savefig('./plots/pola.pdf')





##########################
# Auswertung Wellenlänge #
##########################
print('\n\n\n-----------------------------------------------------------\n','Wellenlängenbestimmung', '\n', '-----------------------------------------------------------\n\n\n')

n, abstand= np.genfromtxt('wellenlaenge.txt', unpack=True)
wellenlange=lamb( 1e-5, n, 83.0e-2,(abstand/2)*1e-2) # dictenoary with 'theta' and 'lambda'

# Tabelle erstellen
l.Latexdocument('./table/wellenlaenge.tex').tabular(
data = [n, abstand, wellenlange['theta'], wellenlange['lambda']*1e9], #Data incl. unpuarray
header = ['n {}', 'd / \centi\meter', '\\theta / \\radian', '\lambda / \\nano\meter'],
places = [0, 1, 1, 1],
caption = 'Aufgenommene Messwerte für die Wellenlängenbestimmung. Der Winkel $\theta$ und die Wellenlänge $\lambda$ werden mit den Gleichung \eqref{} und \eqref{} bestimmt. Der Abstand zum Schirm beträgt $l=\SI{83}{\centi\meter}$ und der Gitterabstand $a=\SI{1e-5}{\meter}$.',
label = 'wellenlänge')

mittelwert_wellenlaenge= ufloat( np.mean(wellenlange['lambda']),sem(wellenlange['lambda']) )
print('Gemittelte Wellenlange: ', mittelwert_wellenlaenge,'\n')

lambda_theo=632.8e-9
delta_lambda= (mittelwert_wellenlaenge/lambda_theo-1)*100
print('Abweichung zur Theorie: ', delta_lambda , '\n')






##################################################
# Auswertung Stabilitätsbediungung konkav/konkav #
##################################################

print('\n\n\n-----------------------------------------------------------\n','Stabilitätsbedinung Konkav/konkav', '\n', '-----------------------------------------------------------\n\n\n')

abstand_kk, I_kk= np.genfromtxt('stabilitaets_bed_kon_kon.txt', unpack=True)


c=g_1g_2(91,r_k2,r_k2 )

test= (I_kk*c)/(max(I_kk))
# Tabelle erstellen

l.Latexdocument('./table/konkon.tex').tabular(
data = [abstand_kk, I_kk,test], #Data incl. unpuarray
header = [' d / \centi\meter', ' I_p / \milli\\ampere',' (I_p\cdot c)/I_{p,max} '],
places = [0, 2,2],
caption = 'Aufgenommene Messwerte für die Untersuchung der Stabilitätsbedingung bei Konkav-Konkave Konfiguration. Der Umskalierungsfaktor hat den Wert $c=\\num{0.12}$.',
label = 'konkon')



#params_stab_kk, cov_stab_kk = curve_fit(quadrat,abstand_kk,I_kk)



params_stab_kk, cov_stab_kk = curve_fit(quadrat,abstand_kk,test)
error_stab_kk= np.sqrt(np.diag(cov_stab_kk))

print('c_kk: ',c)
print(' Quadrat: ', ufloat(params_stab_kk[0], error_stab_kk[0]),'\n')
print(' Linear: ', ufloat(params_stab_kk[1], error_stab_kk[1]),'\n')
print(' Konstante: ', ufloat(params_stab_kk[2],error_stab_kk[2]),'\n')

#Plot
x=np.linspace(abstand_kk[0]-1, abstand_kk[-1]+1,1000)

plt.clf()
plt.plot(abstand_kk,test,'.',label=r'$\mathrm{Messwerte}$')
plt.plot(x, quadrat(x,params_stab_kk[0],params_stab_kk[1], params_stab_kk[2]), label=r'$\mathrm{Fit}$')
plt.plot(x, g_1g_2(x,r_k2,r_k2),label=r'$\mathrm{Theorie:} r_{k2}, r_{K2}$')

plt.xlabel(r'$d / mm $')
plt.ylabel(r'$ g_1g_2$')
plt.grid()
plt.legend()
#plt.show()
plt.savefig('./plots/konkon.pdf')





##################################################
# Auswertung Stabilitätsbediungung konkav/flach #
##################################################

print('\n\n\n-----------------------------------------------------------\n','Stabilitätsbedinung Konkav/flach', '\n', '-----------------------------------------------------------\n\n\n')

abstand_kf, I_kf= np.genfromtxt('stabilitaets_bed_kon_flach.txt', unpack=True)

# tabelle erstellen


max_I_kf=max(I_kf)
c_kf=g_1g_2(75,r_ebene,r_k2 )

test=(I_kf*c_kf)/ max_I_kf

params_stab_kf, cov_stab_kf = curve_fit(g,abstand_kf,test)
error_stab_kf= np.sqrt(np.diag(cov_stab_kf))

l.Latexdocument('./table/konflach.tex').tabular(
data = [abstand_kf, I_kf, test], #Data incl. unpuarray
header = [' d / \centi\meter', ' I_p / \milli\\ampere',' (I_p\cdot c)/I_{p,max} '],
places = [0, 2, 2],
    caption = 'Aufgenommene Messwerte für die Untersuchung der Stabilitätsbedingung bei Konkav-Flache Konfiguration. Der Umskalierungsfaktor hat den Wert $c=\\num{0.46}$.',
label = 'konflach')




#params_stab_kf, cov_stab_kf = curve_fit(g,abstand_kf,I_kf)




print('c_kf: ',c_kf)
print(' linear: ', ufloat(params_stab_kf[0],error_stab_kf[0]),'\n')
print(' konstante: ', ufloat(params_stab_kf[1],error_stab_kf[1]),'\n')


#Plot
x=np.linspace(abstand_kf[0]-1, abstand_kf[-1]+1,1000)

plt.clf()
plt.plot(abstand_kf,test,'.',label=r'$\mathrm{Messwerte}$')
plt.plot(x, g(x,params_stab_kf[0],params_stab_kf[1]), label=r'$\mathrm{Fit}$')
plt.plot(x, g_1g_2(x,r_ebene,r_k2),label=r'$\mathrm{Theorie:} \, flach, r_{K1}$')
plt.xlabel(r'$d / mm $')
plt.ylabel(r'$ g_1g_2$')
plt.grid()
plt.legend()
#plt.show()
plt.savefig('./plots/konflach.pdf')
