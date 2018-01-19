import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math as m
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
from uncertainties.unumpy import log#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
r = l.Latexdocument('results.tex')



#Matrix zur Beschreibung der verwendeten Projektionen
s = m.sqrt(2)
A = np.matrix([[1, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 1],
               [0, s, 0, s, 0, 0, 0, 0, 0],
               [0, 0, s, 0, s, 0, s, 0, 0],
               [0, 0, 0, 0, 0, s, 0, s, 0],
               [0, 0, 0, s, 0, 0, 0, s, 0],
               [s, 0, 0, 0, s, 0, 0, 0, s],
               [0, s, 0, 0, 0, s, 0, 0, 0],
               [1, 0, 0, 1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 1]])
#print(np.linalg.det((A.T * A).I))


def cov_y(y):
    V = np.zeros((len(y), len(y)))
    np.fill_diagonal(V, y**2)
    return np.matrix(V)

def mu(V, y):
    W = V.I
    y = np.matrix(y).T
    mu = ((A.T * (W * A)).I * (A.T * (W * y))).T
    return mu

def cov_mu(V):
    W = V.I
    return (A.T * (W * A)).I

#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS
#a = [1.1, 2.2, 3.3]
#b = [0.1, 0.2, 0.3]
#c = [3, 4, 5]
#d = [6, 7, 8] #EXAMPLE ARRAYS
#
#f = unp.uarray(a, b) #EXAMPLE UARRAY

#l.Latexdocument('latex_example.tex').tabular(
#data = [c, d, f], #Data incl. unpuarray
#header = ['\Delta Q / \giga\elementarycharge', 'T_1 / \micro\second', 'T_1 / \micro\second'],
#places = [1, 1, (1.1, 1.1)],
#caption = 'testcaption.',
#label = 'testlabel')
##############################################




#Leermessung
I_leer_unscaled = np.zeros(12)
I_leer_unscaled[[0, 1, 2, 9, 10, 11]] = 15964
I_leer_unscaled[[3, 5, 6, 8]] = 15653
I_leer_unscaled[[4, 7]] = 16881
t_leer = np.zeros(12)
t_leer[[0, 1, 2, 9, 10, 11]] = 100
t_leer[[3, 5, 6, 8]] = 100.64
t_leer[[4, 7]] = 106.80

#Zählraten mit Fehler und normierte Werte
I_leer_unscaled = unp.uarray(I_leer_unscaled, np.sqrt(I_leer_unscaled))
I_leer_norm = I_leer_unscaled/t_leer



#Aluminium
c_von_alu, c_bis_alu, counts_alu, t_alu = np.genfromtxt('aluminium/alu_data.txt', unpack = True)
I_0_alu = I_leer_norm * t_alu
counts_alu = unp.uarray(counts_alu, np.sqrt(counts_alu))
arg_log_alu = I_0_alu / counts_alu
y_alu = log(I_0_alu / counts_alu)

V_y_alu = cov_y(stds(y_alu))
mu_alu = mu(V_y_alu, noms(y_alu))
V_mu_alu = cov_mu(V_y_alu)
mu_err_alu = np.sqrt(np.diag(V_mu_alu))
mu_alu = unp.uarray(mu_alu, mu_err_alu)
mu_alu_mean = np.mean(mu_alu)
r.app('mu_{\ce{Al}}', Q_(mu_alu_mean, '1/cm'))




#Blei
c_von_blei, c_bis_blei, counts_blei, t_blei = np.genfromtxt('blei/data_blei.txt', unpack = True)
I_0_blei = I_leer_norm * t_blei
counts_blei = unp.uarray(counts_blei, np.sqrt(counts_blei))
arg_log_blei = I_0_blei / counts_blei
y_blei = log(I_0_blei / counts_blei)

V_y_blei = cov_y(stds(y_blei))
mu_blei = mu(V_y_blei, noms(y_blei))
V_mu_blei = cov_mu(V_y_blei)
mu_err_blei = np.sqrt(np.diag(V_mu_blei))
mu_blei = unp.uarray(mu_blei, mu_err_blei)
mu_blei_mean = np.mean(mu_blei)
r.app('mu_{\ce{Pb}}', Q_(mu_blei_mean, '1/cm'))




#Unbekannt
c_von_unb, c_bis_unb, counts_unb, t_unb = np.genfromtxt('unbekannt/data_unb.txt', unpack = True)
I_0_unb = I_leer_norm * t_unb
counts_unb = unp.uarray(counts_unb, np.sqrt(counts_unb))
arg_log_unb = I_0_unb / counts_unb
y_unb = log(I_0_unb / counts_unb)

V_y_unb = cov_y(stds(y_unb))
mu_unb = mu(V_y_unb, noms(y_unb))
V_mu_unb = cov_mu(V_y_unb)
mu_err_unb = np.sqrt(np.diag(V_mu_unb))
mu_unb = unp.uarray(mu_unb, mu_err_unb)
#delta_alu = mu_unb - mu_alu_mean
#delta_blei = mu_unb - mu_blei_mean
mu_unc = (A.T * A).I * A.T * unp.matrix(y_unb).T
print(mu_unb)
print(mu_unc)



r.makeresults()
