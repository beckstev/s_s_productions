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
print(np.linalg.det((A.T * A)))


def cov_y(y):
    V = np.zeros((len(y), len(y)))
    np.fill_diagonal(V, y**2)
    return np.matrix(V)

def mu(V, y):
    W = V.I
    y = np.matrix(y).T
    mu = ((A.T * (W * A)).I * (A.T * (W * y))).T
    return np.array(mu)

def cov_mu(V):
    W = V.I
    return (A.T * (W * A)).I

def weight_mid(x):
    s_x = stds(x)
    x = noms(x)
    mid_x = sum(x/s_x**2) / sum(1/s_x**2)
    del_x = np.sqrt(1 / sum(1/s_x**2))
    return ufloat(mid_x, del_x)




#Leermessung
I_leer_unscaled = np.zeros(12)
I_leer_unscaled[[0, 1, 2, 9, 10, 11]] = 15964
I_leer_unscaled[[3, 5, 6, 8]] = 15653
I_leer_unscaled[[4, 7]] = 16881
t_leer = np.zeros(12)
t_leer[[0, 1, 2, 9, 10, 11]] = 100
t_leer[[3, 5, 6, 8]] = 100.64
t_leer[[4, 7]] = 106.80
channel_leer = unp.uarray([53, 53, 53], [58, 58, 58])




#Zählraten mit Fehler und normierte Werte
I_leer_unscaled = unp.uarray(I_leer_unscaled, np.sqrt(I_leer_unscaled))
I_leer_norm = I_leer_unscaled/t_leer


#l.Latexdocument('tabs/leermessung.tex').tabular(
#data = [np.array([1, 4, 5]), channel_leer, t_leer[[0, 3, 4]], I_leer_unscaled[[0, 3, 4]]], #Data incl. unpuarray
#header = ['Projektion / ', 'Kanal / ', 't / \second', 'N_0'],
#places = [0, (1.0, 1.0), 2, (5.0, 3.0)],
#caption = 'Aufgenommene Daten der Leermessung. Messzeit $t$, Counts $N_0$.',
#label = 'leer')




#Aluminium
c_von_alu, c_bis_alu, counts_alu, t_alu = np.genfromtxt('aluminium/alu_data.txt', unpack = True)
channel_alu = unp.uarray(c_von_alu, c_bis_alu)
I_0_alu = I_leer_norm * t_alu
counts_alu = unp.uarray(counts_alu, np.sqrt(counts_alu))
arg_log_alu = I_0_alu / counts_alu
y_alu = log(I_0_alu / counts_alu)


V_y_alu = cov_y(stds(y_alu))
mu_alu = mu(V_y_alu, noms(y_alu))[0]
V_mu_alu = cov_mu(V_y_alu)
mu_err_alu = np.array(np.sqrt(np.diag(V_mu_alu)))
mu_alu = unp.uarray(mu_alu, mu_err_alu)


mu_alu_mean = weight_mid(mu_alu)#np.mean(mu_alu)

r.app('\mu_{\ce{Al}}', Q_(mu_alu_mean, '1/cm'))
I = np.linspace(1, 12, 12)



#l.Latexdocument('tabs/alu.tex').tabular(
#data = [I, channel_alu, t_alu, counts_alu, I_0_alu, y_alu], #mu_alu
#header = ['\\text{Projektion} / ', '\\text{Kanal} / ', 't / \second', 'N / ', 'N_0 / ', 'y / ' ],#'\mu / \centi\meter^{-1}'
#places = [0, (2.0, 2.0), 2, (4.0, 2.0), (4.0, 2.0), (1.2, 1.2)],#, (2.2, 2.2)
#caption = 'Aufgenommene Messdaten und berechnete Größen zur Untersuchung des Aluminiumwürfels.',
#label = 'alu')



#Blei
c_von_blei, c_bis_blei, counts_blei, t_blei = np.genfromtxt('blei/data_blei.txt', unpack = True)
channel_blei = unp.uarray(c_von_blei, c_bis_blei)
I_0_blei = I_leer_norm * t_blei
counts_blei = unp.uarray(counts_blei, np.sqrt(counts_blei))
arg_log_blei = I_0_blei / counts_blei
y_blei = log(I_0_blei / counts_blei)

V_y_blei = cov_y(stds(y_blei))
mu_blei = mu(V_y_blei, noms(y_blei))[0]
V_mu_blei = cov_mu(V_y_blei)
mu_err_blei = np.array(np.sqrt(np.diag(V_mu_blei)))
mu_blei = unp.uarray(mu_blei, mu_err_blei)
mu_blei_mean = weight_mid(mu_blei)
r.app('\mu_{\ce{Pb}}', Q_(mu_blei_mean, '1/cm'))

#l.Latexdocument('tabs/blei.tex').tabular(
#data = [I, channel_blei, t_blei, counts_blei, I_0_blei, y_blei], #mu_alu
#header = ['\\text{Projektion} / ', '\\text{Kanal} / ', 't / \second', 'N / ', 'N_0 / ', 'y / ' ],#'\mu / \centi\meter^{-1}'
#places = [0, (2.0, 2.0), 2, (4.0, 2.0), (6.0, 3.0), (1.2, 1.2)],#, (2.2, 2.2)
#caption = 'Aufgenommene Messdaten und berechnete Größen zur Untersuchung des Bleiwürfels.',
#label = 'blei')
print(mu_blei)


#l.Latexdocument('tabs/mu_pb_al.tex').tabular(
#data = [np.linspace(1, 9, 9), mu_alu, mu_blei], #mu_alu
#header = ['Würfel / ', '\mu / \centi\meter^{-1}', '\mu / \centi\meter^{-1}'],#'\mu / \centi\meter^{-1}'
#places = [0, (1.2, 1.2), (1.2, 1.2)],#, (2.2, 2.2)
#caption = 'Bestimmte Absorptionskoeffizienten für Aluminium und Blei.',
#label = 'results_mu')


#Unbekannt
c_von_unb, c_bis_unb, counts_unb, t_unb = np.genfromtxt('unbekannt/data_unb.txt', unpack = True)
channel_unb = unp.uarray(c_von_unb, c_bis_unb)
I_0_unb = I_leer_norm * t_unb
counts_unb = unp.uarray(counts_unb, np.sqrt(counts_unb))
arg_log_unb = I_0_unb / counts_unb
y_unb = log(I_0_unb / counts_unb)

V_y_unb = cov_y(stds(y_unb))
mu_unb = mu(V_y_unb, noms(y_unb))[0]
V_mu_unb = cov_mu(V_y_unb)
mu_err_unb = np.sqrt(np.diag(V_mu_unb))
mu_unb = unp.uarray(mu_unb, mu_err_unb)
#delta_alu = mu_unb - mu_alu_mean
#delta_blei = mu_unb - mu_blei_mean
mu_unc = (A.T * A).I * A.T * unp.matrix(y_unb).T

#l.Latexdocument('tabs/unb.tex').tabular(
#data = [I, channel_unb, t_unb, counts_unb, I_0_unb, y_unb], #mu_alu
#header = ['\\text{Projektion} / ', '\\text{Kanal} / ', 't / \second', 'N / ', 'N_0 / ', 'y / ' ],#'\mu / \centi\meter^{-1}'
#places = [0, (2.0, 2.0), 2, (5.0, 3.0), (6.0, 3.0), (1.2, 1.2)],#, (2.2, 2.2)
#caption = 'Aufgenommene Messdaten und berechnete Größen zur Untersuchung des Unbekannten Würfels.',
#label = 'unb')


#Vergleich mit Literaturwerten
mu_blei_lit = 1.245
mu_alu_lit  = 0.202
delta_blei_lit = noms(abs(mu_unb - mu_blei_lit))
delta_alu_lit = noms(abs(mu_unb - mu_alu_lit))

delta_blei_exp = noms(abs(mu_unb - mu_blei_mean.n))
delta_alu_exp = noms(abs(mu_unb - mu_alu_mean.n))

print(mu_unb)
print(delta_alu_exp)
print(mu_alu_mean.n)


#l.Latexdocument('tabs/delta_mu.tex').tabular(
#data = [np.linspace(1, 9, 9), mu_unb, delta_alu_exp, delta_alu_lit, delta_blei_exp, delta_blei_lit], #mu_alu
##header = ['Würfel / ', r'\Delta\mu_{\ce{Al}, exp} / \centi\meter^{-1}', '\Delta\mu_{\ce{Al}, lit} / \centi\meter^{-1}', '\Delta\mu_{\ce{Pb}, exp} / \centi\meter^{-1}', '\Delta\mu_{\ce{Pb}, lit} / \centi\meter^{-1}'],#'\mu / \centi\meter^{-1}'
#header = ['a', 'b', 'c', 'd', 'e', 'f'],
#places = [0, (2.2, 2.2), 2, 2, 2, 2],#, (2.2, 2.2)
#caption = 'Bestimmte Absorptionskoeffizienten des unbekannten Würfels mit absoluten Abweichungen zu den theoretischen, bzw. experimentell bestimmten Werten.',
#label = 'delta_mu')


r.makeresults()
