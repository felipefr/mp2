#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:16:01 2024

@author: frocha
"""

import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

plt.rc("text", usetex = True)
plt.rc("font", family = 'serif')
plt.rc("font", size = 12)

def rungekutta4(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def Ec_rdot(rdot):
    return 0.5*rdot**2

def Eff(r,beta):
    return beta/(r**2) - 1.0/r

def dEff(r,beta):
    return (r - 2*beta)/r**3


def gravitational_system(y, t, beta):
    return np.array([y[1], -dEff(y[0],beta)])    


G = 6.67*10**(-11)
MT = 6*10**24
RT = 6400*10**3

tau = np.sqrt(RT**3/(MT*G))

# really needed ==> etat lies
r0 = 3.0
Em_0 = -0.1
beta = 0.75

# only for making in physical
m = 100

# Consequences
rdot0 = -np.sqrt(2*(Em_0-Eff(r0, beta)))
rdot0_real = RT*rdot0/tau
L0 = np.sqrt(2*beta*G*MT*m**2*RT)
Eref = G*MT*m/RT
tau = np.sqrt(RT**3/(G*MT))

y0 = np.array([r0,rdot0])
Tmax = 100.0
t = np.linspace(0, Tmax, 1000)

sol = rungekutta4(gravitational_system, y0, t, args=(beta,))

plt.figure(1)
plt.title('$r(t)$')
plt.plot(t, sol[:, 0])
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.figure(2)
plt.title('$\dot{r}(t)$')
plt.plot(t, sol[:, 1])
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

plt.figure(3)
plt.title('$Eeff(t)$')
plt.plot(t, Eff(sol[:, 0], beta))
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

fig = plt.figure(4)
camera = Camera(fig)

r = np.linspace(0.5,10.0,100)
Eff_eval = Eff(r, beta)
sol_ = sol[0::10,0]
t_ = t[0::10]

param_str = '$\dot{r}_0 = %.2e m/s$, \n'%rdot0_real
param_str += '$L_0 = %.2e kg m^2/s$, \n'%L0
param_str += '$m = %.2e kg$, \n'%m
param_str += '$E_{ref} = %.2e J$, \n'%Eref
# param_str += '$\\tau = %.2e s$, \n'%tau
param_str += '$R_T = %.2e m$, \n'%RT
param_str += '$M_T = %.2e kg$'%MT
plt.grid()
plt.ylabel('$E_{eff}(\\bar{r})/E_{ref}$')
plt.xlabel('$\\bar{r} = r/R_T$')
plt.title('Etats liés')
for i in range(len(sol_)):
    plt.plot(r, Eff_eval, color = 'black')
    plt.plot([np.min(r), np.max(r)],[Em_0,Em_0], '--', color = 'blue')
    plt.text(4,0.25, param_str, bbox={'facecolor':'blue', 'alpha':0.3, 'pad':10})
    plt.plot([sol_[i]], [Eff(sol_[i],beta)], 'o', color = 'red')
    plt.tight_layout()
    camera.snap()
    
animation = camera.animate()
animation.save('etat_lies.mp4', dpi = 500, fps = 5)    


    
