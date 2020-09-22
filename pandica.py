# -*- coding: utf-8 -*-
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
'''
Created on Mon Nov  6 23:34:42 2017
deterministic method, semi implicit method
@author: john Arul and parth
'''

# =============================================================================
# sodium fire is simulated with 100% Na2O
# aerosol mass is mass of sodium + mass of oxygen burnt + RN aerosol mass
# later on the fission product reaction needs to be considered in future
# =============================================================================
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import isotope
# scipy.optimize.newton_krylov(F, xin, iter=None, rdiff=None, method='lgmres',
# inner_maxiter=20, inner_M=None, outer_k=10, verbose=False, maxiter=None, \
# f_tol=None,
# f_rtol=None, x_tol=None, x_rtol=None, tol_norm=None, line_search='armijo',
# callback=None, **kw)

"""
import numpy as np
from scipy.optimize import newton_krylov
from numpy import cosh, zeros_like, mgrid, zeros
# solve
guess = zeros((nx, ny), float)
sol = newton_krylov(residual, guess, method='lgmres', verbose=1)
print('Residual: %g' % abs(residual(sol)).max())
# visualize
import matplotlib.pyplot as plt
x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
plt.pcolor(x, y, sol)
plt.colorbar()
plt.show()
"""

# from scipy.optimize import newton_krylov
# from isotope import *
# import cProfile
# pr = cProfile.Profile()
# pr.enable()

# specify the path for the outputs
# path = '/home/parth/external_gits/containment/results/'
# path = "H:\\mega\\codes\\CONTAINMENT\\results\\"
path = ""

# =============================================================================
#                                   Functions
# =============================================================================
@jit(nopython=True)
def Dcf(u, T):
    rr = (3.0/(4.0*Pi)*u)**(1.0/3.0)
    res = kB*T/(6.0*Pi*mu*rr)*Cs(u)
    return res

# @jit(nopython=True)
# def Kn(d):
#     #d = (v*6.0/Pi)**(1.0/3.0)
#     return 2.0*mfp/d


@jit(nopython=True)
def Cs(v):
    d = (v*6.0/Pi)**(1.0/3.0)
    kn = 2.0 * mfp/d
    val = 1.0 + kn*(1.142 + 0.588 * np.exp(-0.999/kn))
    return val


@jit(nopython=True)
def eps_0(v, u):
    beta = 1.5    # 0.5
    ex = 1.0/3.0
    ldu = (u)**(ex)
    ldv = (v)**(ex)
    mv = min(u, v)
    eps = mv**ex/(ldu + ldv)
    eps = beta * eps * eps
    return eps

# =============================================================================
#                       Agglomeration kernals
# =============================================================================


@jit(nopython=True)
def KG(u, v):
    """
    Agglomeration via gravitation
    """
    con1 = 2.0 * Pi/9.0*(3.0/(4.0 * Pi))**(4.0/3.0) * \
        gamma**2/chi * (g * rhom/mu)
    absv = abs(Cs(v) * v**(2.0/3.0) - Cs(u) * u**(2.0/3.0))
    res = con1 * eps_0(u, v) * (v**(1.0/3.0) + u**(1.0/3.0))**2 * absv
    return res


@jit(nopython=True)
def KB(u, v, T):
    """
    Aerosol coagulation via brownian diffusion
    """
    con1 = 2.0*kB*T/(3.0*mu)
    res = con1*(v**(1.0/3.0)+u**(1.0/3.0)) * \
        (Cs(v)/v**(1.0/3.0)+Cs(u)/u**(1.0/3.0))
    res = res*gamma/chi
    # print(Kn(u), Kn(v), Cs(u), Cs(v))
    # else:
    return res


@jit(nopython=True)
def KTD(u, v):
    """
    Agglomeration due to turbulent diffusion
    """
    Z = 1.29  # 1.29 to 5.65
    res = 3.0*Z/(4.0*Pi)*(eps_T*rhog/mu)**(0.5)*(v**(1.0/3.0)+u**(1.0/3.0))**3
    return res


@jit(nopython=True)
def KTI(u, v):
    """
    Agglomeration due to turbulent inertia
    """
    Zp = 0.188  # 0.188 to 0.204
    S = 1       # 0.1 sticking coefficient
    res = Zp*rhom/mu*((eps_T**3)*rhog/mu)**(0.25)
    res = S*eps_0(u, v)*res*(v**(1.0/3.0)+u**(1.0/3.0))**2 * \
        abs(u**(2.0/3.0)-v**(2.0/3.0))
    return res


@jit(nopython=True)
def K(u, v, T):
    """
    Total agglomeration kernel
    """
    # K0=1.0  # 4.0*kB*T/(3.0*mu)
    # retv= Ksm()
    # return 1E-3
    # retv=KG(u,v) + KB(u,v, T) + math.sqrt(KTD(u,v)**2+KTI(u,v)**2)
    retv = (KG(u, v) + KB(u, v, T) + KTD(u, v) + KTI(u, v))
    # print(KG(u,v),KB(u,v,T),KTD(u,v),KTI(u,v))
    # test kernal Jacobson p499
    # retv = 8.0*kB*Tvar/3.0/mu
    return retv
# =============================================================================

# =============================================================================
#                           Removal mechanisms
# =============================================================================


@jit(nopython=True)
def PR(u, T):  # Diffusion
    """
    Deposition via browian diffusion
    """
    r = (6.0*u/Pi)**(1.0/3.0)/2.0
    retv1 = kB*T*Aw/(6.0*Pi*mu*Del*chi*V)
    retv = retv1*(1.0/r+Cs(u)*mfp*(1.0/r**2))
    # retv=retv*alpha**(1.0/3.0)
    return retv


@jit(nopython=True)
def GR(u, T):   # Gravitational
    """
    Deposition via gravitational sedimentation
    """
    r = ((6.0*u/Pi)**(1.0/3.0))*0.5
    retv1 = 2.0*g*rhom*Af/(9.0*mu*V*chi)
    retv = retv1*(r**2+Cs(u)*mfp*r)
    # retv=retv*alpha**(1.0/3.0)*r*r
    return retv


@jit(nopython=True)
def TR(u, T):   # Thermophoretic
    """
    Deposition via thermophoretic diffusion
    """
    retv = 3.0*mu*Aw*gradT*Kt/(2.0*rhom*V*T)
    return retv


@jit(nopython=True)
def R(u, T):
    """
    Total removal
    """
    return PR(u, T) + GR(u, T) + TR(u, T)


# @jit(nopython=True)
def Ksm(T):
    """
    """
    val = 8.0*kB*T/(3.0*mu)
    return val
# =============================================================================


def init_dist(no_bins, gmmd, rhom, deld, d, v, total_mass):
    """
    INPUT:
    N: Total number of bins
    deld: diameter width in the bin
    d: diameter according to the geometric binnning
    v: volume of different bins
    gmmd: Geometric mass mean diameter
    total_mass: Total mass of the aerosol in the containment

    Returns:
    Generates initial distribution (n(v)dv) with the mean (GMMD) and standard 
    variation specified. 
    """
    sigma = 2
    s = np.log(sigma)

    m = (total_mass * deld/np.sqrt(2*np.pi)/d/s *
         np.exp(- (np.log(d/gmmd))**2 /2/s**2))    # mass distribution
    n = m/rhom/v     # this is n(v) dv = n(d) * dd
    return n


def mean_size(n):
    """
    Calculates the mean size of the distibution
    returns mean diameter
    """

    mean = 0.0
    norm = sum(n)
    for i in range(len(n)):
        # dlogv=math.log(bin_ratio)
        mean += n[i]/norm*vol[i]

    meand = (mean*6.0/Pi)**(1.0/3.0)
    return meand


@jit(nopython=True)
def f(i, j, k):
    Vij = fv(i) + fv(j)

    # rv = 0.0
    if (k < NB) and (Vij >= fv(k)) and (Vij < fv(k+1)):
        rv = (fv(k+1)-Vij)/(fv(k+1)-fv(k)) * fv(k)/Vij

    elif (k > 1) and (Vij > fv(k-1)) and (Vij < fv(k)):
        rv = 1.0-f(i, j, k-1)
        # rv=1.0-(fv(k)-Vij)/(fv(k)-fv(k-1))*fv(k-1)/Vij

    elif (k == NB) and (Vij >= fv(k)):
        rv = 1.0
    else:
        rv = 0

    # print "\n f=",i,j,k,rv
    return rv


@jit(nopython=True)
def fv(i):   # bin mid volumes
    """
    Returns
    """
    return v0*br**i * (1.0/br+1.0)/2.0
# data


@jit(nopython=True, parallel=True)
def Func(x):
    vv = np.zeros(NB)
    for k in np.linspace(1, NB, NB, dtype=int):
        term1 = 0.0
        term2 = 0.0
        for j in np.linspace(1, k-1, k-1, dtype=int):
            term1 += K(fv(k-j), fv(j), Tvar)*x[k-j-1]*x[j-1]
        for j in np.linspace(1, NB, NB, dtype=int):
            term2 += K(fv(k), fv(j), Tvar)*x[k-1]*x[j-1]

        vv[k-1] = -x[k-1]+xp[k-1]+dt/2.0+term1-dt*term2

    return vv


@jit(nopython=True)
def loof_f(n, nux, nu, nx):
    if aglo == 1:
        for k in range(1, NB+1):  # n is previous nx current

            term1 = 0.0
            for j in range(1, k+1):
                for i in range(1, k):
                    term1 += f(i, j, k)*K(fv(i), fv(j), Tvar)*nux[i-1]*n[j-1]
            term1 = nu[k-1]+term1*dt

            term2 = 0.0
            for j in range(1, NB+1):
                term2 += (1.0-f(k, j, k))*K(fv(k), fv(j), Tvar)*n[j-1]

            term2 = 1.0+dt*term2

            # if k>0:
            nux[k-1] = term1/term2
            nx[k-1] = nux[k-1]/vol[k-1]
    return n, nux, nx
# =============================================================================


"""
program starts
"""

Pi = np.pi
mfp = 0.066E-6     # m
g = 9.8            # m/s2
rhom = 2270.0      # kg/m3 # density of the sodium monoxide (Na2O)
rhog = 1.2
mu = 20.7071E-6    # N.s/ m2;
chi = 1.2          # 1.5
gamma = 1.0
alpha = 0.5        # in epsilon -efficiency
kB = 1.38064E-23
Del = 0.118e-6     # 1e-4 m ;
gradT = 100.0      # K/m;
Kt = 0.1

eps_T = 0.05  # ?????


Af = 1400          # 1400.0  # m2
Aw = 5600          # 16665.0  # m2
V = 74000.0        # 77000.0#86146 #m3
# T=342;
# diff len from Progress in modeling in-containment source term with ASTEC-Na

# lambd = 0.0
# rhom=[2270 1530 12400 2600 6800 6240 3600 6506 6145 11600] kg/ m3;

# data for code
paramf = open("panda.txt", "r")
paramf.readline()
param = paramf.readline()
NBtxt, STtxt, dttxt = param.split()
paramf.readline()
param = paramf.readline()
ditxt, d1txt, dmedtxt = param.split()
paramf.close()


NB = int(NBtxt)  # 50
HNB = NB*NB


SimTime = float(STtxt)  # 3600#24
dt = float(dttxt)  # 10.0


d0 = float(ditxt)*1.0E-6      # micro meter
d1 = float(d1txt)*1e-6        # micro meter
v0 = 1.0/6.0*np.pi*d0**3
v1 = 1.0/6.0*np.pi*d1**3

dmi = float(dmedtxt)*1.0E-6   # initial mean diameter
vm = Pi/6.0*dmi**3            # initial volume of the aerosol


# ============================================================================
# assumption na2O to be modified as a combination of na2o and na2o2
Mna = 350  # kg
monoxide = 1

if monoxide == 1:
    # monomoxide
    mass_oxygen = Mna/2 * 16/23    # na2O, 350 = Mna
    total_mass = Mna + mass_oxygen
else:
    # peroxide
    mass_oxygen = Mna * 16/23    # na2O, 350 = Mna
    total_mass = Mna + mass_oxygen


# ============================================================================
# calculate initial aerosol distribution
# distribute mass over lognormal distribution
# 0 to N-1
# Nless=NB/2

br = (v1/v0)**(1.0/float(NB-1))

n0 = np.zeros(NB)
n = np.zeros(NB)
nx = np.zeros(NB)
nu = np.zeros(NB)
nux = np.zeros(NB)
dia = np.zeros(NB)
dv = np.zeros(NB)
vol = np.zeros(NB)

ddv = 1.0  # mcube to smaller volume

for i in np.linspace(1, NB, NB, dtype=int):
    vol[i-1] = fv(i)    # volume of the bin
    dia[i-1] = (6 * vol[i-1]/np.pi)**(1/3)

dv = 2 * vol * (br-1)/(1+br)    # bin volume width
deld = dia * 2**(1/3) * (br**(1/3)-1)/(1+br)**(1/3)
# Divided by the containment volume to get the number concentration
n0 = init_dist(NB, dmi, rhom, deld, dia, vol, total_mass)/V

NSIME = int(SimTime/dt)

# =============================================================================
#                       Read temperature and pressure
# =============================================================================
# read temperature from input file
# temp_according to inst spray fire
tf = open("containment_temperature.txt", 'r')
# tf.readline()
tdata = tf.readlines()
t = []
temp_air = []
for line in tdata:
    cols = line.split()
    t.append(float(cols[0]))
    temp_air.append(float(cols[1]))

# # if you want constant temp
# for i in range(len(temp_air)):
#    temp_air[i] = 303.0
# read pressure input generated from pfire code
pf = open('containment_pressure.txt', 'r')
# pf.readline()
pdata = pf.readlines()
t1 = []
pressure_air = []   # pressure in the containment
for line in pdata:
    cols = line.split()
    t1.append(float(cols[0]))
    pressure_air.append(float(cols[1]) + 1e5)   # pressure in bars
# ============================================================================

# print("sum=", sum(n))

n = n0.copy()
nx = n.copy()
nu = vol*n
nux[0] = nu[0]*vol[0]
TN = []
TM = []


# testing of aerosol mass conservation
# mass_ratio_check = sum(np.array(n) * rhom * np.array(vol))*V/total_mass
# print mass_ratio_check
fms = sum(n * vol * rhom)*V/total_mass
print('fms0=', fms)

NS = NSIME

aglo = 1
removal = 1

for ti in range(NS):
    # Tvar=300.0
    Tvar = np.interp(ti * dt, t, temp_air)
    # print(Tvar, 'Tvar')

    # if ti * dt/3600 > 0.5:
    #     aglo = 1
    # print(aglo)
    print(ti, dt*ti)
    n, nux, nx = loof_f(n, nux, nu, nx)

    if aglo == 2:

        try:
            xp = n.copy()  # previous value
            nx = newton_krylov(Func, n, method='lgmres',
                               verbose='True', maxiter=10, f_tol=None)
        except Exception:
            # print Exception
            print("convergence error")
            # pass

    fracm = sum(nx*vol)/sum(n*vol)
    fracn = sum(nx)/sum(n)
    print("\n step nfrac =", fracn)
    print("\n step mfrac=", fracm)

# ---------------------------removal process
    if removal == 1:
        for i in range(NB):
            vv = vol[i]
            # print(R(vv, Tvar))
            nx[i] = nx[i]/(1.0+R(vv, Tvar)*dt)
# -------------------------curr values = new values
    n = nx.copy()
    nu = n*vol
# ---------------------------------------------------

    fms = sum(n * vol) * rhom*V/total_mass
    print("\nsuspended mass frac=", fms)

    # n[:] = np.array(nu[:])/np.array(vol[:])
    TN.append(sum(n))   # sum of all sizes
    TM.append(fms*total_mass)

# test case

# Ninit = 1.0E12
# step=100.0
# if aglo ==3:
#
#    for k in np.linspace(1,NB,NB,dtype=int):
#        n[k-1]= Ninit*(0.5*step*K(1,1,298.0)*Ninit)**(k-1)/ \
#                (1.0+0.5*step*K(1,1,298.0)*Ninit)**(k+1)

# =============================================================================

# =============================================================================
#                Plot initial and final aerosol distribution
# =============================================================================
xd = [(6.0*fv(i)/Pi)**(1.0/3.0) for i in range(NB)]

# nr01=[nr0[i]/xd[i] for i in range(NB)]
# n1=[n[i]/xd[i] for i in range(NB)]

# dxv=map(lambda(x):(6.0/Pi*x)**(1.0/3.0),xv)

fl_iso = 0

if fl_iso == 0:

    # plt.ylim(ymin=1.0, ymax=max(n)*10)
    # plt.loglog(xd,nr0,xd,n,'+')
    # print("K", Ksm())
    print("mean n0m,nm =", mean_size(n0), mean_size(n))

    x = [i for i in range(NB)]
    sz = [(v0*br**i * 6.0/Pi)**(1.0/3.0)*1E6 for i in range(NB)]
    n0 = n0+0.1
    n = n+0.1
    plt.figure()
    plt.loglog(sz, n, label='final', color='#ff1453')
    plt.loglog(sz, n0, label='initial', color='#0eaf69')
    plt.ylabel("Number of Aerosols", fontsize=14)
    plt.xlabel("Aerosol Diameter (micron)", fontsize=14)
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', axis='both', color='k')
    plt.grid(which='minor', axis='both', color='#e6e7e4')
    plt.savefig("aerosol_dist.png")

    fi = open(path + 'aero_dist.txt', 'w')
    for i in range(NB):
        fi.write("%E  %E  %E\n" % (sz[i], n0[i], n[i]))
    fi.close()

    # print("%.2E,%.2E, %.2E" % (n0c, sum(nx), sum(n)))


# Na aerosol
fl_iso = 1
if fl_iso == 1:

    TX = TM
    plt.figure()
    NTs = len(TX)
    tx1 = [i*dt/60.0 for i in range(NTs)]
    ax = plt.gca()

    ax.set_xlim(xmin=0, xmax=max(tx1))
    ax.set_ylim(ymin=min(TM), ymax=max(TX))
    plt.plot(tx1, TX)
    # plt.semilogy(tx1, TM)
    plt.savefig("aerosol_evol_mass.png")

    fi = open(path + 'aeros_evol.txt', 'w')
    for i in range(len(tx1)):
        fi.write("%E  %E\n" % (tx1[i], TX[i]))
    fi.close()
# =============================================================================

# =============================================================================
#                          Process & plot isotropic data
# =============================================================================
# fl_iso = 2
if fl_iso == 2:

    # fi=open("leak.txt",'r')
    # x1=[]
    # y1=[]
    # data=fi.readlines()
    # for line in data:
    #     xv,yv=line.split()
    #     x1.append(float(xv))
    #     y1.append(float(yv))
    # fi.close()

    # print isotope.d_group['Xe']
    TN = TM.copy()
    NTs = len(TN)
    released_activity_contain = {}
    act_in_air = {}
    act_dep = {}
    # for j in range(NTs):
    for isot in isotope.d_invent:
        act_dep[isot] = []
    for isot in isotope.d_invent:
        act_in_air[isot] = []
    iso_act = [[0 for i in range(len(isotope.d_invent))] for j in range(NTs)]
    tot_act = [0 for j in range(NTs)]
    tx = [i*dt for i in range(NTs)]
    lv = 0  # *np.interp(tx[i],x1,y1)*0.01
    for j in range(NTs):
        for isot in isotope.d_invent:
            value = isotope.d_invent[isot]
            lambd = value[0]
            act = value[1]*value[2]
            released_activity_contain[isot] = value[1] * value[2]
            elem, iwt = isot.split('-')
            if elem in isotope.d_group['Xe']:
                iso_act[j] += [act*math.exp(-(lambd+lv) * tx[j])]
                # print elem
                act_in_air[isot].append(act * math.exp(-(lambd+lv) * tx[j]))
                act_dep[isot].append(0)
            else:
                iso_act[j] += [act * TN[j]/TN[0] *
                               math.exp(-(lambd + lv)
                                        * tx[j])]  # add a list element
                act_in_air[isot].append(
                    act*TN[j]/TN[0]*math.exp(-(lambd+lv) * tx[j]))

                act_dep[isot].append(act*(1 - TN[j]/TN[0])
                                     * (math.exp(-(lambd+lv) * tx[j])))
        tot_act[j] = sum(iso_act[j])

    txm = [sec/60.0 for sec in tx]
    mass_dep = {}
    for i in act_in_air:    # conversion from Bq to kg
        act_in_air[i] = np.array(
            act_in_air[i])/isotope.d_invent[i][0] * \
            isotope.molar_mass[i]/6.022e23/1e3
        mass_dep[i] = np.array(act_dep[i])/isotope.d_invent[i][0] * \
            isotope.molar_mass[i]/6.022e23/1e3
        released_activity_contain[i] = np.array(
            released_activity_contain[i]) / isotope.d_invent[i][0] *\
            isotope.molar_mass[i]/6.022e23/1e3
# ============================================================================
#                        Activity in suspended form
# ============================================================================
    el_in_air = {}
    plt.figure()
    for i in act_in_air:
        el_in_air[i.split('-')[0]] = []
    el_deposited = {}
    for i in act_in_air:
        el_deposited[i.split('-')[0]] = []
    for i in act_in_air:
        el_deposited[i.split('-')[0]] += [act_dep[i]]
    for i in el_in_air:
        el_deposited[i] = sum(np.array(el_deposited[i]))
        plt.plot(tx, el_deposited[i], label=i)
    el_released = {}
    for i in act_in_air:
        el_released[i.split('-')[0]] = []
    for i in act_in_air:
        el_released[i.split('-')[0]] += [act_in_air[i]]
    for i in el_released:
        el_released[i] = sum(np.array(el_released[i]))

# =============================================================================
#                              Text file write
# =============================================================================
    file = open(path + 'mass_in_contain_air_el.txt', 'w',  1024)
    file1 = open(path + 'temp_in_containment.txt', 'w', 1024)
    file2 = open(path + 'mass_deposited_in_contain_el.txt', 'w', 1024)
    file3 = open(path + 'pres_in_containment.txt', 'w', 1024)
    k12 = 0
    # file4 = open (path + 'total_aerosol_mass_released.txt',  'w', 1024)
    str0 = ''
    str1 = ''
    str2 = ''
    str3 = ''
    str0 += f'{"time":15s}'
    str2 += f'{"time":15s}'
    sus_dep_el_name = ['I', 'Cs', 'Rb', 'Te', 'Sr', 'Ba',
                       'Ru', 'La', 'Zr', 'Cm', 'Ce', 'U', 'Np', 'Pu']
    # for i34 in el_in_air:
    for i34 in sus_dep_el_name:
        str0 += f'{i34:15s}'
        str2 += f'{i34:15s}'
    str0 += f'\n'
    str2 += f'\n'
    for k12 in range(NTs):
        str0 += f'\n{tx[k12]:15.8e}'
        str2 += f'\n{tx[k12]:15.8e}'
        str1 += f'{tx[k12]:15.8e}{np.interp(tx[k12], t, temp_air):15.6e}\n'
        str3 += f'{tx[k12]:15.8e}\
            {np.interp(tx[k12], t1, pressure_air) :15.6e}\n'
        for i in sus_dep_el_name:
            str0 += f'{el_in_air[i][k12]:15.8e}'

            str2 += f'{el_deposited[i][k12]:15.8e}'

        # k12 += 1
    file.write(str0)
    file1.write(str1)
    file2.write(str2)
    file3.write(str3)
    # file4.write(str0)
    file.close()
    file1.close()
    file2.close()
    file3.close()

    # activity isotope wise result write
    f1 = open(path + 'act_in_containment_air_iso.txt', 'w')
    f2 = open(path + 'mass_deposited_in_containment_iso.txt', 'w')
    str1 = ''
    str2 = ''
    str1 = 'Time\t'
    str2 = 'Time\t'
    containment_iso_name = ['Kr-83m', 'Kr-85', 'Kr-85m', 'Kr-87', 'Kr-88',
                            'Kr-89', 'Xe-131m', 'Xe-133', 'Xe-133m', 'Xe-135',
                            'Xe-135m', 'Xe-137', 'Xe-138', 'I-131', 'I-132',
                            'I-133', 'I-134', 'I-135', 'Cs-134', 'Cs-137',
                            'Rb-88', 'Te-131m', 'Te-132', 'Sr-89', 'Sr-90',
                            'Ba-140', 'Ru-103', 'Ru-106', 'La-140', 'Zr-95',
                            'Cm-242', 'Cm-243', 'Cm-244', 'Ce-141', 'Ce-144',
                            'U-237', 'U-239', 'Np-238', 'Np-239', 'Pu-239',
                            'Pu-240', 'Pu-241', 'Pu-242']
    # for i2 in act_in_air:
    for i2 in containment_iso_name:
        str1 += f'{i2:15s}'
        str2 += f'{i2:15s}'
    str1 += '\n'
    str2 += '\n'
    for i in range(len(tx)):
        str1 += f'\n{tx[i]:15.6e}'
        str2 += f'\n{tx[i]:15.6e}'
        for j in containment_iso_name:
            str1 += f'{act_in_air[j][i]:15.6e}'
            str2 += f'{mass_dep[j][i]:15.6e}'
    f1.write(str1)
    f2.write(str2)
    f1.close()
    f2.close()

    ax = plt.gca()
    ax2 = ax.twinx()
    ax.set_xlim(xmax=max(txm))
    ax.set_ylim(ymin=1.0E11, ymax=1.0E20)
    ax.set_ylabel("Total Activity (Bq)")
    ax.set_xlabel("Time (min)")
    ax.grid(which='major', axis='both')
    ax.semilogy(txm, tot_act)
    plt.savefig('activity.png')

    fi = open(path + 'activity.txt', 'w')
    for i in range(len(txm)):
        fi.write("%E  %E\n" % (txm[i], tot_act[i]))
    fi.close()
    plt.figure()
    plt.plot(np.array(tx)/3600, np.array(TM)/TM[0]*total_mass)
    plt.show()
    f2 = open(path + 'total_aerosol_mass.txt', 'w')
    f2.write('Time \t\t total_aerosol_mass(Kg)\n')
    str1 = ''
    for i in range(len(TM)):
        str1 += f'{tx[i]:15.6e}{TM[i]/TM[0]*total_mass:15.6e}\n'
    f2.write(str1)
    f2.close()

tx = [i*dt/3600 for i in range(NTs)]
f2 = open(path + 'total_aerosol_mass_with_new_init_dist.txt', 'w')
f2.write('Time \t\t total_aerosol_mass(Kg)\n')
str1 = ''
for i in range(len(TM)):
    str1 += f'{tx[i]:15.6e}{TM[i]/TM[0]*total_mass:15.6e}\n'
f2.write(str1)
f2.close()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
# ============================================================================
#                           Aerosol removal kernal plot
# ============================================================================
switch = False
if switch:
    r_aero = np.linspace(0.01, 10, int(1e5)) * 1e-6
    v_aero = (r_aero)**(3) * 3.14/6
    plt.figure()
    plt.loglog(r_aero*1e6, R(v_aero, 300), label='Total aerosol removal rates',
               color='#f25c5a', marker='^', markerfacecolor='None',
               markersize=7, markevery=0.1)
    plt.loglog(r_aero*1e6, PR(v_aero, 300),
               label='Diffusion aerosol removal rates', color='#87d498',
               marker='o', markerfacecolor='None', markersize=7, markevery=0.1)
    plt.loglog(r_aero*1e6, GR(v_aero, 300),
               label='Gravitational aerosol removal rates',
               color='#ffd166', marker='d', markerfacecolor='None',
               markersize=7, markevery=0.1)
    plt.loglog(r_aero*1e6, np.array([TR(i, 300) for i in v_aero]),
               label='Thermophoretic aerosol removal rates',
               color='#fa761e', marker='p', markerfacecolor='None',
               markersize=7, markevery=0.1)
    plt.legend()
    plt.minorticks_on()
    plt.grid(which='major', axis='both', c='#a5a1a1')
    plt.grid(which='minor', axis='both', c='#e6e7e4')
    plt.xlabel('Aersol diameter, um')
    plt.ylabel('Rates (per seconds)')
    plt.savefig('aerosol_removal_rates_vs_r')
    plt.show()
# =============================================================================
# Comparision of total mass with other aerosol
# =============================================================================
# tp_file = open('tp_total_suspended_aerosol_mass.txt', 'r')
# header = tp_file.readline()
# data = tp_file.readlines()
# time_tp = []
# mass_tp = []
# for line in data:
#     cols = line.split()
#     time_tp.append(float(cols[0]))
#     mass_tp.append(float(cols[1]))

# xjtu_file = open('xjtu_total_aerosol_mass.txt', 'r')
# header = xjtu_file.readline()
# data = xjtu_file.readlines()
# time_xjtu = []
# mass_xjtu = []
# for line in data:
#     cols = line.split()
#     time_xjtu.append(float(cols[0]))
#     mass_xjtu.append(float(cols[1]))
# ciae_file = open('ciae_total_aerosol_mass.txt', 'r')
# header = ciae_file.readline()
# data = ciae_file.readlines()
# time_ciae = []
# mass_ciae = []
# for line in data:
#     cols = line.split()
#     time_ciae.append(float(cols[0]))
#     mass_ciae.append(float(cols[1]))

# #igcar_file_0_1_5um = open('total_aerosol_mass_0.1_5um.txt', 'r')
# #header = igcar_file_0_1_5um.readline()
# #data = igcar_file_0_1_5um.readlines()
# #time_igcar_0_1_5um = []
# #mass_igcar_0_1_5um = []
# # for line in data:
# #    cols = line.split()
# #    time_igcar_0_1_5um.append(float(cols[0]))
# #    mass_igcar_0_1_5um.append(float(cols[1]))
# #
# #igcar_file_0_01_5um = open('total_aerosol_mass_0.01_5um.txt', 'r')
# #header = igcar_file_0_01_5um.readline()
# #data = igcar_file_0_01_5um.readlines()
# #time_igcar_0_01_5um = []
# #mass_igcar_0_01_5um = []
# # for line in data:
# #    cols = line.split()
# #    time_igcar_0_01_5um.append(float(cols[0]))
# #    mass_igcar_0_01_5um.append(float(cols[1]))

# igcar_file = open('total_aerosol_mass.txt', 'r')
# header = igcar_file.readline()
# data = igcar_file.readlines()
# time_igcar = []
# mass_igcar = []
# for line in data:
#     cols = line.split()
#     time_igcar.append(float(cols[0]))
#     mass_igcar.append(float(cols[1]))
# plt.figure()
# plt.plot(np.array(tx1)/60, TM, label='TM', c='#ff0000')
# plt.plot(np.array(time_xjtu)/3600, mass_xjtu, label='xjtu', color='#408080')
# plt.plot(np.array(time_tp)/3600, mass_tp, label='tp', color='#ff0080')
# # plt.plot(np.array(time_igcar)/3600, mass_igcar, label='IGcar', color='#0000ff')
# plt.plot(np.array(time_ciae)/3600, mass_ciae, label='ciae', color='#000000')
# # plt.plot(np.array(time_igcar_0_1_5um)/3600,
# #  mass_igcar_0_1_5um, label='igcar:0.1-5um', color='#0080c0')
# # plt.plot(np.array(time_igcar_0_01_5um)/3600,
# #  mass_igcar_0_01_5um, label='igcar:0.01-5um', color='#8000ff')
# plt.title('Total aerosol mass suspended in the containment')
# plt.minorticks_on()
# plt.grid(which='major', axis='both', color='#a5a1a1')
# plt.grid(which='minor', axis='both', color='#fafaff')
# plt.legend()
# plt.xlabel('Time (Hours)', fontsize=14)
# plt.ylabel('Mass (Kg)', fontsize=14)
# plt.xlim((0, 24))
# plt.ylim(bottom=0)
# plt.tight_layout()
# plt.savefig('mass_comparision.png')
# plt.show()
