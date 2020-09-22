# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 18:58:07 2017

@author: john

# read manipulate isotopic data
"""
import re
isotope = []
halflife = []
unit = []
core = []
molar_mass = {}    # gram/mol
# Act=[[0 ] for j in]
# with open ('activity.txt') as in_data:

in_data = open('core_activity.txt')
line = in_data.readline()
lines = in_data.readlines()
in_data.close

for line in lines:

    cols = line.split()
    isotope.append(cols[0])
    molar_mass[cols[0]] = float(cols[0].split('-')[1].replace('m', ''))
    halflife.append(float(cols[1]))
    unit.append(cols[2])
    core.append(float(cols[4]))  # 100.0 MWd/kg

lambd = [0.0 for i in range(len(halflife))]
sph = 3600.0

for i in range(len(isotope)):

    if unit[i] == 'd':
        lambd[i] = halflife[i]*24.0*sph
    elif unit[i] == 'm':
        lambd[i] = halflife[i]*60.0
    elif unit[i] == 'h':
        lambd[i] = halflife[i]*sph
    elif unit[i] == 'y':
        lambd[i] = halflife[i]*365.0*24.0*sph
    else:
        print("unit error\n")

for i in range(len(lambd)):
    lambd[i] = 0.693/lambd[i]

d_invent = {}
for iso, l, co in zip(isotope, lambd, core):
    d_invent.update({iso: [l, co]})


in_data = open('RF.txt', 'r')
in_data.readline()
lines = in_data.readlines()
in_data.close

d_eRF = {}  # group representative release fractions

for line in lines:
    cols = line.split()
    rf = float(cols[1])
    rom = float(cols[2])  # material density
    d_eRF.update({cols[0]: [rf, rom]})


in_data = open('group.txt', 'r')
lines = in_data.readlines()
in_data.close

d_group = {}

for line in lines:
    (el, elv) = line.split(':')
    elv = elv[:-1]
    elv = elv.split(',')
    d_group.update({el: elv})

# d_group=dict(zip(elemd,Gr))  #  test

d_eRF1 = {}  # expanded elemnt wise RF

for key in d_group:

    [rf, rom] = d_eRF[key]
    elist = d_group[key]

    for x in elist:
        d_eRF1.update({x: [rf, rom]})

# print d_eRF1

# dict_inventory : Bq,lambda/s,

for key in d_invent:

    vals = d_invent[key]
    (elem, no) = key.split('-')

    if elem in d_eRF1:
        [rf, rom] = d_eRF1[elem]
        vals = vals+[rf]
        d_invent.update({key: vals})
    else:
        print("error: element %s not found in RF file" % (elem))
        rf = 1.0
        vals = vals+[rf]
        d_invent.update({key: vals})

    # print d_invent[key]


def conv_bq_kg(inventory, molar_mass, lembda):
    "converts the RN activity (Bq) to mass Kg"
    Na = 6.022e23    # avogrado number
    mass_in_kg = inventory / lembda * molar_mass / Na / 1e3
    return mass_in_kg
