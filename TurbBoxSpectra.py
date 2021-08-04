######################################################################
"""
   Helmholtz-Hodge Decomposition Using FFT

"""
######################################################################
# Author: Ammar Khallouf
# Date: 23.03.2021
# Version: 1.0
######################################################################
""" *** Description ***

This script performs the Helmholtz-Hodge decomposition for a given 3D vector field using fourier analysis.

The decomposition results in a Poisson Problem which is solved in spectral space and returned to physical space by

an inverse fourier transform. Upon solving the problem, the vector field is sperated into a divergence free component

(incompressible) and a divergent component (compressible).

Multiple I/O file formats are supported.
"""
######################################################################
""" *** Important Notes ***

1. The 3D Vector field is defined on a structured cartesian grid with (Nx,Ny,Nz) number of grids. (Preferably Power of 2, e.g 32,64,128)

2. Boundary conditions are periodic in FFT.

3. Helmholtz-Hodge decomposition assumes a relatively smooth field with decay conditions (i.e small scales have little energy).

4. For an even number of grid points (Nx,Ny,Nz), the Nyquist frequency arises at N/2 which makes the decomposed field complex (i.e imagenary part is present).

5. For a sufficently smooth field, the imagenary part caused by Nyquist frequency mode should be negligible.

"""
######################################################################
"""
*** References ***

1. https://handwiki.org/wiki/Helmholtz_decomposition

2. http://math.mit.edu/~stevenj/fft-deriv.pdf

"""
######################################################################

#*********Read Input Data in Binary Form*********#

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.signal import argrelextrema
import time
import sys
import math
from pylab import *
import random
from IPython.core.display import display, Math
from scipy import signal
matplotlib.rcParams['text.usetex'] = True
#******************************************************#

# Read as h5 file format

hf = h5py.File('windgen_incompressible.h5', 'r')


# Read Velocity field components (Vx,Vy,Vz) accordingly from h5 datasets ('u','v','w')

Vx= np.array(hf.get('u')) 
Vy= np.array(hf.get('v')) 
Vz= np.array(hf.get('w')) 

# Read dimensions of the domain.

# lx  = hf.get('lx')[0]
# ly  = hf.get('ly')[0]
# lz  = hf.get('lz')[0]

lx  = 10080.0
ly  = 1260.0
lz  = 1260.0

# Number of grids in each direction NX,NY,NZ

NX,NY,NZ= Vx.shape

# Spacing of grids in each direction DX,DY,DZ

dx = lx /float(NX - 1)
dy = ly /float(NY - 1)
dz = lz /float(NZ - 1)


# Check for correct dimesnion

if Vx.ndim != 3:
    print('Not a 3D field in the required form!')

hf.close()

#*********Calculate 1D Spectra along stramwise distance *********#

# define spectral analysis input

kmin = 2*pi/lx

kmax = 2*pi*NX/lx

k = np.arange(kmin,kmax,0.000001)

# Length scale (L)
L=40.0

# Spectrum intensity coefficent (alphaeps^2/3)
alphaeps = 0.02

# Theoretical Von Karman streamwise spectra component (x-direction)
Fx = 2*(9/55)*alphaeps*L**(5/3)*(1/(1+(L**2)*k**2)**(5.0/6.0))

# Theoretical Von Karman corssflow/vertical spectra components (y/z-directions)
Fyz = 2*(3/110)*alphaeps*L**(5/3)*((3+8*(L**2)*k**2)/(1+(L**2)*k**2)**(11.0/6.0))

# Fix a position Z (vertical distance from the wall), lets say at middle of box
ordinate_height = int((NZ/2)-1)

# Loop over all span wise coordinates for a fixed height and produce average of the fourier coefficents

psd_xy = []

for i in range(0,NY):

    fff_1,psd_1 = signal.periodogram(Vx[:,i,ordinate_height],2*pi/dx)
    psd_xy.append(psd_1)

psd_avg = (np.array(psd_xy).reshape(NY,fff_1.size).sum(axis=0))/NY

#********* Plot Spectra on log-log scale *********#
fig,ax = subplots()
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
ax.axis([0.005, 0.5, 1.e-2, 10])
ax.loglog(k,Fx,"g",linewidth=3,linestyle='--',label='Theoretical')
ax.loglog(fff_1[1:],psd_avg[1:],"b",linewidth=1,label='Simulated ')
ax.set_xlabel('$k_{1} [rad.m^{-1}$]',fontweight='bold',fontsize=16)
ax.set_ylabel('$E_{u}(k_{1})$ $[m^{2}/sec^{2}]$',fontweight='bold',fontsize=16)
ax.set_title('Stream Velocity Spectra', fontsize=16)
ax.grid(b=True, which='minor', color='b', linestyle='--',alpha=0.3)
ax.grid(b=True, which='major', color='m', linestyle='-',alpha=0.6)
ax.legend(loc='upper right',fontsize=16)
plt.xticks(fontweight='bold',fontsize=16)
plt.yticks(fontweight='bold',fontsize=16)
fig.tight_layout()
# plt.savefig("Mann_Spanwise.svg")
# plt.savefig('Mann_Spanwise.pdf') 
plt.show()

