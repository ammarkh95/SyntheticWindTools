######################################################################
"""
   Isotropic Turbulence Generation using FFT

"""
######################################################################
# Author: Ammar Khallouf
# Date: 23.03.2021
# Version: 1.0
######################################################################
""" *** Description ***

This script generates a 3D turbulent velocity field on a structured grid using FFT.

The generated synthetic field has a gaussian pdf and its energy matches a given power spectrum E(k).

The input power spectrum is normalized by a given amplitude 'sigmaV' with maximum wave number 'kmax'

"""
######################################################################
""" *** Important Notes ***

1. The 3D Vector field is generated on a structured cartesian grid with (Nx,Ny,Nz) number of grids. (Preferably Power of 2, e.g 32,64,128)

2. Boundary conditions are periodic in FFT.

3. The imagenary part of the generated field is discarded (only real part extracted)

4. For an even number of grid points (Nx,Ny,Nz), the Nyquist frequency arises at N/2 which makes the decomposed field complex (i.e imagenary part is present).

5. For a sufficently smooth field, the imagenary part caused by Nyquist frequency mode should be negligible.

"""
######################################################################
"""
*** References ***

1. https://www.io-warnemuende.de/tl_files/staff/burchard/pdf/Turb_Chap4_WS08.pdf

"""
######################################################################

import numpy as np
import matplotlib.pyplot as plt

# Input Parameters:

####### Grid Size and Resolution #######

NX,NY, NZ = 256,128,16 

DX,DY,DZ=  4,4,4

####### Kolmogrov Spectrum function #######

sigmaV = 1.  # velocity dispersion

kmax = 2. * np.pi / 400.  # where E(k) peaks

def Ek_kolm(k, kmax, sigmaV):
    """
    Turbulence power spectrum E(k) with -5/3 Kolmogorov spectrum and a low-k cutoff 
    k Ek normalisation given by sigmaV^2
    Ek peaks at kmax
    """
    kc = kmax / 0.6
    E_k = (k/kc)**(- 5./3) * np.exp(-1./(k / kc)) * (sigmaV**2 / 2.71) / kc
    return E_k

####### Von-Karman Spectrum function #######

alpha_eps_pow2_3 = 0.02  # scaling constant

L_scale = 1.  # integral length scale 

def Ek_vkarm(k, alpha_eps_pow2_3, L_scale):
    """
    Von Karman Turbulence power spectrum E(k) (1948)
    """
    E_k = alpha_eps_pow2_3*L_scale**(5/3)*((L_scale**(4)*k**(4))/(1+L_scale**(2)*k**(2))**(17/6))
    return E_k

####### Extract 3D Velocity Spectrum function #######

def vf_3d_spec(k, kmax, sigmaV):
    #! 4. * pi * k**3 * |vf_3d_spec(k)|**2  =  2. * k * Ek
    vf2 = 2. * k * Ek_kolm(k, kmax, sigmaV) / (4. * np.pi * k**3)
    return vf2**0.5

####### Extract Incompressible part of the velocity field #######

def Incomp(vin):
    '''
    Return solenoidal component of vin using Helmholtz decomposition.
    vin must be of shape (3,nx,ny,nz)

    Helmholtz decomposition:
    vin = vr + vd = (curl A) + div \phi

    Return: (curl A)
    '''

    def C(ar, n):
        return np.cos(2.*np.pi*ar/n)

    nd, nx, ny, nz = vin.shape
    l, m, n = np.mgrid[0:nx,0:ny,0:nz]
    Sl = np.sin(2.*np.pi*l/nx)
    Sm = np.sin(2.*np.pi*m/ny)
    Sn = np.sin(2.*np.pi*n/nz)

    # velocity in Fourier space
    Fvin = np.fft.fftn(vin, s=(nx,ny,nz))
    Fvinx, Fviny, Fvinz = Fvin

    # Laplace operator in Fourier space
    Aprime = 2.*(C(2.*l, nx) + C(2.*m, ny) + C(2.*n, nz) - 3.)

    Aprime[np.where(np.isclose(Aprime,0))] = np.inf

    # Rorational component of velocity in Fourier space
    Fvinr = 4.*np.array([Fvinx*(-Sn**2-Sm**2) + Fviny*Sl*Sm + Fvinz*Sl*Sn,\
                         Fvinx*Sl*Sm + Fviny*(-Sn**2-Sl**2) + Fvinz*Sm*Sn,\
                         Fvinx*Sl*Sn + Fviny*Sm*Sn + Fvinz*(-Sm**2-Sl**2)])/Aprime

    return np.nan_to_num(np.fft.ifftn(Fvinr, s=(nx,ny,nz)).real)

####### Generate  wave number space #######

kx = np.fft.fftfreq(NX,DX).reshape(NX, 1, 1) / DX * 2. * np.pi
ky = np.fft.fftfreq(NY,DY).reshape(NY, 1) / DY * 2. * np.pi
kz = np.fft.fftfreq(NZ,DZ) / DZ * 2. * np.pi
kmag = (kx**2 + ky**2 + kz**2)**0.5

# Avoid runtime warning of division by zero

kmag [0,0,0]=1e-8

####### Generate a 3D Turbulent Velocity field #######

v0 = []
for i in range(3):
    vx1 = np.random.randn(NX, NY, NZ)
    vx1 = (vx1 - vx1.mean()) / vx1.std()
    vx_f = np.fft.fftn(vx1)
    vx_f = vx_f * vf_3d_spec(kmag, kmax, sigmaV/3.**0.5) * \
        (2. * np.pi / DX)**(3./2)
    vx_f[0, 0, 0] = 0.
    vx = np.fft.ifftn(vx_f).real
    v0.append(vx)
v0 = np.array(v0).astype('float32')

####### Calculation Checks for generated velocity field #######

v0_tilde = np.array(
    [np.fft.fftn(v0[ii]) * (DX / (2. * np.pi))**3 for ii in range(3)])
ps = (np.abs(v0_tilde)**2).sum(axis=0)
kkk = (kmag.flatten())[1:]
ppp = (ps.flatten())[1:]
ksort = kkk[kkk.argsort()]
psort = ppp[kkk.argsort()]
NK = 50
kh = np.linspace(ksort[0]*1., ksort[-1], NK+1)
khc = ((kh[1:]**3 + kh[:-1]**3)/2.)**(1./3)  # **0.5
pph = np.array([(psort[np.where((ksort > kh[i]) & (ksort < kh[i+1]))]).mean()
                for i in range(kh.size-1)])

# Sampled Spectrum
Ek_ = pph * (2. * np.pi)**3 * (2. * np.pi * khc**2) / \
    DX**3 / NX / NY / NZ

# Input Spectrum
Ek_in = Ek_kolm(khc, kmax, sigmaV)

# print('v2/2:', v0.var() / 2.)
# print('Ek:', Ek_.mean() * np.diff(khc)[0] * NK)

# print
# print('v0.std:',  v0.std())
# print('sigma from 1d spectrum', ((Ek_[1:] * np.diff(khc)).sum() * 2.)**0.5)

# print(Ek_ / Ek_in)

####### Plot generated spectrum vs input spectrum #######

plt.figure()
plt.loglog(khc, Ek_, 'b-', label='Ek, sampled')
plt.loglog(khc, Ek_in, 'k--', label='Ek, input')
plt.xlabel('$k [kpc^{-1}]$', fontsize=18)
plt.ylabel('$E(k)$', fontsize=18)
plt.legend(loc=0)
plt.grid()
plt.show()
plt.figure()
plt.hist(v0.flatten(), 1000)
plt.xlabel('$v_0$', fontsize=16)


####### Continiuty Correction #######

v0_inc = Incomp(v0.reshape((3, NX, NY, NZ), order='F')).astype('float32')

####### Write generated velocity field to a file #######

# Save generated velocity field to one binary file in Fortran format

v0.reshape((3, NX, NY, NZ), order='F').tofile(
    'v0_xyz_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# Save generated velocity field components to multiple binary files in Fortran format

# v0[0].reshape((NX, NY, NZ), order='F').tofile(
#     'vx_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# v0[1].reshape((NX, NY, NZ), order='F').tofile(
#     'vy_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# v0[2].reshape((NX, NY, NZ), order='F').tofile(
#     'vz_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

####### Write incompressible velocity field to a file #######

# Save generated velocity field to one binary file in Fortran format

v0_inc.reshape((3, NX, NY, NZ), order='F').tofile('v0_inc_xyz_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# Save generated velocity field components to multiple binary files in Fortran format

# v0_inc[0].reshape((NX, NY, NZ), order='F').tofile('vx_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# v0_inc[1].reshape((NX, NY, NZ), order='F').tofile('vy_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# v0_inc[2].reshape((NX, NY, NZ), order='F').tofile('vz_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')