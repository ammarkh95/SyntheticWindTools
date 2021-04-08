
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

#******************************************************#

# Option (1) Read as multiple files in binary format

# # Number of grids in each direction NX,NY,NZ

# NX, NY, NZ = 128,128,128

# # Spacing of grids in each direction DX,DY,DZ

# DX, DY, DZ = 0.5,0.5,0.5

# # Read Velocity field components (Vx,Vy,Vz) from the corresponding 3 binary files

# Vx = np.fromfile('u.bin', dtype=np.float32).reshape(NX, NY, NZ)
# Vy = np.fromfile('v.bin', dtype=np.float32).reshape(NX, NY, NZ)
# Vz = np.fromfile('w.bin', dtype=np.float32).reshape(NX, NY, NZ)

# # Check for correct dimesnion

# if Vx.ndim != 3:
#     print('Not a 3D field in the required form!')

#******************************************************#

# Option (2) Read as a single file in binary format

# # Number of grids in each direction NX,NY,NZ

# NX, NY, NZ = 256,128,16

# # Spacing of grids in each direction DX,DY,DZ

# DX, DY, DZ = 4,4,4

# # Read Velocity field components (Vx,Vy,Vz) accordingly from one binary file

# U = np.fromfile('v0_xyz_256_128_16.dat', dtype=np.float32).reshape(3, NX, NY, NZ)

# # Check for correct dimesnion

# if U.shape[0] != 3:
#     print('Not a 3D field in required form!')

# # Extract 3D Vector field components Vx,Vy,Vz

# Vx,Vy,Vz = U[0],U[1],U[2]

#******************************************************#

# Option (3) Read as h5 file format

# Read Velocity field components (Vx,Vy,Vz) accordingly from h5 datasets ('u','v','w')

hf = h5py.File('windgen.h5', 'r')

Vx= np.array(hf.get('u')) 
Vy= np.array(hf.get('v')) 
Vz= np.array(hf.get('w')) 

# Number of grids in each direction NX,NY,NZ

NX,NY,NZ= Vx.shape

# Spacing of grids in each direction DX,DY,DZ

DX, DY, DZ = 0.5,0.5,0.5

# Check for correct dimesnion

if Vx.ndim != 3:
    print('Not a 3D field in the required form!')

hf.close()

#*********Perfrom multi-dimensional FFT analysis*********#

print(' ! Helmholtz Decomposition Process Started !\n')

# FFT of vector field to spectral space
Vx_spec = np.fft.fftn(Vx)
Vy_spec = np.fft.fftn(Vy)
Vz_spec = np.fft.fftn(Vz)

# Wave number space components
kx = np.fft.fftfreq(NX,DX)[:,None,None]
ky = np.fft.fftfreq(NY,DY)[None,:,None]
kz = np.fft.fftfreq(NZ,DZ)[None,None,:]

# Calculate magnitudes of wave number vectors
k_mag = kx**2 + ky**2 + kz**2

# Set the first component to 1 to avoid division by zero. (i.e discard k=0 component)
k_mag[0,0,0] = 1.

# Calculate the scalar field (phi) in spectral space whose potential represent the compressible part
phi_spec = (Vx_spec * kx + Vy_spec * ky + Vz_spec * kz) / k_mag  # * 1j

# Extract the compressible vector field components in physical space (i.e IFFT of scalar potential)
V_compressible_x = np.fft.ifftn(phi_spec * kx)
V_compressible_y = np.fft.ifftn(phi_spec * ky)
V_compressible_z = np.fft.ifftn(phi_spec * kz)

# Extract the incompressible vector field components in physical space (i.e Subtract the compressible part from original field)
V_incompressible_x = Vx - V_compressible_x
V_incompressible_y = Vy - V_compressible_y
V_incompressible_z = Vz - V_compressible_z

# *********Calculation checks*********

# Check the max divergence value for the original vector field.
div_original = np.fft.ifftn((np.fft.fftn(Vx) * kx + np.fft.fftn(Vy)
                             * ky + np.fft.fftn(Vz) * kz) * 1j * 2. * np.pi)

# Check the max divergence value for the extracted incompressible vector field.
div_incompressible = np.fft.ifftn((np.fft.fftn(V_incompressible_x) * kx + np.fft.fftn(V_incompressible_y)
                                   * ky + np.fft.fftn(V_incompressible_z) * kz) * 1j * 2. * np.pi)

print(' ! Helmholtz Decomposition Performed Successfully !\n')

print('##### Calculation Checks #####\n')

print('Max Divergence for Original Velocity:', abs(div_original).max())

print('Max Divergence for Corrected Velocity :', abs(div_incompressible).max())

# Check the power in Incompressible and Compressible components
# print('variance:')

# print('Incompressible x,y,z:', V_incompressible_x.var(),
#       V_incompressible_y.var(), V_incompressible_z.var())

# print('Compressible x,y,z:', V_compressible_x.var(),
#       V_compressible_y.var(), V_compressible_z.var())

#*********Post-Processing and Data Output*********#

# Create Meshing grids for plots

X = np.arange(0, NX*DX,DX)
Y = np.arange(0, NY*DY,DY)
Z = np.arange(0, NZ*DZ,DZ)

# ######## Stramlines plot at a slice ########
# plt.figure()
# plt.title('J.Mann Velocity Streamlines')
# plt.grid()
# plt.streamplot(Y, X, Vx[:,:,0].real, Vz[:,:,0].real,
#                density=2, linewidth=None, color='#A13BEC')
# plt.figure()
# plt.title('Incompressible Velocity Streamlines')
# plt.grid()
# plt.streamplot(Y, X, V_incompressible_x[:,:,0].real, V_incompressible_z[:,:,0].real, density=2,
#                linewidth=None, color='#A30000')
# plt.figure()
# plt.streamplot(Y, X, V_compressible_x[:,:,0].real, V_compressible_z[:,:,0].real, density=2,
#                linewidth=None, color='#00A352')
# plt.title('Compressible Velocity Streamlines')
# plt.grid()
# plt.pause(10)
# plt.show()

######## Vector magnitudes evaluation at a slice ########
Vf_mag = np.sqrt(Vx[0, :, :]**2 + Vy[0, :, :]**2 + Vz[0, :, :]**2)

Vsel_mag = np.sqrt(V_incompressible_x[0, :, :]**2 +
                   V_incompressible_y[0, :, :]**2 + V_incompressible_z[0, :, :]**2)

Vcomp_mag = np.sqrt(V_compressible_x[0, :, :] ** 2 +
                    V_compressible_y[0, :, :]**2 + V_compressible_z[0, :, :]**2)

# print('one check:', np.min(Vx[0,:,:]))

######## Contour plots at a slice in XY Plane ########
plt.figure(figsize=(8,3))
plt.contourf(X, Y, np.rot90(Vx[:,:,0],3), 600, cmap='jet')
plt.colorbar()
plt.title('J.Mann Velocity Compment (X)')
plt.grid()
plt.gca().set_aspect("equal")
plt.tight_layout()

plt.figure(figsize=(8,3))
plt.contourf(X, Y, np.rot90(V_incompressible_x[:,:,0].real,3), 600, cmap='jet')
plt.colorbar()
plt.title('Incompressible Velocity Compment(X)')
plt.grid()
plt.gca().set_aspect("equal")
plt.tight_layout()

plt.figure(figsize=(8,3))
plt.contourf(X, Y, np.rot90(V_compressible_x[:,:,0].real,3), 600, cmap='jet')
plt.colorbar()
plt.title('Compressible Velocity Compment (X)')
plt.grid()
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()

######## Write the incompressible vector field components in binary format ########

# V_incompressible_x.real.astype('float32').tofile(
#     'vx_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# V_incompressible_y.real.astype('float32').tofile(
#     'vy_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')

# V_incompressible_z.real.astype('float32').tofile(
#     'vz_inc_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')
# print('Binary files written successfully !')

######## Write the incompressible vector field in h5 format ########
hfw = h5py.File('windgen_incompressible.h5', 'w')
hfw.create_dataset('u',data=V_incompressible_x.real)
hfw.create_dataset('v',data=V_incompressible_y.real)
hfw.create_dataset('w',data=V_incompressible_z.real)
hfw.close()
print('HDF5 file written successfully !')