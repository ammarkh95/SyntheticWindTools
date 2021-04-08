import numpy as np
import matplotlib.pyplot as plt

def curl(vin):
    '''
    Return curl of vin. The calculation is done in Fourier space.
    vin must be of shape (3,nx,ny,nz).
    '''
    nd, nx, ny, nz = vin.shape
    l, m, n = np.mgrid[0:nx,0:ny,0:nz]
    Sl = np.sin(2.*np.pi*l/nx)
    Sm = np.sin(2.*np.pi*m/ny)
    Sn = np.sin(2.*np.pi*n/nz)

    Fvin = np.fft.fftn(vin, s=(nx,ny,nz))

    zero_arr = np.zeros((nx, ny, nz))

    # curl matrix in Fourier space
    Fcurl = 2j*np.array([[zero_arr,-Sn,Sm],\
                          [Sn,zero_arr,-Sl],\
                          [-Sm,Sl,zero_arr]])

    Fcurlv = np.einsum('ij...,j...->i...', Fcurl, Fvin)/2.

    return np.nan_to_num(np.fft.ifftn(Fcurlv, s=(nx,ny,nz)))


def solenoidal(vin):
    '''
    Perform Helmholtz decomposition on vin. Return solenoidal component. Needs numpy.fft.
    vin must be of shape (3,nx,ny,nz).

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


# Number of grids in each direction NX,NY,NZ

NX, NY, NZ = 64,32,32

# Option 1 Read Velocity field components (Vx,Vy,Vz) accordingly from one binary file

# U = np.fromfile('v0_xyz_256_128_16.dat', dtype=np.float32).reshape(3, NX, NY, NZ)

# Option 2 Read Velocity field components (Vx,Vy,Vz) from the corresponding 3 binary files

U_x = np.fromfile('test_u.bin', dtype=np.float32).reshape(NX, NY, NZ)
U_y = np.fromfile('test_v.bin', dtype=np.float32).reshape(NX, NY, NZ)
U_z = np.fromfile('test_w.bin', dtype=np.float32).reshape(NX, NY, NZ)

U= np.array([U_x,U_y,U_z])

U_divfree = solenoidal(U).astype('float32')

# Write 3D Incompressible Vector field components Vx,Vy,Vz

U_divfree.reshape((3, NX, NY, NZ), order='F').tofile(
    'v0_inc_xyz_'+str(NX)+'_'+str(NY)+'_'+str(NZ)+'.dat')