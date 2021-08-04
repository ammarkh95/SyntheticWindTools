C############################################################################
C Chair of Structural Analysis - TUM (Statik)
C Date: 15.07.21
C Contributor: Ammar Khallouf
C Version: 1.0
C############################################################################

C Description:

C This small program converts generated binary files from "TurbGen" to Plot3d visualization format (ASCII files)
C This allows to visualize the gnerated turbulence box in post-processinng softwares like "Paraview" 

C The input parameters read directly from the command line are given in the following order:

C NX,NY,NZ: No. of grids along each direction
C LX,LY,LZ: Dimensions of the turbulence box (i.e side length along each direction) 
C fileU.bin, fileV.bin, fileW.bin: output binary files from "TurbGen" for the velocity components (u,v,w)

C Example input from the command line:

C For a turbulence box with (128 x 32 x 32) grids and a domain size (LX=256 m, LY=64 m, LZ= 64 m) 
C with generated turbulence files:  sim-u.bin, sim-v.bin, sim-w.bin
C The call should be like this:

C     TurbView 128 32 32 256 64 64 sim-u.bin sim-v.bin sim-w.bin

!############################################################################

      program TurbView
      implicit none
      integer::i,j,k,recnr,h,l,recordlength,ifile
      integer::nt,nx,ny,nz,ndata
      real*4::dummy
      integer,parameter::idp=kind(1d0),idpo=4
      real(idp),allocatable,dimension(:,:,:)::x,y,z,u,v,w
      integer ifirst,ilast,namelenin,namelenout
      integer,parameter::wordlen=1024  
      character(len=wordlen)::infile,text
      character(len=wordlen),dimension(3)::infiles
      integer,dimension(3)::n,ifileslen
      real(idp),dimension(3)::Lbox,Delta,Shift

C---- User input from command line -------------------------------------
      infiles=''
      do i=1,3
        call getarg(i,text);read(text,*)n(i)
      enddo
      do i=1,3
        call getarg(3+i,text);read(text,*)Lbox(i)
      enddo
      do i=1,3
        call getarg(6+i,text);read(text,*)infiles(i)
        ifileslen(i)=Len_Trim(infiles(i))
      enddo


      write(*,10)' n1=',n(1),'  n2=',n(2),'  n3=',n(3)

   10 format(a,i10,a,i10,a,i10)

      allocate(x(n(1),n(2),n(3)),
     &         y(n(1),n(2),n(3)),
     &         z(n(1),n(2),n(3)),
     &         u(n(1),n(2),n(3)),
     &         v(n(1),n(2),n(3)),
     &         w(n(1),n(2),n(3)))
     
C---- Read file --------------------------------------------------------

      print*,' Reading files : ',infiles(1)(1:ifileslen(1))
      open(unit=1,file=infiles(1)(1:ifileslen(1)),
     &     access='direct',form='unformatted',recl=1)
      do i=1,n(1);do j=1,n(2); do k=1,n(3);
        recnr=k+(j-1)*n(3)+(i-1)*n(2)*n(3)
        read(1,rec=recnr)dummy
        u(i,j,k)=dble(dummy)
      enddo;enddo;enddo
      close(1)

      print*,' Reading files : ',infiles(2)(1:ifileslen(2))
      open(unit=1,file=infiles(2)(1:ifileslen(2)),
     &     access='direct',form='unformatted',recl=1)
      do i=1,n(1);do j=1,n(2); do k=1,n(3);
        recnr=k+(j-1)*n(3)+(i-1)*n(2)*n(3)
        read(1,rec=recnr)dummy
        v(i,j,k)=dble(dummy)
      enddo;enddo;enddo
      close(1)

      print*,' Reading files : ',infiles(3)(1:ifileslen(3))
      open(unit=1,file=infiles(3)(1:ifileslen(3)),
     &     access='direct',form='unformatted',recl=1)
      do i=1,n(1);do j=1,n(2);do k=1,n(3);
        recnr=k+(j-1)*n(3)+(i-1)*n(2)*n(3)
        read(1,rec=recnr)dummy
        w(i,j,k)=dble(dummy)
      enddo;enddo;enddo
      close(1)
      print*,' Finished reading the input files '

      do k=1,n(3)
      do j=1,n(2)
      do i=1,n(1)
        x(i,j,k)=Lbox(1)/(n(1)-1)*(i-1)
      enddo;enddo;enddo
      do k=1,n(3)
      do j=1,n(2)
      do i=1,n(1)
        y(i,j,k)=Lbox(2)/(n(2)-1)*(j-1)
      enddo;enddo;enddo
      do k=1,n(3)
      do j=1,n(2)
      do i=1,n(1)
        z(i,j,k)=Lbox(3)/(n(3)-1)*(k-1)
      enddo;enddo;enddo

      open(unit=10,form='unformatted',file='Turb_Box.xyz')
      write(10)1
      write(10)n(1),n(2),n(3)
      write(10)
     &   (((real(x(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3)),
     &   (((real(y(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3)),
     &   (((real(z(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3))
      close(10)

      
      open(unit=10,form='formatted',file='Turb_Box.nam')
      write(10,*)'u-velocity ; velocity'
      write(10,*)'v-velocity'
      write(10,*)'w-velocity'
      close(10)

      open(unit=10,form='unformatted',file='Turb_Box.f')
      write(10)1
      write(10)n(1),n(2),n(3),3
      write(10)
     &   (((real(u(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3)),
     &   (((real(v(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3)),
     &   (((real(w(i,j,k),idpo),i=1,n(1)),j=1,n(2)),k=1,n(3))
      close(10)
      print*,' Finished writing Plot3D format files'

      stop 
      end
           
