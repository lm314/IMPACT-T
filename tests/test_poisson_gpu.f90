! B1 test: openBC3D_gpu against openBC3Dnew (FFTW, 1 rank), and rank
! independence of the GPU solve.  Not part of the CMake build; see
! tests/README.md for how to build and run.
!
!   test_poisson_gpu NX NY NZ [NPY NPZ [DUMPFILE]]
!
! NPY*NPZ must equal the number of MPI ranks (default 1x1).
! With 1 rank: compares phi and phi_img with openBC3Dnew for three bunch
! positions, checks rhoImg == 0 for flagImage=0, and times both solvers
! (set SKIPCPU=1 to skip openBC3Dnew, e.g. at 512x512x64).
! With DUMPFILE: also writes the gathered phi, phi_img (zmin = NZ*hz) as
! raw binary on rank 0, to be compared byte-for-byte across rank counts.
program test_poisson_gpu
  use FieldQuantclass
  use PoissonGPUclass
  implicit none
  include 'mpif.h'
  integer :: nx,ny,nz,npy,npz,myid,nproc,ierr,myidy,myidz
  character(len=256) :: arg,dumpfile
  character(len=8) :: skip
  integer :: i,j,k,iz,r,n,stat
  integer, allocatable :: pytable(:),pztable(:),ypzstable(:),xpystable(:)
  integer :: y0,z0,inny,innz
  double precision, allocatable :: rho(:,:,:),rimg(:,:,:),rho0(:,:,:)
  double precision, allocatable :: rhoc(:,:,:),rimgc(:,:,:),rimgb(:,:,:)
  double precision :: hx,hy,hz,zmin,zshift,t0,tgpu,tcpu,e1,e2
  double precision, parameter :: zfac(3) = (/0.0d0, 1.0d0, 1.0d4/)
  character(len=10) :: zname(3) = (/"cathode   ","near      ","far       "/)
  logical :: ok, docpu

  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
  call get_command_argument(1,arg); read(arg,*) nx
  call get_command_argument(2,arg); read(arg,*) ny
  call get_command_argument(3,arg); read(arg,*) nz
  npy = 1; npz = 1; dumpfile = ""
  call get_command_argument(4,arg,status=stat)
  if(stat.eq.0 .and. len_trim(arg).gt.0) read(arg,*) npy
  call get_command_argument(5,arg,status=stat)
  if(stat.eq.0 .and. len_trim(arg).gt.0) read(arg,*) npz
  call get_command_argument(6,dumpfile,status=stat)
  if(stat.ne.0) dumpfile = ""
  call get_environment_variable("SKIPCPU",skip)
  docpu = (nproc.eq.1 .and. trim(skip).ne."1")
  if(npy*npz.ne.nproc) then
    if(myid.eq.0) print*,"FAIL: npy*npz must equal the number of ranks"
    call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
  endif
  myidy = mod(myid,npy)
  myidz = myid/npy

  hx = 1.1d-4; hy = 0.9d-4; hz = 1.7d-4
  allocate(pytable(0:npy-1),pztable(0:npz-1),ypzstable(0:npz-1),xpystable(0:npy-1))
  do i = 0, npy-1
    pytable(i) = ny/npy
    if(i.lt.mod(ny,npy)) pytable(i) = pytable(i)+1
  enddo
  do i = 0, npz-1
    pztable(i) = nz/npz
    if(i.lt.mod(nz,npz)) pztable(i) = pztable(i)+1
  enddo
  y0 = sum(pytable(0:myidy-1))
  z0 = sum(pztable(0:myidz-1))
  inny = pytable(myidy)
  innz = pztable(myidz)
  ! +1 is from the real to complex fft; only used by openBC3Dnew (1 rank).
  xpystable(0) = nx+1
  ypzstable(0) = 2*ny

  allocate(rho(nx,inny,innz),rimg(nx,inny,innz),rho0(nx,inny,innz))
  do k = 1, innz
    do j = 1, inny
      do i = 1, nx
        rho0(i,j,k) = rhofun(i,y0+j,z0+k)
      enddo
    enddo
  enddo

  ok = .true.
  if(docpu) then
    allocate(rhoc(nx,ny,nz),rimgc(nx,ny,nz),rimgb(nx,ny,nz))
    if(myid.eq.0) print'(a,3i5)',"grid",nx,ny,nz
    do iz = 1, 3
      zmin = zfac(iz)*nz*hz
      if(iz.eq.3) zmin = 1.0d4*hz
      zshift = -(2*zmin+(nz-1)*hz)
      rho = rho0
      call MPI_BARRIER(MPI_COMM_WORLD,ierr)
      t0 = MPI_WTIME()
      call openBC3D_gpu(nx,ny,nz,rho,hx,hy,hz,0,0,1,1,MPI_COMM_WORLD, &
           pztable,pytable,nx,ny,nz,rimg,1,zshift)
      tgpu = MPI_WTIME()-t0
      rhoc = rho0
      t0 = MPI_WTIME()
      call openBC3Dnew(nx,ny,nz,rhoc,hx,hy,hz,nx+1,2*ny,0,0,1,1, &
           MPI_COMM_WORLD,MPI_COMM_WORLD,MPI_COMM_WORLD, &
           pztable,pytable,ypzstable,xpystable,nx,ny,nz,rimgc,1,zshift)
      tcpu = MPI_WTIME()-t0
      e1 = maxval(abs(rho-rhoc))/maxval(abs(rhoc))
      e2 = maxval(abs(rimg-rimgc))/maxval(abs(rimgc))
      print'(a,a,a,es10.3,a,es10.3,a,2f10.3)',"zmin=",zname(iz)," phi err",e1, &
           "  phi_img err",e2,"  t_gpu,t_cpu(s)",tgpu,tcpu
      ! phi' must match to 1e-12.  The image potential's integrated Green's
      ! function is an 8-corner difference of large antiderivatives, so its
      ! rounding noise grows with (distance/cell)^3 on CPU and GPU alike:
      ! require 1e-11 at the cathode, 2e-10 near it, and at 'far' that the
      ! GPU is no further from a brute-force point-kernel sum than the CPU
      ! is (only for grids small enough to sum directly).
      if(e1.gt.1.0d-12) ok = .false.
      if(iz.eq.1 .and. e2.gt.1.0d-11) ok = .false.
      if(iz.eq.2 .and. e2.gt.2.0d-10) ok = .false.
      if(iz.gt.1 .and. nx*ny*nz.le.40000) then
        call brute_img(zmin,rimgb)
        e1 = maxval(abs(rimg-rimgb))/maxval(abs(rimgb))
        e2 = maxval(abs(rimgc-rimgb))/maxval(abs(rimgb))
        print'(a,2es10.3)',"      vs brute-force point-kernel image: gpu, cpu =",e1,e2
        if(iz.eq.3 .and. e1.gt.2.0d0*e2) ok = .false.
      endif
    enddo
    ! flagImage = 0 must give exactly 0
    rho = rho0
    call openBC3D_gpu(nx,ny,nz,rho,hx,hy,hz,0,0,1,1,MPI_COMM_WORLD, &
         pztable,pytable,nx,ny,nz,rimg,0,zshift)
    if(any(rimg.ne.0.0d0)) then
      ok = .false.
      print*,"FAIL: rhoImg /= 0 for flagImage=0"
    else
      print*,"flagImage=0: rhoImg exactly 0"
    endif
    ! the GPU solve is repeatable (plan state is reused on 2nd call)
    rho = rho0
    call openBC3D_gpu(nx,ny,nz,rho,hx,hy,hz,0,0,1,1,MPI_COMM_WORLD, &
         pztable,pytable,nx,ny,nz,rimg,1,zshift)
  endif

  ! timing (GPU only), image on, zmin = nz*hz
  zmin = nz*hz
  zshift = -(2*zmin+(nz-1)*hz)
  rho = rho0
  call openBC3D_gpu(nx,inny,innz,rho,hx,hy,hz,myidz,myidy,npz,npy,MPI_COMM_WORLD, &
       pztable,pytable,nx,ny,nz,rimg,1,zshift)
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  t0 = MPI_WTIME()
  n = 3
  do r = 1, n
    rho = rho0
    call openBC3D_gpu(nx,inny,innz,rho,hx,hy,hz,myidz,myidy,npz,npy,MPI_COMM_WORLD, &
         pztable,pytable,nx,ny,nz,rimg,1,zshift)
  enddo
  tgpu = (MPI_WTIME()-t0)/n
  if(myid.eq.0) print'(a,f10.4)',"GPU time per call, image on (s):",tgpu

  if(len_trim(dumpfile).gt.0) call dump_global()

  if(myid.eq.0) then
    if(docpu) then
      if(ok) then
        print*,"RESULT: PASS"
      else
        print*,"RESULT: FAIL"
      endif
    endif
  endif
  call MPI_FINALIZE(ierr)

contains

  ! Image potential by direct summation with the point kernel 1/(4 pi r)
  ! (differs from the integrated Green's function by O((h/r)^2)).
  subroutine brute_img(zmn,out)
    double precision, intent(in) :: zmn
    double precision, intent(out) :: out(nx,ny,nz)
    integer :: i,j,k,a,b,c
    double precision :: dx,dy,dz,acc
    double precision, parameter :: pi = 3.14159265358979323846d0
    do k = 1, nz
    do j = 1, ny
    do i = 1, nx
      acc = 0.0d0
      do c = 1, nz
      do b = 1, ny
      do a = 1, nx
        dx = (i-a)*hx; dy = (j-b)*hy
        dz = 2*zmn + (k+c-2)*hz
        acc = acc + rho0(a,b,c)/(4*pi*sqrt(dx*dx+dy*dy+dz*dz))
      enddo
      enddo
      enddo
      out(i,j,k) = -acc
    enddo
    enddo
    enddo
  end subroutine brute_img

  double precision function rhofun(i,j,k)
    integer, intent(in) :: i,j,k
    double precision :: gx,gy,gz
    integer(8) :: h
    gx = (i-0.4d0*nx)/(0.15d0*nx)
    gy = (j-0.6d0*ny)/(0.2d0*ny)
    gz = (k-0.5d0*nz)/(0.1d0*nz)
    h = mod(int(i,8)*73856093_8 + int(j,8)*19349663_8 + int(k,8)*83492791_8, 1000003_8)
    rhofun = exp(-0.5d0*(gx*gx+gy*gy+gz*gz)) + 0.1d0*dble(h)/1000003.0d0
  end function rhofun

  subroutine dump_global()
    double precision, allocatable :: buf(:),gall(:),glb(:,:,:,:)
    integer, allocatable :: info(:),allinfo(:),counts(:),displs(:)
    integer :: rr,ofs,m,ny0,nz0,yy0,zz0
    allocate(info(4),allinfo(4*nproc),counts(nproc),displs(nproc))
    info = (/y0,inny,z0,innz/)
    call MPI_ALLGATHER(info,4,MPI_INTEGER,allinfo,4,MPI_INTEGER,MPI_COMM_WORLD,ierr)
    ofs = 0
    do rr = 1, nproc
      counts(rr) = 2*nx*allinfo(4*(rr-1)+2)*allinfo(4*(rr-1)+4)
      displs(rr) = ofs
      ofs = ofs + counts(rr)
    enddo
    allocate(buf(2*nx*inny*innz),gall(ofs))
    m = nx*inny*innz
    buf(1:m) = reshape(rho,(/m/))
    buf(m+1:2*m) = reshape(rimg,(/m/))
    call MPI_GATHERV(buf,2*m,MPI_DOUBLE_PRECISION,gall,counts,displs, &
         MPI_DOUBLE_PRECISION,0,MPI_COMM_WORLD,ierr)
    if(myid.eq.0) then
      allocate(glb(nx,ny,nz,2))
      do rr = 1, nproc
        yy0 = allinfo(4*(rr-1)+1); ny0 = allinfo(4*(rr-1)+2)
        zz0 = allinfo(4*(rr-1)+3); nz0 = allinfo(4*(rr-1)+4)
        m = nx*ny0*nz0
        glb(:,yy0+1:yy0+ny0,zz0+1:zz0+nz0,1) = &
             reshape(gall(displs(rr)+1:displs(rr)+m),(/nx,ny0,nz0/))
        glb(:,yy0+1:yy0+ny0,zz0+1:zz0+nz0,2) = &
             reshape(gall(displs(rr)+m+1:displs(rr)+2*m),(/nx,ny0,nz0/))
      enddo
      open(20,file=trim(dumpfile),form="unformatted",access="stream",status="replace")
      write(20) glb
      close(20)
      print*,"wrote ",trim(dumpfile)
    endif
  end subroutine dump_global

end program test_poisson_gpu
