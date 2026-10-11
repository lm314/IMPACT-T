! D1.12 check on the CPU (Mac or Sol): kickpt2ptBd_BeamBunch for two particles side
! by side in x at gamma ~ 57 in the bend frame, no dipole field (Param(3) = 0),
! reference angle 0.  The transverse kick must equal the analytic
! e/(4 pi eps0 r^2 gamma) * c Dt dtless / (m c^2/e), to 1e-10 (measured 1.7e-11, the code's constants differ slightly from the analytic ones).  Also checks that
! the particle with z < 0 is a source and a target (bend-frame z starts at the entrance).
! Link like test_pt2ptfields (README): all objects of build-mpi except main/AccSimulator/Input/Output.
program test_bendnbody_cpu
  use BeamBunchclass
  use BeamLineElemclass
  use Dipoleclass
  use Dataclass
  use NumConstclass
  use PhysConstclass
  use Timerclass
  implicit none
  include 'mpif.h'
  type (BeamBunch) :: b
  type (Dipole), target :: dp
  type (BeamLineElem) :: be(1)
  type (fielddata) :: fld(Maxoverlap)
  integer :: idr(3,1), ierr, i, nfail
  double precision :: gam, r, rays(9,2), ezw(4), dt, expd, got, qe, e0
  nfail = 0
  call MPI_INIT(ierr)
  call constructT_PhysConst(2.5d-13,9.3d9)
  call construct_Timer(0.0d0)
  do i = 1, Maxoverlap
    call initt_Data(fld(i))
  enddo
  call construct_Dipole(dp,4,20,4,0.3d0)
  call setparam_Dipole(dp,3,0.0d0)
  call setparam_Dipole(dp,5,0.0125d0)
  be(1) = assign_BeamLineElem(dp)
  idr(:,1) = (/4,15,1/)
  qe = 1.602176634d-19
  e0 = Epsilon0   ! the code's own value (8.854187817e-12)
  gam = 57.0d0
  r = 1.0d-6
  dt = 1.0d0
  rays = 0.0d0
  ! two particles at x = +-r/2, one with z < 0
  rays(1,1) = -0.5d0*r/Scxlt
  rays(1,2) =  0.5d0*r/Scxlt
  rays(5,1) = -2.0d-6/Scxlt
  rays(5,2) = -2.0d-6/Scxlt
  rays(6,:) = sqrt(gam**2-1.0d0)
  rays(7,:) = -1.0d0/0.511005d6
  rays(8,:) = 1.0d0
  ezw = 0.0d0
  call construct_BeamBunch(b,1.0d-5,1.0d6,0.511005d6,-1.0d0,2,0.0d0)
  call kickpt2ptBd_BeamBunch(2,rays,1.0d-10,-1.0d0,0.511005d6,dt,be,idr,1,fld, &
                             1.0d0,0.0d0,2.0d0*qe,1.0d-12,2,2,0,ezw,4,0.0d0,1.0d0)
  expd = qe/(4.0d0*Pi*e0*r*r*gam)*Clight*Scxlt/Clight/0.511005d6*dt
  ! (c Dt) = Scxlt
  expd = qe/(4.0d0*Pi*e0*r*r*gam)*Scxlt/0.511005d6*dt
  got = rays(2,2)
  write(*,'(a,2es22.14,a,es10.3)') "dpx particle at +x: got, expected ",got,expd,"  rel diff ",abs(got-expd)/expd
  if(abs(got-expd)/expd.gt.1.0d-10) nfail = nfail + 1
  got = -rays(2,1)
  if(abs(got-expd)/expd.gt.1.0d-10) nfail = nfail + 1
  write(*,'(a,es10.3)') "mirror particle rel diff ",abs(got-expd)/expd
  if(nfail.eq.0) then
    print*,"test_bendnbody_cpu: PASS"
  else
    print*,"test_bendnbody_cpu: FAIL"
  endif
  call MPI_FINALIZE(ierr)
end program test_bendnbody_cpu
