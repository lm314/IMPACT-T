! X2 (b) test: the GPU field kind 6 (type 101, DTL) against getfldt_DTL (1 rank,
! 1 GPU).  Not part of the CMake build; see tests/README.md.
!
! Run in a directory holding the field files rfdata<ID> of the cells to test
! (Fourier coefficients, one number a line; the nine Ex1 files rfdata852 ...
! rfdata981 of the IMPACT-Z port).  The IDs and the cell lengths below are
! Ex1's.  For every cell, 5 times across an RF period, 1e5 random points in
! the cell (radius up to the 6.5 mm aperture, plus points on the two end
! planes), device against host, per component, scaled by that component's
! maximum.  Also the case of an element with a missing slot upload (the
! table builder must refuse), and the field-free zone of bgrad.
module tdtl_kernels
  use cudafor
  use FieldmapGPUclass
contains
  attributes(global) subroutine k_eval(x,y,z,n,res)
    implicit none
    integer, value :: n
    real(8), device :: x(n),y(n),z(n),res(6,n)
    real(8) :: ef(6)
    integer :: i
    i = (blockIdx%x-1)*blockDim%x + threadIdx%x
    if(i.gt.n) return
    call extfld_dev(x(i),y(i),z(i),1,ef,i)
    res(1,i) = ef(1)
    res(2,i) = ef(2)
    res(3,i) = ef(3)
    res(4,i) = ef(4)
    res(5,i) = ef(5)
    res(6,i) = ef(6)
  end subroutine k_eval
end module tdtl_kernels

program test_dtl_gpu
  use cudafor
  use BeamLineElemclass
  use DTLclass
  use Dataclass
  use NumConstclass
  use PhysConstclass
  use Timerclass
  use FieldmapGPUclass
  use tdtl_kernels
  implicit none
  include 'mpif.h'

  integer, parameter :: ncell = 9, n = 100000, nt = 5
  integer :: ids(ncell) = (/852,868,884,900,916,932,949,965,981/)
  double precision :: clen(ncell) = (/7.886300d-02,8.034760d-02,8.183570d-02, &
      8.332720d-02,8.482240d-02,8.631960d-02,8.781910d-02,8.932100d-02,9.082460d-02/)
  double precision :: q1(ncell) = (/2.049580d-02,2.049580d-02,2.049580d-02, &
      2.141160d-02,2.141160d-02,2.141160d-02,2.141160d-02,2.141160d-02,2.141160d-02/)
  double precision :: g1(ncell) = (/6.804960d+01,-6.720420d+01,6.635880d+01, &
      -6.347630d+01,6.269390d+01,-6.195370d+01,6.121340d+01,-6.051670d+01,5.982000d+01/)
  double precision :: g2(ncell) = (/-6.720420d+01,6.635880d+01,-6.347630d+01, &
      6.269390d+01,-6.195370d+01,6.121340d+01,-6.051670d+01,5.982000d+01,-5.916420d+01/)
  type (DTL), target :: dt(ncell)
  type (fielddata) :: fld(Maxoverlap)
  type (BeamLineElem) :: be(1)
  double precision :: zbe(2,1), prm(25), pos(4), ef(6), tt, zedge, worst, worst_all
  double precision :: dmax, smax
  double precision, allocatable :: x(:),y(:),z(:),hr(:,:),cr(:,:)
  real(8), device, allocatable :: xd(:),yd(:),zd(:),rd(:,:)
  integer :: idr(3,1), ic, it, i, c, ierr, nfail, seed, slot
  logical :: ok
  character(len=80) :: why

  call MPI_INIT(ierr)
  nfail = 0
  call constructT_PhysConst(2.5d-13,3.24d8)
  call construct_Timer(0.0d0)
  do i = 1, Maxoverlap
    call initt_Data(fld(i))
  enddo
  allocate(x(n),y(n),z(n),hr(6,n),cr(6,n),xd(n),yd(n),zd(n),rd(6,n))
  seed = 4711
  worst_all = 0.0d0
  zedge = 2.50337d0
  do ic = 1, ncell
    slot = mod(ic-1,Maxoverlap)+1
    call read1t_Data(fld(slot),ids(ic))
    call construct_DTL(dt(ic),20,20,101,clen(ic))
    prm = 0.0d0
    prm(1:10) = (/zedge,2503370.0d0,3.24d8,30.0d0+7.0d0*ic,dble(ids(ic)),6.5d-3, &
            q1(ic),g1(ic),q1(ic),g2(ic)/)
    call setparam_DTL(dt(ic),prm)
    be(1) = assign_BeamLineElem(dt(ic))
    zbe(1,1) = zedge
    zbe(2,1) = zedge + clen(ic)
    idr(:,1) = (/101,ids(ic),slot/)
    ! 1: without the upload the table must be refused
    call build_active_table(be,zbe,idr,fld,1,1,0.0d0,ok,why)
    if(ok) then
      print*,"FAIL: cell ",ids(ic),": table built with no upload"
      nfail = nfail + 1
    endif
    call fldslot_upload(fld,slot,5)
    do i = 1, n
      x(i) = 6.5d-3*sqrt(nrand())*(2.0d0*nrand()-1.0d0)
      y(i) = 6.5d-3*sqrt(nrand())*(2.0d0*nrand()-1.0d0)
      z(i) = zedge + clen(ic)*nrand()
    enddo
    x(1) = 0.0d0; y(1) = 0.0d0; z(1) = zbe(1,1)
    x(2) = 1.0d-3; y(2) = -2.0d-3; z(2) = zbe(2,1)
    x(3) = 0.0d0; y(3) = 0.0d0; z(3) = zedge + 0.5d0*clen(ic)
    ! points either side of the two gradient zones' edges
    z(4) = zedge + q1(ic)*(1.0d0-1.0d-12)
    z(5) = zedge + q1(ic)*(1.0d0+1.0d-12)
    z(6) = zedge + clen(ic) - q1(ic)*(1.0d0+1.0d-12)
    z(7) = zedge + clen(ic) - q1(ic)*(1.0d0-1.0d-12)
    worst = 0.0d0
    do it = 1, nt
      tt = 1.2d-9 + (it-1)*dble(0.62d-9)
      call build_active_table(be,zbe,idr,fld,1,1,tt,ok,why)
      if(.not.ok) then
        print*,"FAIL: cell ",ids(ic),": active table: ",trim(why)
        nfail = nfail + 1
        cycle
      endif
      xd = x; yd = y; zd = z
      call k_eval<<<(n+127)/128,128>>>(xd,yd,zd,n,rd)
      ierr = cudaDeviceSynchronize()
      hr = rd
      do i = 1, n
        pos = (/x(i),y(i),z(i),tt/)
        call getfldt_BeamLineElem(be(1),pos,ef,fld(slot))
        cr(:,i) = ef
      enddo
      do c = 1, 6
        smax = maxval(abs(cr(c,:)))
        dmax = maxval(abs(hr(c,:)-cr(c,:)))
        if(smax.gt.0.0d0) worst = max(worst,dmax/smax)
      enddo
    enddo
    write(*,'(a,i4,a,es10.3)') "cell ",ids(ic),"  max diff / component max ",worst
    if(worst.gt.1.0d-13) then
      print*,"   FAIL (tol 1e-13)"
      nfail = nfail + 1
    endif
    worst_all = max(worst_all,worst)
    zedge = zedge + clen(ic)
  enddo
  write(*,'(a,es10.3)') "worst over the nine cells: ",worst_all
  if(nfail.eq.0) then
    print*,"test_dtl_gpu: ALL PASS"
  else
    print*,"test_dtl_gpu: FAILURES: ",nfail
  endif
  call MPI_FINALIZE(ierr)

contains

  function nrand() result(r)
    double precision :: r
    seed = mod(seed*1103515245_8+12345_8,2147483648_8)
    r = dble(seed)/2147483648.0d0
  end function nrand

end program test_dtl_gpu
