! D1 test: each GPU field kind / bend kernel against the CPU routine it ports,
! on the same input (1 rank, 1 GPU).  Not part of the CMake build; see
! tests/README.md.  IMPACTT_GPU_PORT_PLAN.md, D1.9 item 1.
!
! Run in a directory holding rfdata111 (quadrupole profile), rfdata15 (dipole
! Enge file) and 1T7.T7 (the CXLS deflector map, z header 0..0.108):
!
!   fld_quad     vs getfld_Quadrupole     (modes 1, 2, 3; skew; RF; 5 times)
!   fld_dipole   vs getfldt_Dipole        (CXLS faces and tilted faces)
!   fld_emcart   vs getfldt_EMfldCart     (5 RF phases, both ends, out of map)
module tbend_kernels
  use cudafor
  use FieldmapGPUclass
contains
  !> the active-table path of the step kernels: extfld_dev on a one-element table
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
end module tbend_kernels

program test_bend_gpu
  use cudafor
  use BeamLineElemclass
  use Quadrupoleclass
  use Dipoleclass
  use EMfldCartclass
  use Dataclass
  use NumConstclass
  use PhysConstclass
  use Timerclass
  use FieldmapGPUclass
  use tbend_kernels
  implicit none
  include 'mpif.h'

  type (Quadrupole), target :: qd(8)
  type (Dipole), target :: dp(2)
  type (EMfldCart), target :: ec(1)
  type (fielddata) :: fld(Maxoverlap)
  integer :: ierr, nfail, it

  call MPI_INIT(ierr)
  nfail = 0
  call constructT_PhysConst(2.5d-13,9.3d9)
  call construct_Timer(0.0d0)
  do it = 1, Maxoverlap
    call initt_Data(fld(it))
  enddo

  call test_quads()
  call test_dipoles()
  call test_cart()

  if(nfail.eq.0) then
    print*,"test_bend_gpu: ALL PASS"
  else
    print*,"test_bend_gpu: FAILURES: ",nfail
  endif
  call MPI_FINALIZE(ierr)

contains

  function nrand(seed) result(r)
    integer, intent(inout) :: seed
    double precision :: r
    seed = mod(seed*1103515245_8+12345_8,2147483648_8)
    r = dble(seed)/2147483648.0d0
  end function nrand

  !> Evaluate the one-element table on the device and the CPU routine on the
  !> host at the same points.  The device gates on [zlo,zhi]; so does the host
  !> (kick2t does that for the CPU).  Returns the largest per-component
  !> relative difference (scaled by that component's largest value) and the
  !> fraction of bit-identical numbers.
  subroutine compare_element(name,be1,itype,fileid,slot,zlo,zhi,tt,x,y,z,n,tol)
    character(len=*), intent(in) :: name
    type (BeamLineElem), intent(in) :: be1
    integer, intent(in) :: itype,fileid,slot,n
    double precision, intent(in) :: zlo,zhi,tt,tol
    double precision, intent(in) :: x(n),y(n),z(n)
    type (BeamLineElem) :: be(1)
    double precision :: zbe(2,1), pos(4), ef(6), rel, dmax, smax, worst
    double precision, allocatable :: hr(:,:), cr(:,:)
    real(8), device, allocatable :: xd(:),yd(:),zd(:),rd(:,:)
    integer :: idr(3,1), i, c, nid, ntot, nz
    logical :: ok
    character(len=80) :: why

    be(1) = be1
    zbe(1,1) = zlo
    zbe(2,1) = zhi
    idr(:,1) = (/itype,fileid,slot/)
    call build_active_table(be,zbe,idr,fld,1,1,tt,ok,why)
    if(.not.ok) then
      print*,"FAIL: ",trim(name),": active table: ",trim(why)
      nfail = nfail + 1
      return
    endif
    allocate(xd(n),yd(n),zd(n),rd(6,n),hr(6,n),cr(6,n))
    xd = x
    yd = y
    zd = z
    call k_eval<<<(n+127)/128,128>>>(xd,yd,zd,n,rd)
    ierr = cudaDeviceSynchronize()
    hr = rd
    cr = 0.0d0
    do i = 1, n
      if(z(i).ge.zlo .and. z(i).le.zhi) then
        pos = (/x(i),y(i),z(i),tt/)
        call getfldt_BeamLineElem(be(1),pos,ef,fld(slot))
        cr(:,i) = ef
      endif
    enddo
    worst = 0.0d0
    nid = 0
    ntot = 0
    nz = 0
    do c = 1, 6
      smax = maxval(abs(cr(c,:)))
      dmax = maxval(abs(hr(c,:)-cr(c,:)))
      if(smax.gt.0.0d0) then
        rel = dmax/smax
        worst = max(worst,rel)
      endif
      do i = 1, n
        ntot = ntot + 1
        if(hr(c,i).eq.cr(c,i)) nid = nid + 1
      enddo
    enddo
    do i = 1, n
      if(z(i).ge.zlo .and. z(i).le.zhi) nz = nz + 1
    enddo
    write(*,'(a34,2x,a,es10.3,a,f8.4,a,i0,a)') name,"max rel diff ",worst, &
      "  bit-identical ",100.0d0*nid/ntot,"%  (",nz," points in range)"
    if(worst.gt.tol) then
      print*,"   FAIL (tol ",tol,")"
      nfail = nfail + 1
    endif
    deallocate(xd,yd,zd,rd,hr,cr)
  end subroutine compare_element

  subroutine test_quads()
    integer :: k, n, seed, i, itt
    double precision, allocatable :: x(:),y(:),z(:)
    double precision :: zedge, len, delta, tt
    character(len=40) :: nm
    n = 100000
    allocate(x(n),y(n),z(n))
    zedge = 8.9723d0
    len = 0.262d0
    delta = 0.02d0
    seed = 4711
    do i = 1, n
      x(i) = 0.02d0*(nrand(seed)-0.5d0)
      y(i) = 0.02d0*(nrand(seed)-0.5d0)
      z(i) = zedge-delta+(len+2*delta)*nrand(seed)
    enddo
    ! both edges, exactly
    z(1) = zedge
    z(2) = zedge+len
    z(3) = zedge - 1.0d-12
    z(4) = zedge + len + 1.0d-12

    call read1tdata_Data(fld(1),111)
    call fldslot_upload(fld,1,1)

    ! 1 plain mode 1; 2 skew; 3 RF + skew; 4 mode 2; 5 mode 2 skew;
    ! 6 mode 3; 7 mode 3 RF
    do k = 1, 7
      call construct_Quadrupole(qd(k),4,20,1,len)
      call setparam_Quadrupole(qd(k),1,zedge)
      call setparam_Quadrupole(qd(k),2,7.86d0)
      call setparam_Quadrupole(qd(k),3,111.0d0)
      call setparam_Quadrupole(qd(k),4,0.0125d0)
    enddo
    call setparam_Quadrupole(qd(2),9,0.1d0)
    call setparam_Quadrupole(qd(3),9,0.1d0)
    call setparam_Quadrupole(qd(3),10,1.0d9)
    call setparam_Quadrupole(qd(3),11,30.0d0)
    call setparam_Quadrupole(qd(4),3,0.2d0)
    call setparam_Quadrupole(qd(5),3,0.2d0)
    call setparam_Quadrupole(qd(5),9,0.1d0)
    call setparam_Quadrupole(qd(6),3,0.0d0)
    call setparam_Quadrupole(qd(7),3,0.0d0)
    call setparam_Quadrupole(qd(7),10,1.0d9)
    call setparam_Quadrupole(qd(7),11,-45.0d0)

    do k = 1, 7
      do itt = 1, 5
        tt = 1.0d-10 + (itt-1)*2.3d-11
        if(k.ne.3 .and. k.ne.7 .and. itt.gt.1) cycle
        write(nm,'(a,i1,a,i1)') "fld_quad case ",k," t",itt
        call compare_element(trim(nm),assign_BeamLineElem(qd(k)),1,111,1, &
                             zedge,zedge+len,tt,x,y,z,n,1.0d-15)
      enddo
    enddo
    deallocate(x,y,z)
  end subroutine test_quads

  subroutine test_dipoles()
    integer :: n, seed, i
    double precision, allocatable :: x(:),y(:),z(:)
    double precision :: f(401)
    n = 100000
    allocate(x(n),y(n),z(n))
    seed = 99
    do i = 1, n
      x(i) = 0.03d0*(nrand(seed)-0.5d0)
      y(i) = 0.01d0*(nrand(seed)-0.5d0)
      z(i) = -0.05d0 + 0.40d0*nrand(seed)
    enddo
    call read1t_Data(fld(2),15)
    call construct_Dipole(dp(1),4,20,4,0.302d0)
    call setparam_Dipole(dp(1),1,5.6547d0)
    call setparam_Dipole(dp(1),3,0.3d0)
    call setparam_Dipole(dp(1),4,15.0d0)
    call setparam_Dipole(dp(1),5,0.0125d0)
    call fldslot_upload(fld,2,3)
    ! points exactly on the four face lines (straight faces, any x)
    do i = 1, 40
      z(i) = fld(2)%Fcoeft(3+mod(i-1,4)*2)*x(i)+fld(2)%Fcoeft(4+mod(i-1,4)*2)
    enddo
    call compare_element("fld_dipole CXLS faces",assign_BeamLineElem(dp(1)),4,15,2, &
                         -1.0d9,1.0d9,0.0d0,x,y,z,n,2.0d-15)

    ! tilted faces: Fcoeft(3), (5), (7), (9) /= 0
    call read1t_Data(fld(3),15)
    fld(3)%Fcoeft(3) = 0.1d0
    fld(3)%Fcoeft(5) = 0.1d0
    fld(3)%Fcoeft(7) = -0.05d0
    fld(3)%Fcoeft(9) = -0.05d0
    call fldslot_upload(fld,3,3)
    call construct_Dipole(dp(2),4,20,4,0.302d0)
    call setparam_Dipole(dp(2),1,5.6547d0)
    call setparam_Dipole(dp(2),3,0.3d0)
    call setparam_Dipole(dp(2),4,15.0d0)
    call setparam_Dipole(dp(2),5,0.0125d0)
    do i = 1, 40
      z(i) = fld(3)%Fcoeft(3+mod(i-1,4)*2)*x(i)+fld(3)%Fcoeft(4+mod(i-1,4)*2)
    enddo
    call compare_element("fld_dipole tilted faces",assign_BeamLineElem(dp(2)),4,15,3, &
                         -1.0d9,1.0d9,0.0d0,x,y,z,n,2.0d-15)
    deallocate(x,y,z)
  end subroutine test_dipoles

  subroutine test_cart()
    integer :: n, seed, i, itype, nout, ipart, iph
    double precision, allocatable :: x(:),y(:),z(:)
    double precision :: zedge, len, tt, xr, yr, pc(3)
    character(len=40) :: nm
    n = 100000
    allocate(x(n),y(n),z(n))
    zedge = 3.7096d0
    len = 0.108d0
    call read3t_Data(fld(4),7)
    call fldslot_upload(fld,4,4)
    print*,"111 map: nx,ny,nz intervals ",fld(4)%NxIntvRfgt,fld(4)%NyIntvRfgt, &
           fld(4)%NzIntvRfgt," x ",fld(4)%XminRfgt,fld(4)%XmaxRfgt," y ", &
           fld(4)%YminRfgt,fld(4)%YmaxRfgt," z ",fld(4)%ZminRfgt,fld(4)%ZmaxRfgt
    xr = 0.98d0*0.5d0*(fld(4)%XmaxRfgt-fld(4)%XminRfgt)
    yr = 0.98d0*0.5d0*(fld(4)%YmaxRfgt-fld(4)%YminRfgt)
    seed = 31337
    do i = 1, n
      x(i) = 0.5d0*(fld(4)%XmaxRfgt+fld(4)%XminRfgt)+xr*(2*nrand(seed)-1)
      y(i) = 0.5d0*(fld(4)%YmaxRfgt+fld(4)%YminRfgt)+yr*(2*nrand(seed)-1)
      z(i) = zedge + len*nrand(seed)
    enddo
    ! the strict test: both ends must give zero on both sides
    z(1) = zedge
    z(2) = zedge+len
    z(3) = zedge - 1.0d-9
    z(4) = zedge + len + 1.0d-9
    do iph = 0, 4
      call construct_EMfldCart(ec(1),1,1,111,len)
      call setparam_EMfldCart(ec(1),1,zedge)
      call setparam_EMfldCart(ec(1),2,0.103d0)
      call setparam_EMfldCart(ec(1),3,9.3d9)
      call setparam_EMfldCart(ec(1),4,72.0d0*iph+5.0d0)
      call setparam_EMfldCart(ec(1),5,7.0d0)
      tt = 1.0d-10 + iph*1.7d-11
      write(nm,'(a,i1)') "fld_emcart phase ",iph
      call compare_element(trim(nm),assign_BeamLineElem(ec(1)),111,7,4, &
                           zedge-1.0d-6,zedge+len+1.0d-6,tt,x,y,z,n,2.0d-15)
    enddo
    nout = cartout_count(ipart,pc)
    if(nout.ne.0) then
      print*,"FAIL: in-map points counted as outside: ",nout
      nfail = nfail + 1
    else
      print*,"fld_emcart in-map points: out-of-map counter 0 (ok)"
    endif

    ! 100 points outside the x range: zero field and counter = 100
    do i = 1, 100
      x(i) = fld(4)%XmaxRfgt*1.5d0 + 0.001d0*i
      y(i) = 0.0d0
      z(i) = zedge + len*0.5d0
    enddo
    call run_out(zedge,len,x,y,z,100)
    deallocate(x,y,z)
  end subroutine test_cart

  subroutine run_out(zedge,len,x,y,z,n)
    double precision, intent(in) :: zedge,len,x(n),y(n),z(n)
    integer, intent(in) :: n
    type (BeamLineElem) :: be(1)
    double precision :: zbe(2,1), pc(3)
    double precision, allocatable :: hr(:,:)
    real(8), device, allocatable :: xd(:),yd(:),zd(:),rd(:,:)
    integer :: idr(3,1), nout, ipart
    logical :: ok
    character(len=80) :: why
    be(1) = assign_BeamLineElem(ec(1))
    zbe(1,1) = zedge-1.0d-6
    zbe(2,1) = zedge+len+1.0d-6
    idr(:,1) = (/111,7,4/)
    call build_active_table(be,zbe,idr,fld,1,1,1.0d-10,ok,why)
    allocate(xd(n),yd(n),zd(n),rd(6,n),hr(6,n))
    xd = x
    yd = y
    zd = z
    call k_eval<<<(n+127)/128,128>>>(xd,yd,zd,n,rd)
    ierr = cudaDeviceSynchronize()
    hr = rd
    nout = cartout_count(ipart,pc)
    write(*,'(a,es10.3,a,i0,a,i0,a,3es12.4)') "fld_emcart out of map: max |field| ", &
      maxval(abs(hr))," counter ",nout," first particle ",ipart," at ",pc
    if(maxval(abs(hr)).ne.0.0d0 .or. nout.ne.100) then
      print*,"   FAIL (expected zero field and counter 100)"
      nfail = nfail + 1
    endif
    deallocate(xd,yd,zd,rd,hr)
  end subroutine run_out

end program test_bend_gpu
