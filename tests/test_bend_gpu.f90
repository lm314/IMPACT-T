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
!   k_kick_bend0 vs kick2tBd0_BeamBunch   (and kickref_Bd: refptcl bit-identical)
!   rottoT_gpu, rotbackT_gpu, convEntr_gpu, convExit_gpu, driftbackhalf_gpu,
!   drifthalfBd_gpu vs the BeamBunch routines
!   diagB_pass1_gpu, diagB_pass2_gpu vs the host loops of diagnostic1avgB_Output
!   bendpt2pt_gpu vs kickpt2ptBd_BeamBunch (N-body in the bend frame, D1.12; with and without CSR)
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
  use ParticleGPUclass
  use BeamBunchclass
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

  call gpu_particles_init(1)
  call test_quads()
  call test_dipoles()
  call test_cart()
  call test_bend_kernels()
  call test_diagB()
  call test_bendnbody()

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

  subroutine report(name,a,b,n,tol)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    double precision, intent(in) :: a(n),b(n),tol
    double precision :: scl,dmax
    integer :: nid,k
    scl = max(maxval(abs(a)),maxval(abs(b)),1.0d-300)
    dmax = maxval(abs(a-b))
    nid = 0
    do k = 1, n
      if(a(k).eq.b(k)) nid = nid + 1
    enddo
    write(*,'(a34,2x,a,es10.3,a,f8.4,a)') name,"max diff/scale ",dmax/scl, &
      "  bit-identical ",100.0d0*nid/n,"%"
    if(dmax/scl.gt.tol) then
      print*,"   FAIL (tol ",tol,")"
      nfail = nfail + 1
    endif
  end subroutine report

  !> compare the momentum kicks (p after - p before), scaled by the largest kick
  subroutine report_kick(name,pa,pb,p0,n,tol)
    character(len=*), intent(in) :: name
    integer, intent(in) :: n
    double precision, intent(in) :: pa(n),pb(n),p0(n),tol
    double precision :: da(n),db(n)
    da = pa - p0
    db = pb - p0
    call report(name,da,db,n,tol)
  end subroutine report_kick

  subroutine copy_bunch(a,b)
    type (BeamBunch), intent(in) :: a
    type (BeamBunch), intent(inout) :: b
    b%Current = a%Current
    b%Mass = a%Mass
    b%Charge = a%Charge
    b%Npt = a%Npt
    b%Nptlocal = a%Nptlocal
    b%refptcl = a%refptcl
    allocate(b%Pts1(9,size(a%Pts1,2)))
    b%Pts1 = a%Pts1
  end subroutine copy_bunch

  subroutine make_bend_bunch(b,n,seed,zlo,zhi,xr)
    type (BeamBunch), intent(inout) :: b
    integer, intent(in) :: n
    integer, intent(inout) :: seed
    double precision, intent(in) :: zlo,zhi,xr
    integer :: k
    call construct_BeamBunch(b,1.0d-5,1.0d6,0.511005d6,-1.0d0,n,0.0d0)
    allocate(b%Pts1(9,n))
    b%Nptlocal = n
    b%Npt = n
    do k = 1, n
      b%Pts1(1,k) = xr*(2*nrand(seed)-1)/Scxlt
      b%Pts1(3,k) = 0.3d0*xr*(2*nrand(seed)-1)/Scxlt
      b%Pts1(5,k) = (zlo+(zhi-zlo)*nrand(seed))/Scxlt
      b%Pts1(2,k) = 0.1d0*(nrand(seed)-0.5d0)
      b%Pts1(4,k) = 0.1d0*(nrand(seed)-0.5d0)
      b%Pts1(6,k) = 45.0d0+0.5d0*(nrand(seed)-0.5d0)
      b%Pts1(7,k) = -1.0d0/0.511005d6
      b%Pts1(8,k) = 1.0d0
      b%Pts1(9,k) = k
    enddo
    b%refptcl = (/0.001d0/Scxlt,4.5d0,0.0d0,0.0d0,0.1d0/Scxlt,45.0d0/)
  end subroutine make_bend_bunch

  subroutine test_bend_kernels()
    type (BeamBunch) :: bc, bg
    type (BeamLineElem) :: be1(1)
    double precision :: zbe1(2,1), rc(6), rg(6), pcc, pcg, cs, ss, cs2, ss2
    double precision :: refc(6), refg(6), tau, tg, dt, p0(9,100000)
    double precision, allocatable :: sv(:)
    integer :: idr1(3,1), n, seed, k
    double precision :: ptref(6), b0, p5
    integer :: slot
    double precision :: dmom

    slot = 2
    b0 = 0.3d0
    p5 = 0.0125d0
    be1(1) = assign_BeamLineElem(dp(1))
    zbe1(1,1) = 5.6547d0
    zbe1(2,1) = 5.6547d0+0.302d0
    idr1(:,1) = (/4,15,slot/)
    tg = 1.0d-10
    dt = 1.0d0

    ! ---- kick, mode none, against kick2tBd0_BeamBunch (refptcl at 0.1 rad)
    n = 100000
    seed = 777
    call make_bend_bunch(bc,n,seed,-0.02d0,0.32d0,0.012d0)
    call copy_bunch(bc,bg)
    bg%refptcl = bc%refptcl
    p0(:,1:n) = bc%Pts1(:,1:n)
    refc = bc%refptcl
    call kick2tBd0_BeamBunch(n,bc%Pts1,tg,bc%Charge,bc%Mass,dt,be1,zbe1,idr1,1,1,fld,refc)
    refg = bg%refptcl
    call kickref_Bd_BeamBunch(refg,tg,bg%Charge,bg%Mass,dt,be1,idr1,1,fld,cs,ss)
    call mark_host_newer(1)
    call kick_bend_gpu(1,bg,.false.,cs,ss,dt,slot,b0,p5,1,1,1,rc,rg)
    call ensure_host(1,bg)
    call report("kickref refptcl (bit-identical)",refc,refg,6,0.0d0)
    ! the momenta are ~45, the kicks ~1e-5: FMA contraction moves the last bit of
    ! the momentum, which is ~1e-10 of a kick.  The pass criterion is the
    ! momentum difference scaled by the momentum; the kick-relative number is
    ! printed for information.
    call report_kick("k_kick_bend0 px",bc%Pts1(2,:),bg%Pts1(2,:),p0(2,:),n,1.0d30)
    call report_kick("k_kick_bend0 py",bc%Pts1(4,:),bg%Pts1(4,:),p0(4,:),n,1.0d30)
    call report_kick("k_kick_bend0 pz",bc%Pts1(6,:),bg%Pts1(6,:),p0(6,:),n,1.0d30)
    dmom = max(maxval(abs(bc%Pts1(2,:)-bg%Pts1(2,:))),maxval(abs(bc%Pts1(4,:)-bg%Pts1(4,:))), &
               maxval(abs(bc%Pts1(6,:)-bg%Pts1(6,:))))/maxval(abs(bc%Pts1(6,:)))
    write(*,'(a34,2x,a,es10.3)') "k_kick_bend0 final momenta","max diff/max|p| ",dmom
    if(dmom.gt.4.0d-16) then
      print*,"   FAIL (tol 4e-16)"
      nfail = nfail + 1
    endif
    call check_gpu_error()
    ! the kicks must be nonzero (the dipole field acts)
    if(maxval(abs(bc%Pts1(2,:)-p0(2,:))).eq.0.0d0 .and. &
       maxval(abs(bc%Pts1(6,:)-p0(6,:))).eq.0.0d0) then
      print*,"   FAIL: the CPU kick is zero, the test does not exercise the dipole"
      nfail = nfail + 1
    endif
    deallocate(bc%Pts1,bg%Pts1)

    ! ---- rotations, range and centroid
    n = 1000000
    seed = 4321
    call make_bend_bunch(bc,n,seed,-0.02d0,0.32d0,0.012d0)
    call copy_bunch(bc,bg)
    ptref = bc%refptcl
    cs = ptref(6)/sqrt(ptref(6)**2 + ptref(2)**2)
    ss = ptref(2)/sqrt(ptref(6)**2 + ptref(2)**2)
    call rottoT_BeamBunch(bc,ptref,rc,pcc)
    call mark_host_newer(1)
    call rottoT_gpu(1,bg,cs,ss,rg,pcg)
    call ensure_host(1,bg)
    call report("rottoT x",bc%Pts1(1,:),bg%Pts1(1,:),n,4.0d-16)
    call report("rottoT px",bc%Pts1(2,:),bg%Pts1(2,:),n,4.0d-16)
    call report("rottoT z",bc%Pts1(5,:),bg%Pts1(5,:),n,4.0d-16)
    call report("rottoT pz",bc%Pts1(6,:),bg%Pts1(6,:),n,4.0d-16)
    call report("rottoT y, py unchanged",bg%Pts1(3:4,:),bc%Pts1(3:4,:),2*n,0.0d0)
    call report("rottoT ptrange",rc,rg,6,4.0d-16)
    call report("rottoT poscent",(/pcc/),(/pcg/),1,5.0d-14)
    deallocate(bc%Pts1,bg%Pts1)
    call make_bend_bunch(bc,n,seed,-0.02d0,0.32d0,0.012d0)
    call copy_bunch(bc,bg)
    call mark_host_newer(1)
    call rotbackT_BeamBunch(bc,ptref)
    call rotbackT_gpu(1,bg,cs,ss)
    call ensure_host(1,bg)
    call report("rotbackT x",bc%Pts1(1,:),bg%Pts1(1,:),n,4.0d-16)
    call report("rotbackT px",bc%Pts1(2,:),bg%Pts1(2,:),n,4.0d-16)
    call report("rotbackT z",bc%Pts1(5,:),bg%Pts1(5,:),n,4.0d-16)
    call report("rotbackT pz",bc%Pts1(6,:),bg%Pts1(6,:),n,4.0d-16)
    deallocate(bc%Pts1,bg%Pts1)

    ! ---- an empty rank
    call construct_BeamBunch(bc,1.0d-5,1.0d6,0.511005d6,-1.0d0,1000,0.0d0)
    allocate(bc%Pts1(9,1))
    bc%Nptlocal = 0
    bc%Npt = 1000
    call copy_bunch(bc,bg)
    bg%Nptlocal = 0
    bg%Npt = 1000
    call rottoT_BeamBunch(bc,ptref,rc,pcc)
    call mark_host_newer(1)
    call rottoT_gpu(1,bg,cs,ss,rg,pcg)
    call report("rottoT empty rank ptrange",rc,rg,6,0.0d0)
    call report("rottoT empty rank poscent",(/pcc/),(/pcg/),1,0.0d0)
    deallocate(bc%Pts1,bg%Pts1)

    ! ---- convEntr, convExit, driftbackhalf, drifthalfBd
    n = 100000
    seed = 55
    call make_bend_bunch(bc,n,seed,-0.02d0,0.32d0,0.012d0)
    call copy_bunch(bc,bg)
    bg%refptcl = bc%refptcl
    call convEntr_BeamBunch(bc,5.6547d0,45.25d0)
    call mark_host_newer(1)
    call convEntr_gpu(1,bg,5.6547d0,45.25d0)
    call ensure_host(1,bg)
    call report("convEntr z",bc%Pts1(5,:),bg%Pts1(5,:),n,0.0d0)
    call report("convEntr refptcl",bc%refptcl,bg%refptcl,6,0.0d0)

    tau = 1.0d0
    call driftbackhalf_BeamBunch(bc,tg,tau)
    call driftbackhalf_gpu(1,bg,tau)
    call ensure_host(1,bg)
    call report("driftbackhalf x",bc%Pts1(1,:),bg%Pts1(1,:),n,4.0d-16)
    call report("driftbackhalf z",bc%Pts1(5,:),bg%Pts1(5,:),n,4.0d-16)
    call report("driftbackhalf refptcl",bc%refptcl,bg%refptcl,6,0.0d0)
    call drifthalfBd_BeamBunch(bc,tg,tau)
    call drifthalfBd_gpu(1,bg,tau)
    call ensure_host(1,bg)
    call report("drifthalfBd x",bc%Pts1(1,:),bg%Pts1(1,:),n,4.0d-16)
    call report("drifthalfBd y",bc%Pts1(3,:),bg%Pts1(3,:),n,4.0d-16)
    call report("drifthalfBd z",bc%Pts1(5,:),bg%Pts1(5,:),n,4.0d-16)
    call report("drifthalfBd refptcl",bc%refptcl,bg%refptcl,6,0.0d0)

    ! reference at the exit angle
    bc%refptcl(2) = 4.4d0
    bc%refptcl(6) = 44.9d0
    bg%refptcl = bc%refptcl
    call convExit_BeamBunch(bc,5.9567d0)
    call convExit_gpu(1,bg,5.9567d0)
    call ensure_host(1,bg)
    call report("convExit x",bc%Pts1(1,:),bg%Pts1(1,:),n,4.0d-16)
    call report("convExit px",bc%Pts1(2,:),bg%Pts1(2,:),n,4.0d-16)
    call report("convExit y",bc%Pts1(3,:),bg%Pts1(3,:),n,4.0d-16)
    call report("convExit z",bc%Pts1(5,:),bg%Pts1(5,:),n,4.0d-16)
    call report("convExit pz",bc%Pts1(6,:),bg%Pts1(6,:),n,4.0d-16)
    call report("convExit refptcl",bc%refptcl,bg%refptcl,6,0.0d0)
    deallocate(bc%Pts1,bg%Pts1)
  end subroutine test_bend_kernels

  !> the per-particle terms of diagnostic1avgB_Output's second pass
  subroutine termsB(q,z0avg,pz0avg,gamavg,t,mx,rr)
    double precision, intent(in) :: q(9),z0avg,pz0avg,gamavg
    double precision, intent(out) :: t(29),mx(6),rr
    double precision :: tmpgam
    tmpgam = sqrt(1.0+q(2)**2+q(4)**2+q(6)**2)
    t = 0.0d0
    t(1) = q(1)
    t(2) = q(2)
    t(3) = q(3)
    t(4) = q(4)
    t(6) = q(6)
    t(7) = q(1)*q(1)
    t(8) = q(2)*q(2)
    t(9) = q(3)*q(3)
    t(10) = q(4)*q(4)
    t(11) = (q(5)-z0avg)**2
    t(12) = (q(6)-pz0avg)**2
    t(13) = q(1)*q(2)
    t(14) = q(3)*q(4)
    t(15) = (q(5)-z0avg)*(q(6)-pz0avg)
    t(16) = q(1)*q(1)*q(1)
    t(17) = q(1)*q(1)*q(1)*q(1)
    t(18) = q(2)*q(2)*q(2)
    t(19) = q(2)*q(2)*q(2)*q(2)
    t(20) = q(3)*q(3)*q(3)
    t(21) = q(3)*q(3)*q(3)*q(3)
    t(22) = q(4)*q(4)*q(4)
    t(23) = q(4)*q(4)*q(4)*q(4)
    t(24) = (q(5)-z0avg)**3
    t(25) = (q(5)-z0avg)**4
    t(26) = (q(6)-pz0avg)**3
    t(27) = (q(6)-pz0avg)**4
    t(28) = tmpgam
    t(29) = (tmpgam-gamavg)**2
    mx(1) = abs(q(1))
    mx(2) = abs(q(2))
    mx(3) = abs(q(3))
    mx(4) = abs(q(4))
    mx(5) = abs(q(5)-z0avg)
    mx(6) = abs(q(6)-pz0avg)
    rr = q(1)**2+q(3)**2
  end subroutine termsB

  subroutine test_diagB()
    type (BeamBunch) :: bg(1)
    integer :: n, seed, i, k
    double precision :: s3(3), g3(3), z0avg, pz0avg, gamavg
    double precision :: sref(29), sabs(29), t(29), mx(6), rr, mref(6), rrref
    double precision :: tg29(29), lmax(6), lcr, err, worst
    n = 1000000
    seed = 2024
    call make_bend_bunch(bg(1),n,seed,-0.02d0,0.32d0,0.012d0)
    ! a rotated, spread-out beam: give z and pz some structure
    do i = 1, n
      bg(1)%Pts1(2,i) = bg(1)%Pts1(2,i) + 0.01d0*bg(1)%Pts1(1,i)*Scxlt/0.012d0
      bg(1)%Pts1(6,i) = bg(1)%Pts1(6,i) + 0.2d0*sin(40.0d0*bg(1)%Pts1(5,i)*Scxlt)
    enddo
    ! pass 1 on the host
    s3 = 0.0d0
    do i = 1, n
      s3(1) = s3(1) + bg(1)%Pts1(5,i)
      s3(2) = s3(2) + bg(1)%Pts1(6,i)
      s3(3) = s3(3) + sqrt(1.0+bg(1)%Pts1(2,i)**2+bg(1)%Pts1(4,i)**2+bg(1)%Pts1(6,i)**2)
    enddo
    call mark_host_newer(1)
    call diagB_pass1_gpu(bg,1,g3)
    worst = 0.0d0
    do k = 1, 3
      err = abs(s3(k)-g3(k))/abs(s3(k))
      worst = max(worst,err)
    enddo
    write(*,'(a34,2x,a,es10.3)') "diagB pass 1 (z, pz, gamma sums)","max rel diff ",worst
    if(worst.gt.2.0d-13) then
      print*,"   FAIL (tol 2e-13)"
      nfail = nfail + 1
    endif
    z0avg = s3(1)/n
    pz0avg = s3(2)/n
    gamavg = s3(3)/n
    ! pass 2 on the host
    sref = 0.0d0
    sabs = 0.0d0
    mref = 0.0d0
    rrref = 0.0d0
    do i = 1, n
      call termsB(bg(1)%Pts1(:,i),z0avg,pz0avg,gamavg,t,mx,rr)
      sref = sref + t
      sabs = sabs + abs(t)
      mref = max(mref,mx)
      rrref = max(rrref,rr)
    enddo
    call diagB_pass2_gpu(bg,1,z0avg,pz0avg,gamavg,tg29,lmax,lcr)
    worst = 0.0d0
    do k = 1, 29
      if(sabs(k).gt.0.0d0) worst = max(worst,abs(sref(k)-tg29(k))/sabs(k))
    enddo
    write(*,'(a34,2x,a,es10.3,a)') "diagB pass 2 (29 sums)","max |diff|/sum|terms| ",worst, &
      "  (tmplc(5) = 0 on the GPU: "//merge("yes","NO ",tg29(5).eq.0.0d0)//")"
    if(worst.gt.2.0d-13 .or. tg29(5).ne.0.0d0) then
      print*,"   FAIL (tol 2e-13)"
      nfail = nfail + 1
    endif
    call report("diagB maxima (equal)",mref,lmax,6,0.0d0)
    call report("diagB max r^2 (equal)",(/rrref/),(/lcr/),1,0.0d0)
    deallocate(bg(1)%Pts1)
  end subroutine test_diagB

  subroutine test_bendnbody()
    type (BeamBunch) :: bc, bg
    type (BeamLineElem) :: be1(1)
    integer :: idr1(3,1), n, seed, k, icsr, nzw
    double precision :: refc(6), refg(6), cs, ss, tg, dt, totchrg, r0, zminw, hzw0, zmaxw
    double precision :: ezw(64), kmax, dk, pscale
    double precision, allocatable :: p0(:,:)
    character(len=40) :: nm
    n = 20000
    seed = 8080
    nzw = 64
    be1(1) = assign_BeamLineElem(dp(1))
    idr1(:,1) = (/4,15,2/)
    tg = 1.0d-10
    dt = 1.0d0
    totchrg = 1.0d-10
    r0 = 1.0d-4
    allocate(p0(9,n))
    do icsr = 0, 1
      call make_bend_bunch(bc,n,seed,-0.02d0,0.32d0,0.012d0)   ! includes z < 0
      call copy_bunch(bc,bg)
      p0 = bc%Pts1
      refc = bc%refptcl
      call kickref_Bd_BeamBunch(refc,tg,bc%Charge,bc%Mass,dt,be1,idr1,1,fld,cs,ss)
      zminw = minval(bc%Pts1(5,:)) - 1.0d-3
      zmaxw = maxval(bc%Pts1(5,:)) + 1.0d-3
      hzw0 = (zmaxw-zminw)*1.0000001d0/(nzw-1)
      do k = 1, nzw
        ezw(k) = 1.0d8*sin(0.3d0*k)   ! large enough that the wake kick is of the size of the space-charge kick
      enddo
      call kickpt2ptBd_BeamBunch(n,bc%Pts1,tg,bc%Charge,bc%Mass,dt,be1,idr1,1,fld, &
                                 cs,ss,totchrg,r0,n,n,icsr,ezw,nzw,zminw,hzw0)
      call mark_host_newer(1)
      call bendpt2pt_gpu(1,bg,cs,ss,dt,2,0.3d0,0.0125d0,totchrg,r0,n,n,icsr,ezw,nzw,zminw,hzw0)
      call ensure_host(1,bg)
      write(nm,'(a,i1)') "bend N-body csr=",icsr
      kmax = 0.0d0
      dk = 0.0d0
      do k = 2, 6, 2
        kmax = max(kmax,maxval(abs(bc%Pts1(k,:)-p0(k,:))))
        dk = max(dk,maxval(abs(bc%Pts1(k,:)-bg%Pts1(k,:))))
      enddo
      pscale = maxval(abs(bc%Pts1(6,:)))
      write(*,'(a34,2x,a,es10.3,a,es10.3,a,es10.3)') trim(nm),"max diff/largest kick ",dk/kmax, &
        "  largest kick ",kmax,"  max diff/|p| ",dk/pscale
      if(dk/pscale.gt.4.0d-16) then
        print*,"   FAIL (tol 4e-16 of |p|; the plan's 1e-13 of the largest kick is the momentum floor, see item 2)"
        nfail = nfail + 1
      endif
      call check_gpu_error()
      deallocate(bc%Pts1,bg%Pts1)
    enddo
  end subroutine test_bendnbody

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
