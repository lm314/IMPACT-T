! B2 test: each GPU particle kernel against the CPU routine it ports, on the
! same input (1 rank, 1 GPU).  Not part of the CMake build; see tests/README.md.
!
! Run in a directory holding rfdata1001, rfdata1002, rfdata1003 and 1T1.T7 (written by
! `python3 tests/make_test_inputs.py particle DIR`)
! (Sample4_2Dfield, element type 112).
!
!   extfld_dev / kick_none   vs scatter20t_BeamBunch   (types 105 and 112, 5 times)
!   kickpt2pt_gpu            vs kickpt2ptImg_BeamBunch
!   drifts + emission        vs drifthalf/drifthalforg/driftemission + AccSimulator's loop
!   singlerange_gpu          vs singlerange
!   lost_gpu                 vs lostREC_BeamBunch (0 and 37 lost)
!   deposit_gpu, gradEB_dev  vs reference loops transcribed from the CPU code
program test_particle_gpu
  use cudafor
  use BeamBunchclass
  use BeamLineElemclass
  use SolRFclass
  use EMfldCylclass
  use Dataclass
  use NumConstclass
  use PhysConstclass
  use Timerclass
  use Rangerclass
  use FieldmapGPUclass
  use ParticleGPUclass
  use PoissonGPUclass
  implicit none
  include 'mpif.h'

  type (SolRF), target :: sl(3)
  type (EMfldCyl), target :: em(1)
  type (BeamLineElem) :: be(4)
  double precision :: zbe(2,4)
  integer :: idr(3,4)
  type (fielddata) :: fld(Maxoverlap)
  double precision :: prm(12), prm13(13), blen(4)
  integer :: i, ierr, nfail
  logical :: ok

  call MPI_INIT(ierr)
  nfail = 0
  call constructT_PhysConst(2.5d-13,9.3d9)
  call construct_Timer(0.0d0)
  do i = 1, Maxoverlap
    call initt_Data(fld(i))
  enddo

  ! elements: RF cell (1001), solenoid (1003), RF cell (1002), 112 map
  blen = (/0.0055d0, 0.2d0, 0.012d0, 0.15d0/)
  call construct_SolRF(sl(1),1,1,105,blen(1))
  prm = (/0.0d0,-1.940689688d0,9.3d9,-78.73087547406008d0,1001.0d0,0.15d0, &
          0.0d0,0.0d0,0.0d0,0.0d0,0.0d0,0.0d0/)
  call setparam_SolRF(sl(1),prm)
  be(1) = assign_BeamLineElem(sl(1))
  call construct_SolRF(sl(2),1,1,105,blen(2))
  prm = (/0.0d0,0.0d0,0.0d0,0.0d0,1003.0d0,0.15d0, &
          0.0d0,0.0d0,0.0d0,0.0d0,0.0d0,0.55d0/)
  call setparam_SolRF(sl(2),prm)
  be(2) = assign_BeamLineElem(sl(2))
  call construct_SolRF(sl(3),1,1,105,blen(3))
  prm = (/0.00554419354839d0,-1317.332962815205d0,9.3d9,-213.73087547406007d0, &
          1002.0d0,0.15d0,0.0d0,0.0d0,0.0d0,0.0d0,0.0d0,0.0d0/)
  call setparam_SolRF(sl(3),prm)
  be(3) = assign_BeamLineElem(sl(3))
  call construct_EMfldCyl(em(1),1,1,112,blen(4))
  prm13 = 0.0d0
  prm13(1) = 0.0d0
  prm13(2) = 44.919d0
  prm13(3) = 2856.0499433d6
  prm13(4) = -55.14d0
  prm13(5) = 1.0d0
  prm13(6) = 0.009d0
  call setparam_EMfldCyl(em(1),prm13)
  be(4) = assign_BeamLineElem(em(1))
  zbe(1,1) = 0.0d0;        zbe(2,1) = blen(1)
  zbe(1,2) = 0.0d0;        zbe(2,2) = blen(2)
  zbe(1,3) = 0.00554419354839d0; zbe(2,3) = zbe(1,3)+blen(3)
  zbe(1,4) = 0.0d0;        zbe(2,4) = blen(4)
  idr(:,1) = (/105,1001,1/)
  idr(:,2) = (/105,1003,2/)
  idr(:,3) = (/105,1002,3/)
  idr(:,4) = (/112,1,4/)
  call read1tdata_Data(fld(1),1001)
  call read1tdata_Data(fld(2),1003)
  call read1tdata_Data(fld(3),1002)
  call read2t_Data(fld(4),1)

  call gpu_particles_init(1)
  call fldslot_upload(fld,1,1)
  call fldslot_upload(fld,2,1)
  call fldslot_upload(fld,3,1)
  call fldslot_upload(fld,4,2)

  call test_none_and_pt2pt()
  call test_drifts()
  call test_range_lost()
  call test_mesh()

  if(nfail.eq.0) then
    print*,"test_particle_gpu: ALL PASS"
  else
    print*,"test_particle_gpu: FAILURES: ",nfail
  endif
  call MPI_FINALIZE(ierr)

contains

  function nrand(seed) result(r)
    integer, intent(inout) :: seed
    double precision :: r
    seed = mod(seed*1103515245_8+12345_8,2147483648_8)
    r = dble(seed)/2147483648.0d0
  end function nrand

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
    write(*,'(a28,2x,a,es10.3,a,f8.4,a)') name,"max diff/scale ",dmax/scl, &
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

  subroutine make_bunch(b,n,seed,zlo,zhi,rmax)
    type (BeamBunch), intent(inout) :: b
    integer, intent(in) :: n
    integer, intent(inout) :: seed
    double precision, intent(in) :: zlo,zhi,rmax
    integer :: k
    double precision :: r,ph,gam
    call construct_BeamBunch(b,1.0d-5,1.0d6,0.511005d6,-1.0d0,n,0.0d0)
    allocate(b%Pts1(9,n))
    b%Nptlocal = n
    b%Npt = n
    do k = 1, n
      r = rmax*nrand(seed)
      ph = 6.283185307179586d0*nrand(seed)
      b%Pts1(1,k) = r*cos(ph)/Scxlt
      b%Pts1(3,k) = r*sin(ph)/Scxlt
      b%Pts1(5,k) = (zlo+(zhi-zlo)*nrand(seed))/Scxlt
      b%Pts1(2,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(4,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(6,k) = 0.1d0+1.5d0*nrand(seed)
      b%Pts1(7,k) = -1.0d0/0.511005d6
      b%Pts1(8,k) = 1.0d0
      b%Pts1(9,k) = k
    enddo
    ! some particles exactly on axis, to test r = 0 of the 112 map
    do k = 1, min(n,50)
      b%Pts1(1,k) = 0.0d0
      b%Pts1(3,k) = 0.0d0
    enddo
  end subroutine make_bunch

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

  subroutine test_none_and_pt2pt()
    type (BeamBunch) :: bc, bg
    integer :: n, seed, it
    double precision :: tt
    double precision, allocatable :: p0(:,:)
    logical :: okt
    character(len=40) :: nm
    n = 100000
    allocate(p0(9,n))
    seed = 12345
    ! z from before the first element to past the last one
    call make_bunch(bc,n,seed,-0.002d0,0.22d0,0.0089d0)
    do it = 1, 5
      tt = 1.0d-10 + (it-1)*2.0d-11
      call copy_bunch(bc,bg)
      p0 = bc%Pts1
      call scatter20t_BeamBunch(n,bc%Pts1,tt,bc%Charge,bc%Mass,1.0d0,be,zbe, &
                                idr,4,1,4,fld,0)
      call build_active_table(be,zbe,idr,fld,1,4,tt,okt,nm)
      if(.not.okt) then
        print*,"FAIL: active table: ",trim(nm)
        nfail = nfail + 1
      endif
      call mark_host_newer(1)
      call kick_none_gpu(1,bg,tt,1.0d0)
      call ensure_host(1,bg)
      write(nm,'(a,i1)') "kick_none (extfld) t",it
      call report_kick(trim(nm)//" px",bc%Pts1(2,:),bg%Pts1(2,:),p0(2,:),n,1.0d-12)
      call report_kick(trim(nm)//" py",bc%Pts1(4,:),bg%Pts1(4,:),p0(4,:),n,1.0d-12)
      call report_kick(trim(nm)//" pz",bc%Pts1(6,:),bg%Pts1(6,:),p0(6,:),n,1.0d-12)
      call check_gpu_error()
      deallocate(bg%Pts1)
      ! undo the kick on the CPU copy for the next time
      bc%Pts1(2:6:2,:) = 0.0d0
      call restore_mom(bc,seed)
    enddo
    deallocate(bc%Pts1)
    call test_pt2pt()
  end subroutine test_none_and_pt2pt

  subroutine restore_mom(b,seed)
    type (BeamBunch), intent(inout) :: b
    integer, intent(inout) :: seed
    integer :: k
    do k = 1, b%Nptlocal
      b%Pts1(2,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(4,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(6,k) = 0.1d0+1.5d0*nrand(seed)
    enddo
  end subroutine restore_mom

  subroutine test_pt2pt()
    type (BeamBunch) :: bc, bg
    integer :: n, seed, img
    double precision :: tt, totchrg, r0
    double precision, allocatable :: p0(:,:)
    logical :: okt
    character(len=40) :: nm
    n = 20000
    allocate(p0(9,n))
    seed = 777
    tt = 1.5d-10
    totchrg = n*1.602176634d-19
    r0 = 1.0d-8
    do img = 0, 1
      call make_bunch(bc,n,seed,0.0001d0,0.02d0,0.00005d0)
      call copy_bunch(bc,bg)
      p0 = bc%Pts1
      call kickpt2ptImg_BeamBunch(n,bc%Pts1,tt,bc%Charge,bc%Mass,1.0d0,be,zbe, &
                                  idr,4,1,4,fld,totchrg,r0,r0,n,n,img)
      call mark_host_newer(1)
      call build_active_table(be,zbe,idr,fld,1,4,tt,okt,nm)
      call kickpt2pt_gpu(1,bg,tt,1.0d0,totchrg,r0,r0,n,n,img)
      call ensure_host(1,bg)
      call report_kick("kickpt2pt kick px",bc%Pts1(2,:),bg%Pts1(2,:),p0(2,:),n,1.0d-11)
      call report_kick("kickpt2pt kick py",bc%Pts1(4,:),bg%Pts1(4,:),p0(4,:),n,1.0d-11)
      call report_kick("kickpt2pt kick pz",bc%Pts1(6,:),bg%Pts1(6,:),p0(6,:),n,1.0d-11)
      call check_gpu_error()
      deallocate(bc%Pts1,bg%Pts1)
    enddo
  end subroutine test_pt2pt

  subroutine test_drifts()
    type (BeamBunch) :: bc, bg
    integer :: n, seed, fc, k
    double precision :: tt, tau, betazini, tmpzz, recpgam, dtmp
    n = 200000
    seed = 4242
    tt = 0.0d0
    tau = 1.0d0
    betazini = 0.05d0
    do fc = 0, 1
      call make_bunch(bc,n,seed,-0.0002d0,0.0003d0,0.0001d0)
      ! some with pz < 0 and z < 0
      do k = 1, n, 7
        bc%Pts1(6,k) = -0.2d0
      enddo
      call copy_bunch(bc,bg)
      ! first half drift
      if(fc.eq.1) then
        call drifthalf_BeamBunch(bc,tt,tau,betazini)
      else
        call drifthalforg_BeamBunch(bc,tt,tau)
      endif
      call mark_host_newer(1)
      call drift_start_gpu(1,bg,tau,fc)
      call ensure_host(1,bg)
      call report("drift start x",bc%Pts1(1,:),bg%Pts1(1,:),n,1.0d-15)
      call report("drift start z",bc%Pts1(5,:),bg%Pts1(5,:),n,1.0d-15)
      ! second half drift with emission
      if(fc.eq.1) then
        call drifthalf_BeamBunch(bc,tt,tau,betazini)
        call driftemission_BeamBunch(bc,tt,tau,betazini)
        do k = 1, n
          tmpzz = bc%Pts1(5,k)-tau*betazini
          recpgam = 1.0/sqrt(1.0+bc%Pts1(2,k)**2+bc%Pts1(4,k)**2+bc%Pts1(6,k)**2)
          if(tmpzz.le.0.0d0 .and. bc%Pts1(5,k).ge.0.0d0) then
            dtmp = tau*bc%Pts1(5,k)/(tau*betazini)
            bc%Pts1(1,k) = bc%Pts1(1,k)+dtmp*bc%Pts1(2,k)*recpgam
            bc%Pts1(3,k) = bc%Pts1(3,k)+dtmp*bc%Pts1(4,k)*recpgam
            bc%Pts1(5,k) = dtmp*bc%Pts1(6,k)*recpgam
          endif
        enddo
      else
        call drifthalforg_BeamBunch(bc,tt,tau)
      endif
      call drift_end_gpu(1,bg,tau,fc,betazini)
      call ensure_host(1,bg)
      call report("drift end x",bc%Pts1(1,:),bg%Pts1(1,:),n,1.0d-15)
      call report("drift end y",bc%Pts1(3,:),bg%Pts1(3,:),n,1.0d-15)
      call report("drift end z",bc%Pts1(5,:),bg%Pts1(5,:),n,1.0d-15)
      deallocate(bc%Pts1,bg%Pts1)
    enddo
  end subroutine test_drifts

  subroutine test_range_lost()
    type (BeamBunch) :: bc, bg
    integer :: n, seed, k, nl, nplc, nptot, nplg, nptg, nlost
    double precision :: rc(6),cc(6),rg(6),cg(6)
    n = 1000000
    seed = 99
    call make_bunch(bc,n,seed,0.0d0,0.5d0,0.01d0)
    call copy_bunch(bc,bg)
    call singlerange(bc%Pts1,n,n,rc,cc)
    call mark_host_newer(1)
    call singlerange_gpu(1,bg,n,n,rg,cg)
    call report("range min/max",rc,rg,6,0.0d0)
    call report("centroid",cc,cg,6,1.0d-13)
    deallocate(bc%Pts1,bg%Pts1)

    do nlost = 0, 37, 37
      n = 100000
      call make_bunch(bc,n,seed,0.0001d0,0.5d0,0.01d0)
      ! lose nlost particles: out in x, y, past z, z<0 with pz<0
      do k = 1, nlost
        select case(mod(k,4))
        case(0)
          bc%Pts1(1,5*k) = 1.0d0/Scxlt
        case(1)
          bc%Pts1(3,5*k) = -1.0d0/Scxlt
        case(2)
          bc%Pts1(5,5*k) = 1.0d6/Scxlt
        case(3)
          bc%Pts1(5,5*k) = -0.001d0/Scxlt
          bc%Pts1(6,5*k) = -0.3d0
        end select
      enddo
      call copy_bunch(bc,bg)
      call lostREC_BeamBunch(bc,0.05d0,0.05d0,100.0d0,0.0d0,nplc,nptot)
      call mark_host_newer(1)
      call lost_gpu(1,bg,0,0.05d0,0.05d0,100.0d0,0.0d0,nplg,nptg)
      call ensure_host(1,bg)
      print*,"lost test: expected lost ",nlost," cpu nplc ",nplc," gpu nplc ",nplg
      if(nplc.ne.nplg .or. nptot.ne.nptg .or. bc%Nptlocal.ne.bg%Nptlocal) then
        print*,"   FAIL: counts differ"
        nfail = nfail + 1
      endif
      call report("lost compacted x",bc%Pts1(1,1:nplc),bg%Pts1(1,1:nplc),nplc,0.0d0)
      call report("lost compacted pz",bc%Pts1(6,1:nplc),bg%Pts1(6,1:nplc),nplc,0.0d0)
      if(bc%Current.ne.bg%Current) then
        print*,"   FAIL: Current differs ",bc%Current,bg%Current
        nfail = nfail + 1
      endif
      deallocate(bc%Pts1,bg%Pts1)
    enddo
  end subroutine test_range_lost

  subroutine test_mesh()
    type (BeamBunch) :: bg
    integer, parameter :: nx = 20, ny = 18, nz = 24
    integer :: n, seed, k, i, j, l, ix, jx, kx, flagpos, fp
    double precision :: range(6), msize(3)
    double precision :: ab, cd, ef, ww, hxi, hyi, hzi, tot1, tot2, gammaz
    double precision :: rho(nx,ny,nz), rref(nx,ny,nz)
    double precision, allocatable :: hp(:,:)
    double precision :: phi(nx,ny,nz), egx, egy, egz, betC, hxg, hyg, hzg
    double precision :: ex(nx,ny,nz),ey(nx,ny,nz),ez(nx,ny,nz),bx(nx,ny,nz),by(nx,ny,nz)
    double precision :: exr(nx,ny,nz),eyr(nx,ny,nz),ezr(nx,ny,nz),bxr(nx,ny,nz),byr(nx,ny,nz)
    double precision :: g2(nx,ny,nz)

    n = 300000
    seed = 31
    call make_bunch(bg,n,seed,-0.0005d0,0.003d0,0.0008d0)
    range = (/-0.001d0/Scxlt,0.001d0/Scxlt,-0.001d0/Scxlt,0.001d0/Scxlt, &
              -0.0006d0/Scxlt,0.0035d0/Scxlt/)
    msize = (/(range(2)-range(1))/(nx-1),(range(4)-range(3))/(ny-1), &
              (range(6)-range(5))/(nz-1)/)
    allocate(hp(9,n))
    hp = bg%Pts1(:,1:n)
    do fp = 0, 1
      flagpos = fp
      call mark_host_newer(1)
      call deposit_gpu(1,bg,nx,ny,nz,range,msize,flagpos)
      rho = rho_dev
      call check_gpu_error()
      ! reference: deposit1(p)_Depositor
      rref = 0.0d0
      hxi = 1.0d0/msize(1)
      hyi = 1.0d0/msize(2)
      hzi = 1.0d0/msize(3)
      do k = 1, n
        if(fp.eq.1 .and. .not.(hp(5,k).gt.0.0d0)) cycle
        ix=(hp(1,k)-range(1))*hxi + 1
        ab=((range(1)-hp(1,k))+ix*msize(1))*hxi
        jx=(hp(3,k)-range(3))*hyi + 1
        cd=((range(3)-hp(3,k))+jx*msize(2))*hyi
        kx=(hp(5,k)-range(5))*hzi + 1
        ef=((range(5)-hp(5,k))+kx*msize(3))*hzi
        if(ix.lt.1 .or. ix+1.gt.nx .or. jx.lt.1 .or. jx+1.gt.ny .or. kx.lt.1 .or. kx+1.gt.nz) then
          print*,"test setup: particle outside the grid"
          stop
        endif
        ww = hp(8,k)
        rref(ix,jx,kx) = rref(ix,jx,kx) + ab*cd*ef*ww
        rref(ix,jx+1,kx) = rref(ix,jx+1,kx) + ab*(1.0d0-cd)*ef*ww
        rref(ix,jx+1,kx+1) = rref(ix,jx+1,kx+1)+ab*(1.0d0-cd)*(1.0d0-ef)*ww
        rref(ix,jx,kx+1) = rref(ix,jx,kx+1)+ab*cd*(1.0d0-ef)*ww
        rref(ix+1,jx,kx+1) = rref(ix+1,jx,kx+1)+(1.0d0-ab)*cd*(1.0d0-ef)*ww
        rref(ix+1,jx+1,kx+1) = rref(ix+1,jx+1,kx+1)+(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)*ww
        rref(ix+1,jx+1,kx) = rref(ix+1,jx+1,kx)+(1.0d0-ab)*(1.0d0-cd)*ef*ww
        rref(ix+1,jx,kx) = rref(ix+1,jx,kx)+(1.0d0-ab)*cd*ef*ww
      enddo
      tot1 = sum(rho)
      tot2 = sum(rref)
      write(*,'(a,i1,a,es10.3,a,es10.3)') "deposit flagpos=",fp," max|d|/max|rho| ", &
        maxval(abs(rho-rref))/maxval(abs(rref))," total charge rel diff ",abs(tot1-tot2)/tot2
      if(maxval(abs(rho-rref))/maxval(abs(rref)).gt.1.0d-14 .or. &
         abs(tot1-tot2)/tot2.gt.1.0d-14) then
        print*,"   FAIL"
        nfail = nfail + 1
      endif
    enddo

    ! gradEB_dev against the 1-rank, Flagbc = 1 branches of gradEB_FieldQuant
    gammaz = 3.7d0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx
          phi(i,j,k) = sin(0.3d0*i)*cos(0.2d0*j)+0.01d0*k*k*sin(0.11d0*i*j)
        enddo
      enddo
    enddo
    hxg = 1.0d0/(msize(1)*Scxlt)
    hyg = 1.0d0/(msize(2)*Scxlt)
    hzg = 1.0d0/(msize(3)*Scxlt*gammaz)
    do l = 0, 1
      phi_dev = phi
      ex_dev = 0.0d0
      ey_dev = 0.0d0
      ez_dev = 0.0d0
      bx_dev = 0.0d0
      by_dev = 0.0d0
      call set_device_mesh(nx,ny,nz)
      phi_dev = phi
      ex_dev = 0.0d0
      ey_dev = 0.0d0
      ez_dev = 0.0d0
      bx_dev = 0.0d0
      by_dev = 0.0d0
      call gradEB_dev(nx,ny,nz,phi_dev,hxg,hyg,hzg,gammaz,l,ex_dev,ey_dev,ez_dev,bx_dev,by_dev)
      ex = ex_dev
      ey = ey_dev
      ez = ez_dev
      bx = bx_dev
      by = by_dev
      if(l.ne.1) then
        betC = sqrt(gammaz**2-1.0)/gammaz/Clight
      else
        betC = -sqrt(gammaz**2-1.0)/gammaz/Clight
      endif
      exr = 0.0d0
      eyr = 0.0d0
      ezr = 0.0d0
      bxr = 0.0d0
      byr = 0.0d0
      do k = 1, nz
        do j = 1, ny
          egx = hxg*(1.5d0*phi(1,j,k)-2.0d0*phi(2,j,k)+0.5d0*phi(3,j,k))
          g2(1,j,k) = egx
          do i = 2, nx-1
            g2(i,j,k) = 0.5d0*hxg*(phi(i-1,j,k)-phi(i+1,j,k))
          enddo
          g2(nx,j,k) = hxg*(-0.5d0*phi(nx-2,j,k)+2.0d0*phi(nx-1,j,k)-1.5d0*phi(nx,j,k))
        enddo
      enddo
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            exr(i,j,k) = exr(i,j,k) + gammaz*g2(i,j,k)
            byr(i,j,k) = byr(i,j,k) + gammaz*g2(i,j,k)*betC
          enddo
        enddo
      enddo
      do k = 1, nz
        do i = 1, nx
          g2(i,1,k) = hyg*(1.5d0*phi(i,1,k)-2*phi(i,2,k)+0.5d0*phi(i,3,k))
          do j = 2, ny-1
            g2(i,j,k) = 0.5d0*hyg*(phi(i,j-1,k)-phi(i,j+1,k))
          enddo
          g2(i,ny,k) = hyg*(-0.5d0*phi(i,ny-2,k)+2*phi(i,ny-1,k)-1.5d0*phi(i,ny,k))
        enddo
      enddo
      do k = 1, nz
        do j = 1, ny
          do i = 1, nx
            eyr(i,j,k) = eyr(i,j,k) + gammaz*g2(i,j,k)
            bxr(i,j,k) = bxr(i,j,k) - gammaz*g2(i,j,k)*betC
          enddo
        enddo
      enddo
      do j = 1, ny
        do i = 1, nx
          g2(i,j,1) = hzg*(1.5d0*phi(i,j,1)-2*phi(i,j,2)+0.5d0*phi(i,j,3))
          do k = 2, nz-1
            g2(i,j,k) = 0.5d0*hzg*(phi(i,j,k-1)-phi(i,j,k+1))
          enddo
          g2(i,j,nz) = hzg*(-0.5d0*phi(i,j,nz-2)+2*phi(i,j,nz-1)-1.5d0*phi(i,j,nz))
        enddo
      enddo
      ezr = ezr + g2
      call report("gradEB ex",reshape(exr,(/nx*ny*nz/)),reshape(ex,(/nx*ny*nz/)),nx*ny*nz,1.0d-15)
      call report("gradEB ey",reshape(eyr,(/nx*ny*nz/)),reshape(ey,(/nx*ny*nz/)),nx*ny*nz,1.0d-15)
      call report("gradEB ez",reshape(ezr,(/nx*ny*nz/)),reshape(ez,(/nx*ny*nz/)),nx*ny*nz,1.0d-15)
      call report("gradEB bx",reshape(bxr,(/nx*ny*nz/)),reshape(bx,(/nx*ny*nz/)),nx*ny*nz,1.0d-15)
      call report("gradEB by",reshape(byr,(/nx*ny*nz/)),reshape(by,(/nx*ny*nz/)),nx*ny*nz,1.0d-15)
    enddo
    deallocate(hp)
  end subroutine test_mesh

end program test_particle_gpu
