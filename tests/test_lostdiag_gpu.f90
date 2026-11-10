! Test of the device-side removal of lost particles (lost_gpu -> compact_lost)
! and of the Flagdiag = 1 moments on the device (diagnostic1avg_Output with
! devok, diag1_pass1_gpu, diag1_pass2_gpu), against the host routines
! (1 rank, 1 GPU).  Not part of the CMake build; see tests/README.md.  No
! input files; it writes lostdiag_{host,dev}.<unit> in the run directory.
!
!   A. lost_gpu against lostREC_BeamBunch (idm 0) and lostDFO_BeamBunch
!      (idm 1): beams with losses on every boundary, also exactly on it
!      (x = -xrad, x = xrad, y = +-yrad, r = xrad, z = zleng, z = 0 with
!      pz < 0) and the particles that must stay (z <= 0 with pz >= 0); the
!      first and the last particle lost, a whole block of 256 lost, 0 lost,
!      a few, about 40 %, all; a second loss call on the compacted device
!      beam without an upload between; a beam larger than the device
!      buffers (reallocation).  Survivors: all 9 columns bit-identical, in
!      the host order (ids included); Nptlocal, nplc, nptot, Npt, Current
!      identical; the device copy stays the newer one (no re-upload).
!   B. the device sums of diag1_pass1_gpu / diag1_pass2_gpu against the
!      host loop of diagnostic1avg_Output transcribed here: each of the 35
!      sums within 1e-12 of the sum of the absolute values of its terms
!      (rounding of a sum in another order), the 6 maxima and max r^2
!      bit-identical; the 6 first-pass sums (1e6 particles, all-positive
!      columns, where the host's sequential sum is itself off by up to a few
!      1e-12) within 1e-12 of the sum of |terms| from a compensated (Neumaier)
!      sum, with the host loop's own error printed beside it.
!   C. the real diagnostic1avg_Output, devok true against host, two bunches,
!      at three times (before the losses, after lost_gpu, after a second
!      loss with idm 1): every number of fort.18 and fort.24-32 as written
!      (e16.8), within one unit of the last printed digit (1.5e-7 relative);
!      fort.28's counts identical.  The device run downloads nothing.
module tld_util
  implicit none
contains
  function nrand(seed) result(r)
    integer, intent(inout) :: seed
    double precision :: r
    seed = mod(seed*1103515245_8+12345_8,2147483648_8)
    r = dble(seed)/2147483648.0d0
  end function nrand
end module tld_util

program test_lostdiag_gpu
  use cudafor
  use BeamBunchclass
  use PhysConstclass
  use Timerclass
  use Outputclass
  use ParticleGPUclass
  use tld_util
  implicit none
  include 'mpif.h'
  integer :: ierr, nfail, seed
  logical :: ok

  call MPI_INIT(ierr)
  nfail = 0
  call constructT_PhysConst(2.5d-13,9.3d9)
  call construct_Timer(0.0d0)
  ok = gpu_particles_enabled()
  if(.not.ok) then
    print*,"test_lostdiag_gpu: the GPU particle path is off (IMPACTT_PARTICLES/IMPACTT_POISSON)"
    call MPI_FINALIZE(ierr)
    stop
  endif
  if(.not.loss_on_device()) print*,"note: IMPACTT_LOSS=host is set: test A then checks the host fallback"
  call gpu_particles_init(2)
  seed = 12345

  call test_lost()
  call test_sums()
  call test_diag_files()

  call report_gpu_stats(0)
  call showtime_Timer()
  if(nfail.eq.0) then
    print*,"test_lostdiag_gpu: ALL PASS"
  else
    print*,"test_lostdiag_gpu: FAILURES: ",nfail
  endif
  call MPI_FINALIZE(ierr)

contains

  subroutine new_bunch(b,n)
    type (BeamBunch), intent(inout) :: b
    integer, intent(in) :: n
    call construct_BeamBunch(b,1.0d-5,1.0d6,0.511005d6,-1.0d0,n,0.0d0)
    if(allocated(b%Pts1)) deallocate(b%Pts1)
    allocate(b%Pts1(9,n))
    b%Nptlocal = n
    b%Npt = n
  end subroutine new_bunch

  subroutine copy_bunch(a,b)
    type (BeamBunch), intent(in) :: a
    type (BeamBunch), intent(inout) :: b
    b%Current = a%Current
    b%Mass = a%Mass
    b%Charge = a%Charge
    b%Npt = a%Npt
    b%Nptlocal = a%Nptlocal
    b%refptcl = a%refptcl
    if(allocated(b%Pts1)) deallocate(b%Pts1)
    allocate(b%Pts1(9,size(a%Pts1,2)))
    b%Pts1 = a%Pts1
  end subroutine copy_bunch

  !> a beam in units of Scxlt: x, y uniform in +-fxy*rad, z in
  !> [zlo, zhi]*zleng, pz > 0 except a fraction fneg with pz < 0
  subroutine fill_beam(b,n,rad,zleng,fxy,zlo,zhi,fneg)
    type (BeamBunch), intent(inout) :: b
    integer, intent(in) :: n
    double precision, intent(in) :: rad,zleng,fxy,zlo,zhi,fneg
    integer :: k
    call new_bunch(b,n)
    do k = 1, n
      b%Pts1(1,k) = fxy*rad*(2.0d0*nrand(seed)-1.0d0)/Scxlt
      b%Pts1(3,k) = fxy*rad*(2.0d0*nrand(seed)-1.0d0)/Scxlt
      b%Pts1(5,k) = (zlo+(zhi-zlo)*nrand(seed))*zleng/Scxlt
      b%Pts1(2,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(4,k) = 0.3d0*(nrand(seed)-0.5d0)
      b%Pts1(6,k) = 0.1d0+1.5d0*nrand(seed)
      if(nrand(seed).lt.fneg) b%Pts1(6,k) = -b%Pts1(6,k)
      b%Pts1(7,k) = -1.0d0/0.511005d6
      b%Pts1(8,k) = 1.0d-19*(1.0d0+nrand(seed))
      b%Pts1(9,k) = k
    enddo
  end subroutine fill_beam

  !> particles exactly on the boundaries and the ones that must stay
  subroutine edge_cases(b,rad,zleng,idm)
    type (BeamBunch), intent(inout) :: b
    double precision, intent(in) :: rad,zleng
    integer, intent(in) :: idm
    integer :: n
    double precision :: p1
    n = b%Nptlocal
    p1 = -rad/Scxlt
    b%Pts1(1,1) = 2.0d0*rad/Scxlt          ! first particle lost (x)
    b%Pts1(5,n) = 2.0d0*zleng/Scxlt        ! last particle lost (z)
    b%Pts1(1,10) = p1                      ! x = -xrad exactly (lost; r = rad for idm 1)
    b%Pts1(3,10) = 0.0d0
    b%Pts1(1,11) = rad/Scxlt               ! x = +xrad exactly
    b%Pts1(3,11) = 0.0d0
    b%Pts1(1,12) = 0.0d0                   ! y = -yrad exactly
    b%Pts1(3,12) = -rad/Scxlt
    b%Pts1(1,13) = 0.0d0                   ! y = +yrad exactly
    b%Pts1(3,13) = rad/Scxlt
    b%Pts1(5,14) = zleng/Scxlt             ! z = zleng exactly (lost)
    b%Pts1(5,15) = 0.0d0                   ! z = 0, pz < 0 (lost)
    b%Pts1(6,15) = -0.2d0
    b%Pts1(5,16) = 0.0d0                   ! z = 0, pz = 0 (stays)
    b%Pts1(6,16) = 0.0d0
    b%Pts1(5,17) = -1.0d-3/Scxlt           ! z < 0, pz > 0 (stays: not yet emitted)
    b%Pts1(6,17) = 0.2d0
    b%Pts1(5,18) = -1.0d-3/Scxlt           ! z < 0, pz < 0 (lost)
    b%Pts1(6,18) = -0.2d0
    b%Pts1(1,19) = 0.0d0                   ! inside, just
    b%Pts1(3,19) = 0.999999d0*rad/Scxlt
    if(idm.eq.1) then                      ! r >= rad with |x|, |y| < rad (lost for DFO only)
      b%Pts1(1,20) = 0.8d0*rad/Scxlt
      b%Pts1(3,20) = 0.8d0*rad/Scxlt
    endif
  end subroutine edge_cases

  !> one loss call on both sides and the comparison
  subroutine lost_both(name,bc,bg,idm,rad,zleng)
    character(len=*), intent(in) :: name
    type (BeamBunch), intent(inout) :: bc,bg
    integer, intent(in) :: idm
    double precision, intent(in) :: rad,zleng
    integer :: nplc,nptot,nplg,nptg,n0,nbad,k,st
    n0 = bc%Nptlocal
    if(idm.eq.0) then
      call lostREC_BeamBunch(bc,rad,rad,zleng,0.0d0,nplc,nptot)
    else
      call lostDFO_BeamBunch(bc,rad,rad,zleng,0.0d0,nplc,nptot)
    endif
    call lost_gpu(1,bg,idm,rad,rad,zleng,0.0d0,nplg,nptg)
    st = ptsbuf_state(1)
    call ensure_host(1,bg)
    nbad = 0
    do k = 1, min(nplc,nplg)
      if(any(bc%Pts1(1:9,k).ne.bg%Pts1(1:9,k))) nbad = nbad + 1
    enddo
    write(*,'(a,a32,a,i8,a,i8,a,i8,a,i2,a,i0)') "A ",name," n ",n0," lost ",n0-nplc, &
      " kept ",nplc," state ",st," differing survivors ",nbad
    if(nplc.ne.nplg .or. nptot.ne.nptg .or. bc%Nptlocal.ne.bg%Nptlocal .or. &
       bc%Npt.ne.bg%Npt) then
      print*,"   FAIL: counts differ: cpu ",nplc,nptot,bc%Nptlocal,bc%Npt, &
             " gpu ",nplg,nptg,bg%Nptlocal,bg%Npt
      nfail = nfail + 1
    endif
    if(bc%Current.ne.bg%Current) then
      print*,"   FAIL: Current differs ",bc%Current,bg%Current
      nfail = nfail + 1
    endif
    if(nbad.ne.0) then
      print*,"   FAIL: survivors differ"
      nfail = nfail + 1
    endif
    if(st.eq.2 .and. loss_on_device()) then
      print*,"   FAIL: the device copy is marked stale (host compaction)"
      nfail = nfail + 1
    endif
  end subroutine lost_both

  subroutine test_lost()
    type (BeamBunch) :: bc, bg
    double precision :: rad, zleng
    integer :: idm, k
    character(len=8) :: tag
    rad = 0.01d0
    zleng = 2.0d0
    do idm = 0, 1
      write(tag,'(a,i1)') " idm ",idm
      ! no loss
      call fill_beam(bc,1000003,rad,zleng,0.6d0,0.001d0,0.5d0,0.0d0)
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("none"//tag,bc,bg,idm,rad,zleng)
      ! a few, on the edges
      call fill_beam(bc,1000003,rad,zleng,0.6d0,0.001d0,0.5d0,0.0d0)
      call edge_cases(bc,rad,zleng,idm)
      do k = 1, 37
        bc%Pts1(1,1000+26000*k) = 1.5d0*rad/Scxlt
      enddo
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("edges + 37"//tag,bc,bg,idm,rad,zleng)
      ! about 40 %, a whole block lost, then a second call on the compacted
      ! device beam with a smaller radius (no upload between)
      call fill_beam(bc,1000003,rad,zleng,1.1d0,-0.1d0,1.1d0,0.3d0)
      call edge_cases(bc,rad,zleng,idm)
      do k = 257, 512
        bc%Pts1(5,k) = 3.0d0*zleng/Scxlt
      enddo
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("many"//tag,bc,bg,idm,rad,zleng)
      call lost_both("many, again at 0.7 rad"//tag,bc,bg,idm,0.7d0*rad,zleng)
      call lost_both("many, again, none lost"//tag,bc,bg,idm,0.7d0*rad,zleng)
      ! all lost
      call fill_beam(bc,300,rad,zleng,0.6d0,1.2d0,1.5d0,0.0d0)
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("all"//tag,bc,bg,idm,rad,zleng)
      ! a small beam, then one larger than the device buffers
      call fill_beam(bc,1001,rad,zleng,1.1d0,-0.1d0,1.1d0,0.3d0)
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("1001"//tag,bc,bg,idm,rad,zleng)
      call fill_beam(bc,1500000+idm*500001,rad,zleng,1.1d0,-0.1d0,1.1d0,0.3d0)
      call copy_bunch(bc,bg)
      call mark_host_newer(1)
      call lost_both("larger than the buffers"//tag,bc,bg,idm,rad,zleng)
    enddo
    deallocate(bc%Pts1,bg%Pts1)
  end subroutine test_lost

  !> the host loop of diagnostic1avg_Output (local sums), transcribed, with
  !> the sums of the absolute values of each term for the tolerance
  subroutine host_sums(b,m0,s0,tmplc,absl,localmax,lcrmax)
    type (BeamBunch), intent(in) :: b
    double precision, intent(in) :: m0(6)
    double precision, intent(out) :: s0(6),tmplc(35),absl(35),localmax(6),lcrmax
    double precision :: x0,px0,y0,py0,z0,pz0,z0avg,tmpgam,tmpx,tmpy,t(35)
    integer :: i,j
    s0 = 0.0d0
    do i = 1, b%Nptlocal
      s0(:) = s0(:) + b%Pts1(1:6,i)
    enddo
    x0 = m0(1); px0 = m0(2); y0 = m0(3); py0 = m0(4); z0 = m0(5); pz0 = m0(6)
    z0avg = z0
    tmplc = 0.0d0
    absl = 0.0d0
    localmax = 0.0d0
    lcrmax = 0.0d0
    do i = 1, b%Nptlocal
      tmpgam = sqrt(1.0d0+b%Pts1(2,i)**2+b%Pts1(4,i)**2+b%Pts1(6,i)**2)
      tmpx = b%Pts1(1,i)
      tmpy = b%Pts1(3,i)
      t(22) = tmpgam
      t(23) = tmpgam**2
      t(1) = (tmpx-x0)**2
      t(10) = (tmpx-x0)**3
      t(11) = (tmpx-x0)**4
      t(7) = (tmpx-x0)*(b%Pts1(2,i)-px0)
      t(2) = (b%Pts1(2,i)-px0)**2
      t(12) = (b%Pts1(2,i)-px0)**3
      t(13) = (b%Pts1(2,i)-px0)**4
      t(3) = (tmpy-y0)**2
      t(14) = (tmpy-y0)**3
      t(15) = (tmpy-y0)**4
      t(8) = (tmpy-y0)*(b%Pts1(4,i)-py0)
      t(4) = (b%Pts1(4,i)-py0)**2
      t(16) = (b%Pts1(4,i)-py0)**3
      t(17) = (b%Pts1(4,i)-py0)**4
      t(24) = (tmpx-x0)*(tmpy-y0)
      t(25) = (tmpx-x0)*(b%Pts1(4,i)-py0)
      t(26) = (b%Pts1(2,i)-px0)*(tmpy-y0)
      t(27) = (b%Pts1(2,i)-px0)*(b%Pts1(4,i)-py0)
      t(28) = (tmpx-x0)*(b%Pts1(5,i)-z0)
      t(29) = (tmpx-x0)*(b%Pts1(6,i)-pz0)
      t(30) = (b%Pts1(5,i)-z0)*(b%Pts1(2,i)-px0)
      t(31) = (b%Pts1(2,i)-px0)*(b%Pts1(6,i)-pz0)
      t(32) = (tmpy-y0)*(b%Pts1(5,i)-z0)
      t(33) = (tmpy-y0)*(b%Pts1(6,i)-pz0)
      t(34) = (b%Pts1(5,i)-z0)*(b%Pts1(4,i)-py0)
      t(35) = (b%Pts1(4,i)-py0)*(b%Pts1(6,i)-pz0)
      t(5) = (b%Pts1(5,i)-z0avg)**2
      t(18) = (b%Pts1(5,i)-z0avg)**3
      t(19) = (b%Pts1(5,i)-z0avg)**4
      t(9) = (b%Pts1(5,i)-z0avg)*(b%Pts1(6,i)-pz0)
      t(6) = (b%Pts1(6,i)-pz0)**2
      t(20) = (b%Pts1(6,i)-pz0)**3
      t(21) = (b%Pts1(6,i)-pz0)**4
      tmplc = tmplc + t
      absl = absl + abs(t)
      do j = 1, 4
        if(localmax(j).lt.abs(b%Pts1(j,i))) localmax(j) = abs(b%Pts1(j,i))
      enddo
      if(localmax(5).lt.abs(b%Pts1(5,i)-z0avg)) localmax(5) = abs(b%Pts1(5,i)-z0avg)
      if(localmax(6).lt.abs(b%Pts1(6,i))) localmax(6) = abs(b%Pts1(6,i))
      if(lcrmax.lt.(tmpx**2+tmpy**2)) lcrmax = tmpx**2 + tmpy**2
    enddo
  end subroutine host_sums

  subroutine test_sums()
    type (BeamBunch) :: bc, bg(1)
    double precision :: m0(6),s0(6),g0(6),a0(6),tl(35),gl(35),al(35),lm(6),gm(6),lr,gr,dmax,d
    double precision, volatile :: e0(6),c0(6),tt   ! volatile: -fast must not reassociate the compensation
    integer :: k,kw,n
    n = 1000000
    call fill_beam(bc,n,0.01d0,2.0d0,0.6d0,0.2d0,0.4d0,0.0d0)
    do k = 1, n                       ! an off-axis, correlated beam
      bc%Pts1(1,k) = bc%Pts1(1,k) + 3.0d-3/Scxlt
      bc%Pts1(2,k) = bc%Pts1(2,k) + 2.0d0*bc%Pts1(1,k)*Scxlt
      bc%Pts1(6,k) = 20.0d0 + bc%Pts1(6,k) + 5.0d0*bc%Pts1(5,k)*Scxlt
    enddo
    call copy_bunch(bc,bg(1))
    call mark_host_newer(1)
    call diag1_pass1_gpu(bg,1,g0)
    m0 = 0.0d0
    call host_sums(bc,m0,s0,tl,al,lm,lr)
    a0 = 0.0d0
    e0 = 0.0d0
    c0 = 0.0d0
    do k = 1, n
      a0 = a0 + abs(bc%Pts1(1:6,k))
      do kw = 1, 6                    ! Neumaier compensated sum: the reference
        tt = e0(kw) + bc%Pts1(kw,k)
        if(abs(e0(kw)).ge.abs(bc%Pts1(kw,k))) then
          c0(kw) = c0(kw) + ((e0(kw) - tt) + bc%Pts1(kw,k))
        else
          c0(kw) = c0(kw) + ((bc%Pts1(kw,k) - tt) + e0(kw))
        endif
        e0(kw) = tt
      enddo
    enddo
    e0 = e0 + c0
    write(*,'(a,es10.3,a,es10.3)') "B pass 1, 6 sums, host loop |cpu-exact|/sum|terms| ", &
      maxval(abs(s0-e0)/max(a0,1.0d-300)),"  |gpu-cpu|/sum|terms| ",maxval(abs(g0-s0)/max(a0,1.0d-300))
    dmax = maxval(abs(g0-e0)/max(a0,1.0d-300))
    write(*,'(a,es10.3)') "B pass 1, 6 sums, max |gpu-exact|/sum|terms| ",dmax
    if(dmax.gt.1.0d-12) then
      print*,"   FAIL"
      nfail = nfail + 1
    endif
    m0 = s0/n
    call host_sums(bc,m0,s0,tl,al,lm,lr)
    call diag1_pass2_gpu(bg,1,m0(5),m0,gl,gm,gr)
    dmax = 0.0d0
    kw = 0
    do k = 1, 35
      d = abs(gl(k)-tl(k))/max(al(k),1.0d-300)
      if(d.gt.dmax) then
        dmax = d
        kw = k
      endif
    enddo
    write(*,'(a,es10.3,a,i0,a)') "B pass 2, 35 sums, max |gpu-cpu|/sum|terms| ",dmax," (sum ",kw,")"
    if(dmax.gt.1.0d-12) then
      print*,"   FAIL"
      nfail = nfail + 1
    endif
    write(*,'(a,l1,a,l1)') "B maxima identical ",all(gm.eq.lm)," max r^2 identical ",gr.eq.lr
    if(any(gm.ne.lm) .or. gr.ne.lr) then
      print*,"   FAIL: maxima ",gm,lm,gr,lr
      nfail = nfail + 1
    endif
    if(ptsbuf_state(1).eq.2) then
      print*,"   FAIL: device copy stale after the diagnostic"
      nfail = nfail + 1
    endif
    deallocate(bc%Pts1,bg(1)%Pts1)
  end subroutine test_sums

  subroutine open_diag(tag)
    character(len=*), intent(in) :: tag
    integer :: u
    character(len=64) :: fn
    do u = 18, 32
      if(u.ge.19 .and. u.le.23) cycle
      write(fn,'(a,a,a,i0)') "lostdiag_",tag,".",u
      open(unit=u,file=trim(fn),status='replace')
    enddo
  end subroutine open_diag

  subroutine close_diag()
    integer :: u
    do u = 18, 32
      if(u.ge.19 .and. u.le.23) cycle
      close(u)
    enddo
  end subroutine close_diag

  !> the files of the host and the device run, number by number
  subroutine compare_diag(nlines)
    integer, intent(in) :: nlines
    integer :: u,nc,l,k,ios,nsame,nnum,nbad
    character(len=512) :: la,lb
    double precision :: a(24),b(24),d,dmax
    character(len=64) :: fa,fb
    do u = 18, 32
      if(u.ge.19 .and. u.le.23) cycle
      select case(u)
      case(18,26); nc = 7
      case(28); nc = 5
      case(31); nc = 13
      case(32); nc = 24
      case default; nc = 8
      end select
      write(fa,'(a,i0)') "lostdiag_host.",u
      write(fb,'(a,i0)') "lostdiag_dev.",u
      open(unit=91,file=trim(fa),status='old')
      open(unit=92,file=trim(fb),status='old')
      nsame = 0
      nnum = 0
      nbad = 0
      dmax = 0.0d0
      do l = 1, nlines
        read(91,'(a)',iostat=ios) la
        if(ios.ne.0) exit
        read(92,'(a)',iostat=ios) lb
        if(ios.ne.0) exit
        if(la.eq.lb) nsame = nsame + 1
        read(la,*) a(1:nc)
        read(lb,*) b(1:nc)
        do k = 1, nc
          nnum = nnum + 1
          d = abs(a(k)-b(k))/max(abs(a(k)),1.0d-300)
          if(a(k).eq.b(k)) d = 0.0d0
          dmax = max(dmax,d)
          if(u.eq.28 .and. k.ge.3 .and. a(k).ne.b(k)) nbad = nbad + 1
          if(d.gt.1.5d-7) nbad = nbad + 1
        enddo
      enddo
      close(91)
      close(92)
      write(*,'(a,i2,a,i0,a,i0,a,i0,a,es10.3)') "C fort.",u,": lines identical ",nsame," of ",nlines, &
        ", numbers ",nnum,", max relative difference ",dmax
      if(nbad.gt.0 .or. l.le.nlines) then
        print*,"   FAIL: ",nbad," numbers differ by more than the last printed digit (or a line is missing)"
        nfail = nfail + 1
      endif
    enddo
  end subroutine compare_diag

  subroutine test_diag_files()
    type (BeamBunch) :: hc(2), gb(2)
    integer :: ib,k,nplc,nptot
    double precision :: rad,zleng,t
    rad = 0.01d0
    zleng = 2.0d0
    do ib = 1, 2
      call fill_beam(hc(ib),400000+ib*100001,rad,zleng,1.05d0,0.1d0+0.05d0*ib,0.3d0,0.01d0)
      do k = 1, hc(ib)%Nptlocal
        hc(ib)%Pts1(2,k) = hc(ib)%Pts1(2,k) + 1.5d0*hc(ib)%Pts1(1,k)*Scxlt
        hc(ib)%Pts1(6,k) = 30.0d0 + hc(ib)%Pts1(6,k)
      enddo
      call copy_bunch(hc(ib),gb(ib))
      call mark_host_newer(ib)
    enddo
    do k = 1, 3
      t = 1.0d-9*k
      if(k.eq.2) then
        do ib = 1, 2
          call lostREC_BeamBunch(hc(ib),rad,rad,zleng,0.0d0,nplc,nptot)
          call lost_gpu(ib,gb(ib),0,rad,rad,zleng,0.0d0,nplc,nptot)
        enddo
      else if(k.eq.3) then
        do ib = 1, 2
          call lostDFO_BeamBunch(hc(ib),0.8d0*rad,0.8d0*rad,zleng,0.0d0,nplc,nptot)
          call lost_gpu(ib,gb(ib),1,0.8d0*rad,0.8d0*rad,zleng,0.0d0,nplc,nptot)
        enddo
      endif
      call open_diag_append("host",k)
      call diagnostic1avg_Output(t,hc,2)
      call close_diag()
      call open_diag_append("dev",k)
      call diagnostic1avg_Output(t,gb,2,.true.)
      call close_diag()
      write(*,'(a,i1,a,2i8,a,2i8,a,2i2)') "C time ",k,": host Nptlocal ",hc(1)%Nptlocal,hc(2)%Nptlocal, &
        "  device ",gb(1)%Nptlocal,gb(2)%Nptlocal,"  device states ",ptsbuf_state(1),ptsbuf_state(2)
      if(ptsbuf_state(1).eq.2 .or. ptsbuf_state(2).eq.2) then
        print*,"   FAIL: device copy stale"
        nfail = nfail + 1
      endif
    enddo
    call compare_diag(3)
    do ib = 1, 2
      deallocate(hc(ib)%Pts1,gb(ib)%Pts1)
    enddo
  end subroutine test_diag_files

  subroutine open_diag_append(tag,k)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: k
    integer :: u
    character(len=64) :: fn
    if(k.eq.1) then
      call open_diag(tag)
      return
    endif
    do u = 18, 32
      if(u.ge.19 .and. u.le.23) cycle
      write(fn,'(a,a,a,i0)') "lostdiag_",tag,".",u
      open(unit=u,file=trim(fn),status='old',position='append')
    enddo
  end subroutine open_diag_append

end program test_lostdiag_gpu
