!----------------------------------------------------------------
! ParticleGPUclass: per-particle work of an ordinary time step on the GPU
! (CUDA Fortran).  See IMPACTT_GPU_PORT_PLAN.md, B2.3-B2.5.
!
! The particles live on the device for the whole run; Ebunch(ib)%Pts1 on
! the host is a mirror that is refreshed only when host code needs it
! (ensure_host) and uploaded again after a host write (mark_host_newer).
! Every kernel wrapper calls ensure_device first and marks the device copy
! newer if it writes.
!
! The kernels are ports of the CPU routines named at each one: same
! expressions in the same order.  Constants that live in host module
! variables (Scxlt, Clight, ...) are passed as value arguments.
!----------------------------------------------------------------
      module ParticleGPUclass
        use cudafor
        use iso_c_binding
        use BeamBunchclass
        use PhysConstclass
        use Timerclass
        use FieldmapGPUclass
        use PoissonGPUclass
        implicit none
        private
        public :: gpu_particles_init, gpu_particles_enabled
        public :: ensure_device, ensure_host, ensure_host_all
        public :: mark_host_newer, mark_host_newer_all
        public :: drift_start_gpu, drift_end_gpu
        public :: singlerange_gpu, lost_gpu
        public :: mesh_zero_gpu, mesh_fields_gpu
        public :: kick_mesh_gpu, kick_none_gpu, kickpt2pt_gpu
        public :: check_gpu_error, report_gpu_stats
        public :: count_fallback, ptsbuf_state, deposit_gpu, gradEB_mesh_gpu
        public :: kick_mesh_core, set_device_mesh, rho_dev, phi_dev, phiimg_dev
        public :: ex_dev, ey_dev, ez_dev, bx_dev, by_dev

        type ptsbuf
          real(8), device, allocatable :: p(:,:)  ! (9,cap), same layout as Pts1
          integer :: cap = 0                      ! allocated columns
          integer :: n = 0                        ! valid columns (= Nptlocal)
          integer :: state = 2                    ! 0 in sync, 1 device newer, 2 host newer
        end type ptsbuf
        type(ptsbuf), allocatable, save :: pb(:)

        integer, save :: nbuf = 0
        integer, save :: nsync_up = 0, nsync_down = 0, nfallback = 0
        integer, save :: envflag = -1

        ! mesh arrays (global grid, B2.4)
        integer, save :: mnx = 0, mny = 0, mnz = 0
        real(8), device, allocatable, save :: rho_dev(:,:,:), phi_dev(:,:,:), &
                                              phiimg_dev(:,:,:)
        real(8), device, allocatable, save :: ex_dev(:,:,:), ey_dev(:,:,:), &
                                              ez_dev(:,:,:), bx_dev(:,:,:), &
                                              by_dev(:,:,:)

        ! N-body work arrays
        integer, save :: ncap_nb = 0, ncap_src = 0
        real(8), device, allocatable, save :: srcloc_d(:,:), targets_d(:,:), &
                                              sources_d(:,:)
        real(8), device, allocatable, save :: fex(:), fey(:), fez(:), fbx(:), &
                                              fby(:), fbz(:)

      contains

        !> True unless IMPACTT_PARTICLES or IMPACTT_POISSON is "cpu".
        function gpu_particles_enabled() result(flag)
        implicit none
        logical :: flag
        character(len=32) :: val
        integer :: stat
        if(envflag.lt.0) then
          envflag = 1
          call get_environment_variable("IMPACTT_PARTICLES",val,status=stat)
          if(stat.eq.0) then
            if(trim(val).eq."cpu") envflag = 0
          endif
          if(.not.use_gpu_poisson()) then
            if(envflag.eq.1) print*,"IMPACTT_POISSON=cpu implies IMPACTT_PARTICLES=cpu"
            envflag = 0
          endif
        endif
        flag = (envflag.eq.1)
        end function gpu_particles_enabled

        subroutine gpu_particles_init(nbunch)
        implicit none
        integer, intent(in) :: nbunch
        integer :: i
        if(.not.allocated(pb)) then
          allocate(pb(max(nbunch,1)))
          nbuf = max(nbunch,1)
        endif
        do i = 1, nbuf
          pb(i)%state = 2
        enddo
        call fldmap_init()
        end subroutine gpu_particles_init

        function ptsbuf_state(ib) result(s)
        implicit none
        integer, intent(in) :: ib
        integer :: s
        s = pb(ib)%state
        end function ptsbuf_state

        subroutine sync_gpu()
        implicit none
        integer :: istat
        istat = cudaDeviceSynchronize()
        end subroutine sync_gpu

        subroutine count_fallback()
        implicit none
        nfallback = nfallback + 1
        end subroutine count_fallback

        !-------------------------------------------------------------
        ! residency (B2.3)
        !-------------------------------------------------------------
        subroutine ensure_device(ib,bunch)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        integer :: n,istat
        double precision :: t0
        if(pb(ib)%state.ne.2) return
        call starttime_Timer(t0)
        n = bunch%Nptlocal
        if(n.gt.pb(ib)%cap .or. .not.allocated(pb(ib)%p)) then
          if(allocated(pb(ib)%p)) deallocate(pb(ib)%p)
          pb(ib)%cap = max(size(bunch%Pts1,2),n,1)
          allocate(pb(ib)%p(9,pb(ib)%cap))
        endif
        if(n.gt.0) then
          istat = cudaMemcpy(c_devloc(pb(ib)%p),c_loc(bunch%Pts1(1,1)), &
                             int(9,8)*int(n,8)*8_8,cudaMemcpyHostToDevice)
          if(istat.ne.cudaSuccess) then
            print*,"ParticleGPU: upload failed: ",cudaGetErrorString(istat)
            stop
          endif
        endif
        pb(ib)%n = n
        pb(ib)%state = 0
        nsync_up = nsync_up + 1
        t_syncgpu = t_syncgpu + elapsedtime_Timer(t0)
        end subroutine ensure_device

        subroutine ensure_host(ib,bunch)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        integer :: n,istat
        double precision :: t0
        if(pb(ib)%state.ne.1) return
        call starttime_Timer(t0)
        n = min(pb(ib)%n,size(bunch%Pts1,2))
        if(n.gt.0) then
          istat = cudaMemcpy(c_loc(bunch%Pts1(1,1)),c_devloc(pb(ib)%p), &
                             int(9,8)*int(n,8)*8_8,cudaMemcpyDeviceToHost)
          if(istat.ne.cudaSuccess) then
            print*,"ParticleGPU: download failed: ",cudaGetErrorString(istat)
            stop
          endif
        endif
        pb(ib)%state = 0
        nsync_down = nsync_down + 1
        t_syncgpu = t_syncgpu + elapsedtime_Timer(t0)
        end subroutine ensure_host

        subroutine ensure_host_all(bunch,nb)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        integer :: ib
        do ib = 1, nb
          call ensure_host(ib,bunch(ib))
        enddo
        end subroutine ensure_host_all

        subroutine mark_host_newer(ib)
        implicit none
        integer, intent(in) :: ib
        pb(ib)%state = 2
        end subroutine mark_host_newer

        subroutine mark_host_newer_all(nb)
        implicit none
        integer, intent(in) :: nb
        integer :: ib
        do ib = 1, nb
          pb(ib)%state = 2
        enddo
        end subroutine mark_host_newer_all

        subroutine check_gpu_error()
        implicit none
        integer :: code,info(4),istat
        istat = cudaGetLastError()
        if(istat.ne.cudaSuccess) then
          print*,"ParticleGPU: CUDA error: ",cudaGetErrorString(istat)
          stop
        endif
        code = gpuerr_d
        if(code.ne.0) then
          info = gpuerr_info_d
          if(code.eq.1) then
            print*,"dep - index out of range: particle ",info(2)," dim ",info(4)," index ",info(3)
          else if(code.eq.2) then
            print*,"kick - index out of range: particle ",info(2)," dim ",info(4)," index ",info(3)
          else if(code.eq.3) then
            print*,"ir: particle ",info(2)," ir ",info(3)
          else
            print*,"ParticleGPU: error code ",code
          endif
          stop
        endif
        end subroutine check_gpu_error

        subroutine report_gpu_stats(myid)
        implicit none
        integer, intent(in) :: myid
        if(myid.eq.0) then
          write(*,'(a,i0,a,i0,a,i0)') " GPU particle path: host->device copies ", &
                nsync_up,", device->host copies ",nsync_down, &
                ", CPU fallback steps ",nfallback
        endif
        end subroutine report_gpu_stats

        !-------------------------------------------------------------
        ! drifts and emission (B2.5.1)
        !-------------------------------------------------------------
        attributes(global) subroutine k_drift(p,n,tau,betazini,mode)
        implicit none
        integer, value :: n,mode
        real(8), value :: tau,betazini
        real(8), device :: p(9,n)
        integer :: i
        real(8) :: recpgam,tmpzz,dtmp
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        recpgam = 1.0d0/sqrt(1.0d0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
        if(mode.eq.0) then
          ! drifthalforg_BeamBunch
          p(1,i) = p(1,i)+0.5d0*tau*p(2,i)*recpgam
          p(3,i) = p(3,i)+0.5d0*tau*p(4,i)*recpgam
          p(5,i) = p(5,i)+0.5d0*tau*p(6,i)*recpgam
        else if(mode.eq.1) then
          ! drifthalf_BeamBunch: only z > 0
          if(p(5,i).gt.0.0) then
            p(1,i) = p(1,i)+0.5d0*tau*p(2,i)*recpgam
            p(3,i) = p(3,i)+0.5d0*tau*p(4,i)*recpgam
            p(5,i) = p(5,i)+0.5d0*tau*p(6,i)*recpgam
          endif
        else
          ! drifthalf, driftemission and the first-order emission correction
          ! of AccSimulator, in the CPU order for each particle
          if(p(5,i).gt.0.0) then
            p(1,i) = p(1,i)+0.5d0*tau*p(2,i)*recpgam
            p(3,i) = p(3,i)+0.5d0*tau*p(4,i)*recpgam
            p(5,i) = p(5,i)+0.5d0*tau*p(6,i)*recpgam
          endif
          if(p(5,i).le.0.0d0 .and. p(6,i).ge.0.0d0) then
            p(5,i) = p(5,i)+tau*betazini
          endif
          tmpzz = p(5,i)-tau*betazini
          recpgam = 1.0/sqrt(1.0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          if(tmpzz.le.0.0d0 .and. p(5,i).ge.0.0d0) then
            dtmp = tau*p(5,i)/(tau*betazini)
            p(1,i) = p(1,i)+dtmp*p(2,i)*recpgam
            p(3,i) = p(3,i)+dtmp*p(4,i)*recpgam
            p(5,i) = dtmp*p(6,i)*recpgam
          endif
        endif
        end subroutine k_drift

        !> First half drift of a step: drifthalf (only z > 0) while the beam
        !> is still being emitted, else drifthalforg.
        subroutine drift_start_gpu(ib,bunch,tau,flagcathode)
        implicit none
        integer, intent(in) :: ib,flagcathode
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tau
        integer :: mode
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          mode = 0
          if(flagcathode.eq.1) mode = 1
          call k_drift<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,tau,0.0d0,mode)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        end subroutine drift_start_gpu

        !> Second half drift of a step (B2.5.1).
        subroutine drift_end_gpu(ib,bunch,tau,flagcathode,betazini)
        implicit none
        integer, intent(in) :: ib,flagcathode
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tau,betazini
        integer :: mode
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          mode = 0
          if(flagcathode.eq.1) mode = 2
          call k_drift<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,tau,betazini,mode)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        end subroutine drift_end_gpu

        !-------------------------------------------------------------
        ! range and centroid (B2.5.2)
        !-------------------------------------------------------------
        subroutine range_core(p,n,rmn,rmx,sm)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(out) :: rmn(3),rmx(3),sm(6)
        integer :: i
        real(8) :: xmn,ymn,zmn,xmx,ymx,zmx,s1,s2,s3,s4,s5,s6
        xmn = 1000000.0
        ymn = 1000000.0
        zmn = 1000000.0
        xmx = -1000000.0
        ymx = -1000000.0
        zmx = -1000000.0
        s1 = 0.0d0
        s2 = 0.0d0
        s3 = 0.0d0
        s4 = 0.0d0
        s5 = 0.0d0
        s6 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          xmn = min(xmn,p(1,i))
          ymn = min(ymn,p(3,i))
          zmn = min(zmn,p(5,i))
          xmx = max(xmx,p(1,i))
          ymx = max(ymx,p(3,i))
          zmx = max(zmx,p(5,i))
          s1 = s1 + p(1,i)
          s2 = s2 + p(2,i)
          s3 = s3 + p(3,i)
          s4 = s4 + p(4,i)
          s5 = s5 + p(5,i)
          s6 = s6 + p(6,i)
        enddo
        rmn(1) = xmn
        rmn(2) = ymn
        rmn(3) = zmn
        rmx(1) = xmx
        rmx(2) = ymx
        rmx(3) = zmx
        sm(1) = s1
        sm(2) = s2
        sm(3) = s3
        sm(4) = s4
        sm(5) = s5
        sm(6) = s6
        end subroutine range_core

        !> singlerange (Ranger.f90) on the device particles: reductions on
        !> the GPU, then the same three MPI_ALLREDUCEs and divisions.
        subroutine singlerange_gpu(ib,bunch,nplc,nptot,range,center)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: ib,nplc,nptot
        type (BeamBunch), intent(inout) :: bunch
        double precision, dimension(6), intent(out) :: range,center
        double precision, dimension(3) :: localrange1,localrange2,temp1,temp2
        double precision, dimension(6) :: centerlc
        integer :: i,ierr
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call range_core(pb(ib)%p,pb(ib)%n,localrange1,localrange2,centerlc)
        else
          localrange1 = 1000000.0
          localrange2 = -1000000.0
          centerlc = 0.0
        endif
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        call MPI_ALLREDUCE(localrange1,temp1,3,MPI_DOUBLE_PRECISION,MPI_MIN, &
                           MPI_COMM_WORLD,ierr)
        call MPI_ALLREDUCE(localrange2,temp2,3,MPI_DOUBLE_PRECISION,MPI_MAX, &
                           MPI_COMM_WORLD,ierr)
        call MPI_ALLREDUCE(centerlc,center,6,MPI_DOUBLE_PRECISION,MPI_SUM, &
                           MPI_COMM_WORLD,ierr)
        if(nptot.gt.0) then
          do i = 1, 3
            range(2*i-1) = temp1(i)
            range(2*i) = temp2(i)
            center(2*i-1) = center(2*i-1)/nptot
            center(2*i) = center(2*i)/nptot
          enddo
        else
          do i = 1, 3
            range(2*i-1) = temp1(i)
            range(2*i) = temp2(i)
            center(2*i-1) = center(2*i-1)
            center(2*i) = center(2*i)
          enddo
        endif
        end subroutine singlerange_gpu

        !-------------------------------------------------------------
        ! lost-particle test (B2.5.3)
        !-------------------------------------------------------------
        subroutine count_lost(p,n,idm,p1,p2,p3,p4,p5,p6,rrap,nlost)
        implicit none
        integer, intent(in) :: n,idm
        real(8), device :: p(9,n)
        real(8), intent(in) :: p1,p2,p3,p4,p5,p6,rrap
        integer, intent(out) :: nlost
        integer :: i,nl,cnt
        real(8) :: rr
        logical :: lost
        cnt = 0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          if(idm.eq.0) then
            lost = p(1,i).le.p1 .or. p(1,i).ge.p2 .or. p(3,i).le.p3 .or. &
                   p(3,i).ge.p4
          else
            rr = p(1,i)**2+p(3,i)**2
            lost = rr.ge.rrap
          endif
          if(.not.lost) then
            lost = (p(5,i).le.p5 .and. p(6,i).lt.0.0) .or. p(5,i).ge.p6
          endif
          nl = 0
          if(lost) nl = 1
          cnt = cnt + nl
        enddo
        nlost = cnt
        end subroutine count_lost

        !> lostREC_BeamBunch (idm = 0) or lostDFO_BeamBunch (idm = 1) with
        !> the predicate evaluated on the GPU.  Each rank makes exactly one
        !> MPI_ALLREDUCE (inside the CPU routine if it lost particles, here
        !> otherwise) and updates Current as the CPU routine does.
        subroutine lost_gpu(ib,bunch,idm,xrad,yrad,zleng,zcent,nplc,nptot)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: ib,idm
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: xrad,yrad,zleng,zcent
        integer, intent(out) :: nplc,nptot
        double precision :: p1,p2,p3,p4,p5,p6,rrap,t0
        integer :: nlost,ierr
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        p1 = -xrad/Scxlt
        p2 = xrad/Scxlt
        p3 = -yrad/Scxlt
        p4 = yrad/Scxlt
        p5 = 0.0
        p6 = zleng/Scxlt
        rrap = p1**2
        nlost = 0
        if(pb(ib)%n.gt.0) then
          call count_lost(pb(ib)%p,pb(ib)%n,idm,p1,p2,p3,p4,p5,p6,rrap,nlost)
        endif
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        if(nlost.gt.0) then
          call ensure_host(ib,bunch)
          if(idm.eq.0) then
            call lostREC_BeamBunch(bunch,xrad,yrad,zleng,zcent,nplc,nptot)
          else
            call lostDFO_BeamBunch(bunch,xrad,yrad,zleng,zcent,nplc,nptot)
          endif
          pb(ib)%state = 2
        else
          nplc = bunch%Nptlocal
          call MPI_ALLREDUCE(nplc,nptot,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
          bunch%Current = bunch%Current*nptot/bunch%Npt
          bunch%Npt = nptot
        endif
        end subroutine lost_gpu

        !-------------------------------------------------------------
        ! mesh arrays
        !-------------------------------------------------------------
        subroutine set_device_mesh(nx,ny,nz)
        implicit none
        integer, intent(in) :: nx,ny,nz
        if(nx.eq.mnx .and. ny.eq.mny .and. nz.eq.mnz .and. allocated(rho_dev)) return
        if(allocated(rho_dev)) deallocate(rho_dev,phi_dev,phiimg_dev,ex_dev, &
                                          ey_dev,ez_dev,bx_dev,by_dev)
        mnx = nx
        mny = ny
        mnz = nz
        allocate(rho_dev(nx,ny,nz),phi_dev(nx,ny,nz),phiimg_dev(nx,ny,nz))
        allocate(ex_dev(nx,ny,nz),ey_dev(nx,ny,nz),ez_dev(nx,ny,nz), &
                 bx_dev(nx,ny,nz),by_dev(nx,ny,nz))
        end subroutine set_device_mesh

        subroutine mesh_zero_gpu(nx,ny,nz)
        implicit none
        integer, intent(in) :: nx,ny,nz
        call set_device_mesh(nx,ny,nz)
        ex_dev = 0.0d0
        ey_dev = 0.0d0
        ez_dev = 0.0d0
        bx_dev = 0.0d0
        by_dev = 0.0d0
        end subroutine mesh_zero_gpu

        !-------------------------------------------------------------
        ! deposition (B2.5.4)
        !-------------------------------------------------------------
        attributes(global) subroutine k_deposit(p,n,rho,nx,ny,nz,xmin,ymin,zmin, &
                                                hx,hy,hz,hxi,hyi,hzi,flagpos)
        implicit none
        integer, value :: n,nx,ny,nz,flagpos
        real(8), value :: xmin,ymin,zmin,hx,hy,hz,hxi,hyi,hzi
        real(8), device :: p(9,n)
        real(8), device :: rho(nx,ny,nz)
        integer :: i,ix,jx,kx,ix1,jx1,kx1,old
        real(8) :: ab,cd,ef,ww,dum
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        if(flagpos.eq.1) then
          if(.not.(p(5,i).gt.0.0d0)) return
        endif
        ix=(p(1,i)-xmin)*hxi + 1
        ab=((xmin-p(1,i))+ix*hx)*hxi
        jx=(p(3,i)-ymin)*hyi + 1
        cd=((ymin-p(3,i))+jx*hy)*hyi
        kx=(p(5,i)-zmin)*hzi + 1
        ef=((zmin-p(5,i))+kx*hz)*hzi
        ix1=ix+1
        if( (ix1.gt.nx).or.(ix.lt.1)) then
          old = atomicCAS(gpuerr_d,0,1)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = ix
            gpuerr_info_d(4) = 1
          endif
          return
        endif
        jx1=jx+1
        if( (jx1.gt.ny).or.(jx.lt.1)) then
          old = atomicCAS(gpuerr_d,0,1)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = jx
            gpuerr_info_d(4) = 2
          endif
          return
        endif
        kx1=kx+1
        if( (kx1.gt.nz).or.(kx.lt.1)) then
          old = atomicCAS(gpuerr_d,0,1)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = kx
            gpuerr_info_d(4) = 3
          endif
          return
        endif
        ww = p(8,i)
        dum = atomicAdd(rho(ix,jx,kx), ab*cd*ef*ww)
        dum = atomicAdd(rho(ix,jx1,kx), ab*(1.0d0-cd)*ef*ww)
        dum = atomicAdd(rho(ix,jx1,kx1), ab*(1.0d0-cd)*(1.0d0-ef)*ww)
        dum = atomicAdd(rho(ix,jx,kx1), ab*cd*(1.0d0-ef)*ww)
        dum = atomicAdd(rho(ix1,jx,kx1), (1.0d0-ab)*cd*(1.0d0-ef)*ww)
        dum = atomicAdd(rho(ix1,jx1,kx1), (1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)*ww)
        dum = atomicAdd(rho(ix1,jx1,kx), (1.0d0-ab)*(1.0d0-cd)*ef*ww)
        dum = atomicAdd(rho(ix1,jx,kx), (1.0d0-ab)*cd*ef*ww)
        end subroutine k_deposit

        !> deposit1_Depositor (flagpos = 0) or deposit1p_Depositor (1) onto
        !> the global grid rho_dev; zeroes rho_dev first.
        subroutine deposit_gpu(ib,bunch,nx,ny,nz,range,msize,flagpos)
        implicit none
        integer, intent(in) :: ib,nx,ny,nz,flagpos
        type (BeamBunch), intent(inout) :: bunch
        double precision, dimension(6), intent(in) :: range
        double precision, dimension(3), intent(in) :: msize
        double precision :: hxi,hyi,hzi,t0
        call ensure_device(ib,bunch)
        call set_device_mesh(nx,ny,nz)
        call starttime_Timer(t0)
        hxi = 1.0d0/msize(1)
        hyi = 1.0d0/msize(2)
        hzi = 1.0d0/msize(3)
        rho_dev = 0.0d0
        if(pb(ib)%n.gt.0) then
          call k_deposit<<<(pb(ib)%n+127)/128,128>>>(pb(ib)%p,pb(ib)%n,rho_dev, &
               nx,ny,nz,range(1),range(3),range(5),msize(1),msize(2),msize(3), &
               hxi,hyi,hzi,flagpos)
        endif
        call sync_gpu()
        t_depgpu = t_depgpu + elapsedtime_Timer(t0)
        end subroutine deposit_gpu

        !> Space-charge fields of one bunch on the global grid (B2.4 step
        !> 3.4): deposit, sum over ranks, scale as chgdens_Depositor, solve,
        !> add the E and B fields (and those of the image) to ex_dev..by_dev.
        !> The caller has already zeroed the fields (mesh_zero_gpu).
        subroutine mesh_fields_gpu(ib,bunch,nx,ny,nz,range,msize,gammaz,gbetz, &
                                   flagpos,flagImage,zshift,nranks)
        implicit none
        integer, intent(in) :: ib,nx,ny,nz,flagpos,flagImage,nranks
        type (BeamBunch), intent(inout) :: bunch
        double precision, dimension(6), intent(in) :: range
        double precision, dimension(3), intent(in) :: msize
        double precision, intent(in) :: gammaz,gbetz,zshift
        double precision :: hxi,hyi,hzi,tmp1,tmp2,hx,hy,hz,t0
        integer :: tmpflag

        call deposit_gpu(ib,bunch,nx,ny,nz,range,msize,flagpos)
        call starttime_Timer(t0)
        if(nranks.gt.1) call allreduce_rho_gpu(rho_dev,nx*ny*nz)
        tmp1 = 1.0d0
        hxi = 1.0d0/msize(1)
        hyi = 1.0d0/msize(2)
        hzi = 1.0d0/msize(3)
        tmp2 = tmp1*hxi*hyi*hzi/(Scxlt*Scxlt*Scxlt*Epsilon0)/gammaz
        call scale_rho(rho_dev,nx,ny,nz,tmp2)
        call sync_gpu()
        t_depgpu = t_depgpu + elapsedtime_Timer(t0)

        hx = msize(1)*Scxlt
        hy = msize(2)*Scxlt
        hz = msize(3)*gammaz*Scxlt
        call starttime_Timer(t0)
        call poisson_dev(nx,ny,nz,rho_dev,phi_dev,phiimg_dev,nx,ny,nz,0,0, &
                         hx,hy,hz,flagImage,zshift,.true.)
        call sync_gpu()
        t_poisgpu = t_poisgpu + elapsedtime_Timer(t0)

        call gradEB_mesh_gpu(nx,ny,nz,msize,gammaz,gbetz,flagImage)
        end subroutine mesh_fields_gpu

        !> rho = rho*s, as scale after the sum over ranks.
        subroutine scale_rho(rho,nx,ny,nz,sc)
        implicit none
        integer, intent(in) :: nx,ny,nz
        real(8), device :: rho(nx,ny,nz)
        double precision, intent(in) :: sc
        integer :: i,j,k
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nz
          do j = 1, ny
            do i = 1, nx
              rho(i,j,k) = rho(i,j,k)*sc
            enddo
          enddo
        enddo
        end subroutine scale_rho

        !> gradEB of phi_dev (and of phiimg_dev if flagImage = 1) into the
        !> device fields.
        subroutine gradEB_mesh_gpu(nx,ny,nz,msize,gammaz,gbetz,flagImage)
        implicit none
        integer, intent(in) :: nx,ny,nz,flagImage
        double precision, dimension(3), intent(in) :: msize
        double precision, intent(in) :: gammaz,gbetz
        double precision :: hxi,hyi,hzi,t0
        integer :: tmpflag
        call starttime_Timer(t0)
        hxi = 1.0d0/(msize(1)*Scxlt)
        hyi = 1.0d0/(msize(2)*Scxlt)
        hzi = 1.0d0/(msize(3)*Scxlt*gammaz)
        if(gbetz.gt.0.0d0) then
          tmpflag = 0
        else
          tmpflag = 1
        endif
        call gradEB_dev(nx,ny,nz,phi_dev,hxi,hyi,hzi,gammaz,tmpflag, &
                        ex_dev,ey_dev,ez_dev,bx_dev,by_dev)
        if(flagImage.eq.1) then
          call gradEB_dev(nx,ny,nz,phiimg_dev,hxi,hyi,hzi,gammaz,flagImage, &
                          ex_dev,ey_dev,ez_dev,bx_dev,by_dev)
        endif
        call sync_gpu()
        t_gradgpu = t_gradgpu + elapsedtime_Timer(t0)
        end subroutine gradEB_mesh_gpu

        !-------------------------------------------------------------
        ! kicks (B2.5.6)
        !-------------------------------------------------------------
        !> The Boris update shared by the three kicks, verbatim.
        attributes(device) subroutine boris(px,py,pz,coefLz,ex,ey,ez,bx,by,bz,dt,clight)
        implicit none
        real(8), intent(inout) :: px,py,pz
        real(8), value :: coefLz,ex,ey,ez,bx,by,bz,dt,clight
        real(8) :: umx,umy,umz,upx,upy,upz,recpgamma,tmp,a1,a2,a3,a4,s1,s2,s3
        umx = px + coefLz*ex*0.5d0*dt
        umy = py + coefLz*ey*0.5d0*dt
        umz = pz + coefLz*ez*0.5d0*dt
        recpgamma = 1.0d0/sqrt(1.0d0+umx*umx+umy*umy+umz*umz)
        tmp = 0.5d0*clight*dt*recpgamma*coefLz
        a1 = tmp*bx
        a2 = tmp*by
        a3 = tmp*bz
        a4 = 1.0d0+a1*a1+a2*a2+a3*a3
        s1 = umx + tmp*(umy*bz-umz*by)
        s2 = umy - tmp*(umx*bz-umz*bx)
        s3 = umz + tmp*(umx*by-umy*bx)
        upx = ((1.0d0+a1*a1)*s1+(a1*a2+a3)*s2+(a1*a3-a2)*s3)/a4
        upy = ((a1*a2-a3)*s1+(1.0d0+a2*a2)*s2+(a2*a3+a1)*s3)/a4
        upz = ((a1*a3+a2)*s1+(a2*a3-a1)*s2+(1.0d0+a3*a3)*s3)/a4
        px = upx + coefLz*ex*0.5d0*dt
        py = upy + coefLz*ey*0.5d0*dt
        pz = upz + coefLz*ez*0.5d0*dt
        end subroutine boris

        !> kick2t_BeamBunch on the global grid (jadd = kadd = 0).
        attributes(global) subroutine k_kick_mesh(p,n,exg,eyg,ezg,bxg,byg, &
              nx,ny,nz,xmin,ymin,zmin,hx,hy,hz,hxi,hyi,hzi,tg,dt,nact,scxlt_, &
              clight_)
        implicit none
        integer, value :: n,nx,ny,nz,nact
        real(8), value :: xmin,ymin,zmin,hx,hy,hz,hxi,hyi,hzi,tg,dt,scxlt_,clight_
        real(8), device :: p(9,n)
        real(8), device :: exg(nx,ny,nz),eyg(nx,ny,nz),ezg(nx,ny,nz)
        real(8), device :: bxg(nx,ny,nz),byg(nx,ny,nz)
        integer :: i,ix,jx,kx,ix1,jx1,kx1,old
        real(8) :: ab,cd,ef,exn,eyn,ezn,bxn,byn,coefLz,zz
        real(8) :: ex,ey,ez,bx,by,bz,ef6(6)
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        if(.not.(p(5,i)>0)) return
        ix=(p(1,i)-xmin)*hxi + 1
        ab=((xmin-p(1,i))+ix*hx)*hxi
        jx=(p(3,i)-ymin)*hyi + 1
        cd=((ymin-p(3,i))+jx*hy)*hyi
        kx=(p(5,i)-zmin)*hzi + 1
        ef=((zmin-p(5,i))+kx*hz)*hzi
        ix1 = ix + 1
        if( (ix1.gt.nx).or.(ix.lt.1)) then
          old = atomicCAS(gpuerr_d,0,2)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = ix
            gpuerr_info_d(4) = 1
          endif
          return
        endif
        jx1 = jx + 1
        if( (jx1.gt.ny).or.(jx.lt.1)) then
          old = atomicCAS(gpuerr_d,0,2)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = jx
            gpuerr_info_d(4) = 2
          endif
          return
        endif
        kx1 = kx + 1
        if( (kx1.gt.nz).or.(kx.lt.1)) then
          old = atomicCAS(gpuerr_d,0,2)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = kx
            gpuerr_info_d(4) = 3
          endif
          return
        endif

        exn = (exg(ix,jx,kx)*ab*cd*ef  &
                +exg(ix,jx1,kx)*ab*(1.0d0-cd)*ef &
                +exg(ix,jx1,kx1)*ab*(1.0d0-cd)*(1.0d0-ef) &
                +exg(ix,jx,kx1)*ab*cd*(1.0d0-ef) &
                +exg(ix1,jx,kx1)*(1.0d0-ab)*cd*(1.0d0-ef) &
                +exg(ix1,jx1,kx1)*(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)&
                +exg(ix1,jx1,kx)*(1.0d0-ab)*(1.0d0-cd)*ef &
                +exg(ix1,jx,kx)*(1.0d0-ab)*cd*ef)

        eyn = (eyg(ix,jx,kx)*ab*cd*ef  &
                +eyg(ix,jx1,kx)*ab*(1.0d0-cd)*ef &
                +eyg(ix,jx1,kx1)*ab*(1.0d0-cd)*(1.0d0-ef) &
                +eyg(ix,jx,kx1)*ab*cd*(1.0d0-ef) &
                +eyg(ix1,jx,kx1)*(1.0d0-ab)*cd*(1.0d0-ef) &
                +eyg(ix1,jx1,kx1)*(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)&
                +eyg(ix1,jx1,kx)*(1.0d0-ab)*(1.0d0-cd)*ef &
                +eyg(ix1,jx,kx)*(1.0d0-ab)*cd*ef)

        ezn = ezg(ix,jx,kx)*ab*cd*ef  &
                +ezg(ix,jx1,kx)*ab*(1.0d0-cd)*ef &
                +ezg(ix,jx1,kx1)*ab*(1.0d0-cd)*(1.0d0-ef) &
                +ezg(ix,jx,kx1)*ab*cd*(1.0d0-ef) &
                +ezg(ix1,jx,kx1)*(1.0d0-ab)*cd*(1.0d0-ef) &
                +ezg(ix1,jx1,kx1)*(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)&
                +ezg(ix1,jx1,kx)*(1.0d0-ab)*(1.0d0-cd)*ef &
                +ezg(ix1,jx,kx)*(1.0d0-ab)*cd*ef

        bxn = bxg(ix,jx,kx)*ab*cd*ef  &
                +bxg(ix,jx1,kx)*ab*(1.0d0-cd)*ef &
                +bxg(ix,jx1,kx1)*ab*(1.0d0-cd)*(1.0d0-ef) &
                +bxg(ix,jx,kx1)*ab*cd*(1.0d0-ef) &
                +bxg(ix1,jx,kx1)*(1.0d0-ab)*cd*(1.0d0-ef) &
                +bxg(ix1,jx1,kx1)*(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)&
                +bxg(ix1,jx1,kx)*(1.0d0-ab)*(1.0d0-cd)*ef &
                +bxg(ix1,jx,kx)*(1.0d0-ab)*cd*ef

        byn = byg(ix,jx,kx)*ab*cd*ef  &
                +byg(ix,jx1,kx)*ab*(1.0d0-cd)*ef &
                +byg(ix,jx1,kx1)*ab*(1.0d0-cd)*(1.0d0-ef) &
                +byg(ix,jx,kx1)*ab*cd*(1.0d0-ef) &
                +byg(ix1,jx,kx1)*(1.0d0-ab)*cd*(1.0d0-ef) &
                +byg(ix1,jx1,kx1)*(1.0d0-ab)*(1.0d0-cd)*(1.0d0-ef)&
                +byg(ix1,jx1,kx)*(1.0d0-ab)*(1.0d0-cd)*ef &
                +byg(ix1,jx,kx)*(1.0d0-ab)*cd*ef

        zz = p(5,i)*scxlt_
        call extfld_dev(p(1,i)*scxlt_,p(3,i)*scxlt_,zz,nact,ef6,i)
        coefLz = p(7,i)*scxlt_

        ex = exn+ef6(1)
        ey = eyn+ef6(2)
        ez = ezn+ef6(3)
        bx = bxn+ef6(4)
        by = byn+ef6(5)
        bz = ef6(6)
        call boris(p(2,i),p(4,i),p(6,i),coefLz,ex,ey,ez,bx,by,bz,dt,clight_)
        end subroutine k_kick_mesh

        !> kickpt2ptImg_BeamBunch's external-field and kick part: fields
        !> from pt2ptFields (not yet scaled by coefE) in fxe..fzb.
        attributes(global) subroutine k_kick_pt2pt(p,n,fxe,fye,fze,fxb,fyb,fzb, &
              coefE,tg,dt,nact,scxlt_,clight_)
        implicit none
        integer, value :: n,nact
        real(8), value :: coefE,tg,dt,scxlt_,clight_
        real(8), device :: p(9,n)
        real(8), device :: fxe(n),fye(n),fze(n),fxb(n),fyb(n),fzb(n)
        integer :: i
        real(8) :: coefLz,zz,ex,ey,ez,bx,by,bz,ef6(6)
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        if(.not.(p(5,i).gt.0.0)) return
        zz = p(5,i)*scxlt_
        call extfld_dev(p(1,i)*scxlt_,p(3,i)*scxlt_,zz,nact,ef6,i)
        ex = coefE*fxe(i)+ef6(1)
        ey = coefE*fye(i)+ef6(2)
        ez = coefE*fze(i)+ef6(3)
        bx = coefE*fxb(i)/clight_+ef6(4)
        by = coefE*fyb(i)/clight_+ef6(5)
        bz = coefE*fzb(i)/clight_+ef6(6)
        coefLz = p(7,i)*scxlt_
        call boris(p(2,i),p(4,i),p(6,i),coefLz,ex,ey,ez,bx,by,bz,dt,clight_)
        end subroutine k_kick_pt2pt

        !> scatter20t_BeamBunch: external fields only, every particle.
        attributes(global) subroutine k_kick_none(p,n,tg,dt,nact,scxlt_,clight_)
        implicit none
        integer, value :: n,nact
        real(8), value :: tg,dt,scxlt_,clight_
        real(8), device :: p(9,n)
        integer :: i
        real(8) :: coefLz,zz,ef6(6)
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        zz = p(5,i)*scxlt_
        call extfld_dev(p(1,i)*scxlt_,p(3,i)*scxlt_,zz,nact,ef6,i)
        coefLz = p(7,i)*scxlt_
        call boris(p(2,i),p(4,i),p(6,i),coefLz,ef6(1),ef6(2),ef6(3),ef6(4), &
                   ef6(5),ef6(6),dt,clight_)
        end subroutine k_kick_none

        !> kick2t_BeamBunch with the fields in ex_dev..by_dev.  The active
        !> element table is the one built for this step.
        subroutine kick_mesh_gpu(ib,bunch,nx,ny,nz,range,msize,tg,dt)
        implicit none
        integer, intent(in) :: ib,nx,ny,nz
        type (BeamBunch), intent(inout) :: bunch
        double precision, dimension(6), intent(in) :: range
        double precision, dimension(3), intent(in) :: msize
        double precision, intent(in) :: tg,dt
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call kick_mesh_core(pb(ib)%p,pb(ib)%n,nx,ny,nz,range,msize,tg,dt)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine kick_mesh_gpu

        !> The launch of k_kick_mesh on explicit device arrays (also used
        !> by the unit test with its own fields).
        subroutine kick_mesh_core(p,n,nx,ny,nz,range,msize,tg,dt)
        implicit none
        integer, intent(in) :: n,nx,ny,nz
        real(8), device :: p(9,n)
        double precision, dimension(6), intent(in) :: range
        double precision, dimension(3), intent(in) :: msize
        double precision, intent(in) :: tg,dt
        call k_kick_mesh<<<(n+127)/128,128>>>(p,n,ex_dev,ey_dev,ez_dev,bx_dev, &
             by_dev,nx,ny,nz,range(1),range(3),range(5),msize(1),msize(2), &
             msize(3),1.0d0/msize(1),1.0d0/msize(2),1.0d0/msize(3),tg,dt, &
             ntab_dev,Scxlt,Clight)
        end subroutine kick_mesh_core

        subroutine kick_none_gpu(ib,bunch,tg,dt)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tg,dt
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call k_kick_none<<<(pb(ib)%n+127)/128,128>>>(pb(ib)%p,pb(ib)%n,tg,dt, &
               ntab_dev,Scxlt,Clight)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine kick_none_gpu

        !-------------------------------------------------------------
        ! N-body kick (B2.5.7)
        !-------------------------------------------------------------
        attributes(global) subroutine k_build_src(p,n,srcloc,targets)
        implicit none
        integer, value :: n
        real(8), device :: p(9,n)
        real(8), device :: srcloc(10,n),targets(4,n)
        integer :: i
        real(8) :: gam
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        if(p(5,i).gt.0.0) then
          gam = sqrt(1.0d0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          srcloc(1,i) = p(1,i)
          srcloc(2,i) = p(3,i)
          srcloc(3,i) = p(5,i)
          srcloc(4,i) = p(2,i)/gam
          srcloc(5,i) = p(4,i)/gam
          srcloc(6,i) = p(6,i)/gam
          srcloc(7,i) = gam
          srcloc(8,i) = 1.0d0
          srcloc(9,i) = gam*gam/(gam+1.0d0)
          srcloc(10,i) = gam/(gam+1.0d0)
          targets(1,i) = p(1,i)
          targets(2,i) = p(3,i)
          targets(3,i) = p(5,i)
          targets(4,i) = 1.0d0
        else
          srcloc(1:10,i) = 0.0d0
          targets(1:4,i) = 0.0d0
        endif
        end subroutine k_build_src

        !> kickpt2ptImg_BeamBunch: sources built on the device; with one
        !> rank they are used in place, otherwise gathered through the host.
        subroutine kickpt2pt_gpu(ib,bunch,tg,dt,totchrg,r0,npttot,nptrue, &
                                 flagimgnow)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: ib,npttot,nptrue,flagimgnow
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tg,dt,totchrg,r0
        integer :: n,nproc,myid,ierr,i,sgn,myoffset
        integer, allocatable :: recvcounts(:),displs(:)
        double precision, allocatable :: hsrc(:,:),hall(:,:)
        double precision :: coefE,t0,t1
        logical :: image

        call ensure_device(ib,bunch)
        n = pb(ib)%n
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
        if(bunch%Charge.gt.0.0) then
          sgn = 1
        else
          sgn = -1
        endif
        coefE = 1.0d0/(4*Pi*Epsilon0*Scxlt*Scxlt)*sgn*totchrg/nptrue

        call starttime_Timer(t0)
        if(n.gt.ncap_nb) then
          if(allocated(srcloc_d)) deallocate(srcloc_d,targets_d,fex,fey,fez,fbx,fby,fbz)
          ncap_nb = n
          allocate(srcloc_d(10,n),targets_d(4,n))
          allocate(fex(n),fey(n),fez(n),fbx(n),fby(n),fbz(n))
        endif
        if(n.gt.0) then
          call k_build_src<<<(n+127)/128,128>>>(pb(ib)%p,n,srcloc_d,targets_d)
        endif

        allocate(recvcounts(nproc),displs(nproc))
        call MPI_ALLGATHER(n,1,MPI_INTEGER,recvcounts,1,MPI_INTEGER, &
                           MPI_COMM_WORLD,ierr)
        displs(1) = 0
        do i = 2, nproc
          displs(i) = displs(i-1) + recvcounts(i-1)
        enddo
        myoffset = displs(myid+1)

        image = (flagimgnow.eq.1)
        if(nproc.eq.1) then
          call starttime_Timer(t1)
          call pt2ptFields_dev(n,targets_d,myoffset,npttot,srcloc_d,image,r0, &
                               fex,fey,fez,fbx,fby,fbz)
          t_pt2ptfields = t_pt2ptfields + elapsedtime_Timer(t1)
        else
          if(npttot.gt.ncap_src) then
            if(allocated(sources_d)) deallocate(sources_d)
            ncap_src = npttot
            allocate(sources_d(10,npttot))
          endif
          allocate(hsrc(10,max(n,1)),hall(10,npttot))
          if(n.gt.0) hsrc(:,1:n) = srcloc_d(:,1:n)
          call MPI_ALLGATHERV(hsrc,10*n,MPI_DOUBLE_PRECISION,hall, &
               10*recvcounts,10*displs,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
          sources_d(:,1:npttot) = hall
          deallocate(hsrc,hall)
          call starttime_Timer(t1)
          call pt2ptFields_dev(n,targets_d,myoffset,npttot,sources_d,image,r0, &
                               fex,fey,fez,fbx,fby,fbz)
          t_pt2ptfields = t_pt2ptfields + elapsedtime_Timer(t1)
        endif
        deallocate(recvcounts,displs)

        if(n.gt.0) then
          call k_kick_pt2pt<<<(n+127)/128,128>>>(pb(ib)%p,n,fex,fey,fez,fbx,fby, &
               fbz,coefE,tg,dt,ntab_dev,Scxlt,Clight)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine kickpt2pt_gpu

      end module ParticleGPUclass
