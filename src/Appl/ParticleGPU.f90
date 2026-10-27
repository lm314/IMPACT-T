!----------------------------------------------------------------
! ParticleGPUclass: per-particle work of an ordinary time step on the GPU
! (CUDA Fortran).  See doc/GPU_PORT_DESIGN.md, B2.3-B2.5.
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
        public :: diag_gpu_ready, diag_zsum_gpu, diag_pass1_gpu, diag_pass2_gpu
        public :: count_bend, csr_hist_gpu, csr_addez_gpu, bendpt2pt_gpu
        public :: rottoT_gpu, rotbackT_gpu, convEntr_gpu, convExit_gpu
        public :: driftbackhalf_gpu, drifthalfBd_gpu, kick_bend_gpu
        public :: diagB_pass1_gpu, diagB_pass2_gpu

        type ptsbuf
          real(8), device, allocatable :: p(:,:)  ! (9,cap), same layout as Pts1
          integer :: cap = 0                      ! allocated columns
          integer :: n = 0                        ! valid columns (= Nptlocal)
          integer :: state = 2                    ! 0 in sync, 1 device newer, 2 host newer
        end type ptsbuf
        type(ptsbuf), allocatable, save :: pb(:)

        integer, save :: nbuf = 0
        integer, save :: nsync_up = 0, nsync_down = 0, nfallback = 0
        integer, save :: nbend_gpu = 0, nbend_host = 0
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

        !> one more type-4 bend, run on the GPU or on the host (D1.5)
        subroutine count_bend(ongpu)
        implicit none
        logical, intent(in) :: ongpu
        if(ongpu) then
          nbend_gpu = nbend_gpu + 1
        else
          nbend_host = nbend_host + 1
        endif
        end subroutine count_bend

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
          else if(code.eq.6) then
            print*,"csr - histogram index out of range: particle ",info(2)," index ",info(3)
          else
            print*,"ParticleGPU: error code ",code
          endif
          stop
        endif
        end subroutine check_gpu_error

        subroutine report_gpu_stats(myid)
        implicit none
        integer, intent(in) :: myid
        integer :: ncart,icart
        real(8) :: pcart(3)
        ncart = cartout_count(icart,pcart)
        if(myid.eq.0) then
          write(*,'(a,i0,a,i0,a,i0,a,i0,a,i0,a,i0)') " GPU particle path: host->device copies ", &
                nsync_up,", device->host copies ",nsync_down, &
                ", CPU fallback steps ",nfallback, &
                ", bends on the GPU ",nbend_gpu,", on the host ",nbend_host, &
                ", 111 out-of-map points ",ncart
          if(ncart.gt.0) write(*,'(a,i0,a,3es14.6)') &
            "   first 111 out-of-map point: particle ",icart," at ",pcart
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

        !> k_build_src for the bend frame (D1.12): every particle is emitted
        attributes(global) subroutine k_build_src_all(p,n,srcloc,targets)
        implicit none
        integer, value :: n
        real(8), device :: p(9,n)
        real(8), device :: srcloc(10,n),targets(4,n)
        integer :: i
        real(8) :: gam
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
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
        end subroutine k_build_src_all

        !> the sources, their gather and pt2ptFields_dev of kickpt2ptImg_BeamBunch:
        !> fields (not yet scaled by coefE) in fex..fbz.  allemitted: every
        !> particle is a source and a target (bend frame, D1.12), else only
        !> z > 0.  With one rank the sources are used in place, otherwise
        !> gathered through the host.
        subroutine pt2pt_fields_gpu(ib,bunch,allemitted,image,r0,r0img,npttot)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: ib,npttot
        logical, intent(in) :: allemitted,image
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: r0,r0img
        integer :: n,nproc,myid,ierr,i,myoffset
        integer, allocatable :: recvcounts(:),displs(:)
        double precision, allocatable :: hsrc(:,:),hall(:,:)
        double precision :: t1

        n = pb(ib)%n
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
        if(n.gt.ncap_nb) then
          if(allocated(srcloc_d)) deallocate(srcloc_d,targets_d,fex,fey,fez,fbx,fby,fbz)
          ncap_nb = n
          allocate(srcloc_d(10,n),targets_d(4,n))
          allocate(fex(n),fey(n),fez(n),fbx(n),fby(n),fbz(n))
        endif
        if(n.gt.0) then
          if(allemitted) then
            call k_build_src_all<<<(n+127)/128,128>>>(pb(ib)%p,n,srcloc_d,targets_d)
          else
            call k_build_src<<<(n+127)/128,128>>>(pb(ib)%p,n,srcloc_d,targets_d)
          endif
        endif

        allocate(recvcounts(nproc),displs(nproc))
        call MPI_ALLGATHER(n,1,MPI_INTEGER,recvcounts,1,MPI_INTEGER, &
                           MPI_COMM_WORLD,ierr)
        displs(1) = 0
        do i = 2, nproc
          displs(i) = displs(i-1) + recvcounts(i-1)
        enddo
        myoffset = displs(myid+1)

        if(nproc.eq.1) then
          call starttime_Timer(t1)
          call pt2ptFields_dev(n,targets_d,myoffset,npttot,srcloc_d,image,r0,r0img, &
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
          call pt2ptFields_dev(n,targets_d,myoffset,npttot,sources_d,image,r0,r0img, &
                               fex,fey,fez,fbx,fby,fbz)
          t_pt2ptfields = t_pt2ptfields + elapsedtime_Timer(t1)
        endif
        deallocate(recvcounts,displs)
        end subroutine pt2pt_fields_gpu

        !> kickpt2ptImg_BeamBunch: fields (pt2pt_fields_gpu) and the kick.
        subroutine kickpt2pt_gpu(ib,bunch,tg,dt,totchrg,r0,r0img,npttot,nptrue, &
                                 flagimgnow)
        implicit none
        integer, intent(in) :: ib,npttot,nptrue,flagimgnow
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tg,dt,totchrg,r0,r0img
        integer :: n,sgn
        double precision :: coefE,t0

        call ensure_device(ib,bunch)
        n = pb(ib)%n
        if(bunch%Charge.gt.0.0) then
          sgn = 1
        else
          sgn = -1
        endif
        coefE = 1.0d0/(4*Pi*Epsilon0*Scxlt*Scxlt)*sgn*totchrg/nptrue

        call starttime_Timer(t0)
        call pt2pt_fields_gpu(ib,bunch,.false.,flagimgnow.eq.1,r0,r0img,npttot)

        if(n.gt.0) then
          call k_kick_pt2pt<<<(n+127)/128,128>>>(pb(ib)%p,n,fex,fey,fez,fbx,fby, &
               fbz,coefE,tg,dt,ntab_dev,Scxlt,Clight)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine kickpt2pt_gpu

        !-------------------------------------------------------------
        ! moments for diagnostic1avgZ_Output (C1)
        !-------------------------------------------------------------
        !> True when the particles of this run are device-resident, so
        !> the diagnostic can sum on the device.  False before
        !> gpu_particles_init (the first diagnostic at t = 0 runs on the host).
        function diag_gpu_ready() result(flag)
        implicit none
        logical :: flag
        flag = .false.
        if(envflag.eq.1 .and. allocated(pb)) flag = .true.
        end function diag_gpu_ready

        !> sum of z over this rank's particles (all bunches).
        subroutine diag_zsum_gpu(bunch,nb,zsum)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        double precision, intent(out) :: zsum
        double precision :: part
        integer :: ib
        zsum = 0.0d0
        do ib = 1, nb
          call ensure_device(ib,bunch(ib))
          if(pb(ib)%n.gt.0) then
            call dz_core(pb(ib)%p,pb(ib)%n,part)
            zsum = zsum + part
          endif
        enddo
        end subroutine diag_zsum_gpu

        subroutine dz_core(p,n,zs)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(out) :: zs
        integer :: i
        real(8) :: s
        s = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s = s + p(5,i)
        enddo
        zs = s
        end subroutine dz_core

        !> the six sums of tmplc0 (x, px, y, py, z, pz drifted to z0avg).
        subroutine diag_pass1_gpu(bunch,nb,z0avg,sums)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        double precision, intent(in) :: z0avg
        double precision, intent(out) :: sums(6)
        double precision :: part(6)
        integer :: ib
        sums = 0.0d0
        do ib = 1, nb
          call ensure_device(ib,bunch(ib))
          if(pb(ib)%n.gt.0) then
            call d1_core(pb(ib)%p,pb(ib)%n,z0avg,part)
            sums = sums + part
          endif
        enddo
        end subroutine diag_pass1_gpu

        subroutine d1_core(p,n,z0avg,sm)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(in) :: z0avg
        real(8), intent(out) :: sm(6)
        integer :: i
        real(8) :: s1,s2,s3,s4,s5,s6,tmpgam,recpgamma,deltat,tmpx,tmpy
        s1 = 0.0d0
        s2 = 0.0d0
        s3 = 0.0d0
        s4 = 0.0d0
        s5 = 0.0d0
        s6 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          tmpgam = sqrt(1.0d0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          recpgamma = 1.0d0/tmpgam
          deltat = (p(5,i)-z0avg)/(recpgamma*p(6,i))
          tmpx = p(1,i)-recpgamma*p(2,i)*deltat
          tmpy = p(3,i)-recpgamma*p(4,i)*deltat
          s1 = s1 + tmpx
          s2 = s2 + p(2,i)
          s3 = s3 + tmpy
          s4 = s4 + p(4,i)
          s5 = s5 + p(5,i)
          s6 = s6 + p(6,i)
        enddo
        sm(1) = s1
        sm(2) = s2
        sm(3) = s3
        sm(4) = s4
        sm(5) = s5
        sm(6) = s6
        end subroutine d1_core

        !> the local 41 sums (tmplc, with tmplc(5) left at 0 for the caller),
        !> the six local maxima and the local max r^2, as in the host loop.
        !> m0 = (x0,px0,y0,py0,z0,pz0), the global means from pass 1.
        subroutine diag_pass2_gpu(bunch,nb,z0avg,m0,tmplc,localmax,lcrmax)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        double precision, intent(in) :: z0avg,m0(6)
        double precision, intent(out) :: tmplc(41),localmax(6),lcrmax
        double precision :: a(41),amx(6),ar
        integer :: ib
        tmplc = 0.0d0
        localmax = 0.0d0
        lcrmax = 0.0d0
        do ib = 1, nb
          call ensure_device(ib,bunch(ib))
          if(pb(ib)%n.gt.0) then
            a = 0.0d0
            amx = 0.0d0
            ar = 0.0d0
            call d2a_core(pb(ib)%p,pb(ib)%n,z0avg,a,ar)
            call d2b_core(pb(ib)%p,pb(ib)%n,a,amx)
            call d2c_core(pb(ib)%p,pb(ib)%n,z0avg,m0,a)
            call d2d_core(pb(ib)%p,pb(ib)%n,z0avg,m0(6),a,amx)
            tmplc = tmplc + a
            localmax = max(localmax,amx)
            lcrmax = max(lcrmax,ar)
          endif
        enddo
        end subroutine diag_pass2_gpu

        ! x, y (drifted) sums and powers, x*px, y*py, gamma sums, max r^2
        subroutine d2a_core(p,n,z0avg,a,rmx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(in) :: z0avg
        real(8), intent(inout) :: a(41)
        real(8), intent(out) :: rmx
        integer :: i
        real(8) :: tmpgam,recpgamma,deltat,tmpx,tmpy
        real(8) :: s1,s7,s16,s17,s13,s3,s9,s20,s21,s14,s28,s29,rm
        s1 = 0.0d0
        s7 = 0.0d0
        s16 = 0.0d0
        s17 = 0.0d0
        s13 = 0.0d0
        s3 = 0.0d0
        s9 = 0.0d0
        s20 = 0.0d0
        s21 = 0.0d0
        s14 = 0.0d0
        s28 = 0.0d0
        s29 = 0.0d0
        rm = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          tmpgam = sqrt(1.0d0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          s28 = s28 + tmpgam
          s29 = s29 + tmpgam**2
          recpgamma = 1.0d0/tmpgam
          deltat = (p(5,i)-z0avg)/(recpgamma*p(6,i))
          tmpx = p(1,i)-recpgamma*p(2,i)*deltat
          tmpy = p(3,i)-recpgamma*p(4,i)*deltat
          s1 = s1 + tmpx
          s7 = s7 + tmpx*tmpx
          s16 = s16 + tmpx*tmpx*tmpx
          s17 = s17 + tmpx*tmpx*tmpx*tmpx
          s13 = s13 + tmpx*p(2,i)
          s3 = s3 + tmpy
          s9 = s9 + tmpy*tmpy
          s20 = s20 + tmpy*tmpy*tmpy
          s21 = s21 + tmpy*tmpy*tmpy*tmpy
          s14 = s14 + tmpy*p(4,i)
          rm = max(rm,tmpx**2+tmpy**2)
        enddo
        a(1) = s1
        a(7) = s7
        a(16) = s16
        a(17) = s17
        a(13) = s13
        a(3) = s3
        a(9) = s9
        a(20) = s20
        a(21) = s21
        a(14) = s14
        a(28) = s28
        a(29) = s29
        rmx = rm
        end subroutine d2a_core

        ! momentum sums and powers (px, py raw; pz about its mean as the host
        ! loop does with pz0 from pass 1 is applied by the caller for 12, 26, 27)
        subroutine d2b_core(p,n,a,amx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(inout) :: a(41)
        real(8), intent(out) :: amx(6)
        integer :: i
        real(8) :: s2,s8,s18,s19,s4,s10,s22,s23,s6,m1,m2,m3,m4,m6
        s2 = 0.0d0
        s8 = 0.0d0
        s18 = 0.0d0
        s19 = 0.0d0
        s4 = 0.0d0
        s10 = 0.0d0
        s22 = 0.0d0
        s23 = 0.0d0
        s6 = 0.0d0
        m1 = 0.0d0
        m2 = 0.0d0
        m3 = 0.0d0
        m4 = 0.0d0
        m6 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s2 = s2 + p(2,i)
          s8 = s8 + p(2,i)*p(2,i)
          s18 = s18 + p(2,i)*p(2,i)*p(2,i)
          s19 = s19 + p(2,i)*p(2,i)*p(2,i)*p(2,i)
          s4 = s4 + p(4,i)
          s10 = s10 + p(4,i)*p(4,i)
          s22 = s22 + p(4,i)*p(4,i)*p(4,i)
          s23 = s23 + p(4,i)*p(4,i)*p(4,i)*p(4,i)
          s6 = s6 + p(6,i)
          m1 = max(m1,abs(p(1,i)))
          m2 = max(m2,abs(p(2,i)))
          m3 = max(m3,abs(p(3,i)))
          m4 = max(m4,abs(p(4,i)))
          m6 = max(m6,abs(p(6,i)))
        enddo
        a(2) = s2
        a(8) = s8
        a(18) = s18
        a(19) = s19
        a(4) = s4
        a(10) = s10
        a(22) = s22
        a(23) = s23
        a(6) = s6
        amx(1) = m1
        amx(2) = m2
        amx(3) = m3
        amx(4) = m4
        amx(6) = m6
        end subroutine d2b_core

        ! cross correlations 30-41
        subroutine d2c_core(p,n,z0avg,m0,a)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(in) :: z0avg,m0(6)
        real(8), intent(inout) :: a(41)
        integer :: i
        real(8) :: x0,px0,y0,py0,z0,pz0
        real(8) :: tmpgam,recpgamma,deltat,tmpx,tmpy
        real(8) :: s30,s31,s32,s33,s34,s35,s36,s37,s38,s39,s40,s41
        x0 = m0(1)
        px0 = m0(2)
        y0 = m0(3)
        py0 = m0(4)
        z0 = m0(5)
        pz0 = m0(6)
        s30 = 0.0d0
        s31 = 0.0d0
        s32 = 0.0d0
        s33 = 0.0d0
        s34 = 0.0d0
        s35 = 0.0d0
        s36 = 0.0d0
        s37 = 0.0d0
        s38 = 0.0d0
        s39 = 0.0d0
        s40 = 0.0d0
        s41 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          tmpgam = sqrt(1.0d0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          recpgamma = 1.0d0/tmpgam
          deltat = (p(5,i)-z0avg)/(recpgamma*p(6,i))
          tmpx = p(1,i)-recpgamma*p(2,i)*deltat
          tmpy = p(3,i)-recpgamma*p(4,i)*deltat
          s30 = s30 + (tmpx-x0)*(tmpy-y0)
          s31 = s31 + (tmpx-x0)*(p(4,i)-py0)
          s32 = s32 + (p(2,i)-px0)*(tmpy-y0)
          s33 = s33 + (p(2,i)-px0)*(p(4,i)-py0)
          s34 = s34 + (tmpx-x0)*(p(5,i)-z0)
          s35 = s35 + (tmpx-x0)*(p(6,i)-pz0)
          s36 = s36 + (p(5,i)-z0)*(p(2,i)-px0)
          s37 = s37 + (p(2,i)-px0)*(p(6,i)-pz0)
          s38 = s38 + (tmpy-y0)*(p(5,i)-z0)
          s39 = s39 + (tmpy-y0)*(p(6,i)-pz0)
          s40 = s40 + (p(5,i)-z0)*(p(4,i)-py0)
          s41 = s41 + (p(4,i)-py0)*(p(6,i)-pz0)
        enddo
        a(30) = s30
        a(31) = s31
        a(32) = s32
        a(33) = s33
        a(34) = s34
        a(35) = s35
        a(36) = s36
        a(37) = s37
        a(38) = s38
        a(39) = s39
        a(40) = s40
        a(41) = s41
        end subroutine d2c_core

        ! z about z0avg, pz about pz0, and max |z - z0avg|
        subroutine d2d_core(p,n,z0avg,pz0,a,amx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(in) :: z0avg,pz0
        real(8), intent(inout) :: a(41)
        real(8), intent(inout) :: amx(6)
        integer :: i
        real(8) :: s11,s24,s25,s15,s12,s26,s27,m5
        s11 = 0.0d0
        s24 = 0.0d0
        s25 = 0.0d0
        s15 = 0.0d0
        s12 = 0.0d0
        s26 = 0.0d0
        s27 = 0.0d0
        m5 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s11 = s11 + (p(5,i)-z0avg)**2
          s24 = s24 + ((p(5,i)-z0avg)**3)
          s25 = s25 + (p(5,i)-z0avg)**4
          s15 = s15 + (p(5,i)-z0avg)*(p(6,i)-pz0)
          s12 = s12 + (p(6,i)-pz0)**2
          s26 = s26 + (p(6,i)-pz0)**3
          s27 = s27 + (p(6,i)-pz0)**4
          m5 = max(m5,abs(p(5,i)-z0avg))
        enddo
        a(11) = s11
        a(24) = s24
        a(25) = s25
        a(15) = s15
        a(12) = s12
        a(26) = s26
        a(27) = s27
        amx(5) = m5
        end subroutine d2d_core


        !-------------------------------------------------------------
        ! D1.6: the bend-frame kernels (type-4 bend loop)
        !-------------------------------------------------------------
        !> rottoT_BeamBunch (dir = 1) / rotbackT_BeamBunch (dir = -1) on
        !> rows 1, 2, 5, 6.  All four values are read before any is written.
        attributes(global) subroutine k_rot(p,n,cs,ss,dir)
        implicit none
        integer, value :: n,dir
        real(8), value :: cs,ss
        real(8), device :: p(9,n)
        integer :: i
        real(8) :: t1,t2,t5,t6
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        t1 = p(1,i)
        t2 = p(2,i)
        t5 = p(5,i)
        t6 = p(6,i)
        if(dir.eq.1) then
          p(1,i) = t1*cs - t5*ss
          p(2,i) = t2*cs - t6*ss
          p(5,i) = t1*ss + t5*cs
          p(6,i) = t2*ss + t6*cs
        else
          p(1,i) = t1*cs + t5*ss
          p(2,i) = t2*cs + t6*ss
          p(5,i) = -t1*ss + t5*cs
          p(6,i) = -t2*ss + t6*cs
        endif
        end subroutine k_rot

        !> convEntr: z -= zsh (zsh = zorgin/Scxlt, computed on the host)
        attributes(global) subroutine k_shiftz(p,n,zsh)
        implicit none
        integer, value :: n
        real(8), value :: zsh
        real(8), device :: p(9,n)
        integer :: i
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        p(5,i) = p(5,i) - zsh
        end subroutine k_shiftz

        !> convExit_BeamBunch's particle loop; (x0,y0,z0) = refptcl(1,3,5)
        attributes(global) subroutine k_convexit(p,n,cs,ss,x0,y0,z0,zsh)
        implicit none
        integer, value :: n
        real(8), value :: cs,ss,x0,y0,z0,zsh
        real(8), device :: p(9,n)
        integer :: i
        real(8) :: t1,t2,t3,t5,t6
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        t1 = p(1,i) - x0
        t2 = p(2,i)
        t3 = p(3,i) - y0
        t5 = p(5,i) - z0
        t6 = p(6,i)
        p(1,i) = t1*cs - t5*ss
        p(2,i) = t2*cs - t6*ss
        p(3,i) = t3
        p(5,i) = t1*ss + t5*cs + zsh
        p(6,i) = t2*ss + t6*cs
        end subroutine k_convexit

        !> min and max of x, y, z and the sum of z, from the rotated particles
        subroutine rrange_core(p,n,rmn,rmx,zs)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(out) :: rmn(3),rmx(3),zs
        integer :: i
        real(8) :: xmn,ymn,zmn,xmx,ymx,zmx,s5
        xmn = 1.0e20
        ymn = 1.0e20
        zmn = 1.0e20
        xmx = -1.0e20
        ymx = -1.0e20
        zmx = -1.0e20
        s5 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          xmn = min(xmn,p(1,i))
          ymn = min(ymn,p(3,i))
          zmn = min(zmn,p(5,i))
          xmx = max(xmx,p(1,i))
          ymx = max(ymx,p(3,i))
          zmx = max(zmx,p(5,i))
          s5 = s5 + p(5,i)
        enddo
        rmn(1) = xmn
        rmn(2) = ymn
        rmn(3) = zmn
        rmx(1) = xmx
        rmx(2) = ymx
        rmx(3) = zmx
        zs = s5
        end subroutine rrange_core

        !> rottoT_BeamBunch: rotate to the local frame of the reference
        !> particle (cs, ss computed by the caller from ptref with the CPU's
        !> expressions), then ptrange (min/max of x, y, z of the rotated
        !> particles, +-1.0e20 on a rank with none) and poscent.
        subroutine rottoT_gpu(ib,bunch,cs,ss,ptrange,poscent)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: cs,ss
        double precision, dimension(6), intent(out) :: ptrange
        double precision, intent(out) :: poscent
        double precision :: rmn(3),rmx(3),zs,zsum,t0
        integer :: i,ierr
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call k_rot<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,cs,ss,1)
          call rrange_core(pb(ib)%p,pb(ib)%n,rmn,rmx,zs)
        else
          rmn = 1.0e20
          rmx = -1.0e20
          zs = 0.0d0
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        do i = 1, 3
          ptrange(2*i-1) = rmn(i)
          ptrange(2*i) = rmx(i)
        enddo
        call MPI_ALLREDUCE(zs,zsum,1,MPI_DOUBLE_PRECISION,MPI_SUM, &
                           MPI_COMM_WORLD,ierr)
        poscent = zsum/bunch%Npt
        end subroutine rottoT_gpu

        !> rotbackT_BeamBunch with cs, ss of the same step's ptref
        subroutine rotbackT_gpu(ib,bunch,cs,ss)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: cs,ss
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call k_rot<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,cs,ss,-1)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        end subroutine rotbackT_gpu

        !> convEntr_BeamBunch: particles on the device, refptcl on the host
        subroutine convEntr_gpu(ib,bunch,zorgin,gamin)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: zorgin,gamin
        double precision :: zsh,t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        zsh = zorgin/Scxlt
        if(pb(ib)%n.gt.0) then
          call k_shiftz<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,zsh)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        bunch%refptcl(1) = 0.0
        bunch%refptcl(2) = 0.0
        bunch%refptcl(3) = 0.0
        bunch%refptcl(4) = 0.0
        bunch%refptcl(5) = bunch%refptcl(5) - zorgin/Scxlt
        bunch%refptcl(6) = sqrt(gamin**2  - 1.0d0) !//gamma beta_z
        end subroutine convEntr_gpu

        !> convExit_BeamBunch: particles on the device, refptcl on the host
        subroutine convExit_gpu(ib,bunch,zorgin)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: zorgin
        double precision :: cs,ss,zsh,gamma0,t0
        double precision, dimension(6) :: temp
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        cs = bunch%refptcl(6)/sqrt(bunch%refptcl(6)**2 + bunch%refptcl(2)**2)
        ss = bunch%refptcl(2)/sqrt(bunch%refptcl(6)**2 + bunch%refptcl(2)**2)
        zsh = zorgin/Scxlt
        if(pb(ib)%n.gt.0) then
          call k_convexit<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,cs,ss, &
               bunch%refptcl(1),bunch%refptcl(3),bunch%refptcl(5),zsh)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        temp(2) = bunch%refptcl(2)
        temp(6) = bunch%refptcl(6)
        bunch%refptcl(1) = 0.0
        bunch%refptcl(2) = 0.0
        bunch%refptcl(3) = 0.0
        bunch%refptcl(4) = 0.0
        bunch%refptcl(5) = 0.0 + zorgin/Scxlt
        bunch%refptcl(6) = sqrt(temp(2)**2 + temp(6)**2)
        gamma0 = sqrt(1.0d0+bunch%refptcl(6)**2)
        bunch%refptcl(6) = -gamma0
        end subroutine convExit_gpu

        !> driftbackhalf_BeamBunch: the drifthalforg mode of k_drift with -tau
        !> (a negated tau gives the CPU's subtraction bit for bit); refptcl on
        !> the host.
        subroutine driftbackhalf_gpu(ib,bunch,tau)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tau
        double precision :: recpgam,t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call k_drift<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,-tau,0.0d0,0)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        recpgam = 1.0d0/sqrt(1.0d0+bunch%refptcl(2)**2+bunch%refptcl(4)**2+&
                           bunch%refptcl(6)**2)
        bunch%refptcl(1) = bunch%refptcl(1)-0.5d0*tau*bunch%refptcl(2)*recpgam
        bunch%refptcl(3) = bunch%refptcl(3)-0.5d0*tau*bunch%refptcl(4)*recpgam
        bunch%refptcl(5) = bunch%refptcl(5)-0.5d0*tau*bunch%refptcl(6)*recpgam
        end subroutine driftbackhalf_gpu

        !> drifthalfBd_BeamBunch: k_drift mode 0 (no z > 0 test); refptcl on
        !> the host.
        subroutine drifthalfBd_gpu(ib,bunch,tau)
        implicit none
        integer, intent(in) :: ib
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: tau
        double precision :: recpgam,t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          call k_drift<<<(pb(ib)%n+255)/256,256>>>(pb(ib)%p,pb(ib)%n,tau,0.0d0,0)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_drftgpu = t_drftgpu + elapsedtime_Timer(t0)
        recpgam = 1.0d0/sqrt(1.0d0+bunch%refptcl(2)**2+bunch%refptcl(4)**2+&
                           bunch%refptcl(6)**2)
        bunch%refptcl(1) = bunch%refptcl(1)+0.5d0*tau*bunch%refptcl(2)*recpgam
        bunch%refptcl(3) = bunch%refptcl(3)+0.5d0*tau*bunch%refptcl(4)*recpgam
        bunch%refptcl(5) = bunch%refptcl(5)+0.5d0*tau*bunch%refptcl(6)*recpgam
        end subroutine drifthalfBd_gpu

        !> The per-particle part of kick2tBd/kick2tBd0 after the space-charge
        !> fields: the dipole of the bend alone, rotated into the beam frame,
        !> and the Boris update.  sc = .true. adds the mesh fields (kick2tBd,
        !> bz has no space-charge part); sc = .false. is kick2tBd0.
        attributes(device) subroutine bend_push(p,i,sc,exn,eyn,ezn,bxn,byn, &
                                                cs,ss,dt,slot,b0,p5,scxlt_,clight_)
        implicit none
        real(8), device :: p(9,*)
        integer, value :: i,slot
        logical, value :: sc
        real(8), value :: exn,eyn,ezn,bxn,byn,cs,ss,dt,b0,p5,scxlt_,clight_
        real(8) :: pos1,pos2,pos3,coefLz,ex,ey,ez,bx,by,bz,ef(6)
        pos1 = (cs*p(1,i)+ss*p(5,i))*scxlt_
        pos2 = p(3,i)*scxlt_
        pos3 = (-ss*p(1,i) + cs*p(5,i))*scxlt_
        coefLz = p(7,i)*scxlt_
        call fld_dipole(pos1,pos2,pos3,slot,b0,p5,ef)
        if(sc) then
          ex = exn+ef(1)*cs - ef(3)*ss
          ey = eyn+ef(2)
          ez = ezn+ef(1)*ss + ef(3)*cs
          bx = bxn+ef(4)*cs - ef(6)*ss
          by = byn+ef(5)
          bz = ef(4)*ss + ef(6)*cs
        else
          ex = ef(1)*cs - ef(3)*ss
          ey = ef(2)
          ez = ef(1)*ss + ef(3)*cs
          bx = ef(4)*cs - ef(6)*ss
          by = ef(5)
          bz = ef(4)*ss + ef(6)*cs
        endif
        call boris(p(2,i),p(4,i),p(6,i),coefLz,ex,ey,ez,bx,by,bz,dt,clight_)
        end subroutine bend_push

        !> kick2tBd0_BeamBunch's particle loop: no space charge
        attributes(global) subroutine k_kick_bend0(p,n,cs,ss,dt,slot,b0,p5, &
              scxlt_,clight_)
        implicit none
        integer, value :: n,slot
        real(8), value :: cs,ss,dt,b0,p5,scxlt_,clight_
        real(8), device :: p(9,n)
        integer :: i
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        call bend_push(p,i,.false.,0.0d0,0.0d0,0.0d0,0.0d0,0.0d0,cs,ss,dt, &
                       slot,b0,p5,scxlt_,clight_)
        end subroutine k_kick_bend0

        !> kick2tBd_BeamBunch's particle loop: the five mesh fields of the
        !> rotated beam (global grid, jadd = kadd = 0), every particle.
        attributes(global) subroutine k_kick_bend_mesh(p,n,exg,eyg,ezg,bxg,byg, &
              nx,ny,nz,xmin,ymin,zmin,hx,hy,hz,hxi,hyi,hzi,cs,ss,dt,slot,b0,p5, &
              scxlt_,clight_)
        implicit none
        integer, value :: n,nx,ny,nz,slot
        real(8), value :: xmin,ymin,zmin,hx,hy,hz,hxi,hyi,hzi,cs,ss,dt,b0,p5, &
                          scxlt_,clight_
        real(8), device :: p(9,n)
        real(8), device :: exg(nx,ny,nz),eyg(nx,ny,nz),ezg(nx,ny,nz)
        real(8), device :: bxg(nx,ny,nz),byg(nx,ny,nz)
        integer :: i,ix,jx,kx,ix1,jx1,kx1,old
        real(8) :: ab,cd,ef,exn,eyn,ezn,bxn,byn
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
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

        call bend_push(p,i,.true.,exn,eyn,ezn,bxn,byn,cs,ss,dt,slot,b0,p5, &
                       scxlt_,clight_)
        end subroutine k_kick_bend_mesh

        !> The particle kick of kick2tBd (usemesh) or kick2tBd0 (not) in the
        !> bend frame.  The caller has already kicked refptcl (kickref_Bd) and
        !> passes the cs, ss that call returned.  slot, b0 = Param(3) and p5 =
        !> Param(5) are those of the bend's dipole element; the mesh fields
        !> are ex_dev..by_dev on the grid (nx,ny,nz) of range, msize.
        subroutine kick_bend_gpu(ib,bunch,usemesh,cs,ss,dt,slot,b0,p5,nx,ny,nz, &
                                 range,msize)
        implicit none
        integer, intent(in) :: ib,slot,nx,ny,nz
        logical, intent(in) :: usemesh
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: cs,ss,dt,b0,p5
        double precision, dimension(6), intent(in) :: range
        double precision, dimension(3), intent(in) :: msize
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        if(pb(ib)%n.gt.0) then
          if(usemesh) then
            call k_kick_bend_mesh<<<(pb(ib)%n+127)/128,128>>>(pb(ib)%p,pb(ib)%n, &
                 ex_dev,ey_dev,ez_dev,bx_dev,by_dev,nx,ny,nz,range(1),range(3), &
                 range(5),msize(1),msize(2),msize(3),1.0d0/msize(1), &
                 1.0d0/msize(2),1.0d0/msize(3),cs,ss,dt,slot,b0,p5,Scxlt,Clight)
          else
            call k_kick_bend0<<<(pb(ib)%n+127)/128,128>>>(pb(ib)%p,pb(ib)%n,cs, &
                 ss,dt,slot,b0,p5,Scxlt,Clight)
          endif
        endif
        pb(ib)%state = 1
        call sync_gpu()
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine kick_bend_gpu


        !> kickpt2ptBd_BeamBunch's particle loop: pt2ptFields_dev fields (in
        !> fex..fbz, scaled here by coefE), the dipole of the bend alone, the
        !> optional CSR wake interpolated with the histogram's weights.
        attributes(global) subroutine k_kick_bend_pt2pt(p,n,fx,fy,fz,gx,gy,gz, &
              coefE,csrflag,ezw,nzw,zminw,hzw0,cs,ss,dt,slot,b0,p5,scxlt_,clight_)
        implicit none
        integer, value :: n,nzw,csrflag,slot
        real(8), value :: coefE,zminw,hzw0,cs,ss,dt,b0,p5,scxlt_,clight_
        real(8), device :: p(9,n)
        real(8), device :: fx(n),fy(n),fz(n),gx(n),gy(n),gz(n),ezw(nzw)
        integer :: i,iizz
        real(8) :: pos1,pos2,pos3,coefLz,ex,ey,ez,bx,by,bz,ef(6),eeff,ezc
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        pos1 = (cs*p(1,i)+ss*p(5,i))*scxlt_
        pos2 = p(3,i)*scxlt_
        pos3 = (-ss*p(1,i) + cs*p(5,i))*scxlt_
        coefLz = p(7,i)*scxlt_
        call fld_dipole(pos1,pos2,pos3,slot,b0,p5,ef)
        ezc = coefE*fz(i)
        if(csrflag.eq.1) then
          iizz = (p(5,i)-zminw)/hzw0 + 1
          eeff = ((zminw-p(5,i))+iizz*hzw0)/hzw0
          ezc = ezc + (ezw(iizz)*eeff + ezw(iizz+1)*(1.0d0-eeff))
        endif
        ex = coefE*fx(i)+ef(1)*cs - ef(3)*ss
        ey = coefE*fy(i)+ef(2)
        ez = ezc+ef(1)*ss + ef(3)*cs
        bx = coefE*gx(i)/clight_+ef(4)*cs - ef(6)*ss
        by = coefE*gy(i)/clight_+ef(5)
        bz = coefE*gz(i)/clight_+(ef(4)*ss + ef(6)*cs)
        call boris(p(2,i),p(4,i),p(6,i),coefLz,ex,ey,ez,bx,by,bz,dt,clight_)
        end subroutine k_kick_bend_pt2pt

        !> The N-body particle kick of a bend step (D1.12): pair fields of all
        !> particles in the rotated frame (image off, every particle a source),
        !> then k_kick_bend_pt2pt.  ezwake(1:nzw) is used when csrflag = 1.
        subroutine bendpt2pt_gpu(ib,bunch,cs,ss,dt,slot,b0,p5,totchrg,r0, &
                                 npttot,nptrue,csrflag,ezwake,nzw,zminw,hzw0)
        implicit none
        integer, intent(in) :: ib,slot,npttot,nptrue,csrflag,nzw
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: cs,ss,dt,b0,p5,totchrg,r0,zminw,hzw0
        double precision, intent(in) :: ezwake(nzw)
        real(8), device, allocatable :: ezw_d(:)
        integer :: n,sgn
        double precision :: coefE,t0
        call ensure_device(ib,bunch)
        n = pb(ib)%n
        if(bunch%Charge.gt.0.0) then
          sgn = 1
        else
          sgn = -1
        endif
        coefE = 1.0d0/(4*Pi*Epsilon0*Scxlt*Scxlt)*sgn*totchrg/nptrue
        call starttime_Timer(t0)
        call pt2pt_fields_gpu(ib,bunch,.true.,.false.,r0,r0,npttot)
        allocate(ezw_d(max(nzw,1)))
        ezw_d(1:nzw) = ezwake
        if(n.gt.0) then
          call k_kick_bend_pt2pt<<<(n+127)/128,128>>>(pb(ib)%p,n,fex,fey,fez,fbx, &
               fby,fbz,coefE,csrflag,ezw_d,nzw,zminw,hzw0,cs,ss,dt,slot,b0,p5, &
               Scxlt,Clight)
        endif
        pb(ib)%state = 1
        call sync_gpu()
        deallocate(ezw_d)
        t_kickgpu = t_kickgpu + elapsedtime_Timer(t0)
        end subroutine bendpt2pt_gpu

        !-------------------------------------------------------------
        ! D1.7: diagnostic1avgB_Output on the device
        !-------------------------------------------------------------
        !> the local sums of z, pz and gamma (pass 1 of diagnostic1avgB)
        subroutine diagB_pass1_gpu(bunch,nb,sums)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        double precision, intent(out) :: sums(3)
        double precision :: part(3)
        integer :: ib
        sums = 0.0d0
        do ib = 1, nb
          call ensure_device(ib,bunch(ib))
          if(pb(ib)%n.gt.0) then
            call dB1_core(pb(ib)%p,pb(ib)%n,part)
            sums = sums + part
          endif
        enddo
        end subroutine diagB_pass1_gpu

        subroutine dB1_core(p,n,sm)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(out) :: sm(3)
        integer :: i
        real(8) :: s1,s2,s3
        s1 = 0.0d0
        s2 = 0.0d0
        s3 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s1 = s1 + p(5,i)
          s2 = s2 + p(6,i)
          s3 = s3 + sqrt(1.0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
        enddo
        sm(1) = s1
        sm(2) = s2
        sm(3) = s3
        end subroutine dB1_core

        !> the 29 local sums (tmplc), the six local maxima and the local max
        !> r^2 of diagnostic1avgB_Output (pass 2).  tmplc(5) stays 0 (the host
        !> loop never adds to z0lc).
        subroutine diagB_pass2_gpu(bunch,nb,z0avg,pz0avg,gamavg,tmplc,localmax, &
                                   lcrmax)
        implicit none
        integer, intent(in) :: nb
        type (BeamBunch), dimension(:), intent(inout) :: bunch
        double precision, intent(in) :: z0avg,pz0avg,gamavg
        double precision, intent(out) :: tmplc(29),localmax(6),lcrmax
        double precision :: a(29),amx(6),ar
        integer :: ib
        tmplc = 0.0d0
        localmax = 0.0d0
        lcrmax = 0.0d0
        do ib = 1, nb
          call ensure_device(ib,bunch(ib))
          if(pb(ib)%n.gt.0) then
            a = 0.0d0
            amx = 0.0d0
            ar = 0.0d0
            call dB2x_core(pb(ib)%p,pb(ib)%n,a,amx)
            call dB2y_core(pb(ib)%p,pb(ib)%n,a,amx)
            call dB2z_core(pb(ib)%p,pb(ib)%n,z0avg,pz0avg,gamavg,a,amx,ar)
            tmplc = tmplc + a
            localmax = max(localmax,amx)
            lcrmax = max(lcrmax,ar)
          endif
        enddo
        end subroutine diagB_pass2_gpu

        ! x, px: sums 1, 2, 7, 8, 13, 16-19; maxima of |x|, |px|
        subroutine dB2x_core(p,n,a,amx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(inout) :: a(29),amx(6)
        integer :: i
        real(8) :: s1,s2,s7,s8,s13,s16,s17,s18,s19,m1,m2
        s1 = 0.0d0
        s2 = 0.0d0
        s7 = 0.0d0
        s8 = 0.0d0
        s13 = 0.0d0
        s16 = 0.0d0
        s17 = 0.0d0
        s18 = 0.0d0
        s19 = 0.0d0
        m1 = 0.0d0
        m2 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s1 = s1 + p(1,i)
          s7 = s7 + p(1,i)*p(1,i)
          s16 = s16 + p(1,i)*p(1,i)*p(1,i)
          s17 = s17 + p(1,i)*p(1,i)*p(1,i)*p(1,i)
          s13 = s13 + p(1,i)*p(2,i)
          s2 = s2 + p(2,i)
          s8 = s8 + p(2,i)*p(2,i)
          s18 = s18 + p(2,i)*p(2,i)*p(2,i)
          s19 = s19 + p(2,i)*p(2,i)*p(2,i)*p(2,i)
          m1 = max(m1,abs(p(1,i)))
          m2 = max(m2,abs(p(2,i)))
        enddo
        a(1) = s1
        a(2) = s2
        a(7) = s7
        a(8) = s8
        a(13) = s13
        a(16) = s16
        a(17) = s17
        a(18) = s18
        a(19) = s19
        amx(1) = m1
        amx(2) = m2
        end subroutine dB2x_core

        ! y, py: sums 3, 4, 9, 10, 14, 20-23; maxima of |y|, |py|
        subroutine dB2y_core(p,n,a,amx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(inout) :: a(29),amx(6)
        integer :: i
        real(8) :: s3,s4,s9,s10,s14,s20,s21,s22,s23,m3,m4
        s3 = 0.0d0
        s4 = 0.0d0
        s9 = 0.0d0
        s10 = 0.0d0
        s14 = 0.0d0
        s20 = 0.0d0
        s21 = 0.0d0
        s22 = 0.0d0
        s23 = 0.0d0
        m3 = 0.0d0
        m4 = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          s3 = s3 + p(3,i)
          s9 = s9 + p(3,i)*p(3,i)
          s20 = s20 + p(3,i)*p(3,i)*p(3,i)
          s21 = s21 + p(3,i)*p(3,i)*p(3,i)*p(3,i)
          s14 = s14 + p(3,i)*p(4,i)
          s4 = s4 + p(4,i)
          s10 = s10 + p(4,i)*p(4,i)
          s22 = s22 + p(4,i)*p(4,i)*p(4,i)
          s23 = s23 + p(4,i)*p(4,i)*p(4,i)*p(4,i)
          m3 = max(m3,abs(p(3,i)))
          m4 = max(m4,abs(p(4,i)))
        enddo
        a(3) = s3
        a(4) = s4
        a(9) = s9
        a(10) = s10
        a(14) = s14
        a(20) = s20
        a(21) = s21
        a(22) = s22
        a(23) = s23
        amx(3) = m3
        amx(4) = m4
        end subroutine dB2y_core

        ! z about z0avg, pz about pz0avg, gamma sums, max r^2: sums 6, 11, 12,
        ! 15, 24-29; maxima 5, 6
        subroutine dB2z_core(p,n,z0avg,pz0avg,gamavg,a,amx,rmx)
        implicit none
        integer, intent(in) :: n
        real(8), device :: p(9,n)
        real(8), intent(in) :: z0avg,pz0avg,gamavg
        real(8), intent(inout) :: a(29),amx(6)
        real(8), intent(out) :: rmx
        integer :: i
        real(8) :: s6,s11,s12,s15,s24,s25,s26,s27,s28,s29,m5,m6,rm,tmpgam
        s6 = 0.0d0
        s11 = 0.0d0
        s12 = 0.0d0
        s15 = 0.0d0
        s24 = 0.0d0
        s25 = 0.0d0
        s26 = 0.0d0
        s27 = 0.0d0
        s28 = 0.0d0
        s29 = 0.0d0
        m5 = 0.0d0
        m6 = 0.0d0
        rm = 0.0d0
        !$cuf kernel do <<<*,*>>>
        do i = 1, n
          tmpgam = sqrt(1.0+p(2,i)**2+p(4,i)**2+p(6,i)**2)
          s28 = s28 + tmpgam
          s29 = s29 + (tmpgam-gamavg)**2
          s11 = s11 + (p(5,i)-z0avg)**2
          s24 = s24 + ((p(5,i)-z0avg)**3)
          s25 = s25 + (p(5,i)-z0avg)**4
          s15 = s15 + (p(5,i)-z0avg)*(p(6,i)-pz0avg)
          s6 = s6 + p(6,i)
          s12 = s12 + (p(6,i)-pz0avg)**2
          s26 = s26 + ((p(6,i)-pz0avg)**3)
          s27 = s27 + (p(6,i)-pz0avg)**4
          m5 = max(m5,abs(p(5,i)-z0avg))
          m6 = max(m6,abs(p(6,i)-pz0avg))
          rm = max(rm,p(1,i)**2+p(3,i)**2)
        enddo
        a(6) = s6
        a(11) = s11
        a(12) = s12
        a(15) = s15
        a(24) = s24
        a(25) = s25
        a(26) = s26
        a(27) = s27
        a(28) = s28
        a(29) = s29
        amx(5) = m5
        amx(6) = m6
        rmx = rm
        end subroutine dB2z_core


        !-------------------------------------------------------------
        ! D1.11: CSR line density and wake field on the device
        !-------------------------------------------------------------
        !> the line-density histogram of the CSR block of the host bend loop,
        !> on the rotated z (row 5): iizz and eeff as written there
        attributes(global) subroutine k_csr_hist(p,n,dens,nz,zmin,hzwake)
        implicit none
        integer, value :: n,nz
        real(8), value :: zmin,hzwake
        real(8), device :: p(9,n)
        real(8), device :: dens(nz+1)
        integer :: i,iizz,old
        real(8) :: eeff,dum
        i = (blockIdx%x-1)*blockDim%x + threadIdx%x
        if(i.gt.n) return
        iizz = (p(5,i)-zmin)/hzwake + 1
        eeff = ((zmin-p(5,i))+iizz*hzwake)/hzwake
        if(iizz.lt.1 .or. iizz.gt.nz) then
          old = atomicCAS(gpuerr_d,0,6)
          if(old.eq.0) then
            gpuerr_info_d(2) = i
            gpuerr_info_d(3) = iizz
          endif
          return
        endif
        dum = atomicAdd(dens(iizz),eeff)
        dum = atomicAdd(dens(iizz+1),1.0d0-eeff)
        end subroutine k_csr_hist

        !> denslc(1:nz) = this rank's histogram (the CPU's denszlc, before the
        !> sums over ranks); zmin = grange(5) of the beam domain
        subroutine csr_hist_gpu(ib,bunch,nz,zmin,hzwake,denslc)
        implicit none
        integer, intent(in) :: ib,nz
        type (BeamBunch), intent(inout) :: bunch
        double precision, intent(in) :: zmin,hzwake
        double precision, intent(out) :: denslc(nz)
        real(8), device, allocatable :: dens_d(:)
        double precision, allocatable :: h(:)
        double precision :: t0
        call ensure_device(ib,bunch)
        call starttime_Timer(t0)
        allocate(dens_d(nz+1),h(nz+1))
        dens_d = 0.0d0
        if(pb(ib)%n.gt.0) then
          call k_csr_hist<<<(pb(ib)%n+127)/128,128>>>(pb(ib)%p,pb(ib)%n,dens_d, &
               nz,zmin,hzwake)
        endif
        call sync_gpu()
        h = dens_d
        denslc = h(1:nz)
        deallocate(dens_d,h)
        t_depgpu = t_depgpu + elapsedtime_Timer(t0)
        end subroutine csr_hist_gpu

        !> ez_dev(i,j,k) += ezw(k) on the global grid (kst = k)
        subroutine csr_addez_gpu(nx,ny,nz,ezw)
        implicit none
        integer, intent(in) :: nx,ny,nz
        double precision, intent(in) :: ezw(nz)
        real(8), device, allocatable :: ezw_d(:)
        integer :: i,j,k
        double precision :: t0
        call starttime_Timer(t0)
        allocate(ezw_d(nz))
        ezw_d = ezw
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nz
          do j = 1, ny
            do i = 1, nx
              ez_dev(i,j,k) = ez_dev(i,j,k) + ezw_d(k)
            enddo
          enddo
        enddo
        call sync_gpu()
        deallocate(ezw_d)
        t_depgpu = t_depgpu + elapsedtime_Timer(t0)
        end subroutine csr_addez_gpu

      end module ParticleGPUclass
