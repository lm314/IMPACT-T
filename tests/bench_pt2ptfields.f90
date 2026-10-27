! A2 performance check: pair evaluations per second for pt2ptFields
! (Appl/BeamBunch.f90) at 1M particles, image charge on (2N^2 pair
! evaluations per call, per doc/GPU_PORT_DESIGN.md's "What the code does
! today"). Single rank only -- this measures one GPU's (or, built without
! USE_GPU, one CPU core's) raw kernel rate, not multi-GPU scaling (A3).
! Builds the 10-column source array (x,y,z,betax,betay,betaz,gamma,flag,
! gamma^2/(gamma+1),gamma/(gamma+1)) the same way the production gather
! does, with the last two precomputed once per particle rather than once
! per pair inside pt2ptFields.
! Not part of the CMake build; see tests/README.md for how to build and run.
program bench_pt2ptfields
        use BeamBunchclass, only: pt2ptFields
        use Timerclass, only: starttime_Timer, elapsedtime_Timer
        implicit none
        include 'mpif.h'

        integer, parameter :: NTOT = 1000000
        integer, parameter :: NREP = 5
        double precision, parameter :: R0 = 1.0d-9

        integer :: ierr, myid, nproc, n, irep
        double precision, allocatable :: sources(:,:), targets(:,:)
        double precision, allocatable :: ex(:),ey(:),ez(:),bx(:),by(:),bz(:)
        double precision :: gpx,gpy,gpz,gam
        double precision :: t0, elapsed, pairs_per_call, rate
        double precision :: step_6p24m, run_6p24m_hours

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

        if(nproc.ne.1) then
          if(myid.eq.0) print*,'FAIL: run with 1 rank (single-GPU/core benchmark)'
          call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
        endif

        allocate(sources(10,NTOT),targets(4,NTOT))
        do n = 1, NTOT
          gpx = 0.30d0*sin(0.881d0*n+1.20d0)
          gpy = 0.30d0*cos(0.727d0*n+0.90d0)
          gpz = 0.50d0+0.40d0*sin(0.611d0*n+0.30d0)
          gam = sqrt(1.0d0+gpx**2+gpy**2+gpz**2)
          sources(1,n) = 5.0d-5*sin(0.371d0*n+0.13d0)
          sources(2,n) = 5.0d-5*cos(0.457d0*n+0.71d0)
          sources(3,n) = 5.0d-3+4.5d-3*sin(0.231d0*n+0.50d0)
          sources(4,n) = gpx/gam
          sources(5,n) = gpy/gam
          sources(6,n) = gpz/gam
          sources(7,n) = gam
          sources(8,n) = 1.0d0
          sources(9,n) = gam*gam/(gam+1.0d0)
          sources(10,n) = gam/(gam+1.0d0)
          targets(1,n) = sources(1,n)
          targets(2,n) = sources(2,n)
          targets(3,n) = sources(3,n)
          targets(4,n) = 1.0d0
        enddo

        allocate(ex(NTOT),ey(NTOT),ez(NTOT),bx(NTOT),by(NTOT),bz(NTOT))

        !//one untimed call first, so a lazy first-touch GPU data allocation
        !//or JIT doesn't pollute the timed reps.
        call pt2ptFields(NTOT,targets,0,NTOT,sources,.true.,R0,R0,&
                          ex,ey,ez,bx,by,bz)

        call starttime_Timer(t0)
        do irep = 1, NREP
          call pt2ptFields(NTOT,targets,0,NTOT,sources,.true.,R0,R0,&
                            ex,ey,ez,bx,by,bz)
        enddo
        elapsed = elapsedtime_Timer(t0)

        !//image on: a direct term (all pairs but j==gidx) plus an image
        !//term (all pairs, self included) -- 2*N^2 pair evaluations, per
        !//the plan's own count for this kernel.
        pairs_per_call = 2.0d0*dble(NTOT)*dble(NTOT)
        rate = dble(NREP)*pairs_per_call/elapsed

        !//project to the plan's target case: 6.24M particles (1 pC
        !//one-to-one), assuming the same per-pair rate holds (true while
        !//the kernel stays compute-bound and the working set still fits
        !//the GPU, both true here: 6.24M source rows is ~500 MB).
        step_6p24m = 2.0d0*6.24d6*6.24d6/rate
        run_6p24m_hours = step_6p24m*2500.0d0/3600.0d0

        print '(A,I0)',        'NTOT           = ', NTOT
        print '(A,I0)',        'reps           = ', NREP
        print '(A,ES14.6,A)',  'elapsed        = ', elapsed, ' s'
        print '(A,ES14.6)',    'pairs/call     = ', pairs_per_call
        print '(A,ES14.6,A)',  'rate           = ', rate, ' pairs/s'
        print '(A,ES14.6,A)',  'proj 6.24M step  = ', step_6p24m, ' s'
        print '(A,ES14.6,A)',  'proj 6.24M run (2500 steps) = ', run_6p24m_hours, ' hours'

        call MPI_FINALIZE(ierr)

end program bench_pt2ptfields
