! Profiling variant of bench_pt2ptfields.f90: smaller N and a single
! rep, so an `ncu` full-metric-set capture of the pt2ptFields kernel
! launch stays fast. See doc/GPU_PORT_DESIGN.md's A2 "Profiling" bullet.
program bench_pt2ptfields_prof
        use BeamBunchclass, only: pt2ptFields
        use Timerclass, only: starttime_Timer, elapsedtime_Timer
        implicit none
        include 'mpif.h'

        integer, parameter :: NTOT = 100000
        double precision, parameter :: R0 = 1.0d-9

        integer :: ierr, myid, nproc, n
        double precision, allocatable :: sources(:,:), targets(:,:)
        double precision, allocatable :: ex(:),ey(:),ez(:),bx(:),by(:),bz(:)
        double precision :: gpx,gpy,gpz,gam
        double precision :: t0, elapsed

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

        if(nproc.ne.1) then
          if(myid.eq.0) print*,'FAIL: run with 1 rank'
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

        !//one untimed warm-up call (first-touch GPU allocation), then one
        !//timed call -- ncu profiles whichever kernel launch(es) it's
        !//told to via --launch-skip/--launch-count.
        call pt2ptFields(NTOT,targets,0,NTOT,sources,.true.,R0,&
                          ex,ey,ez,bx,by,bz)

        call starttime_Timer(t0)
        call pt2ptFields(NTOT,targets,0,NTOT,sources,.true.,R0,&
                          ex,ey,ez,bx,by,bz)
        elapsed = elapsedtime_Timer(t0)
        print '(A,I0)',       'NTOT    = ', NTOT
        print '(A,ES14.6,A)', 'elapsed = ', elapsed, ' s'

        call MPI_FINALIZE(ierr)

end program bench_pt2ptfields_prof
