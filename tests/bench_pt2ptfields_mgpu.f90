! A3 performance check: multi-GPU scaling for pt2ptFields. Unlike
! bench_pt2ptfields.f90 (single rank, one GPU), this splits NTOT=1M
! particles into nptlc=NTOT/nproc local targets per rank -- each rank
! still gathers all NTOT particles as sources (MPI_ALLGATHERV), exactly
! as kickpt2ptImg_BeamBunch does in production -- so the total pair work
! (2*NTOT^2) is fixed regardless of nproc, just divided across ranks/GPUs.
! Run with one rank per GPU (srun --gpus-per-task=1) to measure scaling;
! see doc/GPU_PORT_DESIGN.md's A3 check and tests/README.md.
! Not part of the CMake build; see tests/README.md for how to build and run.
program bench_pt2ptfields_mgpu
        use BeamBunchclass, only: pt2ptFields
        use Timerclass, only: starttime_Timer, elapsedtime_Timer
        implicit none
        include 'mpif.h'

        integer, parameter :: NTOT = 1000000
        integer, parameter :: NREP = 5
        double precision, parameter :: R0 = 1.0d-9

        integer :: ierr, myid, nproc, nptlc, myoffset, n, irep, i, gk
        double precision, allocatable :: srcloc(:,:), sources(:,:), targets(:,:)
        double precision, allocatable :: ex(:),ey(:),ez(:),bx(:),by(:),bz(:)
        integer, allocatable :: recvcounts(:), displs(:)
        double precision :: gpx,gpy,gpz,gam
        double precision :: t0, elapsed, elapsed_max, pairs_total, rate

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

        if(mod(NTOT,nproc).ne.0) then
          if(myid.eq.0) print*,'FAIL: NTOT must be divisible by nproc'
          call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
        endif
        nptlc = NTOT/nproc

        !//replicate kickpt2ptImg_BeamBunch's gather: per-rank counts,
        !//displacements, heap source array via ALLGATHERV.
        allocate(recvcounts(nproc),displs(nproc))
        call MPI_ALLGATHER(nptlc,1,MPI_INTEGER,recvcounts,1,MPI_INTEGER,&
             MPI_COMM_WORLD,ierr)
        displs(1) = 0
        do i = 2, nproc
          displs(i) = displs(i-1)+recvcounts(i-1)
        enddo
        myoffset = displs(myid+1)

        allocate(srcloc(10,nptlc),targets(4,nptlc))
        do n = 1, nptlc
          gk = myoffset+n
          gpx = 0.30d0*sin(0.881d0*gk+1.20d0)
          gpy = 0.30d0*cos(0.727d0*gk+0.90d0)
          gpz = 0.50d0+0.40d0*sin(0.611d0*gk+0.30d0)
          gam = sqrt(1.0d0+gpx**2+gpy**2+gpz**2)
          srcloc(1,n) = 5.0d-5*sin(0.371d0*gk+0.13d0)
          srcloc(2,n) = 5.0d-5*cos(0.457d0*gk+0.71d0)
          srcloc(3,n) = 5.0d-3+4.5d-3*sin(0.231d0*gk+0.50d0)
          srcloc(4,n) = gpx/gam
          srcloc(5,n) = gpy/gam
          srcloc(6,n) = gpz/gam
          srcloc(7,n) = gam
          srcloc(8,n) = 1.0d0
          srcloc(9,n) = gam*gam/(gam+1.0d0)
          srcloc(10,n) = gam/(gam+1.0d0)
          targets(1,n) = srcloc(1,n)
          targets(2,n) = srcloc(2,n)
          targets(3,n) = srcloc(3,n)
          targets(4,n) = 1.0d0
        enddo

        allocate(sources(10,NTOT))
        call MPI_ALLGATHERV(srcloc,10*nptlc,MPI_DOUBLE_PRECISION,&
             sources,10*recvcounts,10*displs,MPI_DOUBLE_PRECISION,&
             MPI_COMM_WORLD,ierr)

        allocate(ex(nptlc),ey(nptlc),ez(nptlc),bx(nptlc),by(nptlc),bz(nptlc))

        !//one untimed warm-up call per rank.
        call pt2ptFields(nptlc,targets,myoffset,NTOT,sources,.true.,R0,R0,&
                          ex,ey,ez,bx,by,bz)

        call MPI_BARRIER(MPI_COMM_WORLD,ierr)
        call starttime_Timer(t0)
        do irep = 1, NREP
          call pt2ptFields(nptlc,targets,myoffset,NTOT,sources,.true.,R0,R0,&
                            ex,ey,ez,bx,by,bz)
        enddo
        elapsed = elapsedtime_Timer(t0)

        !//wall-clock parallel time is set by the slowest rank.
        call MPI_ALLREDUCE(elapsed,elapsed_max,1,MPI_DOUBLE_PRECISION,MPI_MAX,&
             MPI_COMM_WORLD,ierr)

        !//total pair work is independent of nproc: each rank does
        !//nptlc*NTOT*2 pairs (direct+image), summed over nproc ranks of
        !//nptlc=NTOT/nproc gives 2*NTOT^2 regardless of how it's split.
        pairs_total = dble(NREP)*2.0d0*dble(NTOT)*dble(NTOT)
        rate = pairs_total/elapsed_max

        if(myid.eq.0) then
          print '(A,I0)',       'nproc          = ', nproc
          print '(A,I0)',       'NTOT           = ', NTOT
          print '(A,I0)',       'nptlc per rank = ', nptlc
          print '(A,I0)',       'reps           = ', NREP
          print '(A,ES14.6,A)', 'elapsed_max    = ', elapsed_max, ' s'
          print '(A,ES14.6,A)', 'rate           = ', rate, ' pairs/s'
        endif

        call MPI_FINALIZE(ierr)

end program bench_pt2ptfields_mgpu
