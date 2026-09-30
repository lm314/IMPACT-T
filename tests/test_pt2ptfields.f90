! A1 check: validate pt2ptFields (Appl/BeamBunch.f90) against an independent
! reference implementation of the same pair formulas, at 1 and 4 MPI ranks.
! Not part of the CMake build; see tests/README.md for how to build and run.
program test_pt2ptfields
        use BeamBunchclass, only: pt2ptFields
        implicit none
        include 'mpif.h'

        integer, parameter :: NTOT = 2000
        double precision, parameter :: R0 = 1.0d-9

        integer :: ierr, myid, nproc, nptlc, npttot, myoffset
        integer :: n, i, gk
        double precision, allocatable :: gx(:),gy(:),gz(:),gpx(:),gpy(:),gpz(:)
        double precision, allocatable :: rays(:,:)
        double precision, allocatable :: srcloc(:,:), sources(:,:), targets(:,:)
        double precision, allocatable :: ex(:),ey(:),ez(:),bx(:),by(:),bz(:)
        double precision, allocatable :: exo(:),eyo(:),ezo(:),bxo(:),byo(:),bzo(:)
        double precision, allocatable :: exn(:),eyn(:),ezn(:),bxn(:),byn(:),bzn(:)
        integer, allocatable :: recvcounts(:), displs(:)
        double precision :: gam
        double precision :: diffA, diffB, diffC
        double precision :: refA_ex,refA_ey,refA_ez,refA_bx,refA_by,refA_bz
        double precision :: refB_ex,refB_ey,refB_ez,refB_bx,refB_by,refB_bz
        logical :: allok

        call MPI_INIT(ierr)
        call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)

        if(mod(NTOT,nproc).ne.0) then
          if(myid.eq.0) print*,'FAIL: NTOT must be divisible by nproc'
          call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
        endif
        nptlc = NTOT/nproc
        npttot = NTOT

        allocate(gx(NTOT),gy(NTOT),gz(NTOT),gpx(NTOT),gpy(NTOT),gpz(NTOT))
        do i = 1, NTOT
          gx(i)  = 5.0d-5*sin(0.371d0*i+0.13d0)
          gy(i)  = 5.0d-5*cos(0.457d0*i+0.71d0)
          gz(i)  = 5.0d-3+4.5d-3*sin(0.231d0*i+0.50d0)
          gpx(i) = 0.30d0*sin(0.881d0*i+1.20d0)
          gpy(i) = 0.30d0*cos(0.727d0*i+0.90d0)
          gpz(i) = 0.50d0+0.40d0*sin(0.611d0*i+0.30d0)
        enddo

        allocate(rays(9,nptlc))
        do n = 1, nptlc
          gk = myid*nptlc+n
          rays(1,n) = gx(gk)
          rays(2,n) = gpx(gk)
          rays(3,n) = gy(gk)
          rays(4,n) = gpy(gk)
          rays(5,n) = gz(gk)
          rays(6,n) = gpz(gk)
          rays(7,n) = 0.0d0
          rays(8,n) = 0.0d0
          rays(9,n) = 0.0d0
        enddo

        !//replicate the gather in kickpt2ptImg_BeamBunch: per-rank counts,
        !//displacements, and the heap source array via ALLGATHERV.
        allocate(recvcounts(nproc),displs(nproc))
        call MPI_ALLGATHER(nptlc,1,MPI_INTEGER,recvcounts,1,MPI_INTEGER,&
             MPI_COMM_WORLD,ierr)
        displs(1) = 0
        do i = 2, nproc
          displs(i) = displs(i-1)+recvcounts(i-1)
        enddo
        myoffset = displs(myid+1)

        allocate(srcloc(8,nptlc),targets(4,nptlc))
        do n = 1, nptlc
          gam = sqrt(1.0d0+rays(2,n)**2+rays(4,n)**2+rays(6,n)**2)
          srcloc(1,n) = rays(1,n)
          srcloc(2,n) = rays(3,n)
          srcloc(3,n) = rays(5,n)
          srcloc(4,n) = rays(2,n)/gam
          srcloc(5,n) = rays(4,n)/gam
          srcloc(6,n) = rays(6,n)/gam
          srcloc(7,n) = gam
          srcloc(8,n) = 1.0d0
          targets(1,n) = rays(1,n)
          targets(2,n) = rays(3,n)
          targets(3,n) = rays(5,n)
          targets(4,n) = 1.0d0
        enddo

        allocate(sources(8,npttot))
        call MPI_ALLGATHERV(srcloc,8*nptlc,MPI_DOUBLE_PRECISION,&
             sources,8*recvcounts,8*displs,MPI_DOUBLE_PRECISION,&
             MPI_COMM_WORLD,ierr)

        allocate(ex(nptlc),ey(nptlc),ez(nptlc),bx(nptlc),by(nptlc),bz(nptlc))
        allocate(exn(nptlc),eyn(nptlc),ezn(nptlc),bxn(nptlc),byn(nptlc),bzn(nptlc))
        allocate(exo(nptlc),eyo(nptlc),ezo(nptlc),bxo(nptlc),byo(nptlc),bzo(nptlc))

        !//Check A: image on (before zimage) -> production routine should
        !//match the direct+image reference with the self-image term (no
        !//exclusion of j==gk in the image sum).
        call pt2ptFields(nptlc,targets,myoffset,npttot,sources,.true.,R0,&
                          ex,ey,ez,bx,by,bz)

        !//Check B: image off (after zimage, or Flagimg=0) -> production
        !//routine should match the direct-only reference.
        call pt2ptFields(nptlc,targets,myoffset,npttot,sources,.false.,R0,&
                          exn,eyn,ezn,bxn,byn,bzn)

        diffA = 0.0d0
        diffB = 0.0d0
        diffC = 0.0d0
        do n = 1, nptlc
          gk = myoffset+n
          call ref_fields(gk,npttot,sources,R0,.true.,.false.,&
               refA_ex,refA_ey,refA_ez,refA_bx,refA_by,refA_bz)
          call ref_fields(gk,npttot,sources,R0,.false.,.false.,&
               refB_ex,refB_ey,refB_ez,refB_bx,refB_by,refB_bz)

          diffA = max(diffA, reldiff(ex(n),refA_ex), reldiff(ey(n),refA_ey), &
                              reldiff(ez(n),refA_ez), reldiff(bx(n),refA_bx), &
                              reldiff(by(n),refA_by), reldiff(bz(n),refA_bz))
          diffB = max(diffB, reldiff(exn(n),refB_ex), reldiff(eyn(n),refB_ey), &
                              reldiff(ezn(n),refB_ez), reldiff(bxn(n),refB_bx), &
                              reldiff(byn(n),refB_by), reldiff(bzn(n),refB_bz))
          !//sanity: the image flag must actually change the result.
          diffC = max(diffC, abs(ez(n)-exn(n))+abs(ez(n)-ezn(n)))
        enddo

        if(myid.eq.0) then
          call MPI_REDUCE(MPI_IN_PLACE,diffA,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
          call MPI_REDUCE(MPI_IN_PLACE,diffB,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
          call MPI_REDUCE(MPI_IN_PLACE,diffC,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
        else
          call MPI_REDUCE(diffA,diffA,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
          call MPI_REDUCE(diffB,diffB,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
          call MPI_REDUCE(diffC,diffC,1,MPI_DOUBLE_PRECISION,MPI_MAX,0,&
               MPI_COMM_WORLD,ierr)
        endif

        if(myid.eq.0) then
          print '(A,I2,A,ES10.3)','nproc=',nproc,&
             ' CheckA (image on, matches direct+self-image ref) max rel diff = ',diffA
          print '(A,I2,A,ES10.3)','nproc=',nproc,&
             ' CheckB (image off, matches direct-only ref)      max rel diff = ',diffB
          print '(A,I2,A,ES10.3)','nproc=',nproc,&
             ' CheckC (image term is nonzero, sanity)            max |ez_on-ez_off| = ',diffC
          allok = (diffA.lt.1.0d-10).and.(diffB.lt.1.0d-10).and.(diffC.gt.1.0d-3)
          if(allok) then
            print *,'PASS'
          else
            print *,'FAIL'
          endif
        endif

        !//Check D: analytic self-image of a single particle at rest at
        !//height z: image field should be e/(4 pi eps0 (2z)^2) toward the
        !//cathode. pt2ptFields returns the raw (unscaled by coefE) sum, so
        !//for a single particle at rest (gamma=1, beta=0) that is exactly
        !//-1/(2z)^2 in ez.
        if(myid.eq.0) call check_single_particle_image()

        call MPI_FINALIZE(ierr)

contains

        double precision function reldiff(a,b)
        implicit none
        double precision, intent(in) :: a,b
        reldiff = abs(a-b)/max(1.0d0,abs(b))
        end function reldiff

        !//Independent reference: direct term always excludes j==gk; the
        !//image term excludes j==gk only when exclude_self_image is true
        !//(the pre-A1 behavior). Same pair formulas as pt2ptFields, but
        !//summed in one big loop rather than fused local/remote passes,
        !//so this checks the fused production kernel against an
        !//independently-organized computation of the same physics.
        subroutine ref_fields(gk,npttot,sources,r0,image_on,exclude_self_image,&
                               oex,oey,oez,obx,oby,obz)
        implicit none
        integer, intent(in) :: gk,npttot
        double precision, dimension(8,npttot), intent(in) :: sources
        double precision, intent(in) :: r0
        logical, intent(in) :: image_on,exclude_self_image
        double precision, intent(out) :: oex,oey,oez,obx,oby,obz
        integer :: j
        double precision :: rijx,rijy,rijz,rijx2,rijy2,rijz2,rij2
        double precision :: betajx,betajy,betajz,gammaj,tmp1,tmp2,tmp3
        double precision :: tx,ty,tz

        tx = sources(1,gk)
        ty = sources(2,gk)
        tz = sources(3,gk)
        oex = 0.0d0; oey = 0.0d0; oez = 0.0d0
        obx = 0.0d0; oby = 0.0d0; obz = 0.0d0

        do j = 1, npttot
          if(j.eq.gk) cycle
          gammaj = sources(7,j)
          betajx = sources(4,j)
          betajy = sources(5,j)
          betajz = sources(6,j)
          rijx = tx-sources(1,j)
          rijy = ty-sources(2,j)
          rijz = tz-sources(3,j)
          tmp1 = gammaj**2*(rijx*betajx+rijy*betajy+rijz*betajz)/(gammaj+1)
          rijx2 = rijx+tmp1*betajx
          rijy2 = rijy+tmp1*betajy
          rijz2 = rijz+tmp1*betajz
          rij2 = sqrt(rijx2**2+rijy2**2+rijz2**2)
          tmp2 = gammaj*(betajx*rijx2+betajy*rijy2+betajz*rijz2)/(gammaj+1)
          tmp3 = merge(rij2**3,r0**3,rij2.gt.r0)
          oex = oex+gammaj*(rijx2-tmp2*betajx)/tmp3
          oey = oey+gammaj*(rijy2-tmp2*betajy)/tmp3
          oez = oez+gammaj*(rijz2-tmp2*betajz)/tmp3
          obx = obx+gammaj*(betajy*rijz2-betajz*rijy2)/tmp3
          oby = oby+gammaj*(betajz*rijx2-betajx*rijz2)/tmp3
          obz = obz+gammaj*(betajx*rijy2-betajy*rijx2)/tmp3
        enddo

        if(image_on) then
          do j = 1, npttot
            if(exclude_self_image.and.j.eq.gk) cycle
            gammaj = sources(7,j)
            betajx =  sources(4,j)
            betajy =  sources(5,j)
            betajz = -sources(6,j)
            rijx = tx-sources(1,j)
            rijy = ty-sources(2,j)
            rijz = tz+sources(3,j)
            tmp1 = gammaj**2*(rijx*betajx+rijy*betajy+rijz*betajz)/(gammaj+1)
            rijx2 = rijx+tmp1*betajx
            rijy2 = rijy+tmp1*betajy
            rijz2 = rijz+tmp1*betajz
            rij2 = sqrt(rijx2**2+rijy2**2+rijz2**2)
            tmp2 = gammaj*(betajx*rijx2+betajy*rijy2+betajz*rijz2)/(gammaj+1)
            tmp3 = merge(rij2**3,r0**3,rij2.gt.r0)
            oex = oex-gammaj*(rijx2-tmp2*betajx)/tmp3
            oey = oey-gammaj*(rijy2-tmp2*betajy)/tmp3
            oez = oez-gammaj*(rijz2-tmp2*betajz)/tmp3
            obx = obx-gammaj*(betajy*rijz2-betajz*rijy2)/tmp3
            oby = oby-gammaj*(betajz*rijx2-betajx*rijz2)/tmp3
            obz = obz-gammaj*(betajx*rijy2-betajy*rijx2)/tmp3
          enddo
        endif

        end subroutine ref_fields

        subroutine check_single_particle_image()
        implicit none
        double precision, dimension(4,1) :: t1
        double precision, dimension(8,1) :: s1
        double precision, dimension(1) :: e1x,e1y,e1z,b1x,b1y,b1z
        double precision :: z0,analytic,diff

        z0 = 3.7d-3
        t1(1,1) = 0.0d0
        t1(2,1) = 0.0d0
        t1(3,1) = z0
        t1(4,1) = 1.0d0
        s1(1,1) = 0.0d0
        s1(2,1) = 0.0d0
        s1(3,1) = z0
        s1(4,1) = 0.0d0
        s1(5,1) = 0.0d0
        s1(6,1) = 0.0d0
        s1(7,1) = 1.0d0
        s1(8,1) = 1.0d0

        call pt2ptFields(1,t1,0,1,s1,.true.,R0,e1x,e1y,e1z,b1x,b1y,b1z)

        analytic = -1.0d0/(2.0d0*z0)**2
        diff = abs(e1z(1)-analytic)/abs(analytic)
        print '(A,ES14.6,A,ES14.6,A,ES10.3)','CheckD single-particle self-image: computed=',&
           e1z(1),' analytic=',analytic,' rel diff=',diff
        if(diff.lt.1.0d-10) then
          print *,'PASS'
        else
          print *,'FAIL'
        endif
        end subroutine check_single_particle_image

end program test_pt2ptfields
