! The blocked CPU pair loop of pt2ptFields (Appl/BeamBunch.f90, pt2pt_nblk sources per block) against a
! copy of the unblocked loop: the fields must be bit-identical for every block size, with the image on and
! off, a rank offset (targets nptlc < npttot, myoffset > 0, so the self-skip lands inside a block) and
! sources that are not emitted (flag 0). One rank; built without OpenACC (a GPU build keeps the unblocked
! loop, so there the test only compares it with itself).
! Not part of the CMake build; see tests/README.md for how to build and run.
program test_pt2pt_block
        use BeamBunchclass, only: pt2ptFields, pt2pt_nblk
        implicit none
        include 'mpif.h'

        integer, parameter :: NTOT = 3001, NLOC = 1200, OFFSET = 900
        integer, parameter :: NB = 7
        integer, dimension(NB), parameter :: BLOCKS = (/1, 7, 1000, 4096, 3000, 3001, 100000/)
        double precision, parameter :: R0 = 1.0d-9

        integer :: ierr, n, ib, im, nbad, nbadall
        double precision, allocatable :: sources(:,:), targets(:,:)
        double precision, allocatable :: f(:,:), g(:,:)
        double precision :: gpx, gpy, gpz, gam
        logical :: image

        call MPI_INIT(ierr)
        allocate(sources(10,NTOT),targets(4,NLOC),f(NLOC,6),g(NLOC,6))
        do n = 1, NTOT
          gpx = 0.30d0*sin(0.881d0*n+1.20d0)
          gpy = 0.30d0*cos(0.727d0*n+0.90d0)
          gpz = 0.50d0+0.40d0*sin(0.611d0*n+0.30d0)
          gam = sqrt(1.0d0+gpx**2+gpy**2+gpz**2)
          sources(1,n) = 5.0d-5*sin(0.371d0*n+0.13d0)
          sources(2,n) = 5.0d-5*cos(0.457d0*n+0.71d0)
          sources(3,n) = 5.0d-3+4.5d-3*sin(0.231d0*n+0.50d0)
          !//two particles closer than r0, so the cutoff branch is taken too
          if(n.eq.2) sources(1:3,n) = sources(1:3,1) + 1.0d-10
          sources(4,n) = gpx/gam
          sources(5,n) = gpy/gam
          sources(6,n) = gpz/gam
          sources(7,n) = gam
          sources(8,n) = 1.0d0
          if(mod(n,97).eq.0) sources(8,n) = 0.0d0
          sources(9,n) = gam*gam/(gam+1.0d0)
          sources(10,n) = gam/(gam+1.0d0)
        enddo
        do n = 1, NLOC
          targets(1:3,n) = sources(1:3,OFFSET+n)
          targets(4,n) = sources(8,OFFSET+n)
        enddo

        nbadall = 0
        do im = 0, 1
          image = im.eq.1
          call reference(NLOC,targets,OFFSET,NTOT,sources,image,R0,g)
          do ib = 1, NB
            pt2pt_nblk = BLOCKS(ib)
            call pt2ptFields(NLOC,targets,OFFSET,NTOT,sources,image,R0,R0, &
                             f(:,1),f(:,2),f(:,3),f(:,4),f(:,5),f(:,6))
            nbad = count(f.ne.g)
            nbadall = nbadall + nbad
            print '(A,L2,A,I7,A,I8,A,ES10.3)', 'image', image, '  block', BLOCKS(ib), &
                  '  values differing', nbad, '  max |rel diff|', maxval(abs(f-g)/max(abs(g),tiny(1.0d0)))
          enddo
        enddo
        if(nbadall.eq.0) then
          print*, 'PASS: blocked fields bit-identical to the unblocked loop for every block size'
        else
          print*, 'FAIL: ', nbadall, ' field values differ'
        endif
        call MPI_FINALIZE(ierr)

      contains

        !> the unblocked loop of pt2ptFields as it was before blocking (one running sum per target)
        subroutine reference(nptlc,targets,myoffset,npttot,sources,image,r0,f)
        integer, intent(in) :: nptlc,npttot,myoffset
        double precision, dimension(4,nptlc), intent(in) :: targets
        double precision, dimension(10,npttot), intent(in) :: sources
        logical, intent(in) :: image
        double precision, intent(in) :: r0
        double precision, dimension(nptlc,6), intent(out) :: f
        integer :: n,j,gidx
        double precision :: rijx,rijy,rijz,rijx2,rijy2,rijz2,rij2sq,rinv,rinv3
        double precision :: betajx,betajy,betajz,gammaj,c1j,c2j,tmp1,tmp2
        double precision :: sumex,sumey,sumez,sumbx,sumby,sumbz
        double precision :: tx,ty,tz,r0sq,r0inv

        r0sq = r0*r0
        r0inv = 1.0d0/r0
        do n = 1, nptlc
          sumex = 0.0d0
          sumey = 0.0d0
          sumez = 0.0d0
          sumbx = 0.0d0
          sumby = 0.0d0
          sumbz = 0.0d0
          if(targets(4,n).gt.0.0d0) then
            gidx = myoffset + n
            tx = targets(1,n)
            ty = targets(2,n)
            tz = targets(3,n)
            do j = 1, npttot
              if(sources(8,j).gt.0.0d0) then
                gammaj = sources(7,j)
                c1j = sources(9,j)
                c2j = sources(10,j)
                if(j.ne.gidx) then
                  betajx = sources(4,j)
                  betajy = sources(5,j)
                  betajz = sources(6,j)
                  rijx = tx-sources(1,j)
                  rijy = ty-sources(2,j)
                  rijz = tz-sources(3,j)
                  tmp1 = c1j*(rijx*betajx+rijy*betajy+rijz*betajz)
                  rijx2 = rijx + tmp1*betajx
                  rijy2 = rijy + tmp1*betajy
                  rijz2 = rijz + tmp1*betajz
                  rij2sq = rijx2**2+rijy2**2+rijz2**2
                  tmp2 = c2j*(betajx*rijx2+betajy*rijy2+betajz*rijz2)
                  if(rij2sq.gt.r0sq) then
                    rinv = 1.0d0/sqrt(rij2sq)
                  else
                    rinv = r0inv
                  endif
                  rinv3 = rinv*rinv*rinv
                  sumex = sumex + gammaj*(rijx2-tmp2*betajx)*rinv3
                  sumey = sumey + gammaj*(rijy2-tmp2*betajy)*rinv3
                  sumez = sumez + gammaj*(rijz2-tmp2*betajz)*rinv3
                  sumbx = sumbx + gammaj*(betajy*rijz2-betajz*rijy2)*rinv3
                  sumby = sumby + gammaj*(betajz*rijx2-betajx*rijz2)*rinv3
                  sumbz = sumbz + gammaj*(betajx*rijy2-betajy*rijx2)*rinv3
                endif
                if(image) then
                  betajx =  sources(4,j)
                  betajy =  sources(5,j)
                  betajz = -sources(6,j)
                  rijx = tx-sources(1,j)
                  rijy = ty-sources(2,j)
                  rijz = tz+sources(3,j)
                  tmp1 = c1j*(rijx*betajx+rijy*betajy+rijz*betajz)
                  rijx2 = rijx + tmp1*betajx
                  rijy2 = rijy + tmp1*betajy
                  rijz2 = rijz + tmp1*betajz
                  rij2sq = rijx2**2+rijy2**2+rijz2**2
                  tmp2 = c2j*(betajx*rijx2+betajy*rijy2+betajz*rijz2)
                  if(rij2sq.gt.r0sq) then
                    rinv = 1.0d0/sqrt(rij2sq)
                  else
                    rinv = r0inv
                  endif
                  rinv3 = rinv*rinv*rinv
                  sumex = sumex - gammaj*(rijx2-tmp2*betajx)*rinv3
                  sumey = sumey - gammaj*(rijy2-tmp2*betajy)*rinv3
                  sumez = sumez - gammaj*(rijz2-tmp2*betajz)*rinv3
                  sumbx = sumbx - gammaj*(betajy*rijz2-betajz*rijy2)*rinv3
                  sumby = sumby - gammaj*(betajz*rijx2-betajx*rijz2)*rinv3
                  sumbz = sumbz - gammaj*(betajx*rijy2-betajy*rijx2)*rinv3
                endif
              endif
            enddo
          endif
          f(n,1) = sumex
          f(n,2) = sumey
          f(n,3) = sumez
          f(n,4) = sumbx
          f(n,5) = sumby
          f(n,6) = sumbz
        enddo
        end subroutine reference

end program test_pt2pt_block
