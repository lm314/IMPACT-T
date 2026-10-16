!----------------------------------------------------------------
! PoissonGPUclass: open-boundary Poisson solve on one GPU (CUDA Fortran
! + cuFFT), replacing openBC3Dnew in FieldQuantclass when USE_GPU=ON.
! See doc/GPU_PORT_DESIGN.md, B1.1-B1.9.
!
! Every MPI rank gathers the whole rho and solves it redundantly on its
! own GPU, then copies back only its own block.
!----------------------------------------------------------------
      module PoissonGPUclass
        use cudafor
        use cufft
        use PhysConstclass
        implicit none
        private
        public :: openBC3D_gpu, use_gpu_poisson, free_poisson_gpu
        public :: poisson_dev, gradEB_dev, allreduce_rho_gpu

        integer, save :: gnx = 0, gny = 0, gnz = 0
        integer, save :: planR2C = 0, planC2R = 0
        integer, save :: envflag = -1
        real(8), device, allocatable, save :: rd(:,:,:), cor(:,:,:)
        complex(8), device, allocatable, save :: rhohat(:,:,:), ghat(:,:,:)
        real(8), device, allocatable, save :: stage(:,:,:)
        real(8), device, allocatable, save :: blk(:,:,:), blkimg(:,:,:)
        real(8), allocatable, save :: hbuf(:)
        integer, save :: bnx = 0, bny = 0, bnz = 0

      contains

        !> True unless the environment variable IMPACTT_POISSON is "cpu".
        function use_gpu_poisson() result(flag)
        implicit none
        logical :: flag
        character(len=32) :: val
        integer :: stat
        if(envflag.lt.0) then
          envflag = 1
          call get_environment_variable("IMPACTT_POISSON",val,status=stat)
          if(stat.eq.0) then
            if(trim(val).eq."cpu") envflag = 0
          endif
        endif
        flag = (envflag.eq.1)
        end function use_gpu_poisson

        subroutine chkfft(istat,name)
        implicit none
        integer, intent(in) :: istat
        character(len=*), intent(in) :: name
        if(istat.ne.CUFFT_SUCCESS) then
          print*,"PoissonGPU: ",name," failed, cufft status ",istat
          stop
        endif
        end subroutine chkfft

        subroutine chkcuda(name)
        implicit none
        character(len=*), intent(in) :: name
        integer :: istat
        istat = cudaGetLastError()
        if(istat.ne.cudaSuccess) then
          print*,"PoissonGPU: ",name," failed: ",cudaGetErrorString(istat)
          stop
        endif
        end subroutine chkcuda

        subroutine free_poisson_gpu()
        implicit none
        integer :: istat
        if(planR2C.ne.0) then
          istat = cufftDestroy(planR2C)
          istat = cufftDestroy(planC2R)
          planR2C = 0
          planC2R = 0
        endif
        if(allocated(rd)) deallocate(rd)
        if(allocated(cor)) deallocate(cor)
        if(allocated(rhohat)) deallocate(rhohat)
        if(allocated(ghat)) deallocate(ghat)
        if(allocated(stage)) deallocate(stage)
        gnx = 0
        gny = 0
        gnz = 0
        end subroutine free_poisson_gpu

        !> Antiderivative of the Green's function, copied verbatim from
        !> greenf1tIntnew2 (same operations, same order).
        attributes(device) function antider(v1,v2,v3) result(ff0)
        implicit none
        real(8), value :: v1,v2,v3
        real(8) :: ff0
        real(8) :: rr,aa,bb,cc,dd,ee,ff
        rr = sqrt(v1**2+v2**2+v3**2)
        aa = -v3**2*atan(v1*v2/(v3*rr))/2
        bb = -v2**2*atan(v1*v3/(v2*rr))/2
        cc = -v1**2*atan(v2*v3/(v1*rr))/2
        dd = v2*v3*log(v1+rr)
        ee = v1*v3*log(v2+rr)
        ff = v1*v2*log(v3+rr)
        ff0 = aa + bb + cc + dd + ee + ff
        end function antider

        !> Mirror offset of doubled-grid index i (1..2n): 0,1,..,n,n-1,..,1
        attributes(device) function mirror(i,n) result(o)
        implicit none
        integer, value :: i,n
        integer :: o
        o = i - 1
        if(o.gt.n) o = 2*n - o
        end function mirror

        !> 8-corner difference, same sign order as greenf1tIntnew2.
        attributes(device) function dcor(cor,a,b,c,nx,ny) result(d)
        implicit none
        integer, value :: a,b,c,nx,ny
        real(8), device, intent(in) :: cor(0:nx+1,0:ny+1,0:*)
        real(8) :: d
        d = cor(a+1,b+1,c+1)-cor(a,b+1,c+1)-cor(a+1,b,c+1)+ &
            cor(a,b,c+1)-cor(a+1,b+1,c)+cor(a,b+1,c)+ &
            cor(a+1,b,c)-cor(a,b,c)
        end function dcor

        subroutine setup(nx,ny,nz)
        implicit none
        integer, intent(in) :: nx,ny,nz
        integer :: istat
        if(nx.eq.gnx .and. ny.eq.gny .and. nz.eq.gnz) return
        call free_poisson_gpu()
        gnx = nx
        gny = ny
        gnz = nz
        allocate(rd(2*nx,2*ny,2*nz))
        allocate(rhohat(nx+1,2*ny,2*nz))
        allocate(ghat(nx+1,2*ny,2*nz))
        allocate(cor(0:nx+1,0:ny+1,0:2*nz+1))
        allocate(stage(nx,ny,nz))
        ! Fortran (n1,n2,n3), n1 fastest = row-major [n3][n2][n1] to cuFFT.
        istat = cufftPlan3d(planR2C,2*nz,2*ny,2*nx,CUFFT_D2Z)
        call chkfft(istat,"cufftPlan3d(D2Z)")
        istat = cufftPlan3d(planC2R,2*nz,2*ny,2*nx,CUFFT_Z2D)
        call chkfft(istat,"cufftPlan3d(Z2D)")
        end subroutine setup

        !> Poisson solve on device arrays (B1.6 steps 2-9, B2.7).  src_d is the
        !> global (nx,ny,nz) charge density; phi_d and phiimg_d receive the
        !> (onx,ony,onz) block of the potential that starts at (y0,z0) of the
        !> global grid (the image potential, reversed over global z).  With
        !> scale, the extract kernels multiply by hx*hy*hz exactly as
        !> update3Otnew_FieldQuant does.  phiimg_d is untouched if flagImage=0.
        subroutine poisson_dev(nx,ny,nz,src_d,phi_d,phiimg_d,onx,ony,onz,y0,z0, &
                               hx,hy,hz,flagImage,zshift,scale)
        implicit none
        integer, intent(in) :: nx,ny,nz,onx,ony,onz,y0,z0,flagImage
        real(8), device, intent(in) :: src_d(nx,ny,nz)
        real(8), device :: phi_d(onx,ony,onz), phiimg_d(onx,ony,onz)
        double precision, intent(in) :: hx,hy,hz,zshift
        logical, intent(in) :: scale
        integer :: i,j,k,istat,clo
        double precision :: recfourpi,s,hxyz

        call setup(nx,ny,nz)
        recfourpi = 1.0d0/(8.0d0*asin(1.0d0))
        hxyz = hx*hy*hz
        s = 1.0d0/(8.0d0*dble(nx)*dble(ny)*dble(nz))

        ! zero-padded rho -> rd, FFT -> rhohat
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, 2*nz
          do j = 1, 2*ny
            do i = 1, 2*nx
              if(i.le.nx .and. j.le.ny .and. k.le.nz) then
                rd(i,j,k) = src_d(i,j,k)
              else
                rd(i,j,k) = 0.0d0
              endif
            enddo
          enddo
        enddo
        call chkcuda("pad")
        istat = cufftExecD2Z(planR2C,rd,rhohat)
        call chkfft(istat,"cufftExecD2Z(rho)")

        ! direct Green's function
        !$cuf kernel do(3) <<<*,*>>>
        do k = 0, nz+1
          do j = 0, ny+1
            do i = 0, nx+1
              cor(i,j,k) = recfourpi*antider(i*hx-hx/2,j*hy-hy/2,k*hz-hz/2)/hxyz
            enddo
          enddo
        enddo
        call chkcuda("direct corners")
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, 2*nz
          do j = 1, 2*ny
            do i = 1, 2*nx
              rd(i,j,k) = dcor(cor,mirror(i,nx),mirror(j,ny),mirror(k,nz),nx,ny)
            enddo
          enddo
        enddo
        call chkcuda("direct G")
        istat = cufftExecD2Z(planR2C,rd,ghat)
        call chkfft(istat,"cufftExecD2Z(G)")

        ! multiply, inverse FFT, extract the requested block
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, 2*nz
          do j = 1, 2*ny
            do i = 1, nx+1
              ghat(i,j,k) = rhohat(i,j,k)*ghat(i,j,k)*s
            enddo
          enddo
        enddo
        call chkcuda("multiply")
        istat = cufftExecZ2D(planC2R,ghat,rd)
        call chkfft(istat,"cufftExecZ2D")
        if(scale) then
          !$cuf kernel do(3) <<<*,*>>>
          do k = 1, onz
            do j = 1, ony
              do i = 1, onx
                phi_d(i,j,k) = rd(i,y0+j,z0+k)*hx*hy*hz
              enddo
            enddo
          enddo
        else
          !$cuf kernel do(3) <<<*,*>>>
          do k = 1, onz
            do j = 1, ony
              do i = 1, onx
                phi_d(i,j,k) = rd(i,y0+j,z0+k)
              enddo
            enddo
          enddo
        endif
        call chkcuda("extract")

        ! image potential
        if(flagImage.eq.1) then
          !$cuf kernel do(3) <<<*,*>>>
          do k = 0, 2*nz+1
            do j = 0, ny+1
              do i = 0, nx+1
                if(k.le.nz) then
                  cor(i,j,k) = recfourpi*antider(i*hx-hx/2,j*hy-hy/2, &
                               k*hz-hz/2+zshift)/hxyz
                else
                  cor(i,j,k) = recfourpi*antider(i*hx-hx/2,j*hy-hy/2, &
                               zshift-hz*(nz-(k-(nz+1)))-hz/2)/hxyz
                endif
              enddo
            enddo
          enddo
          call chkcuda("image corners")
          !$cuf kernel do(3) <<<*,*>>>
          do k = 1, 2*nz
            do j = 1, 2*ny
              do i = 1, 2*nx
                if(k-1.lt.nz) then
                  clo = k-1
                else
                  clo = k
                endif
                rd(i,j,k) = dcor(cor,mirror(i,nx),mirror(j,ny),clo,nx,ny)
              enddo
            enddo
          enddo
          call chkcuda("image G")
          istat = cufftExecD2Z(planR2C,rd,ghat)
          call chkfft(istat,"cufftExecD2Z(Gimg)")
          !$cuf kernel do(3) <<<*,*>>>
          do k = 1, 2*nz
            do j = 1, 2*ny
              do i = 1, nx+1
                ghat(i,j,k) = -rhohat(i,j,k)*ghat(i,j,k)*s
              enddo
            enddo
          enddo
          call chkcuda("image multiply")
          istat = cufftExecZ2D(planC2R,ghat,rd)
          call chkfft(istat,"cufftExecZ2D(img)")
          if(scale) then
            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, onz
              do j = 1, ony
                do i = 1, onx
                  phiimg_d(i,j,k) = rd(i,y0+j,nz+1-(z0+k))*hx*hy*hz
                enddo
              enddo
            enddo
          else
            !$cuf kernel do(3) <<<*,*>>>
            do k = 1, onz
              do j = 1, ony
                do i = 1, onx
                  phiimg_d(i,j,k) = rd(i,y0+j,nz+1-(z0+k))
                enddo
              enddo
            enddo
          endif
          call chkcuda("image extract")
        endif

        end subroutine poisson_dev

        !> Same arguments as openBC3Dnew's of the same names (the FFT
        !> transpose tables are dropped).  rho is overwritten with the
        !> potential, rhoImg gets the image potential (0 if flagImage=0).
        subroutine openBC3D_gpu(innx,inny,innz,rho,hx,hy,hz,myidz,myidy,npz,npy, &
                                comm2d,pztable,pytable,inxglb,inyglb,inzglb,     &
                                rhoImg,flagImage,zshift)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: innx,inny,innz,inxglb,inyglb,inzglb
        integer, intent(in) :: myidz,myidy,npz,npy,comm2d,flagImage
        integer, dimension(0:npz-1), intent(in) :: pztable
        integer, dimension(0:npy-1), intent(in) :: pytable
        double precision, dimension(innx,inny,innz), intent(inout) :: rho
        double precision, dimension(innx,inny,innz), intent(inout) :: rhoImg
        double precision, intent(in) :: hx,hy,hz,zshift
        integer :: nx,ny,nz,i,j,k,kk,y0,z0,ierr,nprocs,r
        double precision, allocatable :: rhoglb(:,:,:)
        integer, allocatable :: info(:),allinfo(:),counts(:),displs(:)
        double precision, allocatable :: sendbuf(:),recvbuf(:)
        integer :: ofs

        nx = inxglb
        ny = inyglb
        nz = inzglb
        call setup(nx,ny,nz)
        if(bnx.ne.innx .or. bny.ne.inny .or. bnz.ne.innz) then
          if(allocated(blk)) deallocate(blk)
          if(allocated(blkimg)) deallocate(blkimg)
          allocate(blk(innx,inny,innz))
          allocate(blkimg(innx,inny,innz))
          bnx = innx
          bny = inny
          bnz = innz
        endif

        y0 = 0
        do i = 0, myidy-1
          y0 = y0 + pytable(i)
        enddo
        z0 = 0
        do i = 0, myidz-1
          z0 = z0 + pztable(i)
        enddo

        ! 1. gather the global rho (several ranks only), then stage it.
        if(npy*npz.eq.1) then
          stage = rho
        else
          call MPI_COMM_SIZE(comm2d,nprocs,ierr)
          allocate(info(4),allinfo(4*nprocs),counts(nprocs),displs(nprocs))
          info = (/y0,inny,z0,innz/)
          call MPI_ALLGATHER(info,4,MPI_INTEGER,allinfo,4,MPI_INTEGER,comm2d,ierr)
          ofs = 0
          do r = 1, nprocs
            counts(r) = nx*allinfo(4*(r-1)+2)*allinfo(4*(r-1)+4)
            displs(r) = ofs
            ofs = ofs + counts(r)
          enddo
          allocate(recvbuf(ofs),sendbuf(nx*inny*innz))
          sendbuf = reshape(rho,(/nx*inny*innz/))
          call MPI_ALLGATHERV(sendbuf,nx*inny*innz,MPI_DOUBLE_PRECISION,recvbuf, &
                              counts,displs,MPI_DOUBLE_PRECISION,comm2d,ierr)
          allocate(rhoglb(nx,ny,nz))
          do r = 1, nprocs
            if(allinfo(4*(r-1)+1)+allinfo(4*(r-1)+2).gt.ny .or. allinfo(4*(r-1)+3)+allinfo(4*(r-1)+4).gt.nz) then
              print*,"PoissonGPU: gathered block outside the global grid"
              stop
            endif
            kk = displs(r)
            do k = 1, allinfo(4*(r-1)+4)
              do j = 1, allinfo(4*(r-1)+2)
                do i = 1, nx
                  kk = kk + 1
                  rhoglb(i,allinfo(4*(r-1)+1)+j,allinfo(4*(r-1)+3)+k) = recvbuf(kk)
                enddo
              enddo
            enddo
          enddo
          stage = rhoglb
          deallocate(rhoglb,recvbuf,sendbuf,info,allinfo,counts,displs)
        endif

        call poisson_dev(nx,ny,nz,stage,blk,blkimg,innx,inny,innz,y0,z0, &
                         hx,hy,hz,flagImage,zshift,.false.)
        rho = blk
        if(flagImage.eq.1) then
          rhoImg = blkimg
        else
          rhoImg = 0.0d0
        endif

        end subroutine openBC3D_gpu

        !> Sum a device charge-density array over all ranks (B2.4, several
        !> ranks).  Through a host buffer; a GPU-aware MPI or NCCL can replace
        !> it behind this routine.
        subroutine allreduce_rho_gpu(rho_d,n)
        implicit none
        include 'mpif.h'
        integer, intent(in) :: n
        real(8), device :: rho_d(n)
        integer :: ierr
        if(.not.allocated(hbuf)) then
          allocate(hbuf(n))
        else if(size(hbuf).ne.n) then
          deallocate(hbuf)
          allocate(hbuf(n))
        endif
        hbuf = rho_d
        call MPI_ALLREDUCE(MPI_IN_PLACE,hbuf,n,MPI_DOUBLE_PRECISION,MPI_SUM, &
                           MPI_COMM_WORLD,ierr)
        rho_d = hbuf
        end subroutine allreduce_rho_gpu

        !> Add the E and B fields of the potential phi to ex..by (1 rank,
        !> Flagbc=1 branches of gradEB_FieldQuant).  hxi, hyi, hzi and flag
        !> are gradEB_FieldQuant's; flag=1 gives the negative betC.
        subroutine gradEB_dev(nx,ny,nz,phi,hxi,hyi,hzi,gammaz,flag, &
                              ex,ey,ez,bx,by)
        implicit none
        integer, intent(in) :: nx,ny,nz,flag
        real(8), device, intent(in) :: phi(nx,ny,nz)
        real(8), device :: ex(nx,ny,nz),ey(nx,ny,nz),ez(nx,ny,nz)
        real(8), device :: bx(nx,ny,nz),by(nx,ny,nz)
        double precision, intent(in) :: hxi,hyi,hzi,gammaz
        double precision :: betC
        integer :: i,j,k
        double precision :: egx,egy,egz

        if(flag.ne.1) then
          betC = sqrt(gammaz**2-1.0)/gammaz/Clight
        else
          betC = -sqrt(gammaz**2-1.0)/gammaz/Clight
        endif
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, nz
          do j = 1, ny
            do i = 1, nx
              if(i.eq.1) then
                egx = hxi*(1.5d0*phi(1,j,k)-2.0d0*phi(2,j,k)+0.5d0*phi(3,j,k))
              else if(i.eq.nx) then
                egx = hxi*(-0.5d0*phi(nx-2,j,k)+2.0d0*phi(nx-1,j,k)- &
                           1.5d0*phi(nx,j,k))
              else
                egx = 0.5d0*hxi*(phi(i-1,j,k)-phi(i+1,j,k))
              endif
              if(j.eq.1) then
                egy = hyi*(1.5d0*phi(i,1,k)-2*phi(i,2,k)+0.5d0*phi(i,3,k))
              else if(j.eq.ny) then
                egy = hyi*(-0.5d0*phi(i,ny-2,k)+2*phi(i,ny-1,k)- &
                           1.5d0*phi(i,ny,k))
              else
                egy = 0.5d0*hyi*(phi(i,j-1,k)-phi(i,j+1,k))
              endif
              if(k.eq.1) then
                egz = hzi*(1.5d0*phi(i,j,1)-2*phi(i,j,2)+0.5d0*phi(i,j,3))
              else if(k.eq.nz) then
                egz = hzi*(-0.5d0*phi(i,j,nz-2)+2*phi(i,j,nz-1)- &
                           1.5d0*phi(i,j,nz))
              else
                egz = 0.5d0*hzi*(phi(i,j,k-1)-phi(i,j,k+1))
              endif
              ex(i,j,k) = ex(i,j,k) + gammaz*egx
              by(i,j,k) = by(i,j,k) + gammaz*egx*betC
              ey(i,j,k) = ey(i,j,k) + gammaz*egy
              bx(i,j,k) = bx(i,j,k) - gammaz*egy*betC
              ez(i,j,k) = ez(i,j,k) + egz
            enddo
          enddo
        enddo
        call chkcuda("gradEB")
        end subroutine gradEB_dev

      end module PoissonGPUclass
