!----------------------------------------------------------------
! PoissonGPUclass: open-boundary Poisson solve on one GPU (CUDA Fortran
! + cuFFT), replacing openBC3Dnew in FieldQuantclass when USE_GPU=ON.
! See IMPACTT_GPU_PORT_PLAN.md, B1.1-B1.9.
!
! Every MPI rank gathers the whole rho and solves it redundantly on its
! own GPU, then copies back only its own block.
!----------------------------------------------------------------
      module PoissonGPUclass
        use cudafor
        use cufft
        implicit none
        private
        public :: openBC3D_gpu, use_gpu_poisson, free_poisson_gpu

        integer, save :: gnx = 0, gny = 0, gnz = 0
        integer, save :: planR2C = 0, planC2R = 0
        integer, save :: envflag = -1
        real(8), device, allocatable, save :: rd(:,:,:), cor(:,:,:)
        complex(8), device, allocatable, save :: rhohat(:,:,:), ghat(:,:,:)
        real(8), device, allocatable, save :: stage(:,:,:)
        real(8), device, allocatable, save :: blk(:,:,:)
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
        integer :: nx,ny,nz,i,j,k,kk,istat,y0,z0,ierr,nprocs,r,clo
        double precision :: recfourpi,s,hxyz
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
          allocate(blk(innx,inny,innz))
          bnx = innx
          bny = inny
          bnz = innz
        endif

        recfourpi = 1.0d0/(8.0d0*asin(1.0d0))
        hxyz = hx*hy*hz
        s = 1.0d0/(8.0d0*dble(nx)*dble(ny)*dble(nz))

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

        ! 2. zero-padded rho -> rd, FFT -> rhohat
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, 2*nz
          do j = 1, 2*ny
            do i = 1, 2*nx
              if(i.le.nx .and. j.le.ny .and. k.le.nz) then
                rd(i,j,k) = stage(i,j,k)
              else
                rd(i,j,k) = 0.0d0
              endif
            enddo
          enddo
        enddo
        call chkcuda("pad")
        istat = cufftExecD2Z(planR2C,rd,rhohat)
        call chkfft(istat,"cufftExecD2Z(rho)")

        ! 3. direct Green's function
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

        ! 4. multiply, inverse FFT, extract this rank's block
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
        !$cuf kernel do(3) <<<*,*>>>
        do k = 1, innz
          do j = 1, inny
            do i = 1, innx
              blk(i,j,k) = rd(i,y0+j,z0+k)
            enddo
          enddo
        enddo
        call chkcuda("extract")
        rho = blk

        ! 5. image potential
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
          !$cuf kernel do(3) <<<*,*>>>
          do k = 1, innz
            do j = 1, inny
              do i = 1, innx
                blk(i,j,k) = rd(i,y0+j,nz+1-(z0+k))
              enddo
            enddo
          enddo
          call chkcuda("image extract")
          rhoImg = blk
        else
          rhoImg = 0.0d0
        endif

        end subroutine openBC3D_gpu

      end module PoissonGPUclass
