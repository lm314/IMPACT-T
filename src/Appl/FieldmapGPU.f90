!----------------------------------------------------------------
! FieldmapGPUclass: external-field lookup on the GPU for beam line
! elements of type 105 (SolRF with a discrete on-axis file, Param(5)>1000),
! 112 (EMfldCyl), and (D1.2) type 1 (Quadrupole), 111 (EMfldCart) and
! type 4 (Dipole, in ordinary steps).  See doc/GPU_PORT_DESIGN.md, B2.6
! and D1.2.
!
! The CPU dispatch (getfldt_BeamLineElem) is replaced by flat tables:
! device copies of the field data slots (fldmp(1..Maxoverlap)) and a small
! per-step table of the elements the beam currently overlaps.  The device
! routines are verbatim ports of getfldt_SolRF and getfldt_EMfldCyl; the
! host builds every per-step constant (cos, sin, ww, ...) with the CPU
! routines' own expressions, single-precision literals included.
!----------------------------------------------------------------
      module FieldmapGPUclass
        use cudafor
        use Dataclass
        use NumConstclass
        use BeamLineElemclass
        implicit none
        private
        public :: Maxact, fldslot_upload, build_active_table, extfld_dev
        public :: gpuerr_d, gpuerr_info_d, fldmap_init, ntab_dev
        public :: fld_dipole, cartout_count

        integer, parameter :: Maxact = 64

        ! slot data
        real(8), device, allocatable, save :: fcoef_d(:,:,:)  !(4,10002,Maxoverlap)
        real(8), device, allocatable, save :: ezp_d(:), erp_d(:), btp_d(:)
        integer, save :: stride = 0       ! packed elements per slot (kind 2)
        integer, device, save :: stride_d
        integer, save :: slot_which(Maxoverlap) = 0
        integer, save :: slot_nz(Maxoverlap) = 0, slot_nr(Maxoverlap) = 0

        ! dipole slots (D1.2): Fcoeft(1:22), then sin/cos of the entrance and
        ! exit face angles (computed on the host)
        real(8), device, save :: dipc_d(26,Maxoverlap)
        ! 111 maps (D1.2): the six complex arrays of every 111 slot, one after
        ! the other, cart_off(s) elements in
        complex(8), device, allocatable, save :: cex_d(:), cey_d(:), cez_d(:), &
                                                 cbx_d(:), cby_d(:), cbz_d(:)
        integer, save :: cart_off(Maxoverlap) = 0
        ! points outside a 111 map: count, and the first one's particle index
        ! and position
        integer, device, save :: cartout_d
        integer, device, save :: cartfirst_d
        integer, device, save :: cartipart_d
        real(8), device, save :: cartpos_d(3)

        ! per-step active-element table (device).
        ! ti: 1 kind (1 105, 2 112, 3 quadrupole, 4 type 111, 5 dipole), 2 slot,
        !   kind 2: 3 NzIntv, 4 NrIntv;  kind 3: 3 mode, 4 RF flag;
        !   kind 4: 3,4,5 Nx+1, Ny+1, Nz+1 and 6 offset in the packed arrays
        ! tr: 1 zlo, 2 zhi, 3 zedge, 4 escale, 5 len,
        ! 6 bscale, 7 xlrep, 8 tmpcos, 9 tmpsin, 10 hz, 11 hr, 12 zspan
        !   kind 3: 3 zedge, 4 gradient, 5 length, 6 Param(3), 7 Param(4),
        !           8 cos(anglez), 9 sin(anglez), 10 cos(phs)
        !   kind 4: 3 Param(1), 4 length, 5,6 tmptime re/im, 7-9 Xmin Ymin Zmin,
        !           10-12 hx hy hz
        !   kind 5: 3 Param(3) (the field), 4 Param(5) (half gap)
        integer, device, save :: ti_d(6,Maxact)
        real(8), device, save :: tr_d(16,Maxact)
        integer, save :: ntab_dev = 0

        ! error flag: code (1 deposit bounds, 2 kick bounds, 3 ir of 112),
        ! then particle index and offending index
        integer, device, save :: gpuerr_d
        integer, device, save :: gpuerr_info_d(4)

      contains

        subroutine fldmap_init()
        implicit none
        integer :: zero
        integer :: zi(4)
        if(.not.allocated(fcoef_d)) then
          allocate(fcoef_d(4,10002,Maxoverlap))
          zero = 0
          cartout_d = zero
          cartfirst_d = zero
          cartipart_d = zero
          cartpos_d = 0.0d0
        endif
        zero = 0
        zi = 0
        gpuerr_d = zero
        gpuerr_info_d = zi
        end subroutine fldmap_init

        !> Number of particles that were outside a 111 map so far (D1.2).
        integer function cartout_count(ipart,pos)
        implicit none
        integer, intent(out) :: ipart
        real(8), intent(out) :: pos(3)
        integer :: n
        n = cartout_d
        ipart = cartipart_d
        pos = cartpos_d
        cartout_count = n
        end function cartout_count

        !> Copy field data slot s to the device after the host has (re)read
        !> it.  which = 1: read1tdata_Data (105 with a file id > 1000, or a
        !> type-1 quadrupole with a file id >= 100);
        !> which = 2: read2t_Data (112).  fldmp is the whole slot array,
        !> needed to re-pack the other 112 slots if the packed stride grows.
        !> which = 3: read1t_Data of a type-4 dipole (Fcoeft);
        !> which = 4: read3t_Data (111), all 111 slots are re-packed.
        subroutine fldslot_upload(fldmp,s,which)
        implicit none
        type (fielddata), dimension(:), intent(in) :: fldmp
        integer, intent(in) :: s,which
        integer :: nz,nr,need,k
        real(8) :: dtmp(26),k1

        call fldmap_init()
        if(which.eq.1) then
          fcoef_d(:,:,s) = fldmp(s)%Fcoeftdata
          slot_which(s) = 1
        else if(which.eq.2) then
          nz = fldmp(s)%NzIntvRft+1
          nr = fldmp(s)%NrIntvRft+1
          need = nz*nr
          if(need.gt.stride) then
            if(allocated(ezp_d)) deallocate(ezp_d,erp_d,btp_d)
            stride = need
            stride_d = stride
            allocate(ezp_d(Maxoverlap*stride),erp_d(Maxoverlap*stride), &
                     btp_d(Maxoverlap*stride))
            do k = 1, Maxoverlap
              if(k.ne.s .and. slot_which(k).eq.2) call pack_slot(fldmp,k)
            enddo
          endif
          slot_which(s) = 2
          slot_nz(s) = fldmp(s)%NzIntvRft
          slot_nr(s) = fldmp(s)%NrIntvRft
          call pack_slot(fldmp,s)
        else if(which.eq.3) then
          dtmp = 0.0d0
          dtmp(1:22) = fldmp(s)%Fcoeft(1:22)
          k1 = fldmp(s)%Fcoeft(3)
          dtmp(23) = sin(atan(k1))
          dtmp(24) = cos(atan(k1))
          k1 = fldmp(s)%Fcoeft(7)
          dtmp(25) = sin(atan(k1))
          dtmp(26) = cos(atan(k1))
          dipc_d(:,s) = dtmp
          slot_which(s) = 3
        else if(which.eq.4) then
          slot_which(s) = 4
          call pack_cart(fldmp)
        endif
        end subroutine fldslot_upload

        !> Pack the six complex arrays of every 111 slot into six device
        !> arrays, one slot after the other (Fortran order, so that the host
        !> index (ix,jx,kx) of a slot is off+ix+nx1*((jx-1)+ny1*(kx-1))).
        subroutine pack_cart(fldmp)
        implicit none
        type (fielddata), dimension(:), intent(in) :: fldmp
        integer :: k,n,ntot
        ntot = 0
        do k = 1, Maxoverlap
          if(slot_which(k).eq.4) then
            if(associated(fldmp(k)%Exgridt)) then
              cart_off(k) = ntot
              ntot = ntot + size(fldmp(k)%Exgridt)
            endif
          endif
        enddo
        if(allocated(cex_d)) deallocate(cex_d,cey_d,cez_d,cbx_d,cby_d,cbz_d)
        allocate(cex_d(ntot),cey_d(ntot),cez_d(ntot),cbx_d(ntot), &
                 cby_d(ntot),cbz_d(ntot))
        do k = 1, Maxoverlap
          if(slot_which(k).eq.4) then
            if(associated(fldmp(k)%Exgridt)) then
              n = size(fldmp(k)%Exgridt)
              cex_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Exgridt,(/n/))
              cey_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Eygridt,(/n/))
              cez_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Ezgridt,(/n/))
              cbx_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Bxgridt,(/n/))
              cby_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Bygridt,(/n/))
              cbz_d(cart_off(k)+1:cart_off(k)+n) = reshape(fldmp(k)%Bzgridt,(/n/))
            endif
          endif
        enddo
        end subroutine pack_cart

        subroutine pack_slot(fldmp,s)
        implicit none
        type (fielddata), dimension(:), intent(in) :: fldmp
        integer, intent(in) :: s
        integer :: nz,nr,off
        if(.not.associated(fldmp(s)%ezdatat)) return
        nz = fldmp(s)%NzIntvRft+1
        nr = fldmp(s)%NrIntvRft+1
        if(size(fldmp(s)%ezdatat,1).ne.nz .or. size(fldmp(s)%ezdatat,2).ne.nr) return
        off = (s-1)*stride
        ezp_d(off+1:off+nz*nr) = reshape(fldmp(s)%ezdatat,(/nz*nr/))
        erp_d(off+1:off+nz*nr) = reshape(fldmp(s)%erdatat,(/nz*nr/))
        btp_d(off+1:off+nz*nr) = reshape(fldmp(s)%btdatat,(/nz*nr/))
        end subroutine pack_slot

        !> Build the active-element table for elements ibinit..ibend at
        !> time tt (B2.6) with the CPU routines' own expressions.  ok is
        !> false if any element is of a kind that has no GPU port (or there
        !> are too many); then why names the first one.
        subroutine build_active_table(beamelem,zbeamelem,idrfile,fldmp,ibinit, &
                                      ibend,tt,ok,why)
        implicit none
        type (BeamLineElem), dimension(:), intent(in) :: beamelem
        double precision, dimension(:,:), intent(in) :: zbeamelem
        integer, dimension(:,:), intent(in) :: idrfile
        type (fielddata), dimension(:), intent(in) :: fldmp
        integer, intent(in) :: ibinit,ibend
        double precision, intent(in) :: tt
        logical, intent(out) :: ok
        character(len=*), intent(out) :: why
        integer :: ii,n,itype,slot
        integer :: hti(6,Maxact)
        double precision :: htr(16,Maxact)
        double precision :: blength,p1,p2,p3,p4,p5,p12,p9,p10,p11
        integer :: bnseg,bmpstp,bitype
        ! single-precision literals, as in getfldt_SolRF/getfldt_EMfldCyl
        double precision :: clite,pi,ww,theta0,xlrep
        double precision :: twopi,phs,eps
        double precision :: escale,tre,tim
        double complex :: tmptime

        ok = .true.
        why = ""
        n = 0
        hti = 0
        htr = 0.0d0
        do ii = ibinit, ibend
          itype = idrfile(1,ii)
          if(itype.lt.0 .or. itype.eq.0) cycle
          call getparam_BeamLineElem(beamelem(ii),blength,bnseg,bmpstp,bitype)
          if(itype.eq.105) then
            call getparam_BeamLineElem(beamelem(ii),5,p5)
            if(p5.le.1000.0d0) then
              ok = .false.
              why = "type 105 element with a Fourier field file"
              return
            endif
          else if(itype.ne.112 .and. itype.ne.1 .and. itype.ne.111 .and. &
                  itype.ne.4) then
            ok = .false.
            write(why,'(a,i5)') "unported beam line element type ",itype
            return
          endif
          n = n + 1
          if(n.gt.Maxact) then
            ok = .false.
            why = "too many overlapping beam line elements"
            return
          endif
          slot = idrfile(3,ii)
          call getparam_BeamLineElem(beamelem(ii),1,p1)
          call getparam_BeamLineElem(beamelem(ii),2,p2)
          call getparam_BeamLineElem(beamelem(ii),3,p3)
          call getparam_BeamLineElem(beamelem(ii),4,p4)
          htr(1,n) = zbeamelem(1,ii)
          htr(2,n) = zbeamelem(2,ii)
          hti(2,n) = slot
          if(itype.eq.1) then
            ! getfld_Quadrupole
            hti(1,n) = 3
            call getparam_BeamLineElem(beamelem(ii),9,p9)
            call getparam_BeamLineElem(beamelem(ii),10,p10)
            call getparam_BeamLineElem(beamelem(ii),11,p11)
            if(p3.gt.0.0d0 .and. p3.lt.100.0d0) then
              hti(3,n) = 2
            else if(p3.ge.100.0d0) then
              hti(3,n) = 1
              if(slot_which(slot).ne.1) then
                ok = .false.
                why = "type 1 element with a file profile that is not on the device"
                return
              endif
            else
              hti(3,n) = 3
            endif
            htr(3,n) = p1
            htr(4,n) = p2
            htr(5,n) = blength
            htr(6,n) = p3
            htr(7,n) = p4
            htr(8,n) = cos(p9)
            htr(9,n) = sin(p9)
            eps = 1.0d-5
            twopi = 4*asin(1.0d0)
            htr(10,n) = 1.0d0
            if(p10.gt.eps) then
              hti(4,n) = 1
              phs = twopi*p10*tt+p11/360.0d0*twopi
              htr(10,n) = cos(phs)
            endif
          else if(itype.eq.111) then
            ! getfldt_EMfldCart
            if(slot_which(slot).ne.4 .or. .not.associated(fldmp(slot)%Exgridt)) then
              ok = .false.
              why = "type 111 map that is not on the device"
              return
            endif
            hti(1,n) = 4
            hti(3,n) = fldmp(slot)%NxIntvRfgt+1
            hti(4,n) = fldmp(slot)%NyIntvRfgt+1
            hti(5,n) = fldmp(slot)%NzIntvRfgt+1
            hti(6,n) = cart_off(slot)
            htr(3,n) = p1
            htr(4,n) = blength
            escale = p2
            ww = p3*4*asin(1.0)
            theta0 = p4*asin(1.0)/90
            tmptime = dcmplx(cos(ww*tt+theta0),sin(ww*tt+theta0))*escale
            tre = dreal(tmptime)
            tim = dimag(tmptime)
            htr(5,n) = tre
            htr(6,n) = tim
            htr(7,n) = fldmp(slot)%XminRfgt
            htr(8,n) = fldmp(slot)%YminRfgt
            htr(9,n) = fldmp(slot)%ZminRfgt
            htr(10,n) = (fldmp(slot)%XmaxRfgt-fldmp(slot)%XminRfgt)/fldmp(slot)%NxIntvRfgt
            htr(11,n) = (fldmp(slot)%YmaxRfgt-fldmp(slot)%YminRfgt)/fldmp(slot)%NyIntvRfgt
            htr(12,n) = (fldmp(slot)%ZmaxRfgt-fldmp(slot)%ZminRfgt)/fldmp(slot)%NzIntvRfgt
          else if(itype.eq.4) then
            ! getfldt_Dipole in an ordinary step (D1.2, kind 5)
            hti(1,n) = 5
            if(slot_which(slot).ne.3) then
              ok = .false.
              why = "type 4 element whose data are not on the device"
              return
            endif
            call getparam_BeamLineElem(beamelem(ii),5,p5)
            htr(3,n) = p3
            htr(4,n) = p5
          else if(itype.eq.105) then
            hti(1,n) = 1
            htr(3,n) = p1
            htr(4,n) = p2
            htr(5,n) = blength
            call getparam_BeamLineElem(beamelem(ii),12,p12)
            htr(6,n) = p12
            clite = 299792458.e0
            pi = 2*asin(1.0)
            ww = p3*2*pi
            xlrep = ww/clite
            theta0 = p4*asin(1.0)/90
            htr(7,n) = xlrep
            htr(8,n) = cos(ww*tt+theta0)
            htr(9,n) = sin(ww*tt+theta0)
          else
            hti(1,n) = 2
            htr(3,n) = p1
            htr(4,n) = p2
            htr(5,n) = blength
            hti(3,n) = fldmp(slot)%NzIntvRft
            hti(4,n) = fldmp(slot)%NrIntvRft
            ww = p3*4*asin(1.0)
            theta0 = p4*asin(1.0)/90
            htr(8,n) = cos(ww*tt+theta0)*p2
            htr(9,n) = -sin(ww*tt+theta0)*p2
            htr(10,n) = (fldmp(slot)%ZmaxRft-fldmp(slot)%ZminRft)/fldmp(slot)%NzIntvRft
            htr(11,n) = (fldmp(slot)%RmaxRft-fldmp(slot)%RminRft)/fldmp(slot)%NrIntvRft
            htr(12,n) = fldmp(slot)%ZmaxRft-fldmp(slot)%ZminRft
          endif
        enddo
        ntab_dev = n
        if(n.gt.0) then
          ti_d(:,1:n) = hti(:,1:n)
          tr_d(:,1:n) = htr(:,1:n)
        endif
        end subroutine build_active_table

        !> External field at (x,y,z) in meters, summed over the active
        !> elements that contain z, in table order (getfldt_BeamLineElem
        !> summed over the overlapping elements, as kick2t_BeamBunch does).
        attributes(device) subroutine extfld_dev(x,y,zz,nact,ef,ipart)
        implicit none
        real(8), value :: x,y,zz
        integer, value :: nact,ipart
        real(8), intent(out) :: ef(6)
        real(8) :: tf(6)
        integer :: m,kd
        ef(1) = 0.0d0
        ef(2) = 0.0d0
        ef(3) = 0.0d0
        ef(4) = 0.0d0
        ef(5) = 0.0d0
        ef(6) = 0.0d0
        do m = 1, nact
          if( (zz.ge.tr_d(1,m)) .and. (zz.le.tr_d(2,m)) ) then
            kd = ti_d(1,m)
            if(kd.eq.1) then
              call fld_solrf(x,y,zz,m,tf)
            else if(kd.eq.2) then
              call fld_emcyl(x,y,zz,m,tf,ipart)
            else if(kd.eq.3) then
              call fld_quad(x,y,zz,m,tf)
            else if(kd.eq.4) then
              call fld_emcart(x,y,zz,m,tf,ipart)
            else
              call fld_dipole(x,y,zz,ti_d(2,m),tr_d(3,m),tr_d(4,m),tf)
            endif
            ef(1) = ef(1) + tf(1)
            ef(2) = ef(2) + tf(2)
            ef(3) = ef(3) + tf(3)
            ef(4) = ef(4) + tf(4)
            ef(5) = ef(5) + tf(5)
            ef(6) = ef(6) + tf(6)
          endif
        enddo
        end subroutine extfld_dev

        !> getfldt_SolRF, from the in-element test on.  Per-step constants
        !> (xlrep, tmpcos, tmpsin) come from the table.
        attributes(device) subroutine fld_solrf(px,py,pz,m,extfld)
        implicit none
        real(8), value :: px,py,pz
        integer, value :: m
        real(8), intent(out) :: extfld(6)
        real(8) :: zedge,escale,len,xlrep,tmpcos,tmpsin,bscale
        real(8) :: ez1,ezp1,ezpp1,ezppp,f1,f1p,r2
        real(8) :: zstart1,zend1,zlength1,zstart2,zend2,zlc,eps,hz,aa
        real(8) :: bz1,bzp1,bzpp1,bzppp
        real(8) :: tmpex,tmpey,tmpez,tmpbx,tmpby,tmpbz
        real(8) :: clite
        integer :: numpar1,numpar2,i0,iz,iz1,nzl,slot

        clite = 299792458.e0
        zedge = tr_d(3,m)
        escale = tr_d(4,m)
        len = tr_d(5,m)
        bscale = tr_d(6,m)
        xlrep = tr_d(7,m)
        tmpcos = tr_d(8,m)
        tmpsin = tr_d(9,m)
        slot = ti_d(2,m)

        extfld(1) = 0.0d0
        extfld(2) = 0.0d0
        extfld(3) = 0.0d0
        extfld(4) = 0.0d0
        extfld(5) = 0.0d0
        extfld(6) = 0.0d0
        ez1 = 0.0
        ezp1 = 0.0
        ezpp1 = 0.0
        ezppp = 0.0
        if((pz.ge.zedge).and.(pz.le.(zedge+len))) then
          zlc = pz-zedge
          numpar1 = fcoef_d(1,1,slot)+0.1
          zstart1 = fcoef_d(2,1,slot)
          zend1 = fcoef_d(3,1,slot)
          zlength1 = zend1-zstart1
          if(zlength1.gt.1.0d-8) then
            eps = 1.0e-15*zlength1
            nzl = (zlc-zstart1-eps)/zlength1
            zlc = zlc - nzl*zlength1
          endif
          if(numpar1.gt.1) then
            hz = zlength1/(numpar1-1)
          else
            hz = 1.0
          endif

          if( (zlc.ge.zstart1).and.(zlc.le.zend1)) then
            i0 = 1
            iz = (zlc-zstart1)/hz + 1
            if(iz.lt.0) iz = 1
            iz1 = iz + 1
            if(iz1.gt.numpar1) iz1 = numpar1
            aa = (iz*hz-(zlc-zstart1))/hz
            ez1 = fcoef_d(1,iz+i0,slot)*aa+fcoef_d(1,iz1+i0,slot)*(1.0d0-aa)
            ezp1 = fcoef_d(2,iz+i0,slot)*aa+fcoef_d(2,iz1+i0,slot)*(1.0d0-aa)
            ezpp1 = fcoef_d(3,iz+i0,slot)*aa+fcoef_d(3,iz1+i0,slot)*(1.0d0-aa)
            ezppp = fcoef_d(4,iz+i0,slot)*aa+fcoef_d(4,iz1+i0,slot)*(1.0d0-aa)
            ez1=ez1*escale
            ezp1=ezp1*escale
            ezpp1=ezpp1*escale
            ezppp = ezppp*escale
            f1 = -(ezpp1+ez1*xlrep*xlrep)/4
            f1p = -(ezppp+ezp1*xlrep*xlrep)/4
            r2 = px**2+py**2
            tmpex = -px*(ezp1/2+f1p*r2/4)*tmpcos
            tmpey = -py*(ezp1/2+f1p*r2/4)*tmpcos
            tmpez = (ez1+f1*r2)*tmpcos
            tmpbx = py*xlrep/clite*(ez1/2+f1*r2/4)*tmpsin
            tmpby = -px*xlrep/clite*(ez1/2+f1*r2/4)*tmpsin
            tmpbz = 0.0
            extfld(1) = extfld(1) + tmpex
            extfld(2) = extfld(2) + tmpey
            extfld(3) = extfld(3) + tmpez
            extfld(4) = extfld(4) + tmpbx
            extfld(5) = extfld(5) + tmpby
            extfld(6) = extfld(6) + tmpbz
          else
            extfld(1) = 0.0d0
            extfld(2) = 0.0d0
            extfld(3) = 0.0d0
            extfld(4) = 0.0d0
            extfld(5) = 0.0d0
            extfld(6) = 0.0d0
          endif

          i0 = numpar1 + 1
          numpar2 = fcoef_d(1,i0+1,slot) + 0.1
          zstart2 = fcoef_d(2,i0+1,slot)
          zend2 = fcoef_d(3,i0+1,slot)
          if(numpar2.gt.1) then
            hz = (zend2-zstart2)/(numpar2-1)
          else
            hz = 1.0
          endif

          bz1 = 0.0d0
          bzp1 = 0.0d0
          bzpp1 = 0.0d0
          bzppp = 0.0d0
          i0 = numpar1+2
          if( (zlc.ge.zstart2) .and. (zlc.le.zend2) ) then
            iz = (zlc-zstart2)/hz + 1
            if(iz.lt.0) iz = 1
            iz1 = iz + 1
            if(iz1.gt.numpar2) iz1 = numpar2
            aa = (iz*hz-(zlc-zstart2))/hz
            bz1 = fcoef_d(1,iz+i0,slot)*aa+fcoef_d(1,iz1+i0,slot)*(1.0d0-aa)
            bzp1 = fcoef_d(2,iz+i0,slot)*aa+fcoef_d(2,iz1+i0,slot)*(1.0d0-aa)
            bzpp1 = fcoef_d(3,iz+i0,slot)*aa+fcoef_d(3,iz1+i0,slot)*(1.0d0-aa)
            bzppp = fcoef_d(4,iz+i0,slot)*aa+fcoef_d(4,iz1+i0,slot)*(1.0d0-aa)
            r2 = px*px+py*py
            extfld(4) = extfld(4)-bscale*bzp1/2*px+bscale*bzppp*px*r2/16
            extfld(5) = extfld(5)-bscale*bzp1/2*py+bscale*bzppp*py*r2/16
            extfld(6) = extfld(6)+bscale*bz1-bscale*bzpp1*r2/4
          endif
        else
          extfld(1) = 0.0d0
          extfld(2) = 0.0d0
          extfld(3) = 0.0d0
          extfld(4) = 0.0d0
          extfld(5) = 0.0d0
          extfld(6) = 0.0d0
        endif
        end subroutine fld_solrf

        !> getfldt_EMfldCyl, from the in-element test on.
        attributes(device) subroutine fld_emcyl(px,py,pz,m,extfld,ipart)
        implicit none
        real(8), value :: px,py,pz
        integer, value :: m,ipart
        real(8), intent(out) :: extfld(6)
        real(8) :: zedge,len,tmpcos,tmpsin,hz,hr,zspan,zz,efz,efr,er,ez,bt,rr
        integer :: nzi,nri,iz,iz1,ir,ir1,slot,off,nz1,old
        integer :: i00,i10,i11,i01

        extfld(1) = 0.0d0
        extfld(2) = 0.0d0
        extfld(3) = 0.0d0
        extfld(4) = 0.0d0
        extfld(5) = 0.0d0
        extfld(6) = 0.0d0
        zedge = tr_d(3,m)
        len = tr_d(5,m)
        if((pz.gt.zedge).and.(pz.lt.(zedge+len))) then
          tmpcos = tr_d(8,m)
          tmpsin = tr_d(9,m)
          hz = tr_d(10,m)
          hr = tr_d(11,m)
          zspan = tr_d(12,m)
          slot = ti_d(2,m)
          nzi = ti_d(3,m)
          nri = ti_d(4,m)
          zz = pz-zedge
          if(zz.le.0) then
            iz = 1
            efz = zz/hz
          else if(zz.ge.zspan) then
            iz = nzi
            efz = (zz-(iz-1)*hz)/hz
          else
            iz = zz/hz + 1
            efz = (zz-(iz-1)*hz)/hz
          endif
          iz1 = iz+1

          rr = sqrt(px*px+py*py)
          ir = rr/hr + 1
          if(rr.eq.0.0) then
            rr = 1.0e-12
          endif
          if(ir.gt.nri) then
            old = atomicCAS(gpuerr_d,0,3)
            if(old.eq.0) then
              gpuerr_info_d(2) = ipart
              gpuerr_info_d(3) = ir
            endif
            return
          endif
          efr = (rr-(ir-1)*hr)/hr
          ir1 = ir+1

          nz1 = nzi+1
          off = (slot-1)*stride_d
          i00 = off + (ir-1)*nz1 + iz
          i10 = off + (ir-1)*nz1 + iz1
          i11 = off + (ir1-1)*nz1 + iz1
          i01 = off + (ir1-1)*nz1 + iz
          er = (erp_d(i00)*(1.0-efz)*(1.0-efr)+&
               erp_d(i10)*efz*(1.0-efr)+&
               erp_d(i11)*efz*efr+erp_d(i01)*&
               (1.0-efz)*efr)*tmpcos
          ez = (ezp_d(i00)*(1.0-efz)*(1.0-efr)+ezp_d(i10)*&
               efz*(1.0-efr)+&
               ezp_d(i11)*efz*efr+ezp_d(i01)*&
               (1.0-efz)*efr)*tmpcos
          bt = (btp_d(i00)*(1.0-efz)*(1.0-efr)+btp_d(i10)*&
               efz*(1.0-efr)+&
               btp_d(i11)*efz*efr+btp_d(i01)*&
               (1.0-efz)*efr)*tmpsin

          extfld(1) = er*px/rr
          extfld(2) = er*py/rr
          extfld(3) = ez
          extfld(4) = -bt*py/rr
          extfld(5) = bt*px/rr
          extfld(6) = 0.0
        endif
        end subroutine fld_emcyl


        !> getfld_Quadrupole (Quadrupole.f90), from zz = pos(3)-zedge on, with
        !> the three fringe routines inlined.  Mode (ti 3): 1 file profile
        !> (getfldfrgAna3), 2 Enge (getfldfrgAna2), 3 hard edge.  The skew
        !> rotation and the RF factor use per-step constants from the table.
        attributes(device) subroutine fld_quad(px,py,pz,m,extfld)
        implicit none
        real(8), value :: px,py,pz
        integer, value :: m
        real(8), intent(out) :: extfld(6)
        real(8) :: zedge,bgrad,bgradp,bgradpp,len,p3,p4,bb,dd
        real(8) :: zz,tmp1x,tmp2x,t1,t2,t3,cosa,sina
        real(8) :: z10,z20,z3,z4,tmpz,c1,c2,s1,s2,s3,s4,tmp1
        real(8) :: hz,aa,zstart1,zend1,zlength1,zlc
        integer :: mode,slot,i0,numpar1,iz,iz1

        mode = ti_d(3,m)
        slot = ti_d(2,m)
        zedge = tr_d(3,m)
        bb = tr_d(4,m)
        len = tr_d(5,m)
        p3 = tr_d(6,m)
        p4 = tr_d(7,m)
        cosa = tr_d(8,m)
        sina = tr_d(9,m)
        zz = pz-zedge
        bgrad = 0.0d0
        bgradp = 0.0d0
        bgradpp = 0.0d0
        if(mode.eq.2) then
          ! getfldfrgAna2_Quadrupole
          c1 = -0.00004d0
          c2 = 4.518219d0
          dd = 2*p4
          z10 = (len - p3)/2
          z20 = len - z10
          z3 = z10 + 1.5*dd
          z4 = z20 - 1.5*dd
          if(zz.lt.z3) then
            tmpz = -(zz - z10)/dd
            s1 = c1 + c2*tmpz
            bgrad = bb/(1.0+exp(s1))
            s2 = -c2/dd
            tmp1 = exp(s1)*s2
            bgradp = -bb*tmp1/(1.0+exp(s1))**2
            s3 = 0.0
            s4 = exp(s1)*(s3+s2*s2)
            bgradpp = bb*(-s4/(1.0+exp(s1))**2+2*tmp1*tmp1/(1.0+exp(s1))**3)
          else if(zz.gt.z4) then
            tmpz = (zz - z20)/dd
            s1 = c1 + c2*tmpz
            bgrad = bb/(1.0+exp(s1))
            s2 = c2/dd
            tmp1 = exp(s1)*s2
            bgradp = -bb*tmp1/(1.0+exp(s1))**2
            s3 = 0.0
            s4 = exp(s1)*(s3 + s2*s2)
            bgradpp = bb*(-s4/(1.0+exp(s1))**2+2*tmp1*tmp1/(1.0+exp(s1))**3)
          else
            bgrad = bb
            bgradp = 0.0
            bgradpp = 0.0
          endif
        else if(mode.eq.1) then
          ! getfldfrgAna3_Quadrupole, which takes pos(3) and subtracts zedge itself
          if((pz.ge.zedge).and.(pz.le.(zedge+len))) then
            zlc = pz-zedge
            numpar1 = fcoef_d(1,1,slot)+0.1
            zstart1 = fcoef_d(2,1,slot)
            zend1 = fcoef_d(3,1,slot)
            zlength1 = zend1-zstart1
            if(numpar1.gt.1) then
              hz = zlength1/(numpar1-1)
            else
              hz = 1.0
            endif
            if( (zlc.ge.zstart1).and.(zlc.le.zend1)) then
              i0 = 1
              iz = (zlc-zstart1)/hz + 1
              if(iz.lt.0) iz = 1
              iz1 = iz + 1
              if(iz1.gt.numpar1) iz1 = numpar1
              aa = (iz*hz-(zlc-zstart1))/hz
              bgrad = fcoef_d(1,iz+i0,slot)*aa+fcoef_d(1,iz1+i0,slot)*(1.0d0-aa)
              bgradp = fcoef_d(2,iz+i0,slot)*aa+fcoef_d(2,iz1+i0,slot)*(1.0d0-aa)
              bgradpp = fcoef_d(3,iz+i0,slot)*aa+fcoef_d(3,iz1+i0,slot)*(1.0d0-aa)
              bgrad=bgrad*bb
              bgradp=bgradp*bb
              bgradpp=bgradpp*bb
            endif
          endif
        else
          bgrad = bb
        endif

        tmp1x = px*cosa + py*sina
        tmp2x = -px*sina + py*cosa

        extfld(1) = 0.0
        extfld(2) = 0.0
        extfld(3) = 0.0
        if(mode.le.2) then
          t1 = bgrad*tmp2x - &
                      bgradpp*(tmp2x**3+3*tmp1x**2*tmp2x)/12
          t2 = bgrad*tmp1x - &
                      bgradpp*(tmp1x**3+3*tmp1x*tmp2x**2)/12
          t3 = bgradp*tmp1x*tmp2x
        else
          t1 = bgrad*tmp2x
          t2 = bgrad*tmp1x
          t3 = 0.0
        endif

        extfld(4) = t1*cosa - t2*sina
        extfld(5) = t1*sina + t2*cosa
        extfld(6) = t3

        if(ti_d(4,m).eq.1) then
          extfld(4) = extfld(4)*tr_d(10,m)
          extfld(5) = extfld(5)*tr_d(10,m)
          extfld(6) = extfld(6)*tr_d(10,m)
        endif
        end subroutine fld_quad

        !> getfldt_EMfldCart (EMfldCart.f90), from the in-element test on.  The
        !> CPU has no bounds check; a point outside the map's grid gets zero
        !> field here and is counted in cartout_d (D1.2).
        attributes(device) subroutine fld_emcart(px,py,pz,m,extfld,ipart)
        implicit none
        real(8), value :: px,py,pz
        integer, value :: m,ipart
        real(8), intent(out) :: extfld(6)
        real(8) :: hx,hy,hz,zz,ab,cd,ef,xmin,ymin,zmin
        integer :: ix,jx,kx,ix1,jx1,kx1,nx1,ny1,nz1,off,old,k
        complex(8) :: tmptime
        extfld(1) = 0.0d0
        extfld(2) = 0.0d0
        extfld(3) = 0.0d0
        extfld(4) = 0.0d0
        extfld(5) = 0.0d0
        extfld(6) = 0.0d0
        if((pz.gt.tr_d(3,m)).and.(pz.lt.(tr_d(3,m)+tr_d(4,m)))) then
          tmptime = cmplx(tr_d(5,m),tr_d(6,m),kind=8)
          nx1 = ti_d(3,m)
          ny1 = ti_d(4,m)
          nz1 = ti_d(5,m)
          off = ti_d(6,m)
          xmin = tr_d(7,m)
          ymin = tr_d(8,m)
          zmin = tr_d(9,m)
          hx = tr_d(10,m)
          hy = tr_d(11,m)
          hz = tr_d(12,m)
          ix = (px-xmin)/hx + 1
          ab = ((xmin-px)+ix*hx)/hx
          jx = (py-ymin)/hy + 1
          cd = ((ymin-py)+jx*hy)/hy
          zz = pz-tr_d(3,m)
          kx = (zz-zmin)/hz + 1
          ef = ((zmin-zz)+kx*hz)/hz
          if(ix.lt.1 .or. ix.gt.nx1 .or. jx.lt.1 .or. jx.gt.ny1 .or. &
             kx.lt.1 .or. kx.gt.nz1) then
            k = atomicAdd(cartout_d,1)
            old = atomicCAS(cartfirst_d,0,1)
            if(old.eq.0) then
              cartipart_d = ipart
              cartpos_d(1) = px
              cartpos_d(2) = py
              cartpos_d(3) = pz
            endif
            return
          endif
          ! a point exactly on the upper edge has weight 0 on the next cell
          ix1 = min(ix + 1,nx1)
          jx1 = min(jx + 1,ny1)
          kx1 = min(kx + 1,nz1)
          extfld(1) = dble(c8sum(cex_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
          extfld(2) = dble(c8sum(cey_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
          extfld(3) = dble(c8sum(cez_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
          extfld(4) = dble(c8sum(cbx_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
          extfld(5) = dble(c8sum(cby_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
          extfld(6) = dble(c8sum(cbz_d,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef)*tmptime)
        endif
        end subroutine fld_emcart

        !> The eight-term trilinear sum of getfldt_EMfldCart, in its term order.
        attributes(device) function c8sum(a,off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1,ab,cd,ef) result(r)
        implicit none
        complex(8), device :: a(*)
        integer, value :: off,nx1,ny1,ix,jx,kx,ix1,jx1,kx1
        real(8), value :: ab,cd,ef
        complex(8) :: r
        r = a(off+ix+nx1*((jx-1)+ny1*(kx-1)))*ab*cd*ef  &
           +a(off+ix+nx1*((jx1-1)+ny1*(kx-1)))*ab*(1.d0-cd)*ef &
           +a(off+ix+nx1*((jx1-1)+ny1*(kx1-1)))*ab*(1.d0-cd)*(1.d0-ef) &
           +a(off+ix+nx1*((jx-1)+ny1*(kx1-1)))*ab*cd*(1.d0-ef) &
           +a(off+ix1+nx1*((jx-1)+ny1*(kx1-1)))*(1.d0-ab)*cd*(1.d0-ef) &
           +a(off+ix1+nx1*((jx1-1)+ny1*(kx1-1)))*(1.d0-ab)*(1.d0-cd)*(1.d0-ef) &
           +a(off+ix1+nx1*((jx1-1)+ny1*(kx-1)))*(1.d0-ab)*(1.d0-cd)*ef &
           +a(off+ix1+nx1*((jx-1)+ny1*(kx-1)))*(1.d0-ab)*cd*ef
        end function c8sum

        !> getfldt_Dipole (Dipole.f90), from dd1 on.  b0 = Param(3) (the
        !> field), p5 = Param(5) (the half gap).  The face-angle sines and
        !> cosines are per-element constants, uploaded with the slot (dipc_d
        !> 23:26).  Components 1-3 are zero.
        attributes(device) subroutine fld_dipole(px,py,pz,slot,b0,p5,ef)
        implicit none
        real(8), value :: px,py,pz,b0,p5
        integer, value :: slot
        real(8), intent(out) :: ef(6)
        real(8) :: zz,z1,z2,z3,z4,xx1,zz1,ss,k1
        real(8) :: dd,s1,s2,s3,s4,tmpz,tmp1,bb,bbp,bbpp,dd1,dd2
        real(8) :: cc(8)

        dd1 = dipc_d(11,slot)/2
        dd2 = dipc_d(12,slot)/2
        cc(1) = dipc_d(13,slot)
        cc(2) = dipc_d(14,slot)
        cc(3) = dipc_d(15,slot)
        cc(4) = dipc_d(16,slot)
        cc(5) = dipc_d(17,slot)
        cc(6) = dipc_d(18,slot)
        cc(7) = dipc_d(19,slot)
        cc(8) = dipc_d(20,slot)
        zz = pz
        ef(1) = 0.0d0
        ef(2) = 0.0d0
        ef(3) = 0.0d0
        ef(4) = 0.0d0
        ef(5) = 0.0d0
        ef(6) = 0.0d0
        z1 = dipc_d(3,slot)*px + dipc_d(4,slot)
        z2 = dipc_d(5,slot)*px + dipc_d(6,slot)
        z3 = dipc_d(7,slot)*px + dipc_d(8,slot)
        z4 = dipc_d(9,slot)*px + dipc_d(10,slot)
        dd = 2*p5
        if((zz.ge.z1).and.(zz.le.z2)) then
          k1 = dipc_d(3,slot)
          xx1 = (px+pz*k1-dipc_d(4,slot)*k1)/(1.0+k1*k1)
          zz1 = k1*xx1+dipc_d(4,slot)
          ss = sqrt((xx1-px)**2+(zz1-pz)**2)
          tmpz = -(ss-dd1)/dd
          s1 = cc(1)+cc(2)*tmpz+cc(3)*tmpz**2+cc(4)*tmpz**3+cc(5)*tmpz**4+&
               cc(6)*tmpz**5+cc(7)*tmpz**6+cc(8)*tmpz**7
          bb = 1./(1.0+exp(s1))
          s2 = -(cc(2)+2*cc(3)*tmpz+3*cc(4)*tmpz**2+4*cc(5)*tmpz**3+ &
                 5*cc(6)*tmpz**4+6*cc(7)*tmpz**5+7*cc(8)*tmpz**6)/dd
          tmp1 = exp(s1)*s2
          bbp = -tmp1/(1.0+exp(s1))**2
          s3 = (2*cc(3)+6*cc(4)*tmpz+12*cc(5)*tmpz**2+&
                20*cc(6)*tmpz**3+30*cc(7)*tmpz**4+42*cc(8)*tmpz**5)/dd/dd
          s4 = exp(s1)*(s3+s2*s2)
          bbpp = (-s4/(1.0+exp(s1))**2+2*tmp1*tmp1/(1.0+exp(s1))**3)

          ef(4) = -b0*(bbp*py)*dipc_d(23,slot)
          ef(5) = b0*(bb-bbpp/2*py*py)
          ef(6) = b0*(bbp*py)*dipc_d(24,slot)
        else if((zz.gt.z2).and.(zz.lt.z3)) then
          ef(5) = b0
        else if((zz.ge.z3).and.(zz.le.z4)) then
          k1 = dipc_d(7,slot)
          xx1 = (px+pz*k1-dipc_d(8,slot)*k1)/(1.0+k1*k1)
          zz1 = k1*xx1+dipc_d(8,slot)
          ss = sqrt((xx1-px)**2+(zz1-pz)**2)
          tmpz = (ss-dd2)/dd
          s1 = cc(1)+cc(2)*tmpz+cc(3)*tmpz**2+cc(4)*tmpz**3+cc(5)*tmpz**4+&
               cc(6)*tmpz**5+cc(7)*tmpz**6+cc(8)*tmpz**7
          bb = 1.0d0/(1.0+exp(s1))
          s2 = (cc(2)+2*cc(3)*tmpz+3*cc(4)*tmpz**2+4*cc(5)*tmpz**3+ &
                 5*cc(6)*tmpz**4+6*cc(7)*tmpz**5+7*cc(8)*tmpz**6)/dd
          tmp1 = exp(s1)*s2
          bbp = -tmp1/(1.0+exp(s1))**2
          s3 = (2*cc(3)+6*cc(4)*tmpz+12*cc(5)*tmpz**2+&
                20*cc(6)*tmpz**3+30*cc(7)*tmpz**4+42*cc(8)*tmpz**5)/dd/dd
          s4 = exp(s1)*(s3 + s2*s2)
          bbpp = (-s4/(1.0+exp(s1))**2+2*tmp1*tmp1/(1.0+exp(s1))**3)

          ef(4) = -b0*(bbp*py)*dipc_d(25,slot)
          ef(5) = b0*(bb-bbpp/2*py*py)
          ef(6) = b0*(bbp*py)*dipc_d(26,slot)
        endif
        end subroutine fld_dipole

      end module FieldmapGPUclass
