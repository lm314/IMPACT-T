!----------------------------------------------------------------
! FieldmapGPUclass: external-field lookup on the GPU for beam line
! elements of type 105 (SolRF with a discrete on-axis file, Param(5)>1000)
! and 112 (EMfldCyl).  See IMPACTT_GPU_PORT_PLAN.md, B2.6.
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

        integer, parameter :: Maxact = 64

        ! slot data
        real(8), device, allocatable, save :: fcoef_d(:,:,:)  !(4,10002,Maxoverlap)
        real(8), device, allocatable, save :: ezp_d(:), erp_d(:), btp_d(:)
        integer, save :: stride = 0       ! packed elements per slot (kind 2)
        integer, device, save :: stride_d
        integer, save :: slot_which(Maxoverlap) = 0
        integer, save :: slot_nz(Maxoverlap) = 0, slot_nr(Maxoverlap) = 0

        ! per-step active-element table (device).  ti: 1 kind, 2 slot,
        ! 3 NzIntv, 4 NrIntv; tr: 1 zlo, 2 zhi, 3 zedge, 4 escale, 5 len,
        ! 6 bscale, 7 xlrep, 8 tmpcos, 9 tmpsin, 10 hz, 11 hr, 12 zspan
        integer, device, save :: ti_d(4,Maxact)
        real(8), device, save :: tr_d(12,Maxact)
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
        if(.not.allocated(fcoef_d)) allocate(fcoef_d(4,10002,Maxoverlap))
        zero = 0
        zi = 0
        gpuerr_d = zero
        gpuerr_info_d = zi
        end subroutine fldmap_init

        !> Copy field data slot s to the device after the host has (re)read
        !> it.  which = 1: read1tdata_Data (105 with a file id > 1000);
        !> which = 2: read2t_Data (112).  fldmp is the whole slot array,
        !> needed to re-pack the other 112 slots if the packed stride grows.
        subroutine fldslot_upload(fldmp,s,which)
        implicit none
        type (fielddata), dimension(:), intent(in) :: fldmp
        integer, intent(in) :: s,which
        integer :: nz,nr,need,k

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
        endif
        end subroutine fldslot_upload

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
        integer :: hti(4,Maxact)
        double precision :: htr(12,Maxact)
        double precision :: blength,p1,p2,p3,p4,p5,p12
        integer :: bnseg,bmpstp,bitype
        ! single-precision literals, as in getfldt_SolRF/getfldt_EMfldCyl
        double precision :: clite,pi,ww,theta0,xlrep

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
          else if(itype.ne.112) then
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
          htr(3,n) = p1
          htr(4,n) = p2
          htr(5,n) = blength
          hti(2,n) = slot
          if(itype.eq.105) then
            hti(1,n) = 1
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
            else
              call fld_emcyl(x,y,zz,m,tf,ipart)
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

      end module FieldmapGPUclass
