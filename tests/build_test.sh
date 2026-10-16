#!/bin/bash
# Build one of the test programs in this folder against an existing CMake build
# of IMPACT-T: the test is compiled with that build's compiler, flags and module
# files, and linked with all of its object files except the main program.
#
#   tests/build_test.sh BUILD_DIR tests/test_pt2ptfields.f90 [OUTPUT]
#
# BUILD_DIR is a configured and built CMake directory (cmake -S src -B BUILD_DIR
# ...; cmake --build BUILD_DIR). Use an MPI build (-DUSE_MPI=ON) for every test;
# for the GPU tests also -DUSE_GPU=ON (nvfortran). OUTPUT defaults to the test's
# name next to the source.
set -euo pipefail

build=$(cd "$1" && pwd)
src=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
out=${3:-${src%.f90}}

tdir=$(ls -d "$build"/CMakeFiles/ImpactTexe*.dir | head -1)
[ -f "$tdir/link.txt" ] || { echo "no $tdir/link.txt: build $build first" >&2; exit 1; }

flag() { sed -n "s/^$1 = //p" "$tdir/flags.make"; }
fflags="$(flag Fortran_FLAGS) $(flag Fortran_DEFINES) $(flag Fortran_INCLUDES)"

# link.txt: <compiler> <link flags> "obj.o" ... -o <exe> <libraries>
read -r -a tok < "$tdir/link.txt"
fc=${tok[0]}
ldflags=() objs=() libs=() seen_o=0 after_exe=0
for ((i = 1; i < ${#tok[@]}; i++)); do
    t=${tok[i]//\"/}
    if [ "$after_exe" = 1 ]; then libs+=("$t")
    elif [ "$t" = "-o" ]; then i=$((i + 1)); after_exe=1
    elif [[ "$t" == *.o ]]; then
        seen_o=1
        [[ "$t" == */Contrl/main.f90.o ]] || objs+=("$build/$t")
    elif [ "$seen_o" = 0 ]; then ldflags+=("$t")
    fi
done

cd "$build"
# shellcheck disable=SC2086
"$fc" $fflags -I"$build" "${ldflags[@]}" -o "$out" "$src" "${objs[@]}" "${libs[@]}"
echo "built $out"
