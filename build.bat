@echo off
set pwd=%~dp0
echo %pwd%
set pwd2=D:/BaiduSyncdisk/
set "pwd2=%pwd2:/=/%"
echo %pwd2%

echo  ---------------------
set path=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin;%path%   
echo %path% 
mkdir build
mkdir install

if not exist %pwd%\build\gklib (
    mkdir build\gklib
)
cd build\gklib
if not exist %pwd%\install\gklib (
    echo  -----------build gklib----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install  %pwd%/GKlib
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\metis (
    mkdir build\metis
)
cd build\metis
if not exist %pwd%\install\metis (
    echo  -----------build metis----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/metis  %pwd%/METIS-5.2.1.1
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\glog-0.7.1 (
    mkdir build\glog-0.7.1
)
cd build\glog-0.7.1
if not exist %pwd%\install\glog-0.7.1 (
    echo  -----------build glog-0.7.1----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DBUILD_TESTING:BOOL="0" ^
    -DWITH_GFLAGS:BOOL="0"  ^
    -DWITH_THREADS:BOOL="1"  ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/glog-0.7.1  %pwd%/glog-0.7.1
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\cgal-5.6.1 (
    mkdir build\cgal-5.6.1
)
cd build\cgal-5.6.1
if not exist %pwd%\install\cgal-5.6.1 (
    echo  -----------build cgal-5.6.1----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DWITH_tests:BOOL="0" ^
    -DWITH_demos:BOOL="0" ^
    -DWITH_examples:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_TESTING:BOOL="0" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/cgal-5.6.1  %pwd%/cgal-5.6.1
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\eigen-3.4.0 (
    mkdir build\eigen-3.4.0
)
cd build\eigen-3.4.0
if not exist %pwd%\install\eigen-3.4.0 (
    echo  -----------build eigen-3.4.0----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DEIGEN_TEST_AVX:BOOL="0"  ^
    -DEIGEN_TEST_SSSE3:BOOL="1"  ^
    -DEIGEN_TEST_SSE3:BOOL="1"  ^
    -DEIGEN_TEST_AVX2:BOOL="0"  ^
    -DCUDA_PROPAGATE_HOST_FLAGS:BOOL="0"  ^
    -DEIGEN_TEST_SSE4_1:BOOL="1"  ^
    -DEIGEN_TEST_AVX512:BOOL="0"  ^
    -DCUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE:BOOL="0"  ^
    -DEIGEN_TEST_AVX512DQ:BOOL="0"  ^
    -DEIGEN_TEST_OPENMP:BOOL="1"  ^
    -DEIGEN_TEST_SSE4_2:BOOL="1"  ^
    -DCUDA_HOST_COMPILATION_CPP:BOOL="0"  ^
    -DEIGEN_TEST_SSE2:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/eigen-3.4.0  %pwd%/eigen-3.4.0
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\ceres-solver-2.2.0 (
    mkdir build\ceres-solver-2.2.0
)
cd build\ceres-solver-2.2.0
if not exist %pwd%\install\ceres-solver-2.2.0 (
    echo  -----------build ceres-solver-2.2.0----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DEigen3_DIR:PATH=%pwd%install/eigen-3.4.0/share/eigen3/cmake ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DUSE_CUDA:BOOL="0"  ^
    -DMINIGLOG_MAX_LOG_LEVEL:STRING="2"  ^
    -DLAPACK:BOOL="1"  ^
    -DBUILD_BENCHMARKS:BOOL="1"  ^
    -DMINIGLOG:BOOL="1"  ^
    -DSuiteSparse_SPQR_LIBRARY:FILEPATH="SuiteSparse_SPQR_LIBRARY-NOTFOUND"  ^
    -DSUITESPARSE:BOOL="1" ^
    -DMINIGLOG:BOOL="0"  ^
    -Dglog_DIR:PATH=%pwd%install/glog-0.7.1/lib/cmake/glog  ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/ceres-solver-2.2.0  %pwd%/ceres-solver-2.2.0
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\lz4-1.9.4 (
    mkdir build\lz4-1.9.4
)
cd build\lz4-1.9.4
if not exist %pwd%\install\lz4-1.9.4 (
    echo  -----------build lz4-1.9.4----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DBUILD_STATIC_LIBS:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/lz4-1.9.4  %pwd%/lz4-1.9.4/build/cmake
    ninja install -j12 
)  
cd ../..
rem =======================================================================

if not exist %pwd%\build\zlib-1.2.13 (
    mkdir build\zlib-1.2.13
)
cd build\zlib-1.2.13
if not exist %pwd%\install\zlib-1.2.13 (
    echo  -----------build zlib-1.2.13----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DBUILD_TESTS:BOOL="1" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/zlib-1.2.13  %pwd%/zlib-1.2.13
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\hdf5-hdf5-1_14_3 (
    mkdir build\hdf5-hdf5-1_14_3
)
cd build\hdf5-hdf5-1_14_3
if not exist %pwd%\install\hdf5-hdf5-1_14_3 (
    echo  -----------build hdf5-hdf5-1_14_3----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/install/zlib-1.2.13/lib/libzlib.dll.a  ^
    -DHDF5_USE_FILE_LOCKING:BOOL="1"  ^
    -DCMAKE_CXX_STANDARD=11 ^
    -DH5_HAVE_ALARM:INTERNAL=0 -DH5_HAVE_ASPRINTF:INTERNAL=0 -DH5_HAVE_VASPRINTF:INTERNAL=0 ^
    -DHDF5_TEST_CPP:BOOL="0"  ^
    -DHDF5_BUILD_TOOLS:BOOL="0"  ^
    -DHDF5_TEST_FORTRAN:BOOL="0"  ^
    -DHDF5_TEST_EXAMPLES:BOOL="0"  ^
    -DZLIB_DIR:PATH=%pwd%/install/zlib-1.2.13/lib ^
    -DHDF5_TEST_PARALLEL:BOOL="0"  ^
    -DHDF5_TEST_SWMR:BOOL="0"  ^
    -DHDF5_BUILD_CPP_LIB:BOOL="1"  ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/install/zlib-1.2.13/lib/libzlib.dll.a  ^
    -DHDF5_BUILD_EXAMPLES:BOOL="0"  ^
    -DHDF5_TEST_JAVA:BOOL="0"  ^
    -DBUILD_TESTING:BOOL="0"  ^
    -DZLIB_USE_EXTERNAL:BOOL="1"  ^
    -DHDF5_TEST_SERIAL:BOOL="0"  ^
    -DHDF5_TEST_TOOLS:BOOL="0"  ^
    -DZLIB_INCLUDE_DIR:PATH=%pwd%/install/zlib-1.2.13/include ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/hdf5-hdf5-1_14_3  %pwd%/hdf5-hdf5-1_14_3
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\build\flann-1.9.2 (
    mkdir build\flann-1.9.2
)
cd build\flann-1.9.2
if not exist %pwd%\install\flann-1.9.2 (
    echo  -----------build flann-1.9.2----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DHDF5_DIR:PATH=%pwd%/install/hdf5-hdf5-1_14_3/cmake ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%/install/lz4-1.9.4/include ^
    -DLZ4_LINK_LIBRARIES:FILEPATH=%pwd%/install/lz4-1.9.4/lib/liblz4.dll.a  ^
    -DBUILD_TESTS:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="0" ^
    -DUSE_OPENMP:BOOL="0" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/flann-1.9.2  %pwd%/flann-1.9.2
    ninja install -j12 
)  
cd ../..


rem =======================================================================

if not exist %pwd%\build\sqlite-amalgamation-3460100 (
    mkdir build\sqlite-amalgamation-3460100
)
cd build\sqlite-amalgamation-3460100
if not exist %pwd%\install\sqlite-amalgamation-3460100 (
    echo  -----------build sqlite-amalgamation-3460100----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/sqlite-amalgamation-3460100  %pwd%/sqlite-amalgamation-3460100
    ninja install -j12 
)  
cd ../..


rem =======================================================================

if not exist %pwd%\build\glew-2.1.0 (
    mkdir build\glew-2.1.0
)
cd build\glew-2.1.0
if not exist %pwd%\install\glew-2.1.0 (
    echo  -----------build glew-2.1.0----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/glew-2.1.0  %pwd%/glew-2.1.0/build/cmake 
    ninja install -j12 
)  
cd ../..


rem =======================================================================

if not exist %pwd%\build\colmap-3.10 (
    mkdir build\colmap-3.10
)
cd build\colmap-3.10
if not exist %pwd%\install\colmap-3.10 (
    echo  -----------build colmap-3.10----------
    cmake -G Ninja ^
    -DCMAKE_CXX_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-g++.exe ^
    -DCMAKE_C_COMPILER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-gcc.exe ^
    -DCMAKE_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_CXX_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_C_COMPILER_AR=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ar.exe ^
    -DCMAKE_C_COMPILER_RANLIB=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ranlib.exe ^
    -DCMAKE_DLLTOOL=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-dlltool.exe ^
    -DCMAKE_NM=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-nm.exe ^
    -DCMAKE_LINKER=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-ld ^
    -DCMAKE_STRIP=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin/x86_64-w64-mingw32-strip.exe ^
    -DBoost_DIR:PATH="D:/ucl360/library2019share/boost185\lib\cmake\Boost-1.85.0"  ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/colmap-3.10  D:/repo/colmap-3.10
    ninja install -j12 
)  
cd ../..

rem =======================================================================

if not exist %pwd%\install\spdlog-1.15.0 (
    echo  -----------build spdlog-1.15.0----------
    cmake -G Ninja  -B %pwd%\build\spdlog-1.15.0  -S %pwd%\spdlog-1.15.0    ^
    -DSPDLOG_BUILD_EXAMPLE_HO:BOOL="0"  ^
    -DSPDLOG_BUILD_WARNINGS:BOOL="0"  ^
    -DCMAKE_EXPORT_BUILD_DATABASE:BOOL="0"  ^
    -DSPDLOG_BUILD_EXAMPLE:BOOL="0"  ^
    -DSPDLOG_BUILD_SHARED:BOOL="0"  ^
    -DSPDLOG_BUILD_ALL:BOOL="0"  ^
    -DSPDLOG_BUILD_TESTS_HO:BOOL="0"  ^
    -DSPDLOG_BUILD_TESTS:BOOL="0"  ^
    -DSPDLOG_BUILD_PIC:BOOL="0"  ^
    -DSPDLOG_BUILD_BENCH:BOOL="0" ^
    -DSPDLOG_BUILD_SHARED:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/spdlog-1.15.0 
    cd  %pwd%\build\spdlog-1.15.0 
    ninja install -j16
    cd %pwd%
) 

rem  -DFREEIMAGE_LIBRARIES:FILEPATH="D:/repo/colmapThird/FreeImage3180Win32Win64/x64/FreeImage.lib" 
rem  -DCeres_DIR:PATH="D:\repo\colmapThird\install\ceres-solver-2.2.0\lib\cmake\Ceres" 
rem  -DMPFR_INCLUDE_DIR:PATH="D:/repo/colmapThird/auxiliary/gmp/include" 
rem  -DLZ4_INCLUDE_DIRS:PATH="D:\repo\colmapThird\install\lz4-1.9.4\include" 
rem  -DCGAL_DIR:PATH="D:\repo\colmapThird\install\cgal-5.6.1\lib\cmake\CGAL" 
rem  -DGlew_DIR:PATH="D:\repo\colmapThird\install\glew-2.1.0\lib\cmake\glew" 
rem  -DGMP_INCLUDE_DIR:PATH="D:/repo/colmapThird/auxiliary/gmp/include" 
rem  -DFLANN_INCLUDE_DIRS:PATH="D:\repo\colmapThird\install\flann-1.9.2\include" 
rem  -Dboost_graph_DIR:PATH="D:/ucl360/library2019share/boost185/lib/cmake/boost_graph-1.85.0" 
rem  -DFLANN_LIBRARY_DIR_HINTS:PATH="" 
rem  -DGMP_LIBRARY_DEBUG:FILEPATH="D:/repo/colmapThird/auxiliary/gmp/lib/libgmp-10.lib" 
rem  -DSQLite3_LIBRARY:FILEPATH="D:/repo/colmapThird/install/sqlite-amalgamation-3460100/lib/libSQLITE3.dll.a" 
rem  -DBoost_DIR:PATH="D:\ucl360\library2019share\boost185\lib\cmake\Boost-1.85.0" 
rem  -DSQLite3_INCLUDE_DIR:PATH="D:/repo/colmapThird/install/sqlite-amalgamation-3460100/include" 
rem  -DFLANN_LIBRARIES:FILEPATH="D:/repo/colmapThird/install/flann-1.9.2/lib/libflann_cpp_s.a" 
rem  -DFLANN_INCLUDE_DIR_HINTS:PATH="" 
rem  -DMPFR_LIBRARIES:FILEPATH="D:/repo/colmapThird/auxiliary/gmp/lib/libmpfr-4.lib" 
rem  -DMETIS_LIBRARIES:FILEPATH="D:/repo/colmapThird/install/metis/lib/libmetis.a" 
rem  -DGMP_LIBRARY_RELEASE:FILEPATH="D:/repo/colmapThird/auxiliary/gmp/lib/libgmp-10.lib" 
rem  -Dglog_DIR:PATH="D:\repo\colmapThird\install\glog-0.7.1\lib\cmake\glog" 
rem  -DLZ4_LIBRARIES:FILEPATH="D:/repo/colmapThird/install/lz4-1.9.4/lib/liblz4.a" 
rem  -DCUDA_ENABLED:BOOL="0" 
rem  -DMETIS_INCLUDE_DIRS:PATH="D:\repo\colmapThird\install\metis\include" 
rem  -DFREEIMAGE_INCLUDE_DIRS:PATH="D:\repo\colmapThird\FreeImage3180Win32Win64\x64" 







rem   python D:\repo\Instant-NGP-for-GTX-1000\scripts\colmap2nerf.py --colmap_db  D:\BaiduNetdiskWorkspace\mvs_mvg_bat\workSpace\unre.db   --text D:\BaiduNetdiskWorkspace\mvs_mvg_bat\workSpace --images  D:\BaiduNetdiskWorkspace\mvs_mvg_bat\workSpace/images