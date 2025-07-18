@echo off
set pwd=%~dp0
echo %pwd%
set pwd2=D:/BaiduSyncdisk/
set "pwd2=%pwd2:/=/%"
echo %pwd2%

echo  ---------------------
echo  -----------need set QT_QPA_PLATFORM_PLUGIN_PATH----------
set QT_QPA_PLATFORM_PLUGIN_PATH=C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/plugins/platforms
::set path=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin;%path%   
echo %path% 
mkdir build
mkdir install


if not exist %pwd%\install\jsoncpp-1.9.6\Release (
    echo  -----------build jsoncpp-1.9.6----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\jsoncpp-1.9.6  -S %pwd%\jsoncpp-1.9.6 -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install\jsoncpp-1.9.6\Release
    -DJSONCPP_WITH_CMAKE_PACKAGE:BOOL="1" ^
    -DJSONCPP_STATIC_WINDOWS_RUNTIME:BOOL="0" ^
    -DBUILD_OBJECT_LIBS:BOOL="1" ^
    -DJSONCPP_WITH_EXAMPLE:BOOL="0" ^
	-DBUILD_TESTING:BOOL="0"  ^
	-DJSONCPP_WITH_EXAMPLE:BOOL="0"   ^
	-DJSONCPP_WITH_TESTS:BOOL="0"   ^
	-DJSONCPP_WITH_POST_BUILD_UNITTEST:BOOL="0"   ^
    -DJSONCPP_WITH_PKGCONFIG_SUPPORT:BOOL="0" ^
    -DJSONCPP_WITH_WARNING_AS_ERROR:BOOL="0" ^
    -DJSONCPP_WITH_TESTS:BOOL="0" ^
    -DJSONCPP_WITH_STRICT_ISO:BOOL="1" ^
    -DBUILD_STATIC_LIBS:BOOL="1" ^
    -DCMAKE_BUILD_TYPE="Release" 
    TIMEOUT /T 1
    msbuild %pwd%\build\jsoncpp-1.9.6\INSTALL.vcxproj  -t:Rebuild -p:Configuration=Release

    echo  -----------build jsoncpp-1.9.6----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\jsoncpp-1.9.6  -S %pwd%\jsoncpp-1.9.6 -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install\jsoncpp-1.9.6\Debug 
    -DJSONCPP_WITH_CMAKE_PACKAGE:BOOL="1" ^
    -DJSONCPP_STATIC_WINDOWS_RUNTIME:BOOL="0" ^
    -DBUILD_OBJECT_LIBS:BOOL="1" ^
	-DBUILD_TESTING:BOOL="0"  ^
	-DJSONCPP_WITH_EXAMPLE:BOOL="0"   ^
	-DJSONCPP_WITH_TESTS:BOOL="0"   ^
	-DJSONCPP_WITH_POST_BUILD_UNITTEST:BOOL="0"   ^
    -DJSONCPP_WITH_PKGCONFIG_SUPPORT:BOOL="0" ^
    -DJSONCPP_WITH_WARNING_AS_ERROR:BOOL="0" ^
    -DJSONCPP_WITH_TESTS:BOOL="0" ^
    -DJSONCPP_WITH_STRICT_ISO:BOOL="1" ^
    -DBUILD_STATIC_LIBS:BOOL="1" ^
    -DCMAKE_BUILD_TYPE="Debug" 
    TIMEOUT /T 1
    msbuild %pwd%\build\jsoncpp-1.9.6\INSTALL.vcxproj  -t:Rebuild -p:Configuration=Debug
)

if not exist %pwd%\install\GKlib (
    echo  -----------build gklib----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\gklib  -S %pwd%\GKlib  -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install 
    -DGKREGEX:BOOL="1" -DGKRAND:BOOL="1" ^
    -DBUILD_SHARED_LIBS:BOOL="0" ^
    -DBUILD_TESTING:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" 
    TIMEOUT /T 1
    msbuild %pwd%\build\gklib\INSTALL.vcxproj  -t:Rebuild -p:Configuration=Release
)


rem =======================================================================

if not exist %pwd%\install\METIS-5.2.1.1 (
    echo  -----------build METIS----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\METIS-5.2.1.1  -S %pwd%\METIS-5.2.1.1 ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/METIS-5.2.1.1  ^
    -DGKREGEX:BOOL="1" ^
    -DGKRAND:BOOL="1" ^
    -DBUILD_TESTING:BOOL="0" ^
    -DOPENMP:BOOL="1" ^
    -DCMAKE_BUILD_TYPE="Release" 
    TIMEOUT /T 1
    msbuild %pwd%\build\METIS-5.2.1.1\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\glog-0.7.1 (
    echo  -----------build glog-0.7.1----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\glog-0.7.1  -S %pwd%\glog-0.7.1  -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/glog-0.7.1  ^
    -DBUILD_TESTING:BOOL="0" ^
    -DWITH_GFLAGS:BOOL="0"  ^
    -DWITH_THREADS:BOOL="1"  ^
    -DBUILD_TESTING:BOOL="0" ^
    -DWITH_GTEST:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" 
    TIMEOUT /T 1
    msbuild %pwd%\build\glog-0.7.1\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    msbuild %pwd%\build\glog-0.7.1\INSTALL.vcxproj -t:Rebuild -p:Configuration=Debug
)
rem =======================================================================

if not exist %pwd%\install\eigen-3.4.0 (
    echo  -----------build eigen-3.4.0----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\eigen-3.4.0  -S %pwd%\eigen-3.4.0     ^
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
	-DCUDA_VERSION:STRING="11.7"  ^
	-DCUDA_BUILD_CUBIN:BOOL="0"  ^
	-DBUILD_TESTING:BOOL="0"  ^
	-DEIGEN_DOC_USE_MATHJAX:BOOL="0"  ^
	-DCUDA_USE_STATIC_CUDA_RUNTIME:BOOL="0"  ^
	-DCUDA_64_BIT_DEVICE_CODE:BOOL="0"  ^
	-DEIGEN_BUILD_DOC:BOOL="0"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/eigen-3.4.0 
    TIMEOUT /T 1
    msbuild %pwd%\build\eigen-3.4.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\PoseLib-2.0.4 (
    echo  -----------build PoseLib-2.0.4----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\PoseLib-2.0.4  -S %pwd%\PoseLib-2.0.4     ^
    -DBUILD_SHARED_LIBS:BOOL="0" ^
    -DEigen3_DIR:PATH="%pwd%install/eigen-3.4.0/share/eigen3/cmake"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/PoseLib-2.0.4
    TIMEOUT /T 1
    msbuild %pwd%\build\PoseLib-2.0.4\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\cgal-5.6.1 (
    echo  -----------build cgal-5.6.1----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\cgal-5.6.1  -S %pwd%\cgal-5.6.1     ^
    -DBUILD_DOC:BOOL="0" ^
    -DCGAL_REPORT_DUPLICATE_FILES:BOOL="0" ^
    -DCMAKE_SKIP_INSTALL_RPATH:BOOL="0" ^
    -DCGAL_DEV_MODE:BOOL="0" ^
	-DBUILD_DOC:BOOL="0"  ^
    -DWITH_tests:BOOL="0" ^
    -DBUILD_TESTING:BOOL="0" ^
    -DCGAL_TEST_DRAW_FUNCTIONS:BOOL="0" ^
    -DCGAL_ENABLE_CHECK_HEADERS:BOOL="0" ^
    -DCGAL_CTEST_DISPLAY_MEM_AND_TIME:BOOL="0" ^
    -DCMAKE_VERBOSE_MAKEFILE:BOOL="0" ^
    -DCMAKE_SKIP_RPATH:BOOL="0" ^
    -DWITH_examples:BOOL="0" ^
    -DWITH_demos:BOOL="0" ^
    -DCMAKE_CONFIGURATION_TYPES:STRING="Release"  ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/cgal-5.6.1 
    TIMEOUT /T 1
    msbuild %pwd%\build\cgal-5.6.1\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    xcopy   %pwd%auxiliary\gmp %pwd%install\cgal-5.6.1\include\CGAL\Installation\auxiliary\gmp\ /s /e /y
)
rem =======================================================================

if not exist %pwd%\install\opencv480 (
    echo  -----------build opencv480----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\opencv480  -S %pwd%\opencv480     ^
    -DBUILD_opencv_apps:BOOL="0" ^
    -DBUILD_WITH_DEBUG_INFO:BOOL="0" ^
    -DBUILD_opencv_flann:BOOL="1" ^
    -DBUILD_opencv_world:BOOL="1" ^
    -DBUILD_opencv_gapi:BOOL="1" ^
    -DINSTALL_PDB:BOOL="0" ^
    -DBUILD_opencv_features2d:BOOL="1" ^
    -DBUILD_DOCS:BOOL="0"              ^
    -DBUILD_PNG:BOOL="1"               ^
    -DBUILD_opencv_ts:BOOL="1"         ^
    -DBUILD_opencv_stitching:BOOL="1"  ^
    -DBUILD_PERF_TESTS:BOOL="0"        ^
    -DBUILD_opencv_imgcodecs:BOOL="1"  ^
    -DBUILD_ITT:BOOL="1"               ^
    -DBUILD_opencv_calib3d:BOOL="1"    ^
    -DBUILD_opencv_core:BOOL="1"       ^
    -DBUILD_opencv_imgproc:BOOL="1"    ^
    -DBUILD_opencv_video:BOOL="1"      ^
    -DBUILD_WITH_STATIC_CRT:BOOL="1"   ^
    -DBUILD_SHARED_LIBS:BOOL="1"       ^
    -DBUILD_PACKAGE:BOOL="1"           ^
    -DBUILD_TESTS:BOOL="0"             ^
    -DBUILD_WEBP:BOOL="1"              ^
    -DBUILD_OPENJPEG:BOOL="1"          ^
    -DBUILD_JAVA:BOOL="0"              ^
    -DBUILD_EXAMPLES:BOOL="0"          ^
    -DBUILD_opencv_java_bindings_generator:BOOL="0"    ^
    -DBUILD_opencv_videoio:BOOL="1"                    ^
    -DBUILD_opencv_python_tests:BOOL="0"               ^
    -DBUILD_ZLIB:BOOL="1"                              ^
    -DBUILD_WITH_DYNAMIC_IPP:BOOL="0"                  ^
    -DBUILD_opencv_js_bindings_generator:BOOL="0"      ^
    -DBUILD_PROTOBUF:BOOL="1"                          ^
    -DBUILD_opencv_objc_bindings_generator:BOOL="1"    ^
    -DBUILD_JPEG:BOOL="1"                              ^
    -DBUILD_IPP_IW:BOOL="1"                            ^
    -DINSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL:BOOL="0"  ^
    -DBUILD_opencv_python_bindings_generator:BOOL="0"  ^
    -DBUILD_opencv_highgui:BOOL="1"   ^
    -DBUILD_TBB:BOOL="1"              ^
    -DBUILD_JASPER:BOOL="0"           ^
    -DBUILD_TIFF:BOOL="1"             ^
    -DBUILD_OPENEXR:BOOL="0"          ^
    -DBUILD_opencv_ml:BOOL="1"        ^
    -DBUILD_opencv_photo:BOOL="1"     ^
    -DBUILD_opencv_python3:BOOL="0"   ^
    -DBUILD_USE_SYMLINKS:BOOL="0"     ^
    -DBUILD_opencv_objdetect:BOOL="1" ^
    -DBUILD_FAT_JAVA_LIB:BOOL="0"     ^
    -DBUILD_opencv_dnn:BOOL="1"       ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/opencv480 
    TIMEOUT /T 1
    msbuild %pwd%\build\opencv480\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\lz4-1.9.4 (
    echo  -----------build lz4-1.9.4----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\lz4-1.9.4  -S %pwd%\lz4-1.9.4\build\cmake     ^
    -DBUILD_SHARED_LIBS:BOOL="0" ^
    -DBUILD_STATIC_LIBS:BOOL="1" ^
	-DBUILD_TESTS:BOOL="0"  ^
	-DBUILD_EXAMPLES:BOOL="0"   ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/lz4-1.9.4
    TIMEOUT /T 1
    msbuild %pwd%\build\lz4-1.9.4\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\zlib-1.2.13 (
    echo  -----------build zlib-1.2.13----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\zlib-1.2.13  -S %pwd%\zlib-1.2.13     ^
	-DBUILD_TESTS:BOOL="0"  ^
	-DBUILD_EXAMPLES:BOOL="0"   ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/zlib-1.2.13
    TIMEOUT /T 1
    msbuild %pwd%\build\zlib-1.2.13\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================


if not exist %pwd%\install\protobuf-3.20.0-rc3 (
    echo  -----------build protobuf-3.20.0-rc3----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\protobuf-3.20.0-rc3  -S %pwd%\protobuf-3.20.0-rc3/cmake     ^
    -Dprotobuf_BUILD_SHARED_LIBS:BOOL="1"   ^
    -Dprotobuf_BUILD_TESTS:BOOL="0"    ^
    -DZLIB_INCLUDE_DIR:PATH=%pwd%/install/zlib-1.2.13/include    ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%/install/protobuf-3.20.0-rc3/     ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlib.lib    ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlib.lib    ^
    -Dprotobuf_WITH_ZLIB:BOOL="1" 
    TIMEOUT /T 1
    msbuild %pwd%\build\protobuf-3.20.0-rc3\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    msbuild %pwd%\build\protobuf-3.20.0-rc3\INSTALL.vcxproj -t:Rebuild -p:Configuration=Debug
    copy %pwd%\install\zlib-1.2.13\bin\zlib.dll  %pwd%install\protobuf-3.20.0-rc3\bin
)
rem =======================================================================


if not exist %pwd%\install\hdf5-hdf5-1_14_3 (
    echo  -----------build hdf5-hdf5-1_14_3----------
    cmake   -G "Visual Studio 16 2019"   -B %pwd%\build\hdf5-hdf5-1_14_3  -S %pwd%\hdf5-hdf5-1_14_3    ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlibstatic.lib  ^
    -DHDF5_USE_FILE_LOCKING:BOOL="1"  ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DH5_HAVE_ALARM:INTERNAL=0  ^
	-DH5_HAVE_ASPRINTF:INTERNAL=0 ^
	-DH5_HAVE_VASPRINTF:INTERNAL=0 ^
    -DHDF5_TEST_CPP:BOOL="0"  ^
    -DHDF5_BUILD_TOOLS:BOOL="0"  ^
    -DHDF5_TEST_FORTRAN:BOOL="0"  ^
    -DHDF5_TEST_EXAMPLES:BOOL="0"  ^
    -DZLIB_DIR:PATH=%pwd%/install/zlib-1.2.13/lib ^
    -DHDF5_TEST_PARALLEL:BOOL="0"  ^
    -DHDF5_TEST_SWMR:BOOL="0"  ^
    -DHDF5_BUILD_CPP_LIB:BOOL="1"  ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlibstatic.lib  ^
    -DHDF5_BUILD_EXAMPLES:BOOL="0"  ^
    -DHDF5_TEST_JAVA:BOOL="0"  ^
    -DBUILD_TESTING:BOOL="0"  ^
    -DZLIB_USE_EXTERNAL:BOOL="1"  ^
    -DHDF5_TEST_SERIAL:BOOL="0"  ^
    -DHDF5_TEST_TOOLS:BOOL="0"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/hdf5-hdf5-1_14_3
    TIMEOUT /T 1
    msbuild %pwd%\build\hdf5-hdf5-1_14_3\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)

rem =======================================================================


if not exist %pwd%\install\sqlite-amalgamation-3460100 (
    echo  -----------build sqlite-amalgamation-3460100----------
    lib /DEF:%pwd%/sqlite-dll-win-x64-3460100/sqlite3.def  /OUT:%pwd%/sqlite-dll-win-x64-3460100/sqlite3.lib /MACHINE:x64
    mkdir %pwd%\install\sqlite-amalgamation-3460100
    xcopy   %pwd%\sqlite-dll-win-x64-3460100 %pwd%\install\sqlite-amalgamation-3460100 /s /e /y
)
rem =======================================================================

if not exist %pwd%\install\ceres-solver-2.2.0 (
    echo  -----------build ceres-solver-2.2.0----------
    cmake   -G "Visual Studio 16 2019"   -B %pwd%\build\ceres-solver-2.2.0  -S %pwd%\ceres-solver-2.2.0    ^
    -DEigen3_DIR:PATH=%pwd%install/eigen-3.4.0/share/eigen3/cmake ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DUSE_CUDA:BOOL="0"  ^
    -DMINIGLOG_MAX_LOG_LEVEL:STRING="2"  ^
    -DLAPACK:BOOL="1"  ^
	-DBUILD_TESTING:BOOL="0"  ^
	-DBUILD_EXAMPLES:BOOL="0"  ^
	-DBUILD_DOCUMENTATION:BOOL="0"  ^
	-DBUILD_BENCHMARKS:BOOL="0"  ^
    -DMINIGLOG:BOOL="1"  ^
    -DSuiteSparse_SPQR_LIBRARY:FILEPATH="SuiteSparse_SPQR_LIBRARY-NOTFOUND"  ^
    -DSUITESPARSE:BOOL="1" ^
    -DMINIGLOG:BOOL="0"  ^
    -Dglog_DIR:PATH=%pwd%install/glog-0.7.1/lib/cmake/glog  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/ceres-solver-2.2.0
    TIMEOUT /T 1
    msbuild %pwd%\build\ceres-solver-2.2.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    msbuild %pwd%\build\ceres-solver-2.2.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Debug
)
rem =======================================================================

if not exist %pwd%\install\flann-1.9.2 (
    echo  -----------build flann-1.9.2----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\flann-1.9.2  -S %pwd%\flann-1.9.2    ^
    -DHDF5_DIR:PATH=%pwd%/install/hdf5-hdf5-1_14_3/cmake ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%/install/lz4-1.9.4/include ^
    -DLZ4_LINK_LIBRARIES:FILEPATH=%pwd%/install/lz4-1.9.4/lib/lz4.lib  ^
    -DBUILD_TESTS:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="0" ^
    -DUSE_OPENMP:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/flann-1.9.2
    TIMEOUT /T 1
    msbuild %pwd%\build\flann-1.9.2\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\glew-2.1.0 (
    echo  -----------build glew-2.1.0----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\glew-2.1.0  -S %pwd%\glew-2.1.0\build\cmake    ^
    -DHDF5_DIR:PATH=%pwd%/install/hdf5-hdf5-1_14_3/cmake ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%/install/lz4-1.9.4/include ^
    -DLZ4_LINK_LIBRARIES:FILEPATH=%pwd%/install/lz4-1.9.4/lib/lz4.lib  ^
    -DBUILD_TESTS:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="0" ^
    -DUSE_OPENMP:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/glew-2.1.0
    TIMEOUT /T 1
    msbuild %pwd%\build\glew-2.1.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
rem =======================================================================

if not exist %pwd%\install\colmap-3.10 (
    echo  -----------build colmap-3.10----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\colmap-3.10  -S %pwd%\colmap-3.10    ^
    -DSQLite3_INCLUDE_DIR:PATH=%pwd%install/sqlite-amalgamation-3460100   ^
    -DJSONCPP_LIBRARIES_DIRS:FILEPATH=%pwd%install/jsoncpp-1.9.6/lib   ^
    -DJSONCPP_INCLUDE_DIRS:PATH=%pwd%install/jsoncpp-1.9.6/include  ^
    -DFREEIMAGE_LIBRARIES:FILEPATH=%pwd%FreeImage3180Win32Win64/x64/FreeImage.lib   ^
    -Dglog_DIR:PATH=%pwd%install/glog-0.7.1/lib/cmake/glog   ^
    -DBoost_DIR:PATH="D:/ucl360/library2019share/boost185/lib/cmake/Boost-1.85.0"   ^
    -DFLANN_LIBRARIES:FILEPATH=%pwd%install/flann-1.9.2/lib/flann_cpp_s.lib   ^
    -DFLANN_INCLUDE_DIRS:PATH=%pwd%install/flann-1.9.2/include   ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%install/lz4-1.9.4/include   ^
    -DCeres_DIR:PATH=%pwd%install/ceres-solver-2.2.0/lib/cmake/Ceres   ^
    -DFREEIMAGE_INCLUDE_DIRS:PATH=%pwd%FreeImage3180Win32Win64/x64   ^
    -DLZ4_LIBRARIES:FILEPATH=%pwd%install/lz4-1.9.4/lib/lz4.lib   ^
    -DSQLite3_LIBRARY:FILEPATH=%pwd%install/sqlite-amalgamation-3460100/sqlite3.lib  ^
    -DHDF5_DIR:PATH=%pwd%install/hdf5-hdf5-1_14_3/cmake ^
    -DMPFR_LIBRARIES:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libmpfr-4.lib  ^
    -DGMP_LIBRARY_DEBUG:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libgmp-10.lib   ^
    -DGMP_INCLUDE_DIR:PATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/include  ^
    -DMETIS_LIBRARIES:FILEPATH=%pwd%install/METIS-5.2.1.1/lib/metis.lib  ^
    -DCGAL_DIR:PATH=%pwd%install/cgal-5.6.1/lib/cmake/CGAL   ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/colmap-3.10   ^
    -DCUDA_ENABLED:BOOL="0"  ^
    -DMETIS_INCLUDE_DIRS:PATH=%pwd%install/METIS-5.2.1.1/include  ^
    -DGMP_LIBRARY_RELEASE:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libgmp-10.lib  ^
    -DMPFR_INCLUDE_DIR:PATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/include  ^
    -DGlew_DIR:PATH=%pwd%install/glew-2.1.0/lib/cmake/glew  ^
    -DGK_LIBRARIES:FILEPATH=%pwd%install/GKlib/lib/GKlib.lib ^
    -DQt5_DIR:PATH="C:\Qt\Qt5.14.2\5.14.2\msvc2017_64\lib\cmake\Qt5"  ^
    -DIDXTYPEWIDTH=32 ^
    -DREALTYPEWIDTH=32 ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/colmap-3.10
    TIMEOUT /T 1
    msbuild %pwd%\build\colmap-3.10\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
copy %pwd%\FreeImage3180Win32Win64\x64\FreeImage.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\install\jsoncpp-1.9.6-release\bin\jsoncpp.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\install\glog-0.7.1\bin\glog.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\install\glew-2.1.0\bin\glew32.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\install\sqlite-amalgamation-3460100\sqlite3.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\auxiliary\gmp\lib\libgmp-10.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\auxiliary\gmp\lib\libmpfr-4.dll  %pwd%\install\colmap-3.10\bin
copy %pwd%\install\jsoncpp-1.9.6\bin\jsoncpp.dll  %pwd%\install\colmap-3.10\bin
rem =======================================================================
if not exist %pwd%\install\colmap-3.11.1 (
    echo  -----------build colmap-3.11.1----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\colmap-3.11.1  -S %pwd%\colmap-3.11.1    ^
    -DEigen3_DIR:PATH=%pwd%install/eigen-3.4.0/share/eigen3/cmake   ^
    -DCERES_INCLUDE_DIRS:PATH=%pwd%install/ceres-solver-2.2.0/include   ^
    -DCERES_LIBRARIES:FILEPATH=%pwd%install/ceres-solver-2.2.0/lib/ceres.lib ^
    -DJSONCPP_INCLUDE_DIRS:PATH=%pwd%install/jsoncpp-1.9.6/include  ^
    -DJSONCPP_LIBRARIES_DIRS:PATH=%pwd%install/jsoncpp-1.9.6/lib   ^
    -DJSONCPP_LIBRARIES:STRING=jsoncpp_static.lib   ^
    -DGLOG_INCLUDE_DIRS:PATH=%pwd%install/glog-0.7.1/include   ^
    -DGLOG_LIBRARIES:FILEPATH=%pwd%install/glog-0.7.1/lib/glog.lib^
    -DBOOST_INCLUDE_DIRS:PATH=D:/ucl360/library2019share/boost185/include/boost-1_85 ^
    -DBOOST_LIBRARIES_DIR:PATH=D:/ucl360/library2019share/boost185/lib ^
    -DBOOST_PROGRAM_OPTIONS_LIBRARIES:FILEPATH=D:/ucl360/library2019share/boost185/lib/libboost_program_options-vc142-mt-s-x64-1_85.lib ^
    -DBOOST_GRAPH_LIBRARIES:FILEPATH=D:/ucl360/library2019share/boost185/lib/libboost_graph-vc142-mt-s-x64-1_85.lib ^
    -DBOOST_SYSTEM_LIBRARIES:FILEPATH=D:/ucl360/library2019share/boost185/lib/libboost_system-vc142-mt-s-x64-1_85.lib ^
    -DPOSELIB_INCLUDE_DIRS:PATH=%pwd%install/PoseLib-2.0.4/include  ^
    -DPOSELIB_LIBRARIES:FILEPATH=%pwd%install/PoseLib-2.0.4/lib/PoseLib.lib ^
    -DSQLITE_INCLUDE_DIRS:PATH=%pwd%install/sqlite-amalgamation-3460100   ^
    -DSQLITE_LIBRARIES:FILEPATH=%pwd%install/sqlite-amalgamation-3460100/sqlite3.lib ^
    -DFREEIMAGE_INCLUDE_DIRS:PATH=%pwd%FreeImage3180Win32Win64/x64 ^
    -DFREEIMAGE_LIBRARIES:FILEPATH=%pwd%FreeImage3180Win32Win64/x64/FreeImage.lib ^
    -DFLANN_DIR_INCLUDE_DIRS:PATH=%pwd%install/flann-1.9.2/include ^
    -DFLANN_DIR_LIBRARIES:FILEPATH=%pwd%install/flann-1.9.2/lib/flann_cpp_s.lib ^
    -DLZ4_DIR_INCLUDE_DIRS:PATH=%pwd%install/lz4-1.9.4/include   ^
    -DLZ4_DIR_LIBRARIES:FILEPATH=%pwd%install/lz4-1.9.4/lib/lz4.lib  ^
    -DMETIS_DIR_INCLUDE_DIRS:PATH=%pwd%install/METIS-5.2.1.1/include  ^
    -DMETIS_LIBRARIES:FILEPATH=%pwd%install/METIS-5.2.1.1/lib/metis.lib ^
    -DGKLIB_LIBRARIES:FILEPATH=%pwd%install/GKlib/lib/GKlib.lib ^
    -DQt5_DIRS:PATH=C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/include ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/colmap-3.11.1
    TIMEOUT /T 1
    msbuild %pwd%\build\colmap-3.11.1\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
)
copy %pwd%\FreeImage3180Win32Win64\x64\FreeImage.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\install\jsoncpp-1.9.6-release\bin\jsoncpp.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\install\glog-0.7.1\bin\glog.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\install\glew-2.1.0\bin\glew32.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\install\sqlite-amalgamation-3460100\sqlite3.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\auxiliary\gmp\lib\libgmp-10.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\auxiliary\gmp\lib\libmpfr-4.dll  %pwd%\install\colmap-3.11.1\bin
copy %pwd%\install\jsoncpp-1.9.6\bin\jsoncpp.dll  %pwd%\install\colmap-3.11.1\bin
rem =======================================================================


if not exist %pwd%\install\test (
    echo  -----------build test----------
    cmake  -G "Visual Studio 16 2019"  -B %pwd%\build\test  -S %pwd%\test    ^
    -DEigen3_DIR:PATH=%pwd%install/eigen-3.4.0/share/eigen3/cmake ^
    -DSQLite3_INCLUDE_DIR:PATH=%pwd%install/sqlite-amalgamation-3460100   ^
    -DFREEIMAGE_LIBRARIES:FILEPATH=%pwd%FreeImage3180Win32Win64/x64/FreeImage.lib   ^
    -Dglog_DIR:PATH=%pwd%install/glog-0.7.1/lib/cmake/glog   ^
    -DBoost_DIR:PATH="D:/ucl360/library2019share/boost185/lib/cmake/Boost-1.85.0"   ^
    -DFLANN_LIBRARIES:FILEPATH=%pwd%install/flann-1.9.2/lib/flann_cpp_s.lib   ^
    -DFLANN_INCLUDE_DIRS:PATH=%pwd%install/flann-1.9.2/include   ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%install/lz4-1.9.4/include   ^
    -DCeres_DIR:PATH=%pwd%install/ceres-solver-2.2.0/lib/cmake/Ceres   ^
    -DFREEIMAGE_INCLUDE_DIRS:PATH=%pwd%FreeImage3180Win32Win64/x64   ^
    -DLZ4_LIBRARIES:FILEPATH=%pwd%install/lz4-1.9.4/lib/lz4.lib   ^
    -DSQLite3_LIBRARY:FILEPATH=%pwd%install/sqlite-amalgamation-3460100/sqlite3.lib  ^
    -DHDF5_DIR:PATH=%pwd%install/hdf5-hdf5-1_14_3/cmake ^
    -DMPFR_LIBRARIES:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libmpfr-4.lib  ^
    -DGMP_LIBRARY_DEBUG:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libgmp-10.lib   ^
    -DGMP_INCLUDE_DIR:PATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/include  ^
    -DMETIS_LIBRARIES:FILEPATH=%pwd%install/METIS-5.2.1.1/lib/metis.lib  ^
    -DCGAL_DIR:PATH=%pwd%install/cgal-5.6.1/lib/cmake/CGAL   ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/test   ^
    -DCUDA_ENABLED:BOOL="0"  ^
    -DMETIS_INCLUDE_DIRS:PATH=%pwd%install/METIS-5.2.1.1/include  ^
    -DGMP_LIBRARY_RELEASE:FILEPATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/lib/libgmp-10.lib  ^
    -DMPFR_INCLUDE_DIR:PATH=%pwd%install/cgal-5.6.1/include/CGAL/Installation/auxiliary/gmp/include  ^
    -DGlew_DIR:PATH=%pwd%install/glew-2.1.0/lib/cmake/glew  ^
    -DGK_LIBRARIES:FILEPATH=%pwd%install/GKlib/lib/GKlib.lib ^
    -DQt5_DIR:PATH="C:\Qt\Qt5.14.2\5.14.2\msvc2017_64\lib\cmake\Qt5"  ^
    -DIDXTYPEWIDTH=32 ^
    -DREALTYPEWIDTH=32 ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/test
    TIMEOUT /T 1
    msbuild %pwd%\build\test\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
     
)

rem =======================================================================

if not exist %pwd%\install\spdlog-1.15.0 (
    echo  -----------build spdlog-1.15.0----------
    cmake  -G "Visual Studio 16 2019"    -B %pwd%\build\spdlog-1.15.0  -S %pwd%\spdlog-1.15.0    ^
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

    TIMEOUT /T 1
    msbuild %pwd%\build\spdlog-1.15.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    cd %pwd%
) 


if not exist %pwd%\install\libigl-2.5.0 (
    echo  -----------build libigl-2.5.0----------
    cmake  -G "Visual Studio 16 2019"    -B %pwd%\build\libigl-2.5.0  -S %pwd%\libigl-2.5.0    ^
    -DEMBREE_STATIC_LIB:BOOL="0"   ^
    -DLIBIGL_BUILD_TUTORIALS:BOOL="0"   ^
    -DLIBIGL_COPYLEFT_CORE:BOOL="0"   ^
    -DEMBREE_RAY_PACKETS:BOOL="0"   ^
    -DLIBIGL_OPENGL:BOOL="0"   ^
    -DUSE_MSVC_RUNTIME_LIBRARY_DLL:BOOL="0"   ^
    -DLIBIGL_STB:BOOL="0"   ^
    -DBUILD_TESTING:BOOL="0"   ^
    -DLIBIGL_BUILD_TESTS:BOOL="0"   ^
    -DLIBIGL_EMBREE:BOOL="0"   ^
    -DLIBIGL_USE_STATIC_LIBRARY:BOOL="0"   ^
    -DLIBIGL_GLFW:BOOL="0"   ^
    -DLIBIGL_XML:BOOL="0"   ^
    -DFETCHCONTENT_SOURCE_DIR_EIGEN:PATH=%pwd%install/eigen-3.4.0/include/eigen3   ^
    -DLIBIGL_GLFW_TESTS:BOOL="0"   ^
    -DCATCH_INSTALL_DOCS:BOOL="0"   ^
    -DLIBIGL_INSTALL:BOOL="1"   ^
    -DLIBIGL_IMGUI:BOOL="0"   ^
    -DLIBIGL_COPYLEFT_TETGEN:BOOL="0"   ^
    -DLIBIGL_PREDICATES:BOOL="0"   ^
    -DLIBIGL_SPECTRA:BOOL="0"   ^
    -DLIBIGL_COPYLEFT_CGAL:BOOL="0"   ^
    -DBUILD_SHARED_LIBS:BOOL="1"   ^
    -DCATCH_BUILD_TESTING:BOOL="0"   ^
    -DLIBIGL_RESTRICTED_TRIANGLE:BOOL="0"   ^
    -DLIBIGL_COPYLEFT_COMISO:BOOL="0" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/libigl-2.5.0

    TIMEOUT /T 1
    msbuild %pwd%\build\libigl-2.5.0\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    cd %pwd%
) 


if not exist %pwd%\install\flatbuffers-24.12.23 (
    echo  -----------build flatbuffers-24.12.23----------
    cmake  -G "Visual Studio 16 2019"    -B %pwd%\build\flatbuffers-24.12.23  -S %pwd%\flatbuffers-24.12.23    ^
    -DFLATBUFFERS_BUILD_TESTS:BOOL="0"  ^
    -DFLATBUFFERS_BUILD_SHAREDLIB:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/flatbuffers-24.12.23

    TIMEOUT /T 1
    msbuild %pwd%\build\flatbuffers-24.12.23\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    cd %pwd%
) 

if not exist %pwd%\install\ncnn-20250503 (
    echo  -----------build ncnn-20250503----------
    cmake  -G "Visual Studio 16 2019"    -B %pwd%\build\ncnn-20250503  -S %pwd%\ncnn-20250503   ^
    -DNCNN_BUILD_BENCHMARK:BOOL="0"  ^
    -DNCNN_BUILD_EXAMPLES:BOOL="0"   ^
    -DNCNN_SHARED_LIB:BOOL="1"   ^
    -DNCNN_ASAN:BOOL="0"   ^
    -DNCNN_BUILD_TESTS:BOOL="0"   ^
    -DNCNN_SYSTEM_GLSLANG:BOOL="0"   ^
    -Dprotobuf_DIR:PATH=%pwd%/install/protobuf-3.20.0-rc3/cmake   ^
    -DNCNN_BUILD_TOOLS:BOOL="1"   ^
    -DNCNN_VULKAN:BOOL="0"   ^
    -DProtobuf_LIBRARY_RELEASE:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotobuf.lib    ^
    -DProtobuf_PROTOC_EXECUTABLE:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/bin/protoc.exe    ^
    -DProtobuf_LITE_LIBRARY_DEBUG:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotobuf-lited.lib    ^
    -DProtobuf_LITE_LIBRARY_RELEASE:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotobuf-lite.lib    ^
    -DProtobuf_PROTOC_LIBRARY_RELEASE:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotoc.lib    ^
    -DProtobuf_PROTOC_LIBRARY_DEBUG:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotocd.lib    ^
    -DProtobuf_INCLUDE_DIR:PATH=%pwd%/install/protobuf-3.20.0-rc3/include"    ^
    -DProtobuf_LIBRARY_DEBUG:FILEPATH=%pwd%/install/protobuf-3.20.0-rc3/lib/libprotobufd.lib   ^
    -DZLIB_INCLUDE_DIR:PATH=%pwd%/install/zlib-1.2.13/include    ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlib.lib    ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/install/zlib-1.2.13/lib/zlib.lib   ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/ncnn-20250503
    TIMEOUT /T 1
    msbuild %pwd%\build\ncnn-20250503\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
rem    msbuild %pwd%\build\ncnn-20250503\INSTALL.vcxproj -t:Rebuild -p:Configuration=Debug
    cd %pwd%
) 



if not exist %pwd%\install\BA_exe (
    echo  -----------build BA_exe----------
    cmake  -G "Visual Studio 16 2019"    -B %pwd%build\BA_exe  -S %pwd%BA   ^
    -DCERES_INCLUDE_DIR:PATH=%pwd%install/ceres-solver-2.2.0/include   ^
    -DCERES_LIB_DIR:PATH=%pwd%install/ceres-solver-2.2.0/lib   ^
    -DGLOG_INCLUDE_DIR:PATH=%pwd%install/glog-0.7.1/include   ^
    -DGLOG_LIB_DIR:PATH=%pwd%install/glog-0.7.1/lib   ^
    -DJSONCPP_INCLUDE_DIR:PATH=%pwd%install/jsoncpp-1.9.6/include   ^
    -DJSONCPP_LIB_DIR:PATH=%pwd%install/jsoncpp-1.9.6/lib   ^
    -DEIGEN_INCLUDE_DIR:PATH=%pwd%install/eigen-3.4.0/include/eigen3   ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/BA_exe

    TIMEOUT /T 1
    msbuild %pwd%\build\BA_exe\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release
    cd %pwd%
) 

copy %pwd%\install\jsoncpp-1.9.6-release\bin\jsoncpp.dll  %pwd%install\BA_exe
copy %pwd%\install\glog-0.7.1\bin\glog.dll  %pwd%install\BA_exe
pause