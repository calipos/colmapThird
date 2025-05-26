@echo off
set pwd=%~dp0
echo %pwd%

 
echo %path% 
mkdir buildmingw
mkdir installmingw
 


if not exist %pwd%\installmingw\jsoncpp-1.9.6 (
    echo  -----------build jsoncpp-1.9.6----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\jsoncpp-1.9.6  -S %pwd%\jsoncpp-1.9.6 -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw\jsoncpp-1.9.6 ^
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
    cd  %pwd%\buildmingw\jsoncpp-1.9.6
    ninja install -j16
    cd %pwd%
)

rem if not exist %pwd%\installmingw\GKlib (
rem     echo  -----------build gklib----------
rem     cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\gklib  -S %pwd%\GKlib -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw ^
rem     -DGKREGEX:BOOL="1" -DGKRAND:BOOL="1" ^
rem     -DBUILD_SHARED_LIBS:BOOL="0" ^
rem     -DBUILD_TESTING:BOOL="0" ^
rem     -DCMAKE_BUILD_TYPE="Release" 
rem     cd %pwd%\buildmingw\gklib
rem     ninja install -j16
rem     cd %pwd%
rem )

 

if not exist %pwd%\installmingw\glog-0.7.1 (
    echo  -----------build glog-0.7.1----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\glog-0.7.1  -S %pwd%\glog-0.7.1 ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/glog-0.7.1  ^
    -DBUILD_TESTING:BOOL="0" ^
    -DWITH_GFLAGS:BOOL="0"  ^
    -DWITH_THREADS:BOOL="1"  ^
    -DBUILD_TESTING:BOOL="0" ^
    -DWITH_GTEST:BOOL="0" ^
    -DWITH_UNWIND:STRING="none" ^
    -DCMAKE_BUILD_TYPE="Release" 
    cd  %pwd%\buildmingw\glog-0.7.1
    ninja install -j16
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\eigen-3.4.0 (
    echo  -----------build eigen-3.4.0----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\eigen-3.4.0  -S %pwd%\eigen-3.4.0     ^
	-DEIGEN_TEST_SSE3:BOOL="0"  ^
    -DEIGEN_TEST_SSE4_1:BOOL="0"  ^
    -DEIGEN_TEST_SSE4_2:BOOL="0"  ^
    -DEIGEN_TEST_AVX512:BOOL="0"  ^
    -DEIGEN_TEST_AVX512DQ:BOOL="0"  ^
    -DEIGEN_TEST_AVX:BOOL="0"  ^
    -DEIGEN_TEST_AVX2:BOOL="0"  ^
    -DEIGEN_TEST_SSSE3:BOOL="0"  ^
    -DEIGEN_TEST_SSE2:BOOL="0"  ^
    -DCUDA_PROPAGATE_HOST_FLAGS:BOOL="0"  ^
    -DCUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE:BOOL="0"  ^
    -DEIGEN_TEST_OPENMP:BOOL="1"  ^
    -DCUDA_HOST_COMPILATION_CPP:BOOL="0"  ^
	-DCUDA_VERSION:STRING="11.7"  ^
	-DCUDA_BUILD_CUBIN:BOOL="0"  ^
	-DBUILD_TESTING:BOOL="0"  ^
	-DEIGEN_DOC_USE_MATHJAX:BOOL="0"  ^
	-DCUDA_USE_STATIC_CUDA_RUNTIME:BOOL="0"  ^
	-DCUDA_64_BIT_DEVICE_CODE:BOOL="0"  ^
	-DEIGEN_BUILD_DOC:BOOL="0"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/eigen-3.4.0 
    cd  %pwd%\buildmingw\eigen-3.4.0
    ninja install -j16
    cd %pwd%
)
rem =======================================================================
if not exist %pwd%\installmingw\PoseLib-2.0.4 (
    echo  -----------build PoseLib-2.0.4----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\PoseLib-2.0.4  -S %pwd%\PoseLib-2.0.4     ^
    -DBUILD_SHARED_LIBS:BOOL="0" ^
    -DEigen3_DIR:PATH="%pwd%installmingw/eigen-3.4.0/share/eigen3/cmake"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH="%pwd%installmingw/PoseLib-2.0.4"
    cd  %pwd%\buildmingw\PoseLib-2.0.4
    ninja install -j16
    cd %pwd%
)
rem =======================================================================



if not exist %pwd%\installmingw\cgal-5.6.1 (
    echo  -----------build cgal-5.6.1----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\cgal-5.6.1  -S %pwd%\cgal-5.6.1     ^
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
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/cgal-5.6.1 
    cd  %pwd%\buildmingw\cgal-5.6.1
    ninja install -j16
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\opencv480 (
    echo  -----------build opencv480----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\opencv480  -S %pwd%\opencv480     ^
    -DWITH_OBSENSOR=OFF  ^
    -DBUILD_opencv_apps:BOOL="0" ^
    -DBUILD_WITH_DEBUG_INFO:BOOL="0" ^
    -DBUILD_opencv_flann:BOOL="1" ^
    -DBUILD_opencv_world:BOOL="1" ^
    -DBUILD_opencv_gapi:BOOL="1" ^
    -DCPU_BASELINE:STRING="SSSE3" ^
    -DCPU_DISPATCH:STRING="SSSE3" ^
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
    -DEigen3_DIR:PATH="%pwd%installmingw/eigen-3.4.0/share/eigen3/cmake"  ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/opencv480 
    cd  %pwd%\buildmingw\opencv480
    ninja install -j12
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\lz4-1.9.4 (
    echo  -----------build lz4-1.9.4----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\lz4-1.9.4  -S %pwd%\lz4-1.9.4\build\cmake     ^
    -DBUILD_SHARED_LIBS:BOOL="0" ^
    -DBUILD_STATIC_LIBS:BOOL="1" ^
	-DBUILD_TESTS:BOOL="0"  ^
	-DBUILD_EXAMPLES:BOOL="0"   ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/lz4-1.9.4
    cd  %pwd%\buildmingw\lz4-1.9.4
    ninja install -j16
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\zlib-1.2.13 (
    echo  -----------build zlib-1.2.13----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\zlib-1.2.13  -S %pwd%\zlib-1.2.13     ^
	-DBUILD_TESTS:BOOL="0"  ^
	-DBUILD_EXAMPLES:BOOL="0"   ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/zlib-1.2.13
    cd  %pwd%\buildmingw\zlib-1.2.13
    ninja install -j16
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\hdf5-hdf5-1_14_3 (
    echo  -----------build hdf5-hdf5-1_14_3----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\hdf5-hdf5-1_14_3  -S %pwd%\hdf5-hdf5-1_14_3    ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/installmingw/zlib-1.2.13/lib/zlibstatic.lib  ^
    -DHDF5_USE_FILE_LOCKING:BOOL="1"  ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DH5_HAVE_ALARM:INTERNAL=0  ^
	-DH5_HAVE_ASPRINTF:INTERNAL=0 ^
	-DH5_HAVE_VASPRINTF:INTERNAL=0 ^
    -DHDF5_TEST_CPP:BOOL="0"  ^
    -DHDF5_BUILD_TOOLS:BOOL="0"  ^
    -DHDF5_TEST_FORTRAN:BOOL="0"  ^
    -DHDF5_TEST_EXAMPLES:BOOL="0"  ^
    -DZLIB_DIR:PATH=%pwd%/installmingw/zlib-1.2.13/lib ^
    -DHDF5_TEST_PARALLEL:BOOL="0"  ^
    -DHDF5_TEST_SWMR:BOOL="0"  ^
    -DHDF5_BUILD_CPP_LIB:BOOL="1"  ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/installmingw/zlib-1.2.13/lib/zlibstatic.lib  ^
    -DHDF5_BUILD_EXAMPLES:BOOL="0"  ^
    -DHDF5_TEST_JAVA:BOOL="0"  ^
    -DBUILD_TESTING:BOOL="0"  ^
    -DZLIB_USE_EXTERNAL:BOOL="1"  ^
    -DHDF5_TEST_SERIAL:BOOL="0"  ^
    -DHDF5_TEST_TOOLS:BOOL="0"  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/hdf5-hdf5-1_14_3
    cd  %pwd%\buildmingw\hdf5-hdf5-1_14_3
    ninja install -j16
    cd %pwd%
)

if not exist %pwd%\installmingw\ceres-solver-2.2.0 (
    echo  -----------build ceres-solver-2.2.0----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\ceres-solver-2.2.0  -S %pwd%\ceres-solver-2.2.0    ^
    -DEigen3_DIR:PATH=%pwd%installmingw/eigen-3.4.0/share/eigen3/cmake ^
    -DCMAKE_CXX_FLAGS:STRING="-std=gnu++17" ^
    -DUSE_CUDA:BOOL="0"  ^
    -DBUILD_SHARED_LIBS:BOOL="1" ^
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
    -Dglog_DIR:PATH=%pwd%installmingw/glog-0.7.1/lib/cmake/glog  ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/ceres-solver-2.2.0
    cd  %pwd%\buildmingw\ceres-solver-2.2.0
    ninja install -j16 -v
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\flann-1.9.2 (
    echo  -----------build flann-1.9.2----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\flann-1.9.2  -S %pwd%\flann-1.9.2    ^
    -DHDF5_DIR:PATH=%pwd%/installmingw/hdf5-hdf5-1_14_3/cmake ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%/installmingw/lz4-1.9.4/include ^
    -DLZ4_LINK_LIBRARIES:FILEPATH=%pwd%/installmingw/lz4-1.9.4/lib/liblz4.a  ^
    -DBUILD_TESTS:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="0" ^
    -DUSE_OPENMP:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/flann-1.9.2
    cd  %pwd%\buildmingw\flann-1.9.2
    ninja install -j16
    cd %pwd%
)
rem =======================================================================

if not exist %pwd%\installmingw\glew-2.1.0 (
    echo  -----------build glew-2.1.0----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\glew-2.1.0  -S %pwd%\glew-2.1.0\build\cmake    ^
    -DHDF5_DIR:PATH=%pwd%/installmingw/hdf5-hdf5-1_14_3/cmake ^
    -DLZ4_INCLUDE_DIRS:PATH=%pwd%/installmingw/lz4-1.9.4/include ^
    -DLZ4_LINK_LIBRARIES:FILEPATH=%pwd%/installmingw/lz4-1.9.4/lib/lz4.a  ^
    -DBUILD_TESTS:BOOL="0" ^
    -DBUILD_DOC:BOOL="0" ^
    -DBUILD_PYTHON_BINDINGS:BOOL="0" ^
    -DBUILD_MATLAB_BINDINGS:BOOL="0" ^
    -DBUILD_EXAMPLES:BOOL="0" ^
    -DUSE_OPENMP:BOOL="0" ^
    -DCMAKE_BUILD_TYPE="Release" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/glew-2.1.0
    cd  %pwd%\buildmingw\glew-2.1.0
    ninja install -j16
    cd %pwd%
) 
rem =======================================================================

if not exist %pwd%\installmingw\spdlog-1.15.0 (
    echo  -----------build spdlog-1.15.0----------
    cmake -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++  -B %pwd%\buildmingw\spdlog-1.15.0  -S %pwd%\spdlog-1.15.0    ^
    -DSPDLOG_BUILD_EXAMPLE_HO:BOOL="0"  ^
    -DSPDLOG_BUILD_WARNINGS:BOOL="0"  ^
    -DCMAKE_EXPORT_BUILD_DATABASE:BOOL="0"  ^
    -DSPDLOG_BUILD_EXAMPLE:BOOL="0"  ^
    -DSPDLOG_BUILD_SHARED:BOOL="0"  ^
    -DSPDLOG_BUILD_ALL:BOOL="0"  ^
    -DSPDLOG_MSVC_UTF8:BOOL="0"  ^
    -DSPDLOG_BUILD_TESTS_HO:BOOL="0"  ^
    -DSPDLOG_BUILD_TESTS:BOOL="0"  ^
    -DSPDLOG_BUILD_PIC:BOOL="0"  ^
    -DSPDLOG_BUILD_BENCH:BOOL="0" ^
    -DSPDLOG_BUILD_SHARED:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/spdlog-1.15.0 
    cd  %pwd%\buildmingw\spdlog-1.15.0 
    ninja install -j16
    cd %pwd%
) 




if not exist %pwd%\installmingw\libigl-2.5.0 (
    echo  -----------build libigl-2.5.0----------
    cmake  -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++    -B %pwd%\buildmingw\libigl-2.5.0  -S %pwd%\libigl-2.5.0    ^
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
    -DFETCHCONTENT_SOURCE_DIR_EIGEN:PATH="D:/repo/colmapThird/installmingw/eigen-3.4.0/include/eigen3"   ^
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
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/libigl-2.5.0

    cd  %pwd%\buildmingw\libigl-2.5.0
    ninja install -j16
    cd %pwd%
) 




if not exist %pwd%\installmingw\flatbuffers-24.12.23 (
    echo  -----------build flatbuffers-24.12.23----------
    cmake  -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++    -B %pwd%\buildmingw\flatbuffers-24.12.23  -S %pwd%\flatbuffers-24.12.23    ^
    -DFLATBUFFERS_BUILD_TESTS:BOOL="0"  ^
    -DFLATBUFFERS_BUILD_SHAREDLIB:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/flatbuffers-24.12.23
    cd  %pwd%\buildmingw\flatbuffers-24.12.23
    ninja install -j16
    cd %pwd%
) 


if not exist %pwd%\installmingw\cereal-1.3.2 (
    echo  -----------build cereal-1.3.2----------
    cmake  -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++    -B %pwd%\buildmingw\cereal-1.3.2  -S %pwd%\cereal-1.3.2    ^
    -DBUILD_SANDBOX:BOOL="0" ^
    -DBoost_SERIALIZATION_LIBRARY_RELEASE:FILEPATH="D:/ucl360/libraries-mgw/boost_1.85.0/lib/libboost_serialization-mgw14-mt-s-x64-1_85.a" ^
    -DBoost_INCLUDE_DIR:PATH="D:/ucl360/libraries-mgw/boost_1.85.0/include/boost-1_85" ^
    -DBoost_SERIALIZATION_LIBRARY_DEBUG:FILEPATH="D:/ucl360/libraries-mgw/boost_1.85.0/lib/libboost_serialization-mgw14-mt-sd-x64-1_85.a"  ^
    -DBUILD_TESTS:BOOL="0"  ^
    -DBUILD_DOC:BOOL="0" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/cereal-1.3.2
    cd  %pwd%\buildmingw\cereal-1.3.2
    ninja install -j16
    cd %pwd%
) 


if not exist %pwd%\installmingw\protobuf-3.20.0-rc3 (
    echo  -----------build protobuf-3.20.0-rc3----------
    cmake  -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++    -B %pwd%\buildmingw\protobuf-3.20.0-rc3  -S %pwd%\protobuf-3.20.0-rc3\cmake    ^
    -Dprotobuf_BUILD_TESTS:BOOL="0"    ^
    -DZLIB_LIBRARY_RELEASE:FILEPATH=%pwd%/installmingw/zlib-1.2.13/lib/libzlib.dll.a  ^
    -DZLIB_INCLUDE_DIR:PATH=%pwd%/installmingw/zlib-1.2.13/include  ^
    -DZLIB_LIBRARY_DEBUG:FILEPATH=%pwd%/installmingw/zlib-1.2.13/lib/libzlib.dll.a  ^
    -Dprotobuf_BUILD_SHARED_LIBS:BOOL="1"   ^
    -Dprotobuf_WITH_ZLIB:BOOL="1" ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/protobuf-3.20.0-rc3
    cd  %pwd%\buildmingw\protobuf-3.20.0-rc3
    ninja install -j16
    cd %pwd%
) 




if not exist %pwd%\installmingw\BA_exe (
    echo  -----------build BA_exe----------
    cmake  -G Ninja -DCMAKE_C_COMPILER=gcc  -DCMAKE_CXX_COMPILER=g++     -B %pwd%buildmingw\BA_exe  -S %pwd%BA   ^
    -DCERES_INCLUDE_DIR:PATH=%pwd%installmingw/ceres-solver-2.2.0/include   ^
    -DCERES_LIB_DIR:PATH=%pwd%installmingw/ceres-solver-2.2.0/lib   ^
    -DGLOG_INCLUDE_DIR:PATH=%pwd%installmingw/glog-0.7.1/include   ^
    -DGLOG_LIB_DIR:PATH=%pwd%installmingw/glog-0.7.1/lib   ^
    -DJSONCPP_INCLUDE_DIR:PATH=%pwd%installmingw/jsoncpp-1.9.6/include   ^
    -DJSONCPP_LIB_DIR:PATH=%pwd%installmingw/jsoncpp-1.9.6/lib   ^
    -DEIGEN_INCLUDE_DIR:PATH=%pwd%installmingw/eigen-3.4.0/include/eigen3   ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%installmingw/BA_exe
    cd  %pwd%\buildmingw\BA_exe
    ninja install -j16
    cd %pwd%
) 
copy %pwd%\installmingw\glog-0.7.1\bin\libglog.dll  %pwd%\installmingw\BA_exe
copy %pwd%\installmingw\ceres-solver-2.2.0\bin\libceres.dll  %pwd%\installmingw\BA_exe
pause