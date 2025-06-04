@echo off
set pwd=%~dp0
echo %pwd%
 

echo  ---------------------
echo  -----------need set QT_QPA_PLATFORM_PLUGIN_PATH----------
set QT_QPA_PLATFORM_PLUGIN_PATH=C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/plugins/platforms
rem set path=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin;%path%   
echo %path% 
mkdir build
mkdir install

if not exist %pwd%\install\hdf5-hdf5-1_14_3 (
    echo  -----------build hdf5-hdf5-1_14_3----------
    cmake  -B %pwd%\build\hdf5-hdf5-1_14_3  -S %pwd%\hdf5-hdf5-1_14_3    ^
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
