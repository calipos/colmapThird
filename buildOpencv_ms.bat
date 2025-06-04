@echo off
set pwd=%~dp0
echo %pwd%
 

echo  ---------------------
echo  -----------need set QT_QPA_PLATFORM_PLUGIN_PATH----------
set QT_QPA_PLATFORM_PLUGIN_PATH=C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/plugins/platforms
::set path=%pwd2%lib/llvm-mingw-20240221-msvcrt-x86_64/bin;%path%   
echo %path% 
mkdir build
mkdir install


if not exist %pwd%\install\opencv480 (
    echo  -----------build opencv480----------
    cmake  -B %pwd%\build\opencv480  -S %pwd%\opencv480     ^
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
    -DINSTALL_PDB:BOOL="0"                             ^
    -DINSTALL_C_EXAMPLES:BOOL="0"                      ^
    -DINSTALL_PDB_COMPONENT_EXCLUDE_FROM_ALL:BOOL="0"  ^
    -DINSTALL_PYTHON_EXAMPLES:BOOL="0"                 ^
    -DINSTALL_TESTS:BOOL="0"                           ^
    -DCMAKE_BUILD_TYPE="Release"                       ^
    -DCMAKE_INSTALL_PREFIX:PATH=%pwd%install/opencv480 
    TIMEOUT /T 1
)
    rem msbuild %pwd%\build\opencv480\INSTALL.vcxproj -t:Rebuild -p:Configuration=Release    
    msbuild %pwd%\build\opencv480\INSTALL.vcxproj -p:Configuration=Release