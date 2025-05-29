# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


cmake_minimum_required(VERSION 3.10)

################################################################################
# Options
################################################################################

option(SIMD_ENABLED "Whether to enable SIMD optimizations" ON)
option(OPENMP_ENABLED "Whether to enable OpenMP parallelization" ON)
option(IPO_ENABLED "Whether to enable interprocedural optimization" ON)
option(CUDA_ENABLED "Whether to enable CUDA, if available" OFF)
option(GUI_ENABLED "Whether to enable the graphical UI" ON)
option(OPENGL_ENABLED "Whether to enable OpenGL, if available" ON)
option(TESTS_ENABLED "Whether to build test binaries" OFF)
option(ASAN_ENABLED "Whether to enable AddressSanitizer flags" OFF)
option(PROFILING_ENABLED "Whether to enable google-perftools linker flags" OFF)
option(CCACHE_ENABLED "Whether to enable compiler caching, if available" OFF)
option(CGAL_ENABLED "Whether to enable the CGAL library" OFF)
option(LSD_ENABLED "Whether to enable the LSD library" OFF)
option(UNINSTALL_ENABLED "Whether to create a target to 'uninstall' colmap" ON)
option(FETCH_POSELIB "Whether to consume PoseLib using FetchContent or find_package" OFF)
option(ALL_SOURCE_TARGET "Whether to create a target for all source files (for Visual Studio / XCode development)" OFF)

# Propagate options to vcpkg manifest.
if(TESTS_ENABLED)
  enable_testing()
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()
if(CUDA_ENABLED)
    list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
endif()
if(GUI_ENABLED)
    list(APPEND VCPKG_MANIFEST_FEATURES "gui")
endif()
if(CGAL_ENABLED)
    list(APPEND VCPKG_MANIFEST_FEATURES "cgal")
endif()

if(LSD_ENABLED)
    message(STATUS "Enabling LSD support")
    add_definitions("-DCOLMAP_LSD_ENABLED")
else()
    message(STATUS "Disabling LSD support")
endif()

project(COLMAP LANGUAGES C CXX)

set(COLMAP_VERSION "3.11.1")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set_property(GLOBAL PROPERTY GLOBAL_DEPENDS_NO_CYCLES ON)

################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXCompilerFlag)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)

# Build position-independent code, so that shared libraries can link against
# COLMAP's static libraries.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Dependency configuration
################################################################################

set(COLMAP_FIND_QUIETLY FALSE)
#include(cmake/FindDependencies.cmake)

################################################################################
# Compiler specific configuration
################################################################################

if(CMAKE_BUILD_TYPE)
    message(STATUS "Build type specified as ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "Build type not specified, using Release")
    set(CMAKE_BUILD_TYPE Release)
    set(IS_DEBUG OFF)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "ClangTidy")
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(NOT CLANG_TIDY_EXE)
        message(FATAL_ERROR "Could not find the clang-tidy executable, please set CLANG_TIDY_EXE")
    endif()
else()
    unset(CLANG_TIDY_EXE)
endif()

add_definitions("-DIDXTYPEWIDTH=32")
add_definitions("-DREALTYPEWIDTH=32")
add_definitions("-DCOLMAP_GUI_ENABLED")
if(IS_MSVC)
    # Some fixes for the Glog library.
    add_definitions("-DGLOG_USE_GLOG_EXPORT")
    add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES")
    add_definitions("-DGL_GLEXT_PROTOTYPES")
    add_definitions("-DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc  /bigobj")
    # Disable warning: 'initializing': conversion from 'X' to 'Y', possible loss of data
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4267 /wd4305")
    # Enable object level parallel builds in Visual Studio.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
    endif()
endif()

if(IS_GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version 4.8 or older not supported")
    endif()

    # Hide incorrect warnings for uninitialized Eigen variables under GCC.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-maybe-uninitialized")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

if(IS_MACOS)
    # Mitigate CMake limitation, see: https://discourse.cmake.org/t/avoid-duplicate-linking-to-avoid-xcode-15-warnings/9084/10
    add_link_options(LINKER:-no_warn_duplicate_libraries)
endif()

if(IS_DEBUG)
    add_definitions("-DEIGEN_INITIALIZE_MATRICES_BY_NAN")
endif()

if(SIMD_ENABLED)
    message(STATUS "Enabling SIMD support")
else()
    message(STATUS "Disabling SIMD support")
endif()

if(IPO_ENABLED AND NOT IS_DEBUG AND NOT IS_GNU)
    message(STATUS "Enabling interprocedural optimization")
    set_property(DIRECTORY PROPERTY INTERPROCEDURAL_OPTIMIZATION 1)
else()
    message(STATUS "Disabling interprocedural optimization")
endif()

if(ASAN_ENABLED)
    message(STATUS "Enabling ASan support")
    if(IS_CLANG OR IS_GNU)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
        add_link_options(-fsanitize=address)
    else()
        message(FATAL_ERROR "Unsupported compiler for ASan mode")
    endif()
endif()

if(CCACHE_ENABLED)
    find_program(CCACHE ccache)
    if(CCACHE)
        message(STATUS "Enabling ccache support")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    else()
        message(STATUS "Disabling ccache support")
    endif()
else()
    message(STATUS "Disabling ccache support")
endif()

if(PROFILING_ENABLED)
    message(STATUS "Enabling profiling support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -lprofiler -ltcmalloc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
else()
    message(STATUS "Disabling profiling support")
endif()

if(NOT DEFINED Eigen3_DIR)
set(Eigen3_DIR D:/repo/colmapThird/install/eigen-3.4.0/share/eigen3/cmake)
message(STATUS "use default Eigen3_DIR=${Eigen3_DIR}")
else()
message(STATUS "use Eigen3_DIR=${Eigen3_DIR}")
endif()
find_package(Eigen3)


if(NOT DEFINED JSONCPP_INCLUDE_DIRS)
set(JSONCPP_INCLUDE_DIRS D:/repo/colmapThird/install/jsoncpp-1.9.6/include)
message(STATUS "use default JSONCPP_INCLUDE_DIRS=${JSONCPP_INCLUDE_DIRS}")
else()
message(STATUS "use JSONCPP_INCLUDE_DIRS=${JSONCPP_INCLUDE_DIRS}")
endif()
if(NOT DEFINED JSONCPP_LIBRARIES_DIRS)
set(JSONCPP_LIBRARIES_DIRS D:/repo/colmapThird/install/jsoncpp-1.9.6/lib)
message(STATUS "use default JSONCPP_LIBRARIES_DIRS=${JSONCPP_LIBRARIES_DIRS}")
else()
message(STATUS "use JSONCPP_LIBRARIES_DIRS=${JSONCPP_LIBRARIES_DIRS}")
endif()

if(NOT DEFINED CERES_INCLUDE_DIRS)
set(CERES_INCLUDE_DIRS D:/repo/colmapThird/install/ceres-solver-2.2.0/include)
message(STATUS "use default CERES_INCLUDE_DIRS=${CERES_INCLUDE_DIRS}")
else()
message(STATUS "use CERES_INCLUDE_DIRS=${CERES_INCLUDE_DIRS}")
endif()
if(NOT DEFINED CERES_LIBRARIES)
set(CERES_LIBRARIES D:/repo/colmapThird/install/ceres-solver-2.2.0/lib/ceres.lib)
message(STATUS "use default CERES_LIBRARIES=${CERES_LIBRARIES}")
else()
message(STATUS "use CERES_LIBRARIES=${CERES_LIBRARIES}")
endif()
add_library(Ceres::ceres INTERFACE IMPORTED)
include_directories(${CERES_INCLUDE_DIRS})
target_include_directories(Ceres::ceres INTERFACE ${CERES_INCLUDE_DIRS})
target_link_libraries(Ceres::ceres INTERFACE ${CERES_LIBRARIES})

if(NOT DEFINED GLOG_INCLUDE_DIRS)
set(GLOG_INCLUDE_DIRS D:/repo/colmapThird/install/glog-0.7.1/include)
message(STATUS "use default GLOG_INCLUDE_DIRS=${GLOG_INCLUDE_DIRS}")
else()
message(STATUS "use GLOG_INCLUDE_DIRS=${GLOG_INCLUDE_DIRS}")
endif()
if(NOT DEFINED GLOG_LIBRARIES)
set(GLOG_LIBRARIES D:/repo/colmapThird/install/glog-0.7.1/lib/glog.lib)
message(STATUS "use default GLOG_LIBRARIES=${GLOG_LIBRARIES}")
else()
message(STATUS "use GLOG_LIBRARIES=${GLOG_LIBRARIES}")
endif()
add_library(glog::glog INTERFACE IMPORTED)
include_directories(${GLOG_INCLUDE_DIRS})
target_include_directories(glog::glog INTERFACE ${GLOG_INCLUDE_DIRS})
target_link_libraries(glog::glog INTERFACE ${GLOG_LIBRARIES})


if(NOT DEFINED BOOST_INCLUDE_DIRS)
set(BOOST_INCLUDE_DIRS D:/ucl360/library2019share/boost185/include/boost-1_85)
message(STATUS "use default BOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}")
else()
message(STATUS "use BOOST_INCLUDE_DIRS=${BOOST_INCLUDE_DIRS}")
endif()
if(NOT DEFINED BOOST_LIBRARIES_DIR)
set(BOOST_LIBRARIES_DIR D:/ucl360/library2019share/boost185/lib)
message(STATUS "use default BOOST_LIBRARIES_DIR=${BOOST_LIBRARIES_DIR}")
else()
message(STATUS "use BOOST_LIBRARIES_DIR=${BOOST_LIBRARIES_DIR}")
endif()
if(NOT DEFINED BOOST_PROGRAM_OPTIONS_LIBRARIES)
set(BOOST_PROGRAM_OPTIONS_LIBRARIES D:/ucl360/library2019share/boost185/lib/libboost_program_options-vc142-mt-s-x64-1_85.lib)
message(STATUS "use default BOOST_PROGRAM_OPTIONS_LIBRARIES=${BOOST_PROGRAM_OPTIONS_LIBRARIES}")
else()
message(STATUS "use BOOST_PROGRAM_OPTIONS_LIBRARIES=${BOOST_PROGRAM_OPTIONS_LIBRARIES}")
endif()
if(NOT DEFINED BOOST_GRAPH_LIBRARIES)
set(BOOST_GRAPH_LIBRARIES D:/ucl360/library2019share/boost185/lib/libboost_graph-vc142-mt-s-x64-1_85.lib)
message(STATUS "use default BOOST_GRAPH_LIBRARIES=${BOOST_GRAPH_LIBRARIES}")
else()
message(STATUS "use BOOST_GRAPH_LIBRARIES=${BOOST_GRAPH_LIBRARIES}")
endif()
if(NOT DEFINED BOOST_SYSTEM_LIBRARIES)
set(BOOST_SYSTEM_LIBRARIES D:/ucl360/library2019share/boost185/lib/libboost_system-vc142-mt-s-x64-1_85.lib)
message(STATUS "use default BOOST_SYSTEM_LIBRARIES=${BOOST_SYSTEM_LIBRARIES}")
else()
message(STATUS "use BOOST_SYSTEM_LIBRARIES=${BOOST_SYSTEM_LIBRARIES}")
endif()
add_library(Boost::boost INTERFACE IMPORTED)
add_library(Boost::program_options INTERFACE IMPORTED)
add_library(Boost::graph INTERFACE IMPORTED)
add_library(Boost::system INTERFACE IMPORTED)
include_directories(${BOOST_INCLUDE_DIRS})
target_include_directories(Boost::boost INTERFACE ${BOOST_INCLUDE_DIRS})
target_include_directories(Boost::program_options INTERFACE ${BOOST_INCLUDE_DIRS})
target_include_directories(Boost::graph INTERFACE ${BOOST_INCLUDE_DIRS})
target_include_directories(Boost::system INTERFACE ${BOOST_INCLUDE_DIRS})
link_directories(${BOOST_LIBRARIES_DIR}) 
#target_link_libraries(Boost::program_options INTERFACE ${BOOST_PROGRAM_OPTIONS_LIBRARIES})
#target_link_libraries(Boost::graph INTERFACE ${BOOST_GRAPH_LIBRARIES})
#target_link_libraries(Boost::system INTERFACE ${BOOST_SYSTEM_LIBRARIES})

if(NOT DEFINED POSELIB_INCLUDE_DIRS)
set(POSELIB_INCLUDE_DIRS D:/repo/colmapThird/install/PoseLib-2.0.4/include)
message(STATUS "use default POSELIB_INCLUDE_DIRS=${POSELIB_INCLUDE_DIRS}")
else()
message(STATUS "use POSELIB_INCLUDE_DIRS=${POSELIB_INCLUDE_DIRS}")
endif()
if(NOT DEFINED POSELIB_LIBRARIES)
set(POSELIB_LIBRARIES D:/repo/colmapThird/install/PoseLib-2.0.4/lib/PoseLib.lib)
message(STATUS "use default POSELIB_LIBRARIES=${POSELIB_LIBRARIES}")
else()
message(STATUS "use POSELIB_LIBRARIES=${POSELIB_LIBRARIES}")
endif()
add_library(PoseLib::PoseLib INTERFACE IMPORTED)
include_directories(${POSELIB_INCLUDE_DIRS})
target_include_directories(PoseLib::PoseLib INTERFACE ${POSELIB_INCLUDE_DIRS})
target_link_libraries(PoseLib::PoseLib INTERFACE ${POSELIB_LIBRARIES})

if(NOT DEFINED SQLITE_INCLUDE_DIRS)
set(SQLITE_INCLUDE_DIRS D:/repo/colmapThird/install/sqlite-amalgamation-3460100)
message(STATUS "use default SQLITE_INCLUDE_DIRS=${SQLITE_INCLUDE_DIRS}")
else()
message(STATUS "use SQLITE_INCLUDE_DIRS=${SQLITE_INCLUDE_DIRS}")
endif()
if(NOT DEFINED SQLITE_LIBRARIES)
set(SQLITE_LIBRARIES D:/repo/colmapThird/install/sqlite-amalgamation-3460100/sqlite3.lib)
message(STATUS "use default SQLITE_LIBRARIES=${SQLITE_LIBRARIES}")
else()
message(STATUS "use SQLITE_LIBRARIES=${SQLITE_LIBRARIES}")
endif()
add_library(SQLite::SQLite3 INTERFACE IMPORTED)
include_directories(${SQLITE_INCLUDE_DIRS})
target_include_directories(SQLite::SQLite3 INTERFACE ${SQLITE_INCLUDE_DIRS})
target_link_libraries(SQLite::SQLite3 INTERFACE ${SQLITE_LIBRARIES})


if(NOT DEFINED FREEIMAGE_INCLUDE_DIRS)
set(FREEIMAGE_INCLUDE_DIRS D:/repo/colmapThird/FreeImage3180Win32Win64/x64)
message(STATUS "use default FREEIMAGE_INCLUDE_DIRS=${FREEIMAGE_INCLUDE_DIRS}")
else()
message(STATUS "use FREEIMAGE_INCLUDE_DIRS=${FREEIMAGE_INCLUDE_DIRS}")
endif()
if(NOT DEFINED FREEIMAGE_LIBRARIES)
set(FREEIMAGE_LIBRARIES D:/repo/colmapThird/FreeImage3180Win32Win64/x64/FreeImage.lib)
message(STATUS "use default FREEIMAGE_LIBRARIES=${FREEIMAGE_LIBRARIES}")
else()
message(STATUS "use FREEIMAGE_LIBRARIES=${FREEIMAGE_LIBRARIES}")
endif()
add_library(freeimage::FreeImage INTERFACE IMPORTED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})
target_include_directories(freeimage::FreeImage INTERFACE ${FREEIMAGE_INCLUDE_DIRS})
target_link_libraries(freeimage::FreeImage INTERFACE ${FREEIMAGE_LIBRARIES})


#set(Qt5_INCLUDE_DIRS C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/include)
#set(Qt5_Core_LIBRARIES C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/lib/Qt5Core.lib)
#set(Qt5_OpenGL_LIBRARIES C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/lib/Qt5OpenGL.lib)
#set(Qt5_Widgets_LIBRARIES C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/lib/Qt5Widgets.lib)
#add_library(Qt5::Core INTERFACE IMPORTED)
#include_directories(${Qt5_INCLUDE_DIRS})
#target_include_directories(Qt5::Core INTERFACE ${Qt5_INCLUDE_DIRS})
#target_link_libraries(Qt5::Core INTERFACE ${Qt5_Core_LIBRARIES})
#add_library(Qt5::OpenGL INTERFACE IMPORTED)
#target_include_directories(Qt5::OpenGL INTERFACE ${Qt5_INCLUDE_DIRS})
#target_link_libraries(Qt5::OpenGL INTERFACE ${Qt5_OpenGL_LIBRARIES})
#add_library(Qt5::Widgets INTERFACE IMPORTED)
#target_include_directories(Qt5::Widgets INTERFACE ${Qt5_INCLUDE_DIRS})
#target_link_libraries(Qt5::Widgets INTERFACE ${Qt5_Widgets_LIBRARIES})
#include_directories(${Qt5_INCLUDE_DIRS}/QtCore)
#include_directories(${Qt5_INCLUDE_DIRS}/QtOpenGL)
#include_directories(${Qt5_INCLUDE_DIRS}/QtGui)
#include_directories(${Qt5_INCLUDE_DIRS}/QtWidgets)


set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL ${COLMAP_FIND_TYPE})


if(NOT DEFINED FLANN_DIR_INCLUDE_DIRS)
set(FLANN_DIR_INCLUDE_DIRS D:/repo/colmapThird/install/flann-1.9.2/include)
message(STATUS "use default FLANN_DIR_INCLUDE_DIRS=${FLANN_DIR_INCLUDE_DIRS}")
else()
message(STATUS "use FLANN_DIR_INCLUDE_DIRS=${FLANN_DIR_INCLUDE_DIRS}")
endif()
if(NOT DEFINED FLANN_DIR_LIBRARIES)
set(FLANN_DIR_LIBRARIES D:/repo/colmapThird/install/flann-1.9.2/lib/flann_cpp_s.lib)
message(STATUS "use default FLANN_DIR_LIBRARIES=${FLANN_DIR_LIBRARIES}")
else()
message(STATUS "use FLANN_DIR_LIBRARIES=${FLANN_DIR_LIBRARIES}")
endif()
add_library(flann INTERFACE IMPORTED)
include_directories(${FLANN_DIR_INCLUDE_DIRS})
target_include_directories(flann INTERFACE ${FLANN_DIR_INCLUDE_DIRS})
target_link_libraries(flann INTERFACE ${FLANN_DIR_LIBRARIES})


if(NOT DEFINED LZ4_DIR_INCLUDE_DIRS)
set(LZ4_DIR_INCLUDE_DIRS D:/repo/colmapThird/install/lz4-1.9.4/include)
message(STATUS "use default LZ4_DIR_INCLUDE_DIRS=${LZ4_DIR_INCLUDE_DIRS}")
else()
message(STATUS "use LZ4_DIR_INCLUDE_DIRS=${LZ4_DIR_INCLUDE_DIRS}")
endif()
if(NOT DEFINED LZ4_DIR_LIBRARIES)
set(LZ4_DIR_LIBRARIES D:/repo/colmapThird/install/lz4-1.9.4/lib/lz4.lib)
message(STATUS "use default LZ4_DIR_LIBRARIES=${LZ4_DIR_LIBRARIES}")
else()
message(STATUS "use LZ4_DIR_LIBRARIES=${LZ4_DIR_LIBRARIES}")
endif()
add_library(lz4 INTERFACE IMPORTED)
include_directories(${LZ4_DIR_INCLUDE_DIRS})
target_include_directories(lz4 INTERFACE ${LZ4_DIR_INCLUDE_DIRS})
target_link_libraries(lz4 INTERFACE ${LZ4_DIR_LIBRARIES})


if(NOT DEFINED METIS_DIR_INCLUDE_DIRS)
set(METIS_DIR_INCLUDE_DIRS D:/repo/colmapThird/install/METIS-5.2.1.1/include)
message(STATUS "use default METIS_DIR_INCLUDE_DIRS=${METIS_DIR_INCLUDE_DIRS}")
else()
message(STATUS "use METIS_DIR_INCLUDE_DIRS=${METIS_DIR_INCLUDE_DIRS}")
endif()
if(NOT DEFINED METIS_LIBRARIES)
set(METIS_LIBRARIES D:/repo/colmapThird/install/METIS-5.2.1.1/lib/metis.lib)
message(STATUS "use default METIS_LIBRARIES=${METIS_LIBRARIES}")
else()
message(STATUS "use METIS_LIBRARIES=${METIS_LIBRARIES}")
endif()
if(NOT DEFINED GKLIB_LIBRARIES)
set(GKLIB_LIBRARIES D:/repo/colmapThird/install/GKlib/lib/GKlib.lib)
message(STATUS "use default GKLIB_LIBRARIES=${GKLIB_LIBRARIES}")
else()
message(STATUS "use GKLIB_LIBRARIES=${GKLIB_LIBRARIES}")
endif()
add_library(metis INTERFACE IMPORTED)
include_directories(${METIS_DIR_INCLUDE_DIRS})
target_include_directories(metis INTERFACE ${METIS_DIR_INCLUDE_DIRS})
target_link_libraries(metis INTERFACE ${METIS_LIBRARIES})
target_link_libraries(metis INTERFACE ${GKLIB_LIBRARIES})

if(NOT DEFINED Qt5_DIRS)
set(Qt5_DIRS C:/Qt/Qt5.14.2/5.14.2/msvc2017_64/include)
message(STATUS "use default Qt5_DIRS=${Qt5_DIRS}")
else()
message(STATUS "use Qt5_DIRS=${Qt5_DIRS}")
endif()

if(GUI_ENABLED)
    find_package(Qt5 5.4 ${COLMAP_FIND_TYPE} COMPONENTS Core OpenGL Widgets)
    message(STATUS "Found Qt")
    message(STATUS "  Module : ${Qt5Core_DIR}")
    message(STATUS "  Module : ${Qt5OpenGL_DIR}")
    message(STATUS "  Module : ${Qt5Widgets_DIR}")
    if(Qt5_FOUND)
        # Qt5 was built with -reduce-relocations.
        if(Qt5_POSITION_INDEPENDENT_CODE)
            set(CMAKE_POSITION_INDEPENDENT_CODE ON)
            # Workaround for Qt5 CMake config bug under Ubuntu 20.04: https://gitlab.kitware.com/cmake/cmake/-/issues/16915
            if(TARGET Qt5::Core)
                get_property(core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS)
                string(REPLACE "-fPIC" "" new_qt5_core_options "${core_options}")
                set_property(TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_qt5_core_options})
                set_property(TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON")
                if(NOT IS_MSVC)
                    set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC")
                endif()
            endif()
        endif()

        # Enable automatic compilation of Qt resource files.
        set(CMAKE_AUTORCC ON)
    endif()
endif()

if(GUI_ENABLED AND Qt5_FOUND)
    add_definitions("-DCOLMAP_GUI_ENABLED")
    message(STATUS "Enabling GUI support")
else()
    set(GUI_ENABLED OFF)
    message(STATUS "Disabling GUI support")
endif()

if(OPENGL_ENABLED)
    if(NOT GUI_ENABLED)
        message(STATUS "Disabling GUI also disables OpenGL")
        set(OPENGL_ENABLED OFF)
    else()
        add_definitions("-DCOLMAP_OPENGL_ENABLED")
        message(STATUS "Enabling OpenGL support")
    endif()
else()
    message(STATUS "Disabling OpenGL support")
endif()


# Generate source file with version definitions.
include(GenerateVersionDefinitions)

include_directories(src)
link_directories(${COLMAP_LINK_DIRS})

add_subdirectory(src/colmap)
add_subdirectory(src/thirdparty)

################################################################################
# Generate source groups for Visual Studio, XCode, etc.
################################################################################

COLMAP_ADD_SOURCE_DIR(src/colmap/controllers CONTROLLERS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/estimators ESTIMATORS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/exe EXE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/feature FEATURE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/geometry GEOMETRY_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/image IMAGE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/math MATH_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/mvs MVS_SRCS *.h *.cc *.cu)
COLMAP_ADD_SOURCE_DIR(src/colmap/optim OPTIM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/retrieval RETRIEVAL_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/scene SCENE_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/sensor SENSOR_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/sfm SFM_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/tools TOOLS_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/ui UI_SRCS *.h *.cc)
COLMAP_ADD_SOURCE_DIR(src/colmap/util UTIL_SRCS *.h *.cc)

if(LSD_ENABLED)
    COLMAP_ADD_SOURCE_DIR(src/thirdparty/LSD THIRDPARTY_LSD_SRCS *.h *.c)
endif()



# Install batch scripts under Windows.
if(IS_MSVC)
    install(FILES "scripts/shell/COLMAP.bat" "scripts/shell/RUN_TESTS.bat"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "/")
endif()

# Install application meny entry under Linux/Unix.
if(UNIX AND NOT APPLE)
    install(FILES "doc/COLMAP.desktop" DESTINATION "share/applications")
endif()

# Configure the uninstallation script.
if(UNINSTALL_ENABLED)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeUninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake"
                   IMMEDIATE @ONLY)
    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)
    set_target_properties(uninstall PROPERTIES FOLDER ${CMAKE_TARGETS_ROOT_FOLDER})
endif()

set(COLMAP_EXPORT_LIBS
    # Internal.
    colmap_controllers
    colmap_estimators
    colmap_exe
    colmap_feature_types
    colmap_feature
    colmap_geometry
    colmap_image
    colmap_math
    colmap_mvs
    colmap_optim
    colmap_retrieval
    colmap_scene
    colmap_sensor
    colmap_sfm
    colmap_util
    # Third-party.
    colmap_poisson_recon
    colmap_vlfeat
)
if(GUI_ENABLED)
    list(APPEND COLMAP_EXPORT_LIBS
         colmap_ui
    )
endif()


# Add unified interface library target to export.
add_library(colmap INTERFACE)
target_link_libraries(colmap INTERFACE ${COLMAP_EXPORT_LIBS})
set(INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
target_include_directories(
    colmap
    INTERFACE
        $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

install(
    TARGETS colmap ${COLMAP_EXPORT_LIBS}
    EXPORT colmap-targets
    LIBRARY DESTINATION thirdparty/)

# Generate config and version.
include(CMakePackageConfigHelpers)
set(PACKAGE_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/colmap-config.cmake")
set(INSTALL_CONFIG_DIR "share/colmap")
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/colmap-config.cmake.in ${PACKAGE_CONFIG_FILE}
    INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})
install(FILES ${PACKAGE_CONFIG_FILE} DESTINATION ${INSTALL_CONFIG_DIR})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/colmap-config-version.cmake.in"
                "${CMAKE_CURRENT_BINARY_DIR}/colmap-config-version.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/colmap-config-version.cmake"
        DESTINATION "share/colmap")

# Install targets.
install(
    EXPORT colmap-targets
    FILE colmap-targets.cmake
    NAMESPACE colmap::
    DESTINATION ${INSTALL_CONFIG_DIR})

# Install header files.
install(
    DIRECTORY src/colmap
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")
install(
    DIRECTORY src/thirdparty
    DESTINATION include/colmap
    FILES_MATCHING REGEX ".*[.]h|.*[.]hpp|.*[.]inl")

# Install find_package scripts for dependencies.
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    DESTINATION share/colmap
    FILES_MATCHING PATTERN "Find*.cmake")
