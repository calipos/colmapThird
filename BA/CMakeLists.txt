cmake_minimum_required(VERSION 3.18.1) 

#add_definitions(-DGLOG_USE_GLOG_EXPORT)
#add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
#add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
set(CMAKE_BUILD_TYPE release)
project(BA)
set(LIBs )
message("C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
if(MSVC)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /std:c++17 /W1 -D_CRT_SECURE_NO_WARNINGS -DGLOG_USE_GLOG_EXPORT -DGLOG_NO_ABBREVIATED_SEVERITIES -DGOOGLE_GLOG_DLL_DECL= /bigobj")
set(LIBs glog.lib ceres.lib jsoncpp_static.lib)
else()
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -W   -fPIC -std=gnu++17 -O2 -fopenmp -DGLOG_USE_GLOG_EXPORT")
set(LIBs libglog.dll.a libceres.dll.a jsoncpp_static.lib)
endif()

file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h) 
#message(STATUS "cpp_srcs= ${cpp_srcs}")	
#message(STATUS "CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")	
message(STATUS "buildMesh CMAKE_CURRENT_SOURCE_DIR= ${CMAKE_CURRENT_SOURCE_DIR}")	

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CERES_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIR}) 
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${JSONCPP_INCLUDE_DIR})
link_directories(${CERES_LIB_DIR}) 
link_directories(${JSONCPP_LIB_DIR}) 
link_directories(${GLOG_LIB_DIR}) 
add_executable(${PROJECT_NAME}_EXE  ${cpp_srcs})
target_link_libraries ( ${PROJECT_NAME}_EXE PRIVATE
         ${LIBs})
install(TARGETS    ${PROJECT_NAME}_EXE
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
       LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
       ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})

#GLOG_USE_GLOG_EXPORT
#GLOG_NO_ABBREVIATED_SEVERITIES
#GOOGLE_GLOG_DLL_DECL=