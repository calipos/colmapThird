cmake_minimum_required(VERSION 3.18.1)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -W   -fPIC -std=gnu++17 -O2 -fopenmp")
#set(CMAKE_BUILD_TYPE release)
project(BA)

file(GLOB cpp_srcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp) 
#message(STATUS "cpp_srcs= ${cpp_srcs}")	
#message(STATUS "CMAKE_CXX_FLAGS= ${CMAKE_CXX_FLAGS}")	
message(STATUS "buildMesh CMAKE_CURRENT_SOURCE_DIR= ${CMAKE_CURRENT_SOURCE_DIR}")	

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CERES_INCLUDE_DIR})
include_directories(${GLOG_INCLUDE_DIR})
 
link_directories(${DCERES_LIB_DIR}) 
link_directories(${GLOG_LIB_DIR}) 
add_executable(${PROJECT_NAME}_EXE  ${cpp_srcs})
target_link_libraries ( ${PROJECT_NAME}_EXE PRIVATE
         ceres.lib
         glog.lib)
#install(TARGETS ${PROJECT_NAME}  ${PROJECT_NAME}_EXE
#        RUNTIME DESTINATION ${outPath}/install/bin
#       LIBRARY DESTINATION ${outPath}/install/lib
#       ARCHIVE DESTINATION ${outPath}/install/lib)
#install(FILES buildMesh.h DESTINATION  ${outPath}/install/include)    


#GLOG_USE_GLOG_EXPORT
#GLOG_NO_ABBREVIATED_SEVERITIES
#GOOGLE_GLOG_DLL_DECL=