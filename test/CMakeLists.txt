# CMakeList.txt : CMake project for test, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project (test)
include_directories(${FLANN_INCLUDE_DIRS})
find_package(Eigen3   REQUIRED)
# Add source to this project's executable.
add_executable (test "test.cpp" "test.h")
target_link_libraries ( ${PROJECT_NAME} PRIVATE
		Eigen3::Eigen
        ${FLANN_INCLUDE_DIRS})
# TODO: Add tests and install targets if needed.
