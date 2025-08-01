cmake_minimum_required(VERSION 3.1...3.10) # for CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)
macro(ncnn_install_tool toolname)
    install(TARGETS ${toolname} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endmacro()

if(MSVC)
    # warning C4018: '<': signed/unsigned mismatch
    # warning C4244: 'argument': conversion from 'uint64_t' to 'const unsigned int', possible loss of data
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()

#add_subdirectory(caffe)
#add_subdirectory(mxnet)
add_subdirectory(onnx)
add_subdirectory(onnx_pips2)
#add_subdirectory(darknet)
if(NCNN_INT8)
    add_subdirectory(quantize)
else()
    message(WARNING "NCNN_INT8 disabled, quantize tools won't be built")
endif()

add_executable(ncnn2mem ncnn2mem.cpp)
target_link_libraries(ncnn2mem PRIVATE ncnn)
if(NCNN_VULKAN)
    target_link_libraries(ncnn2mem PRIVATE ${Vulkan_LIBRARY})
endif()

add_executable(ncnnoptimize ncnnoptimize.cpp)
target_link_libraries(ncnnoptimize PRIVATE ncnn)
if(NCNN_VULKAN)
    target_link_libraries(ncnnoptimize PRIVATE ${Vulkan_LIBRARY})
endif()

add_executable(ncnnmerge ncnnmerge.cpp)

# add all tools to a virtual project group
set_property(TARGET ncnn2mem PROPERTY FOLDER "tools")
set_property(TARGET ncnnoptimize PROPERTY FOLDER "tools")
set_property(TARGET ncnnmerge PROPERTY FOLDER "tools")
ncnn_install_tool(ncnn2mem)
ncnn_install_tool(ncnnmerge)
ncnn_install_tool(ncnnoptimize)
