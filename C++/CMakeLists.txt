cmake_minimum_required(VERSION 3.18)
project(SuperPointCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(run_superpoint run_superpoint_stbimage.cpp)

target_link_libraries(run_superpoint ${TORCH_LIBRARIES})

if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET run_superpoint POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
        $<TARGET_FILE_DIR:run_superpoint>)
endif()