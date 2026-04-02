cmake_minimum_required(VERSION 3.10)
project(apex_server)
set(CMAKE_CXX_STANDARD 14)

find_package(ncnn REQUIRED)
add_executable(apex_server src/main.cpp)
target_link_libraries(apex_server ncnn)
