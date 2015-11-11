cmake_minimum_required(VERSION 2.8)
project(test)
add_executable( test test.cpp)
target_link_libraries (test opencv_gpu)
target_link_libraries (test opencv_core)
target_link_libraries (test opencv_highgui)
target_link_libraries (test opencv_imgproc)

