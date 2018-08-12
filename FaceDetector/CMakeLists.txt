cmake_minimum_required(VERSION 3.12)
project(FaceDetector)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 17)

add_executable(FaceDetector main.cpp)

target_link_libraries( FaceDetector ${OpenCV_LIBS} )