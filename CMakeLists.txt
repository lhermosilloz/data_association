cmake_minimum_required(VERSION 3.16)
project(data_association)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Create executable
add_executable(my_program main.cpp)

# Link OpenCV libraries
target_link_libraries(my_program ${OpenCV_LIBS})
target_include_directories(my_program PRIVATE ${OpenCV_INCLUDE_DIRS})
