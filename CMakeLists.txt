cmake_minimum_required(VERSION 3.10)
project(StereoTrackAssociation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Create standalone library
add_library(stereo_track_association 
    stereo_track_association.cpp
)
target_link_libraries(stereo_track_association ${OpenCV_LIBS})

# Create adaptive stereo library
add_library(adaptive_stereo_association
    adaptive_stereo_association.cpp
    stereo_track_association.cpp
)
target_link_libraries(adaptive_stereo_association ${OpenCV_LIBS})

# Test program for standalone API
add_executable(test_pipeline 
    test_standalone_pipeline.cpp
)
target_link_libraries(test_pipeline stereo_track_association ${OpenCV_LIBS})

# Enhanced development program (original)
add_executable(enhanced_program 
    main.cpp
)
target_link_libraries(enhanced_program ${OpenCV_LIBS})

# Diagnostic test for stationary scene issues
add_executable(test_stationary
    test_stationary_scenes.cpp
)
target_link_libraries(test_stationary stereo_track_association ${OpenCV_LIBS})

# Adaptive system demonstration
add_executable(test_adaptive
    test_adaptive_system.cpp
)
target_link_libraries(test_adaptive adaptive_stereo_association ${OpenCV_LIBS})

# Install rules
install(TARGETS stereo_track_association adaptive_stereo_association test_pipeline enhanced_program test_stationary test_adaptive
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES stereo_track_association.h adaptive_stereo_association.h
        DESTINATION include)