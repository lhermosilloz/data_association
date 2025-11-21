# Stereo Track Association System

A comprehensive C++ stereo camera track association system with 3D depth calculation.

## Overview

This system implements multi-modal track association for stereo vision setups, combining:
- **Epipolar geometry constraints** for geometric validation
- **Classification confidence matching** for object type consistency  
- **Bounding box similarity** for size-based correspondence
- **Hungarian algorithm optimization** for optimal track pairing
- **3D triangulation** for depth and world coordinate calculation

## Features

### Core Capabilities
- ✅ **Stereo Track Association**: Match objects between Camera 2 and Camera 3
- ✅ **Epipolar Constraint Filtering**: Hard geometric constraint using fundamental matrix
- ✅ **Multi-modal Cost Function**: Combines geometric + classification + size costs
- ✅ **3D Triangulation**: DLT-based stereo reconstruction with depth calculation
- ✅ **Classification Validation**: Enforces mutual exclusivity constraints
- ✅ **Realistic Test Scenarios**: Comprehensive test cases with proper dimensions

### Enhanced Features  
- ✅ **Proper Classification Constraints**: Armed/unarmed, military/civilian, surrender/no-surrender pairs sum to 1.0
- ✅ **Realistic Track Dimensions**: Size scaling based on object type and depth
- ✅ **Depth Validation**: Range checking and triangulation quality assessment
- ✅ **Comprehensive Error Handling**: Robust validation and edge case management

## Standalone API

### Quick Start

```cpp
#include "stereo_track_association.h"

// Initialize associator
StereoTrackAssociator associator;

// Process frame
StereoAssociationResult result = associator.processFrame(frame_data);

// Access results
for (const auto& track : result.corresponded_tracks) {
    std::cout << "Track " << track.camera1_track_id 
              << " <-> " << track.camera2_track_id
              << " at depth " << track.depth_meters << "m" << std::endl;
}
```

### Key Data Structures

```cpp
// Input: Frame with tracks from both cameras
struct FrameData {
    int frame_id;
    std::vector<CameraTracks> camera_tracks;
};

// Output: Correspondences with 3D coordinates
struct StereoAssociationResult {
    std::vector<CorrespondedTrack> corresponded_tracks;
    std::vector<int> unmatched_camera1_tracks;
    std::vector<int> unmatched_camera2_tracks;
    double average_depth;
    int valid_triangulations;
};
```

## Files

### Standalone API
- `stereo_track_association.h` - Clean API header
- `stereo_track_association.cpp` - Implementation 
- `test_standalone_pipeline.cpp` - Usage examples

### Original Development
- `main.cpp` - Complete development system with comprehensive tests
- `data/camera_2.csv`, `data/camera_3.csv` - Sample data files

## Build Instructions

```bash
# Compile standalone API
g++ -o test_pipeline test_standalone_pipeline.cpp stereo_track_association.cpp \
    `pkg-config --cflags --libs opencv4` -std=c++17

# Compile full development system  
g++ -o build/enhanced_program main.cpp `pkg-config --cflags --libs opencv4` -std=c++17
```

## Algorithm Pipeline

1. **Input Validation** → Check classification constraints
2. **Epipolar Gating** → Filter pairs using fundamental matrix  
3. **Cost Matrix** → Combine geometric + size + classification costs
4. **Hungarian Assignment** → Optimal pairing solution
5. **3D Triangulation** → Calculate world coordinates and depth
6. **Result Packaging** → Return correspondences and unmatched tracks

## Camera Configuration

- **Stereo Setup**: Camera 2 and Camera 3 with 125mm baseline
- **Calibration**: Pre-calibrated intrinsics and distortion parameters
- **Coordinate System**: Right-handed, Camera 2 as reference frame
- **Resolution**: 1280x720 working resolution with proper bounds checking

## Dependencies

- **OpenCV 4.x**: Computer vision operations
- **C++17**: Modern language features  
- **Standard Library**: Containers and algorithms

## Performance

- **Computational**: O(n×m) epipolar gating, O(n²) assignment optimization
- **Accuracy**: Sub-pixel geometric constraints, meter-level depth precision
- **Robustness**: Handles occlusion, classification uncertainty, edge cases
