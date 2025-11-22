#include "stereo_track_association.h"
#include <iostream>

// Simple test with tracks that should definitely match
int main() {
    std::cout << "=== Debug Stereo Association Issues ===" << std::endl;
    
    StereoTrackAssociator associator;
    
    // Test with very lenient settings
    associator.setEpipolarThreshold(50.0);  // Very loose
    associator.setMaxAssignmentCost(2.0);   // Very loose
    
    std::cout << "Using lenient settings: 50px epipolar, 2.0 cost threshold" << std::endl;
    
    // Create simple test case
    FrameData frame;
    frame.frame_id = 1;
    
    // Camera 1 tracks - simple positions
    CameraTracks cam1_tracks;
    cam1_tracks.camera_id = 2;
    cam1_tracks.tracks = {
        {101, 80, 120, 400, 300, 0.8f, 0.2f, 0.7f, 0.3f, 0.1f, 0.9f},  // Center-ish
        {102, 85, 125, 600, 200, 0.3f, 0.7f, 0.2f, 0.8f, 0.05f, 0.95f} // Right side
    };
    
    // Camera 2 tracks - positions that should correspond based on 125mm baseline
    CameraTracks cam2_tracks;
    cam2_tracks.camera_id = 3;
    cam2_tracks.tracks = {
        {201, 78, 118, 370, 300, 0.75f, 0.25f, 0.65f, 0.35f, 0.15f, 0.85f}, // Shifted left from 400
        {202, 83, 123, 570, 200, 0.25f, 0.75f, 0.25f, 0.75f, 0.08f, 0.92f}  // Shifted left from 600
    };
    
    frame.camera_tracks.push_back(cam1_tracks);
    frame.camera_tracks.push_back(cam2_tracks);
    
    std::cout << "\nInput tracks:" << std::endl;
    std::cout << "Camera 1: Track 101 at (400,300), Track 102 at (600,200)" << std::endl;
    std::cout << "Camera 2: Track 201 at (370,300), Track 202 at (570,200)" << std::endl;
    std::cout << "Expected: Left shift due to 125mm baseline" << std::endl;
    
    StereoAssociationResult result = associator.processFrame(frame);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Corresponded tracks: " << result.corresponded_tracks.size() << std::endl;
    
    for (const auto& track : result.corresponded_tracks) {
        std::cout << "  Track " << track.camera1_track_id 
                  << " <-> " << track.camera2_track_id
                  << ", Cost: " << track.assignment_cost
                  << ", Epipolar dist: " << track.epipolar_distance << "px"
                  << ", Depth: " << track.depth_meters << "m"
                  << ", Valid: " << (track.is_valid ? "Yes" : "No") << std::endl;
    }
    
    std::cout << "\nUnmatched Camera 1: ";
    for (int id : result.unmatched_camera1_tracks) std::cout << id << " ";
    std::cout << std::endl;
    
    std::cout << "Unmatched Camera 2: ";
    for (int id : result.unmatched_camera2_tracks) std::cout << id << " ";
    std::cout << std::endl;
    
    // Test 2: Single track pair
    std::cout << "\n=== Test 2: Single Perfect Track Pair ===" << std::endl;
    
    FrameData frame2;
    frame2.frame_id = 2;
    
    CameraTracks cam1_single;
    cam1_single.camera_id = 2;
    cam1_single.tracks = {{103, 70, 140, 640, 360, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    
    CameraTracks cam2_single;
    cam2_single.camera_id = 3;
    cam2_single.tracks = {{203, 70, 140, 620, 360, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f}};
    
    frame2.camera_tracks.push_back(cam1_single);
    frame2.camera_tracks.push_back(cam2_single);
    
    std::cout << "Single track test: (640,360) <-> (620,360)" << std::endl;
    
    StereoAssociationResult result2 = associator.processFrame(frame2);
    
    std::cout << "Results: " << result2.corresponded_tracks.size() << " correspondences" << std::endl;
    for (const auto& track : result2.corresponded_tracks) {
        std::cout << "  " << track.camera1_track_id << " <-> " << track.camera2_track_id
                  << ", Epipolar: " << track.epipolar_distance << "px"
                  << ", Depth: " << track.depth_meters << "m" << std::endl;
    }
    
    return 0;
}