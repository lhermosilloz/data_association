#include "stereo_track_association.h"
#include <random>

// Helper function to create realistic tracks with proper classification constraints
TrackInfo createTrack(int id, int x, int y, 
                     float armed_ratio = 0.5f,     
                     float military_ratio = 0.3f,  
                     float surrender_ratio = 0.1f, 
                     int width = 80, int height = 120) {
    
    const float epsilon = 0.01f;
    
    // Armed/Unarmed: exactly one is dominant
    float armed_conf, unarmed_conf;
    if (armed_ratio > 0.5f) {
        armed_conf = std::max(epsilon, std::min(1.0f - epsilon, armed_ratio));
        unarmed_conf = 1.0f - armed_conf;
    } else {
        unarmed_conf = std::max(epsilon, std::min(1.0f - epsilon, 1.0f - armed_ratio));
        armed_conf = 1.0f - unarmed_conf;
    }
    
    // Military/Civilian: exactly one is dominant
    float military_conf, civilian_conf;
    if (military_ratio > 0.5f) {
        military_conf = std::max(epsilon, std::min(1.0f - epsilon, military_ratio));
        civilian_conf = 1.0f - military_conf;
    } else {
        civilian_conf = std::max(epsilon, std::min(1.0f - epsilon, 1.0f - military_ratio));
        military_conf = 1.0f - civilian_conf;
    }
    
    // Surrender/No_surrender: exactly one is dominant
    float surrender_conf, no_surrender_conf;
    if (surrender_ratio > 0.5f) {
        surrender_conf = std::max(epsilon, std::min(1.0f - epsilon, surrender_ratio));
        no_surrender_conf = 1.0f - surrender_conf;
    } else {
        no_surrender_conf = std::max(epsilon, std::min(1.0f - epsilon, 1.0f - surrender_ratio));
        surrender_conf = 1.0f - no_surrender_conf;
    }
    
    return {id, width, height, x, y, 
            armed_conf, unarmed_conf,
            military_conf, civilian_conf,
            surrender_conf, no_surrender_conf};
}

// Helper function to generate realistic dimensions based on depth
std::pair<int, int> generateRealisticDimensions(const std::string& object_type, double depth_meters) {
    double base_width_pixels, base_height_pixels;
    
    if (object_type == "person_standing") {
        base_width_pixels = 60.0;   
        base_height_pixels = 180.0; 
    } else if (object_type == "person_crouching") {
        base_width_pixels = 80.0;   
        base_height_pixels = 120.0; 
    } else if (object_type == "person_prone") {
        base_width_pixels = 160.0;  
        base_height_pixels = 40.0;  
    } else {
        base_width_pixels = 70.0;   
        base_height_pixels = 150.0;
    }
    
    // Scale inversely with distance
    double reference_distance = 5.0;
    double scale_factor = reference_distance / depth_meters;
    
    // Add realistic variation (±10%)
    double variation = 0.1 * (rand() % 21 - 10) / 100.0;
    scale_factor *= (1.0 + variation);
    
    int width = static_cast<int>(base_width_pixels * scale_factor);
    int height = static_cast<int>(base_height_pixels * scale_factor);
    
    // Ensure reasonable bounds
    width = std::max(20, std::min(200, width));
    height = std::max(30, std::min(300, height));
    
    return std::make_pair(width, height);
}

int main() {
    std::cout << "=== Standalone Stereo Track Association Pipeline Test ===" << std::endl;
    
    // Initialize the stereo track associator
    StereoTrackAssociator associator;
    
    // Configure parameters
    associator.setEpipolarThreshold(10.0);
    associator.setMaxAssignmentCost(1.0);
    
    // Test Case 1: Military Personnel Scenario
    {
        std::cout << "\n--- Test 1: Military Personnel Scenario ---" << std::endl;
        
        auto dims_5m = generateRealisticDimensions("person_standing", 5.0);
        auto dims_8m = generateRealisticDimensions("person_standing", 8.0);
        auto dims_12m = generateRealisticDimensions("person_crouching", 12.0);
        
        // Create frame data
        FrameData frame;
        frame.frame_id = 1;
        
        // Camera 2 tracks
        CameraTracks cam2_tracks;
        cam2_tracks.camera_id = 2;
        cam2_tracks.tracks = {
            createTrack(101, 400, 300, 0.95f, 0.85f, 0.15f, dims_5m.first, dims_5m.second),    // Armed military standing
            createTrack(102, 600, 250, 0.90f, 0.80f, 0.20f, dims_8m.first, dims_8m.second),    // Armed military standing  
            createTrack(103, 800, 350, 0.85f, 0.75f, 0.10f, dims_12m.first, dims_12m.second)   // Armed military crouching
        };
        
        // Camera 3 tracks (positioned closer to epipolar lines)
        CameraTracks cam3_tracks;
        cam3_tracks.camera_id = 3;
        cam3_tracks.tracks = {
            createTrack(201, 380, 300, 0.93f, 0.82f, 0.18f, dims_5m.first, dims_5m.second),    // Armed military standing
            createTrack(202, 580, 250, 0.88f, 0.78f, 0.25f, dims_8m.first, dims_8m.second),    // Armed military standing
            createTrack(203, 780, 350, 0.87f, 0.73f, 0.12f, dims_12m.first, dims_12m.second)   // Armed military crouching
        };
        
        frame.camera_tracks.push_back(cam2_tracks);
        frame.camera_tracks.push_back(cam3_tracks);
        
        // Process frame using standalone pipeline
        StereoAssociationResult result = associator.processFrame(frame);
        
        // Display results
        std::cout << "Frame " << result.frame_id << " Results:" << std::endl;
        std::cout << "  Corresponded tracks: " << result.corresponded_tracks.size() << std::endl;
        
        for (const auto& track : result.corresponded_tracks) {
            std::cout << "    Cam1[" << track.camera1_track_id << "] <-> Cam2[" << track.camera2_track_id 
                      << "] - Cost: " << std::fixed << std::setprecision(3) << track.assignment_cost
                      << ", Depth: " << track.depth_meters << "m"
                      << ", Valid: " << (track.is_valid ? "Yes" : "No") << std::endl;
        }
        
        std::cout << "  Unmatched Camera 1 tracks: ";
        for (int id : result.unmatched_camera1_tracks) std::cout << id << " ";
        std::cout << std::endl;
        
        std::cout << "  Unmatched Camera 2 tracks: ";
        for (int id : result.unmatched_camera2_tracks) std::cout << id << " ";
        std::cout << std::endl;
        
        std::cout << "  Average depth: " << std::fixed << std::setprecision(2) << result.average_depth << "m" << std::endl;
        std::cout << "  Valid triangulations: " << result.valid_triangulations << std::endl;
    }
    
    // Test Case 2: Mixed Civilian and Military
    {
        std::cout << "\n--- Test 2: Mixed Civilian/Military Scenario ---" << std::endl;
        
        auto dims_6m = generateRealisticDimensions("person_standing", 6.0);
        auto dims_10m = generateRealisticDimensions("person_crouching", 10.0);
        auto dims_15m = generateRealisticDimensions("person_standing", 15.0);
        
        FrameData frame;
        frame.frame_id = 2;
        
        // Camera 2 tracks
        CameraTracks cam2_tracks;
        cam2_tracks.camera_id = 2;
        cam2_tracks.tracks = {
            createTrack(111, 300, 280, 0.15f, 0.25f, 0.05f, dims_6m.first, dims_6m.second),     // Unarmed civilian
            createTrack(112, 500, 320, 0.80f, 0.70f, 0.30f, dims_10m.first, dims_10m.second),   // Armed military 
            createTrack(113, 750, 290, 0.10f, 0.20f, 0.90f, dims_15m.first, dims_15m.second)    // Unarmed civilian surrendering
        };
        
        // Camera 3 tracks (positioned for better correspondence + one extra unmatched)
        CameraTracks cam3_tracks;
        cam3_tracks.camera_id = 3;
        cam3_tracks.tracks = {
            createTrack(211, 280, 280, 0.12f, 0.22f, 0.08f, dims_6m.first, dims_6m.second),     // Unarmed civilian
            createTrack(212, 480, 320, 0.85f, 0.75f, 0.25f, dims_10m.first, dims_10m.second),   // Armed military
            createTrack(213, 730, 290, 0.08f, 0.18f, 0.85f, dims_15m.first, dims_15m.second),   // Unarmed civilian surrendering
            createTrack(214, 950, 400, 0.50f, 0.60f, 0.20f, dims_10m.first, dims_10m.second)    // Extra unmatched track
        };
        
        frame.camera_tracks.push_back(cam2_tracks);
        frame.camera_tracks.push_back(cam3_tracks);
        
        StereoAssociationResult result = associator.processFrame(frame);
        
        std::cout << "Frame " << result.frame_id << " Results:" << std::endl;
        std::cout << "  Corresponded tracks: " << result.corresponded_tracks.size() << std::endl;
        
        for (const auto& track : result.corresponded_tracks) {
            std::cout << "    Cam1[" << track.camera1_track_id << "] <-> Cam2[" << track.camera2_track_id 
                      << "] - Cost: " << std::fixed << std::setprecision(3) << track.assignment_cost
                      << ", Depth: " << track.depth_meters << "m"
                      << ", Valid: " << (track.is_valid ? "Yes" : "No") << std::endl;
        }
        
        std::cout << "  Unmatched Camera 1 tracks: ";
        for (int id : result.unmatched_camera1_tracks) std::cout << id << " ";
        std::cout << std::endl;
        
        std::cout << "  Unmatched Camera 2 tracks: ";
        for (int id : result.unmatched_camera2_tracks) std::cout << id << " ";
        std::cout << std::endl;
        
        std::cout << "  Statistics: " << result.valid_triangulations << " valid triangulations" << std::endl;
    }
    
    // Test Case 3: Edge case - No valid correspondences
    {
        std::cout << "\n--- Test 3: No Valid Correspondences ---" << std::endl;
        
        FrameData frame;
        frame.frame_id = 3;
        
        // Camera 2 tracks
        CameraTracks cam2_tracks;
        cam2_tracks.camera_id = 2;
        cam2_tracks.tracks = {
            createTrack(121, 100, 100, 0.70f, 0.60f, 0.40f, 50, 80)
        };
        
        // Camera 3 tracks (far from epipolar line)
        CameraTracks cam3_tracks;
        cam3_tracks.camera_id = 3;
        cam3_tracks.tracks = {
            createTrack(221, 1000, 600, 0.75f, 0.65f, 0.35f, 50, 80)  // Too far away
        };
        
        frame.camera_tracks.push_back(cam2_tracks);
        frame.camera_tracks.push_back(cam3_tracks);
        
        StereoAssociationResult result = associator.processFrame(frame);
        
        std::cout << "Frame " << result.frame_id << " Results:" << std::endl;
        std::cout << "  Corresponded tracks: " << result.corresponded_tracks.size() << std::endl;
        std::cout << "  Unmatched Camera 1 tracks: ";
        for (int id : result.unmatched_camera1_tracks) std::cout << id << " ";
        std::cout << std::endl;
        std::cout << "  Unmatched Camera 2 tracks: ";
        for (int id : result.unmatched_camera2_tracks) std::cout << id << " ";
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Standalone Pipeline Test Complete ===" << std::endl;
    std::cout << "\nAPI Summary:" << std::endl;
    std::cout << "- Input: FrameData struct with camera tracks" << std::endl;
    std::cout << "- Output: StereoAssociationResult with:" << std::endl;
    std::cout << "  * corresponded_tracks (with 3D coordinates and depth)" << std::endl;
    std::cout << "  * unmatched_camera1_tracks" << std::endl;
    std::cout << "  * unmatched_camera2_tracks" << std::endl;
    std::cout << "  * Statistics (average depth, triangulation count, etc.)" << std::endl;
    
    return 0;
}