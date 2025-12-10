#include "stereo_track_association.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

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
    // Typical object sizes in meters (width, height)
    std::map<std::string, std::pair<double, double>> object_sizes = {
        {"person_standing", {0.5, 1.7}},
        {"person_crouching", {0.5, 1.0}},
        {"vehicle_small", {1.8, 1.5}},
        {"vehicle_large", {2.5, 3.0}}
    };
    
    auto it = object_sizes.find(object_type);
    if (it == object_sizes.end()) {
        it = object_sizes.find("person_standing");
    }
    
    double real_width = it->second.first;
    double real_height = it->second.second;
    
    // Camera focal length (average of fx, fy from calibration)
    double focal_length = 690.0;  // pixels
    
    // Calculate pixel dimensions: pixel_size = (real_size * focal_length) / depth
    int pixel_width = static_cast<int>((real_width * focal_length) / depth_meters);
    int pixel_height = static_cast<int>((real_height * focal_length) / depth_meters);
    
    // Clamp to reasonable ranges
    pixel_width = std::max(20, std::min(200, pixel_width));
    pixel_height = std::max(30, std::min(300, pixel_height));
    
    return {pixel_width, pixel_height};
}

// Test stationary scene - objects remain in same positions across frames
FrameData createStationaryScene(int frame_id, bool add_noise = false) {
    FrameData frame;
    frame.frame_id = frame_id;
    
    // Generate same object positions every time (stationary)
    std::vector<std::tuple<int, int, int, double, float, float, float>> objects = {
        {400, 300, 5, 5.0, 0.8f, 0.7f, 0.1f},    // Armed military at center
        {200, 250, 8, 8.0, 0.3f, 0.2f, 0.05f},   // Unarmed civilian left
        {600, 350, 12, 12.0, 0.9f, 0.8f, 0.15f}, // Armed military right
        {800, 200, 15, 15.0, 0.2f, 0.1f, 0.02f}  // Unarmed civilian far right
    };
    
    CameraTracks cam2_tracks, cam3_tracks;
    cam2_tracks.camera_id = 2;
    cam3_tracks.camera_id = 3;
    
    std::random_device rd;
    std::mt19937 gen(frame_id); // Use frame_id as seed for reproducible noise
    std::uniform_real_distribution<> noise_dist(-2.0, 2.0);
    
    for (size_t i = 0; i < objects.size(); i++) {
        auto [x, y, cam2_id, depth, armed, military, surrender] = objects[i];
        
        // Add optional noise to simulate tracking jitter
        int noise_x = add_noise ? static_cast<int>(noise_dist(gen)) : 0;
        int noise_y = add_noise ? static_cast<int>(noise_dist(gen)) : 0;
        
        auto dims = generateRealisticDimensions("person_standing", depth);
        
        // Camera 2 track
        cam2_tracks.tracks.push_back(createTrack(
            cam2_id + 100, x + noise_x, y + noise_y, armed, military, surrender, 
            dims.first, dims.second));
            
        // Camera 3 track (stereo correspondence with baseline shift)
        // For 125mm baseline and given depth, calculate expected disparity
        double baseline = 0.125; // meters
        double focal_length = 690.0; // pixels
        int disparity = static_cast<int>((baseline * focal_length) / depth);
        
        cam3_tracks.tracks.push_back(createTrack(
            cam2_id + 200, x - disparity + noise_x, y + noise_y, armed, military, surrender,
            dims.first, dims.second));
    }
    
    frame.camera_tracks.push_back(cam2_tracks);
    frame.camera_tracks.push_back(cam3_tracks);
    
    return frame;
}

// Test moving scene - objects change positions between frames
FrameData createMovingScene(int frame_id, bool add_noise = false) {
    FrameData frame;
    frame.frame_id = frame_id;
    
    // Objects move in predictable patterns based on frame_id
    std::vector<std::tuple<int, int, int, double, float, float, float, int, int>> objects = {
        {400, 300, 5, 5.0, 0.8f, 0.7f, 0.1f, 2, 1},    // Moving right and down
        {200, 250, 8, 8.0, 0.3f, 0.2f, 0.05f, -1, 2},  // Moving left and down
        {600, 350, 12, 12.0, 0.9f, 0.8f, 0.15f, 1, -1}, // Moving right and up
        {800, 200, 15, 15.0, 0.2f, 0.1f, 0.02f, -2, 0}  // Moving left
    };
    
    CameraTracks cam2_tracks, cam3_tracks;
    cam2_tracks.camera_id = 2;
    cam3_tracks.camera_id = 3;
    
    std::random_device rd;
    std::mt19937 gen(frame_id);
    std::uniform_real_distribution<> noise_dist(-2.0, 2.0);
    
    for (size_t i = 0; i < objects.size(); i++) {
        auto [base_x, base_y, cam2_id, depth, armed, military, surrender, vel_x, vel_y] = objects[i];
        
        // Calculate new position based on frame and velocity
        int x = base_x + (frame_id * vel_x);
        int y = base_y + (frame_id * vel_y);
        
        // Keep objects within image bounds
        x = std::max(50, std::min(1200, x));
        y = std::max(50, std::min(650, y));
        
        // Add optional noise
        int noise_x = add_noise ? static_cast<int>(noise_dist(gen)) : 0;
        int noise_y = add_noise ? static_cast<int>(noise_dist(gen)) : 0;
        
        auto dims = generateRealisticDimensions("person_standing", depth);
        
        // Camera 2 track
        cam2_tracks.tracks.push_back(createTrack(
            cam2_id + 100, x + noise_x, y + noise_y, armed, military, surrender,
            dims.first, dims.second));
            
        // Camera 3 track with stereo correspondence
        double baseline = 0.125;
        double focal_length = 690.0;
        int disparity = static_cast<int>((baseline * focal_length) / depth);
        
        cam3_tracks.tracks.push_back(createTrack(
            cam2_id + 200, x - disparity + noise_x, y + noise_y, armed, military, surrender,
            dims.first, dims.second));
    }
    
    frame.camera_tracks.push_back(cam2_tracks);
    frame.camera_tracks.push_back(cam3_tracks);
    
    return frame;
}

// Performance statistics tracking
struct ScenarioStats {
    std::string name;
    int total_frames;
    int total_expected_matches;
    int total_actual_matches;
    int total_valid_triangulations;
    double avg_assignment_cost;
    double avg_depth;
    double avg_processing_time_ms;
    std::vector<double> match_rates;
    
    void print() {
        double match_rate = (total_expected_matches > 0) ? 
            (double)total_actual_matches / total_expected_matches * 100.0 : 0.0;
            
        std::cout << "=== " << name << " Statistics ===" << std::endl;
        std::cout << "  Total frames processed: " << total_frames << std::endl;
        std::cout << "  Expected matches: " << total_expected_matches << std::endl;
        std::cout << "  Actual matches: " << total_actual_matches << std::endl;
        std::cout << "  Match rate: " << std::fixed << std::setprecision(1) << match_rate << "%" << std::endl;
        std::cout << "  Valid triangulations: " << total_valid_triangulations 
                  << " (" << ((total_actual_matches > 0) ? 
                      (double)total_valid_triangulations / total_actual_matches * 100.0 : 0.0) 
                  << "%)" << std::endl;
        std::cout << "  Average assignment cost: " << std::setprecision(3) << avg_assignment_cost << std::endl;
        std::cout << "  Average depth: " << std::setprecision(1) << avg_depth << "m" << std::endl;
        std::cout << "  Average processing time: " << std::setprecision(2) << avg_processing_time_ms << "ms" << std::endl;
        
        // Calculate match rate consistency (standard deviation)
        if (match_rates.size() > 1) {
            double mean_rate = 0.0;
            for (double rate : match_rates) mean_rate += rate;
            mean_rate /= match_rates.size();
            
            double variance = 0.0;
            for (double rate : match_rates) {
                variance += (rate - mean_rate) * (rate - mean_rate);
            }
            variance /= (match_rates.size() - 1);
            double std_dev = sqrt(variance);
            
            std::cout << "  Match rate consistency (std dev): " << std::setprecision(2) << std_dev << "%" << std::endl;
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "=== STATIONARY vs MOVING SCENE DIAGNOSTIC TEST ===" << std::endl;
    std::cout << "Testing stereo track association pipeline consistency" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Initialize associator with different parameter sets
    struct TestConfig {
        std::string name;
        double epipolar_threshold;
        double max_assignment_cost;
    };
    
    std::vector<TestConfig> configs = {
        {"Strict Parameters", 10.0, 1.0},
        {"Lenient Parameters", 50.0, 2.0},
        {"Default Parameters", 30.0, 1.0}
    };
    
    const int NUM_FRAMES = 10;
    const int EXPECTED_MATCHES_PER_FRAME = 4; // We have 4 object pairs in each scene
    
    for (const auto& config : configs) {
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "Testing with " << config.name << " (epipolar: " << config.epipolar_threshold 
                  << "px, max_cost: " << config.max_assignment_cost << ")" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        StereoTrackAssociator associator;
        associator.setEpipolarThreshold(config.epipolar_threshold);
        associator.setMaxAssignmentCost(config.max_assignment_cost);
        
        // Test scenarios
        std::vector<std::tuple<std::string, bool, bool>> scenarios = {
            {"Stationary Scene (No Noise)", false, false},
            {"Stationary Scene (With Noise)", false, true},
            {"Moving Scene (No Noise)", true, false},
            {"Moving Scene (With Noise)", true, true}
        };
        
        std::vector<ScenarioStats> scenario_results;
        
        for (auto [scenario_name, is_moving, add_noise] : scenarios) {
            ScenarioStats stats;
            stats.name = scenario_name;
            stats.total_frames = NUM_FRAMES;
            stats.total_expected_matches = NUM_FRAMES * EXPECTED_MATCHES_PER_FRAME;
            stats.total_actual_matches = 0;
            stats.total_valid_triangulations = 0;
            stats.avg_assignment_cost = 0.0;
            stats.avg_depth = 0.0;
            stats.avg_processing_time_ms = 0.0;
            
            double total_cost = 0.0, total_depth = 0.0, total_time = 0.0;
            
            std::cout << "\n--- " << scenario_name << " ---" << std::endl;
            
            for (int frame = 1; frame <= NUM_FRAMES; frame++) {
                // Create frame data
                FrameData frame_data = is_moving ? 
                    createMovingScene(frame, add_noise) : 
                    createStationaryScene(frame, add_noise);
                
                // Measure processing time
                auto start_time = std::chrono::high_resolution_clock::now();
                StereoAssociationResult result = associator.processFrame(frame_data);
                auto end_time = std::chrono::high_resolution_clock::now();
                
                double frame_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                total_time += frame_time;
                
                // Collect statistics
                int frame_matches = result.corresponded_tracks.size();
                stats.total_actual_matches += frame_matches;
                stats.total_valid_triangulations += result.valid_triangulations;
                
                double frame_match_rate = (EXPECTED_MATCHES_PER_FRAME > 0) ? 
                    (double)frame_matches / EXPECTED_MATCHES_PER_FRAME * 100.0 : 0.0;
                stats.match_rates.push_back(frame_match_rate);
                
                // Calculate average costs and depths for this frame
                double frame_cost = 0.0, frame_depth = 0.0;
                for (const auto& track : result.corresponded_tracks) {
                    total_cost += track.assignment_cost;
                    if (track.is_valid) {
                        frame_depth += track.depth_meters;
                    }
                }
                
                std::cout << "  Frame " << frame << ": " << frame_matches << "/" << EXPECTED_MATCHES_PER_FRAME 
                          << " matches (" << std::fixed << std::setprecision(1) << frame_match_rate << "%), "
                          << result.valid_triangulations << " valid triangulations, "
                          << std::setprecision(1) << frame_time << "ms" << std::endl;
            }
            
            // Calculate averages
            stats.avg_assignment_cost = (stats.total_actual_matches > 0) ? 
                total_cost / stats.total_actual_matches : 0.0;
            stats.avg_depth = (stats.total_valid_triangulations > 0) ? 
                total_depth / stats.total_valid_triangulations : 0.0; // Fixed variable reference
            stats.avg_processing_time_ms = total_time / NUM_FRAMES;
            
            scenario_results.push_back(stats);
        }
        
        // Print detailed statistics for this configuration
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "DETAILED STATISTICS FOR " << config.name << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        for (auto& stats : scenario_results) {
            stats.print();
        }
        
        // Comparative analysis
        std::cout << "COMPARATIVE ANALYSIS:" << std::endl;
        
        // Compare stationary vs moving (no noise)
        if (scenario_results.size() >= 3) {
            auto& stationary_clean = scenario_results[0];
            auto& moving_clean = scenario_results[2];
            
            double stat_rate = (double)stationary_clean.total_actual_matches / stationary_clean.total_expected_matches * 100.0;
            double move_rate = (double)moving_clean.total_actual_matches / moving_clean.total_expected_matches * 100.0;
            
            std::cout << "  Clean scenes: Stationary " << std::setprecision(1) << stat_rate 
                      << "% vs Moving " << move_rate << "%" << std::endl;
                      
            if (stat_rate < move_rate - 5.0) {
                std::cout << "  ⚠️  ISSUE DETECTED: Stationary scenes perform significantly worse!" << std::endl;
            } else if (move_rate < stat_rate - 5.0) {
                std::cout << "  ⚠️  UNEXPECTED: Moving scenes perform worse than stationary!" << std::endl;
            } else {
                std::cout << "  ✅ Performance is consistent between stationary and moving scenes." << std::endl;
            }
        }
        
        // Compare noise sensitivity
        if (scenario_results.size() >= 4) {
            std::cout << "  Noise sensitivity:" << std::endl;
            for (size_t i = 0; i < scenario_results.size(); i += 2) {
                if (i + 1 < scenario_results.size()) {
                    auto& clean = scenario_results[i];
                    auto& noisy = scenario_results[i + 1];
                    
                    double clean_rate = (double)clean.total_actual_matches / clean.total_expected_matches * 100.0;
                    double noisy_rate = (double)noisy.total_actual_matches / noisy.total_expected_matches * 100.0;
                    double degradation = clean_rate - noisy_rate;
                    
                    std::cout << "    " << (i == 0 ? "Stationary" : "Moving") << ": "
                              << std::setprecision(1) << degradation << "% degradation with noise" << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "DIAGNOSTIC CONCLUSIONS:" << std::endl;
    std::cout << "1. Compare stationary vs moving scene performance across configurations" << std::endl;
    std::cout << "2. Identify optimal parameter settings for your use case" << std::endl;
    std::cout << "3. Assess noise sensitivity differences between scenarios" << std::endl;
    std::cout << "4. Look for consistency issues (high standard deviation in match rates)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    return 0;
}