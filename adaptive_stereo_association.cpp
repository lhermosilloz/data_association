#include "adaptive_stereo_association.h"
#include <cmath>
#include <algorithm>
#include <numeric>

AdaptiveStereoTrackAssociator::AdaptiveStereoTrackAssociator() 
    : StereoTrackAssociator(),
      adaptive_mode_enabled_(true),
      movement_threshold_(5.0), // pixels
      history_size_(5) {
    
    // Define parameter sets based on diagnostic findings
    stationary_params_ = {30.0, 1.5, "Stationary-Optimized"}; // Slightly more lenient for stationary
    moving_params_ = {25.0, 1.0, "Moving-Optimized"};         // Tighter for moving objects  
    default_params_ = {30.0, 1.0, "Default"};                 // Safe middle ground
    
    // Initialize statistics
    adaptive_stats_ = {false, 0.0, default_params_.epipolar_threshold, 
                      default_params_.max_assignment_cost, 0, 0.0};
}

StereoAssociationResult AdaptiveStereoTrackAssociator::processFrameAdaptive(const FrameData& frame_data) {
    // Add current frame to history
    frame_history_.push_back(frame_data);
    if (frame_history_.size() > static_cast<size_t>(history_size_)) {
        frame_history_.pop_front();
    }
    
    // Calculate movement if we have previous frames
    double current_movement = 0.0;
    bool is_stationary = false;
    
    if (adaptive_mode_enabled_ && frame_history_.size() >= 2) {
        current_movement = calculateFrameMovement(frame_data, frame_history_[frame_history_.size() - 2]);
        movement_history_.push_back(current_movement);
        
        if (movement_history_.size() > static_cast<size_t>(history_size_)) {
            movement_history_.pop_front();
        }
        
        // Determine if scene is stationary
        is_stationary = isStationaryScene(movement_history_);
    }
    
    // Select optimal parameters
    ParameterSet selected_params = adaptive_mode_enabled_ ? 
        selectOptimalParameters(is_stationary) : default_params_;
    
    // Apply parameters to base class
    setEpipolarThreshold(selected_params.epipolar_threshold);
    setMaxAssignmentCost(selected_params.max_assignment_cost);
    
    // Update statistics
    updateStatistics(is_stationary, current_movement, selected_params);
    
    // Process frame with selected parameters
    StereoAssociationResult result = processFrame(frame_data);
    
    // Add diagnostic information to result (extend the result structure if needed)
    // For now, we'll just return the standard result
    return result;
}

double AdaptiveStereoTrackAssociator::calculateFrameMovement(
    const FrameData& current_frame, const FrameData& previous_frame) {
    
    // Find corresponding cameras
    const CameraTracks* current_cam2 = nullptr;
    const CameraTracks* current_cam3 = nullptr;
    const CameraTracks* previous_cam2 = nullptr;
    const CameraTracks* previous_cam3 = nullptr;
    
    for (const auto& cam : current_frame.camera_tracks) {
        if (cam.camera_id == 2) current_cam2 = &cam;
        else if (cam.camera_id == 3) current_cam3 = &cam;
    }
    
    for (const auto& cam : previous_frame.camera_tracks) {
        if (cam.camera_id == 2) previous_cam2 = &cam;
        else if (cam.camera_id == 3) previous_cam3 = &cam;
    }
    
    if (!current_cam2 || !current_cam3 || !previous_cam2 || !previous_cam3) {
        return 0.0; // Unable to calculate movement
    }
    
    // Calculate movement for each camera
    double cam2_movement = calculateCameraMovement(*current_cam2, *previous_cam2);
    double cam3_movement = calculateCameraMovement(*current_cam3, *previous_cam3);
    
    // Return average movement across both cameras
    return (cam2_movement + cam3_movement) / 2.0;
}

bool AdaptiveStereoTrackAssociator::isStationaryScene(const std::deque<double>& movement_history) {
    if (movement_history.size() < 3) {
        return false; // Need sufficient history
    }
    
    // Calculate average movement over recent history
    double average_movement = std::accumulate(movement_history.begin(), movement_history.end(), 0.0) 
                             / movement_history.size();
    
    // Scene is considered stationary if average movement is below threshold
    return average_movement < movement_threshold_;
}

AdaptiveStereoTrackAssociator::ParameterSet 
AdaptiveStereoTrackAssociator::selectOptimalParameters(bool is_stationary) {
    if (is_stationary) {
        return stationary_params_;
    } else {
        return moving_params_;
    }
}

void AdaptiveStereoTrackAssociator::updateStatistics(
    bool is_stationary, double movement, const ParameterSet& params) {
    
    adaptive_stats_.frames_processed++;
    adaptive_stats_.is_stationary_scene = is_stationary;
    adaptive_stats_.average_movement = movement;
    adaptive_stats_.current_epipolar_threshold = params.epipolar_threshold;
    adaptive_stats_.current_max_cost = params.max_assignment_cost;
    
    // Update stationary ratio (running average)
    static int stationary_count = 0;
    if (is_stationary) stationary_count++;
    
    adaptive_stats_.stationary_ratio = (double)stationary_count / adaptive_stats_.frames_processed;
}

// Utility functions
double calculateTrackMovement(const TrackInfo& track1, const TrackInfo& track2) {
    double dx = track1.bb_center_x - track2.bb_center_x;
    double dy = track1.bb_center_y - track2.bb_center_y;
    return std::sqrt(dx*dx + dy*dy);
}

double calculateCameraMovement(const CameraTracks& current_cam, const CameraTracks& previous_cam) {
    if (current_cam.tracks.empty() || previous_cam.tracks.empty()) {
        return 0.0;
    }
    
    std::vector<double> track_movements;
    
    // Find matching tracks by ID and calculate their movement
    for (const auto& current_track : current_cam.tracks) {
        for (const auto& previous_track : previous_cam.tracks) {
            if (current_track.track_id == previous_track.track_id) {
                double movement = calculateTrackMovement(current_track, previous_track);
                track_movements.push_back(movement);
                break;
            }
        }
    }
    
    if (track_movements.empty()) {
        // If no matching track IDs, use centroid-based movement
        double current_cx = 0, current_cy = 0;
        double previous_cx = 0, previous_cy = 0;
        
        for (const auto& track : current_cam.tracks) {
            current_cx += track.bb_center_x;
            current_cy += track.bb_center_y;
        }
        current_cx /= current_cam.tracks.size();
        current_cy /= current_cam.tracks.size();
        
        for (const auto& track : previous_cam.tracks) {
            previous_cx += track.bb_center_x;
            previous_cy += track.bb_center_y;
        }
        previous_cx /= previous_cam.tracks.size();
        previous_cy /= previous_cam.tracks.size();
        
        double dx = current_cx - previous_cx;
        double dy = current_cy - previous_cy;
        return std::sqrt(dx*dx + dy*dy);
    }
    
    // Return average movement of matched tracks
    return std::accumulate(track_movements.begin(), track_movements.end(), 0.0) / track_movements.size();
}