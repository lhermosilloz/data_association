#pragma once

#include "stereo_track_association.h"
#include <unordered_map>
#include <deque>

// Enhanced stereo track associator with adaptive parameters
class AdaptiveStereoTrackAssociator : public StereoTrackAssociator {
public:
    AdaptiveStereoTrackAssociator();
    
    // Main processing with adaptive parameters
    StereoAssociationResult processFrameAdaptive(const FrameData& frame_data);
    
    // Configuration
    void setAdaptiveMode(bool enabled) { adaptive_mode_enabled_ = enabled; }
    void setMovementThreshold(double threshold) { movement_threshold_ = threshold; }
    void setHistorySize(int size) { history_size_ = size; }
    
    // Statistics and diagnostics
    struct AdaptiveStats {
        bool is_stationary_scene;
        double average_movement;
        double current_epipolar_threshold;
        double current_max_cost;
        int frames_processed;
        double stationary_ratio;
    };
    
    AdaptiveStats getAdaptiveStats() const { return adaptive_stats_; }
    
private:
    bool adaptive_mode_enabled_;
    double movement_threshold_;
    int history_size_;
    
    // Parameter sets for different scenarios
    struct ParameterSet {
        double epipolar_threshold;
        double max_assignment_cost;
        std::string name;
    };
    
    ParameterSet stationary_params_;
    ParameterSet moving_params_;
    ParameterSet default_params_;
    
    // Frame history for movement analysis
    std::deque<FrameData> frame_history_;
    std::deque<double> movement_history_;
    
    // Statistics tracking
    mutable AdaptiveStats adaptive_stats_;
    
    // Helper methods
    double calculateFrameMovement(const FrameData& current_frame, const FrameData& previous_frame);
    bool isStationaryScene(const std::deque<double>& movement_history);
    ParameterSet selectOptimalParameters(bool is_stationary);
    void updateStatistics(bool is_stationary, double movement, const ParameterSet& params);
};

// Utility functions for movement analysis
double calculateTrackMovement(const TrackInfo& track1, const TrackInfo& track2);
double calculateCameraMovement(const CameraTracks& cam1, const CameraTracks& cam2);