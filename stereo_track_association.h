#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <iomanip>

// Forward declarations and data structures
struct TrackInfo {
    int track_id;
    int width;
    int height;
    int bb_center_x;
    int bb_center_y;
    float armed_confidence;
    float unarmed_confidence;
    float military_confidence;
    float civilian_confidence;
    float surrender_confidence;
    float no_surrender_confidence;
};

struct CameraTracks {
    int camera_id;
    std::vector<TrackInfo> tracks;
};

struct FrameData {
    int frame_id;
    std::vector<CameraTracks> camera_tracks;
};

struct Point3D {
    double x, y, z;
    double depth;
    bool is_valid;
    double reprojection_error;
};

struct CorrespondedTrack {
    int camera1_track_id;
    int camera2_track_id;
    float assignment_cost;
    Point3D world_position;
    double depth_meters;
    bool is_valid;              // True if triangulation succeeded and depth is in [0.5m, 100m] range
    double epipolar_distance;   // Epipolar constraint distance in pixels
};

struct StereoAssociationResult {
    int frame_id;
    std::vector<CorrespondedTrack> corresponded_tracks;
    std::vector<int> unmatched_camera1_tracks;
    std::vector<int> unmatched_camera2_tracks;
    double average_depth;
    int valid_triangulations;
    double average_reprojection_error;
};

// Main stereo track association pipeline
class StereoTrackAssociator {
public:
    StereoTrackAssociator();
    
    // Main processing function
    StereoAssociationResult processFrame(const FrameData& frame_data);
    
    // Configuration
    void setEpipolarThreshold(double threshold) { epipolar_threshold_ = threshold; }
    void setMaxAssignmentCost(double max_cost) { max_assignment_cost_ = max_cost; }
    
private:
    double epipolar_threshold_;
    double max_assignment_cost_;
    
    // Internal camera calibration matrices
    cv::Mat K2_, K3_;
    cv::Mat dist2_, dist3_;
    cv::Mat F_2to3_;
    
    // Internal processing methods
    void initializeCameraMatrices();
    cv::Point2d undistortPixel(double x, double y, const cv::Mat& K, const cv::Mat& dist);
    std::unordered_map<int, std::vector<std::pair<int, bool>>> performEpipolarGating(const FrameData& frame);
    std::unordered_map<int, std::unordered_map<int, TrackInfo>> createLookupTable(const FrameData& frame);
    std::unordered_map<int, std::unordered_map<int, float>> buildCostMatrix(
        const FrameData& frame,
        const std::unordered_map<int, std::vector<std::pair<int, bool>>>& binary_mask,
        const std::unordered_map<int, std::unordered_map<int, TrackInfo>>& lookup);
    std::vector<std::pair<int, int>> solveAssignment(
        const std::unordered_map<int, std::unordered_map<int, float>>& cost_matrix,
        const std::vector<TrackInfo>& tracks_cam1,
        const std::vector<TrackInfo>& tracks_cam2);
    Point3D triangulatePoints(const TrackInfo& track_cam1, const TrackInfo& track_cam2);
    
    // Cost computation methods
    double computeEpipolarDistance(const TrackInfo& track1, const TrackInfo& track2);
    double epipolarDistanceToCost(double distance);
    double computeWidthHeightRatioCost(const TrackInfo& track1, const TrackInfo& track2);
    double computeClassificationCost(const TrackInfo& track1, const TrackInfo& track2);
    double computeFinalCost(double epipolar_cost, double ratio_cost, double conf_cost);
};