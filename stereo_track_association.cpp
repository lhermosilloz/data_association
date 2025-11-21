#include "stereo_track_association.h"

StereoTrackAssociator::StereoTrackAssociator() 
    : epipolar_threshold_(10.0), max_assignment_cost_(1.0) {
    initializeCameraMatrices();
}

void StereoTrackAssociator::initializeCameraMatrices() {
    // Camera 2 intrinsics
    K2_ = (cv::Mat_<double>(3,3) << 
        690.25849569, 0.0, 614.21523163,
        0.0, 689.6248818, 338.94900124,
        0.0, 0.0, 1.0);
    
    dist2_ = (cv::Mat_<double>(1,5) << 
        -3.58152939e-01, 1.85211072e-01, 7.39179696e-05, 1.70375285e-04, -5.82790603e-02);
    
    // Camera 3 intrinsics
    K3_ = (cv::Mat_<double>(3,3) << 
        691.76303387, 0.0, 631.403727,
        0.0, 691.18295277, 362.73688706,
        0.0, 0.0, 1.0);
    
    dist3_ = (cv::Mat_<double>(1,5) << 
        -3.57820636e-01, 1.84797837e-01, 2.20893356e-04, 9.55935431e-05, -5.91141748e-02);
    
    // Compute fundamental matrix (Camera 2 to Camera 3)
    // R = I, T = [0.125, 0, 0] (125mm baseline)
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat T = (cv::Mat_<double>(3,1) << 0.125, 0.0, 0.0);
    
    // Essential matrix: E = [T]_x * R
    cv::Mat T_cross = (cv::Mat_<double>(3,3) << 
        0, -T.at<double>(2,0), T.at<double>(1,0),
        T.at<double>(2,0), 0, -T.at<double>(0,0),
        -T.at<double>(1,0), T.at<double>(0,0), 0);
    
    cv::Mat E = T_cross * R;
    
    // Fundamental matrix: F = inv(K3)^T * E * inv(K2)
    cv::Mat K2_inv, K3_inv;
    cv::invert(K2_, K2_inv);
    cv::invert(K3_, K3_inv);
    F_2to3_ = K3_inv.t() * E * K2_inv;
}

cv::Point2d StereoTrackAssociator::undistortPixel(double x, double y, const cv::Mat& K, const cv::Mat& dist) {
    std::vector<cv::Point2f> distorted_points = {cv::Point2f(x, y)};
    std::vector<cv::Point2f> undistorted_points;
    cv::undistortPoints(distorted_points, undistorted_points, K, dist, cv::noArray(), K);
    return cv::Point2d(undistorted_points[0].x, undistorted_points[0].y);
}

double StereoTrackAssociator::computeEpipolarDistance(const TrackInfo& track1, const TrackInfo& track2) {
    // Undistort points
    cv::Point2d p1_ud = undistortPixel(track1.bb_center_x, track1.bb_center_y, K2_, dist2_);
    cv::Point2d p2_ud = undistortPixel(track2.bb_center_x, track2.bb_center_y, K3_, dist3_);
    
    // Compute epipolar line: l = F * p1
    cv::Mat p1_homo = (cv::Mat_<double>(3,1) << p1_ud.x, p1_ud.y, 1.0);
    cv::Mat epiline = F_2to3_ * p1_homo;
    
    double a = epiline.at<double>(0,0);
    double b = epiline.at<double>(1,0);
    double c = epiline.at<double>(2,0);
    
    // Distance from point to line: |ax + by + c| / sqrt(a² + b²)
    double distance = std::abs(a * p2_ud.x + b * p2_ud.y + c) / std::sqrt(a*a + b*b);
    return distance;
}

double StereoTrackAssociator::epipolarDistanceToCost(double distance) {
    if (distance > epipolar_threshold_) {
        return 1e6; // Invalid - hard constraint
    }
    // Smooth cost function: sigmoid-like curve
    return 1.0 / (1.0 + std::exp(-0.5 * (distance - 5.0)));
}

double StereoTrackAssociator::computeWidthHeightRatioCost(const TrackInfo& track1, const TrackInfo& track2) {
    double ratio1 = static_cast<double>(track1.width) / track1.height;
    double ratio2 = static_cast<double>(track2.width) / track2.height;
    
    double ratio_diff = std::abs(ratio1 - ratio2) / std::max(ratio1, ratio2);
    return std::min(1.0, ratio_diff * 2.0); // Scale and clamp
}

double StereoTrackAssociator::computeClassificationCost(const TrackInfo& track1, const TrackInfo& track2) {
    // Compare all three confidence pairs
    double armed_diff = std::abs(track1.armed_confidence - track2.armed_confidence);
    double military_diff = std::abs(track1.military_confidence - track2.military_confidence);
    double surrender_diff = std::abs(track1.surrender_confidence - track2.surrender_confidence);
    
    // Average difference normalized to [0,1]
    return (armed_diff + military_diff + surrender_diff) / 3.0;
}

double StereoTrackAssociator::computeFinalCost(double epipolar_cost, double ratio_cost, double conf_cost) {
    // Weighted combination - epipolar constraint is most important
    const double w_epi = 0.5;   // Geometric constraint (most reliable)
    const double w_ratio = 0.2; // Size similarity (moderate reliability)
    const double w_conf = 0.3;  // Classification similarity (helps disambiguation)
    
    // If epipolar cost is infinite, return infinite (hard constraint)
    if (epipolar_cost > 1e5) {
        return 1e6;
    }
    
    return w_epi * epipolar_cost + w_ratio * ratio_cost + w_conf * conf_cost;
}

std::unordered_map<int, std::vector<std::pair<int, bool>>> StereoTrackAssociator::performEpipolarGating(const FrameData& frame) {
    std::unordered_map<int, std::vector<std::pair<int, bool>>> binary_mask;
    
    // Find camera tracks
    std::vector<TrackInfo> tracks_cam1, tracks_cam2;
    for (const auto& camera_track : frame.camera_tracks) {
        if (camera_track.camera_id == 2) {
            tracks_cam1 = camera_track.tracks;
        } else if (camera_track.camera_id == 3) {
            tracks_cam2 = camera_track.tracks;
        }
    }
    
    // Test all pairs
    for (const auto& track1 : tracks_cam1) {
        for (const auto& track2 : tracks_cam2) {
            double distance = computeEpipolarDistance(track1, track2);
            bool passes_gate = (distance <= epipolar_threshold_);
            
            binary_mask[track1.track_id].emplace_back(track2.track_id, passes_gate);
        }
    }
    
    return binary_mask;
}

std::unordered_map<int, std::unordered_map<int, TrackInfo>> StereoTrackAssociator::createLookupTable(const FrameData& frame) {
    std::unordered_map<int, std::unordered_map<int, TrackInfo>> lookup;
    
    for (const auto& camera_track : frame.camera_tracks) {
        for (const auto& track : camera_track.tracks) {
            lookup[camera_track.camera_id][track.track_id] = track;
        }
    }
    
    return lookup;
}

std::unordered_map<int, std::unordered_map<int, float>> StereoTrackAssociator::buildCostMatrix(
    const FrameData& frame,
    const std::unordered_map<int, std::vector<std::pair<int, bool>>>& binary_mask,
    const std::unordered_map<int, std::unordered_map<int, TrackInfo>>& lookup) {
    
    std::unordered_map<int, std::unordered_map<int, float>> cost_matrix;
    std::set<std::pair<int, int>> processed_pairs;
    
    for (const auto& entry : binary_mask) {
        int idA = entry.first;
        
        if (lookup.at(2).find(idA) == lookup.at(2).end()) {
            continue;
        }
        
        for (const auto& pair : entry.second) {
            if (pair.second) { // If allowed by epipolar gating
                int idB = pair.first;
                
                if (lookup.at(3).find(idB) == lookup.at(3).end()) {
                    continue;
                }
                
                std::pair<int, int> track_pair = std::make_pair(std::min(idA, idB), std::max(idA, idB));
                if (processed_pairs.find(track_pair) != processed_pairs.end()) {
                    continue;
                }
                processed_pairs.insert(track_pair);

                const TrackInfo& trackA = lookup.at(2).at(idA);
                const TrackInfo& trackB = lookup.at(3).at(idB);

                double d_epi = computeEpipolarDistance(trackA, trackB);
                double c_epi = epipolarDistanceToCost(d_epi);
                double c_ratio = computeWidthHeightRatioCost(trackA, trackB);
                double c_conf = computeClassificationCost(trackA, trackB);
                double c_final = computeFinalCost(c_epi, c_ratio, c_conf);

                cost_matrix[idA][idB] = static_cast<float>(c_final);
            }
        }
    }
    
    return cost_matrix;
}

std::vector<std::pair<int, int>> StereoTrackAssociator::solveAssignment(
    const std::unordered_map<int, std::unordered_map<int, float>>& cost_matrix,
    const std::vector<TrackInfo>& tracks_cam1,
    const std::vector<TrackInfo>& tracks_cam2) {
    
    std::vector<std::pair<int, int>> assignments;
    
    // Create pairs sorted by cost (greedy Hungarian approximation)
    std::vector<std::tuple<double, int, int>> cost_pairs;
    
    for (const auto& row : cost_matrix) {
        for (const auto& col : row.second) {
            if (col.second < max_assignment_cost_) {
                cost_pairs.push_back(std::make_tuple(col.second, row.first, col.first));
            }
        }
    }
    
    std::sort(cost_pairs.begin(), cost_pairs.end());
    
    // Track assigned IDs
    std::set<int> assigned_cam1, assigned_cam2;
    
    for (const auto& pair : cost_pairs) {
        int id1 = std::get<1>(pair);
        int id2 = std::get<2>(pair);
        
        if (assigned_cam1.find(id1) == assigned_cam1.end() && 
            assigned_cam2.find(id2) == assigned_cam2.end()) {
            assignments.emplace_back(id1, id2);
            assigned_cam1.insert(id1);
            assigned_cam2.insert(id2);
        }
    }
    
    return assignments;
}

Point3D StereoTrackAssociator::triangulatePoints(const TrackInfo& track_cam1, const TrackInfo& track_cam2) {
    // Undistort points
    cv::Point2d p1_ud = undistortPixel(track_cam1.bb_center_x, track_cam1.bb_center_y, K2_, dist2_);
    cv::Point2d p2_ud = undistortPixel(track_cam2.bb_center_x, track_cam2.bb_center_y, K3_, dist3_);
    
    // Build projection matrices P1 = K2[I|0], P2 = K3[R|t]  
    cv::Mat P1 = K2_ * (cv::Mat_<double>(3,4) << 
        1, 0, 0, 0,
        0, 1, 0, 0, 
        0, 0, 1, 0);
    
    cv::Mat Rt = (cv::Mat_<double>(3,4) << 
        1, 0, 0, 0.125,  // R = I, t = [0.125, 0, 0]
        0, 1, 0, 0,
        0, 0, 1, 0);
    
    cv::Mat P2 = K3_ * Rt;
    
    // Triangulate using DLT
    cv::Mat A = (cv::Mat_<double>(4,4) <<
        p1_ud.x * P1.at<double>(2,0) - P1.at<double>(0,0), 
        p1_ud.x * P1.at<double>(2,1) - P1.at<double>(0,1),
        p1_ud.x * P1.at<double>(2,2) - P1.at<double>(0,2),
        p1_ud.x * P1.at<double>(2,3) - P1.at<double>(0,3),
        
        p1_ud.y * P1.at<double>(2,0) - P1.at<double>(1,0),
        p1_ud.y * P1.at<double>(2,1) - P1.at<double>(1,1), 
        p1_ud.y * P1.at<double>(2,2) - P1.at<double>(1,2),
        p1_ud.y * P1.at<double>(2,3) - P1.at<double>(1,3),
        
        p2_ud.x * P2.at<double>(2,0) - P2.at<double>(0,0),
        p2_ud.x * P2.at<double>(2,1) - P2.at<double>(0,1),
        p2_ud.x * P2.at<double>(2,2) - P2.at<double>(0,2), 
        p2_ud.x * P2.at<double>(2,3) - P2.at<double>(0,3),
        
        p2_ud.y * P2.at<double>(2,0) - P2.at<double>(1,0),
        p2_ud.y * P2.at<double>(2,1) - P2.at<double>(1,1),
        p2_ud.y * P2.at<double>(2,2) - P2.at<double>(1,2),
        p2_ud.y * P2.at<double>(2,3) - P2.at<double>(1,3));
    
    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt);
    cv::Mat X_homogeneous = vt.row(3).t();
    
    Point3D result;
    if (std::abs(X_homogeneous.at<double>(3)) > 1e-6) {
        result.x = X_homogeneous.at<double>(0) / X_homogeneous.at<double>(3);
        result.y = X_homogeneous.at<double>(1) / X_homogeneous.at<double>(3);
        result.z = X_homogeneous.at<double>(2) / X_homogeneous.at<double>(3);
        result.depth = std::sqrt(result.x*result.x + result.y*result.y + result.z*result.z);
        
        // Basic validation
        result.reprojection_error = 0.0; // Simplified for now
        result.is_valid = (result.depth > 0.5 && result.depth < 100.0);
    } else {
        result.x = result.y = result.z = result.depth = 0.0;
        result.reprojection_error = 1e6;
        result.is_valid = false;
    }
    
    return result;
}

StereoAssociationResult StereoTrackAssociator::processFrame(const FrameData& frame_data) {
    StereoAssociationResult result;
    result.frame_id = frame_data.frame_id;
    
    // Get camera tracks
    std::vector<TrackInfo> tracks_cam1, tracks_cam2;
    for (const auto& camera_track : frame_data.camera_tracks) {
        if (camera_track.camera_id == 2) {
            tracks_cam1 = camera_track.tracks;
        } else if (camera_track.camera_id == 3) {
            tracks_cam2 = camera_track.tracks;
        }
    }
    
    if (tracks_cam1.empty() || tracks_cam2.empty()) {
        // No tracks to associate
        for (const auto& track : tracks_cam1) {
            result.unmatched_camera1_tracks.push_back(track.track_id);
        }
        for (const auto& track : tracks_cam2) {
            result.unmatched_camera2_tracks.push_back(track.track_id);
        }
        result.average_depth = 0.0;
        result.valid_triangulations = 0;
        result.average_reprojection_error = 0.0;
        return result;
    }
    
    // 1. Epipolar gating
    auto binary_mask = performEpipolarGating(frame_data);
    auto lookup = createLookupTable(frame_data);
    
    if (binary_mask.empty()) {
        // No valid pairs
        for (const auto& track : tracks_cam1) {
            result.unmatched_camera1_tracks.push_back(track.track_id);
        }
        for (const auto& track : tracks_cam2) {
            result.unmatched_camera2_tracks.push_back(track.track_id);
        }
        result.average_depth = 0.0;
        result.valid_triangulations = 0;
        result.average_reprojection_error = 0.0;
        return result;
    }
    
    // 2. Build cost matrix
    auto cost_matrix = buildCostMatrix(frame_data, binary_mask, lookup);
    
    // 3. Solve assignment
    auto assignments = solveAssignment(cost_matrix, tracks_cam1, tracks_cam2);
    
    // 4. Process assignments and triangulate
    std::set<int> matched_cam1, matched_cam2;
    double total_depth = 0.0;
    double total_reproj_error = 0.0;
    int valid_count = 0;
    
    for (const auto& assignment : assignments) {
        int id1 = assignment.first;
        int id2 = assignment.second;
        
        matched_cam1.insert(id1);
        matched_cam2.insert(id2);
        
        // Find the tracks
        const TrackInfo* track1_ptr = nullptr;
        const TrackInfo* track2_ptr = nullptr;
        
        for (const auto& track : tracks_cam1) {
            if (track.track_id == id1) {
                track1_ptr = &track;
                break;
            }
        }
        
        for (const auto& track : tracks_cam2) {
            if (track.track_id == id2) {
                track2_ptr = &track;
                break;
            }
        }
        
        if (track1_ptr && track2_ptr) {
            Point3D world_pos = triangulatePoints(*track1_ptr, *track2_ptr);
            
            CorrespondedTrack corresponded_track;
            corresponded_track.camera1_track_id = id1;
            corresponded_track.camera2_track_id = id2;
            corresponded_track.assignment_cost = cost_matrix.at(id1).at(id2);
            corresponded_track.world_position = world_pos;
            corresponded_track.depth_meters = world_pos.depth;
            corresponded_track.is_valid = world_pos.is_valid;
            corresponded_track.epipolar_distance = computeEpipolarDistance(*track1_ptr, *track2_ptr);
            
            result.corresponded_tracks.push_back(corresponded_track);
            
            if (world_pos.is_valid) {
                total_depth += world_pos.depth;
                total_reproj_error += world_pos.reprojection_error;
                valid_count++;
            }
        }
    }
    
    // 5. Collect unmatched tracks
    for (const auto& track : tracks_cam1) {
        if (matched_cam1.find(track.track_id) == matched_cam1.end()) {
            result.unmatched_camera1_tracks.push_back(track.track_id);
        }
    }
    
    for (const auto& track : tracks_cam2) {
        if (matched_cam2.find(track.track_id) == matched_cam2.end()) {
            result.unmatched_camera2_tracks.push_back(track.track_id);
        }
    }
    
    // 6. Calculate statistics
    result.average_depth = (valid_count > 0) ? total_depth / valid_count : 0.0;
    result.valid_triangulations = valid_count;
    result.average_reprojection_error = (valid_count > 0) ? total_reproj_error / valid_count : 0.0;
    
    return result;
}