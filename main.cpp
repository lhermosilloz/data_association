#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <string>
#include <set>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <random>
#include <cmath>
#include <opencv2/calib3d.hpp>


struct TrackInfo{
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

// Assignment result structure
struct TrackAssignment {
    int camera1_track_id;
    int camera2_track_id;
    float assignment_cost;
    bool is_valid;
};

struct AssignmentResult {
    int frame_id;
    std::vector<TrackAssignment> assignments;
    std::vector<int> unassigned_camera1_tracks;
    std::vector<int> unassigned_camera2_tracks;
};

// 3D Point structure for stereo triangulation
struct Point3D {
    double x, y, z;
    double depth;
    bool is_valid;
    double reprojection_error;
};

// Enhanced assignment result with depth information
struct TrackAssignmentWithDepth {
    int camera1_track_id;
    int camera2_track_id;
    float assignment_cost;
    bool is_valid;
    Point3D world_position;
    double depth_meters;
    double epipolar_distance;
};

struct EnhancedAssignmentResult {
    int frame_id;
    std::vector<TrackAssignmentWithDepth> assignments;
    std::vector<int> unassigned_camera1_tracks;
    std::vector<int> unassigned_camera2_tracks;
    double average_depth;
    int valid_triangulations;
    double average_reprojection_error;
};

// --- Stereo calibration data (Camera 2 = index 0, Camera 3 = index 1) ---

const cv::Mat K2 = (cv::Mat_<double>(3,3) <<
    690.25849569, 0.0,          614.21523163,
    0.0,          689.62488180, 338.94900124,
    0.0,          0.0,          1.0
);

const cv::Mat dist2 = (cv::Mat_<double>(1,5) <<
    -3.58152939e-01,  1.85211072e-01,
     7.39179696e-05,  1.70375285e-04,
    -5.82790603e-02
);

const cv::Mat K3 = (cv::Mat_<double>(3,3) <<
    691.76303387, 0.0,          631.40372700,
    0.0,          691.18295277, 362.73688706,
    0.0,          0.0,          1.0
);

const cv::Mat dist3 = (cv::Mat_<double>(1,5) <<
    -3.57820636e-01,  1.84797837e-01,
     2.20893356e-04,  9.55935431e-05,
    -5.91141748e-02
);

// R = I, T = [0.125, 0, 0]^T (meters)
const cv::Mat R_2to3 = (cv::Mat_<double>(3,3) <<
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0
);

const cv::Mat t_2to3 = (cv::Mat_<double>(3,1) <<
    0.125, 0.0, 0.0
);

// Fundamental matrix from Cam2 -> Cam3 (computed once)
cv::Mat computeFundamental_2to3()
{
    // [t]_x
    double tx = t_2to3.at<double>(0);
    double ty = t_2to3.at<double>(1);
    double tz = t_2to3.at<double>(2);

    cv::Mat Tx = (cv::Mat_<double>(3,3) <<
         0.0, -tz,  ty,
         tz,  0.0, -tx,
        -ty,  tx,  0.0
    );

    cv::Mat E = Tx * R_2to3;           // Essential matrix
    cv::Mat K2_inv = K2.inv();
    cv::Mat K3_inv = K3.inv();

    cv::Mat F = K3_inv.t() * E * K2_inv;
    return F;
}

// Undistort a single pixel (x, y) with given intrinsics/distortion,
// returning undistorted pixel coords (still in pixel units).
static inline cv::Point2d undistortPixel(
    double x, double y,
    const cv::Mat& K,
    const cv::Mat& dist)
{
    std::vector<cv::Point2f> src(1);
    src[0] = cv::Point2f(static_cast<float>(x), static_cast<float>(y));

    std::vector<cv::Point2f> dst(1);

    // P = K so output is back in pixel coordinates, not normalized
    cv::undistortPoints(src, dst, K, dist, cv::noArray(), K);

    return cv::Point2d(dst[0].x, dst[0].y);
}


// Modified epipolar gating with debug output
std::unordered_map<int, std::vector<std::pair<int, bool>>> EpipolarGatingDebug(const FrameData& frame, bool verbose = false) {
    std::unordered_map<int, std::vector<std::pair<int, bool>>> binary_mask;

    if (frame.camera_tracks.size() < 2) {
        return binary_mask;
    }

    const CameraTracks& camA = frame.camera_tracks[0];
    const CameraTracks& camB = frame.camera_tracks[1];

    const cv::Mat F_2to3 = computeFundamental_2to3();
    const cv::Mat F_3to2 = F_2to3.t();

    // Test multiple thresholds
    std::vector<double> thresholds = {1.0, 2.0, 5.0, 10.0, 20.0, 50.0};
    
    if (verbose) {
        std::cout << "\n--- EPIPOLAR DISTANCE ANALYSIS ---" << std::endl;
    }

    // Direction: camA (2) -> camB (3)
    for (const auto& trackA : camA.tracks) {
        cv::Point2d pA_ud = undistortPixel(
            trackA.bb_center_x,
            trackA.bb_center_y,
            K2, dist2
        );

        cv::Mat xA = (cv::Mat_<double>(3,1) <<
            pA_ud.x,
            pA_ud.y,
            1.0
        );

        cv::Mat lB = F_2to3 * xA;
        double a = lB.at<double>(0);
        double b = lB.at<double>(1);
        double c = lB.at<double>(2);
        double denom = std::sqrt(a*a + b*b) + 1e-12;

        for (const auto& trackB : camB.tracks) {
            cv::Point2d pB_ud = undistortPixel(
                trackB.bb_center_x,
                trackB.bb_center_y,
                K3, dist3
            );

            double num = std::abs(a * pB_ud.x + b * pB_ud.y + c);
            double distance = num / denom;

            if (verbose) {
                std::cout << "Track " << trackA.track_id << " (" << trackA.bb_center_x 
                          << "," << trackA.bb_center_y << ") -> Track " << trackB.track_id 
                          << " (" << trackB.bb_center_x << "," << trackB.bb_center_y 
                          << "): epipolar distance = " << distance << " pixels" << std::endl;
                
                for (double thresh : thresholds) {
                    std::cout << "  Threshold " << thresh << "px: " 
                              << (distance < thresh ? "PASS" : "FAIL") << std::endl;
                }
                std::cout << std::endl;
            }

            // Use more realistic threshold for unrectified images
            bool allowed = (distance < 30.0); // Increased from 5.0
            binary_mask[trackA.track_id].push_back(
                std::make_pair(trackB.track_id, allowed)
            );
        }
    }

    // Direction: camB (3) -> camA (2) - similar analysis
    for (const auto& trackB : camB.tracks) {
        cv::Point2d pB_ud = undistortPixel(
            trackB.bb_center_x,
            trackB.bb_center_y,
            K3, dist3
        );

        cv::Mat xB = (cv::Mat_<double>(3,1) <<
            pB_ud.x,
            pB_ud.y,
            1.0
        );

        cv::Mat lA = F_3to2 * xB;
        double a = lA.at<double>(0);
        double b = lA.at<double>(1);
        double c = lA.at<double>(2);
        double denom = std::sqrt(a*a + b*b) + 1e-12;

        for (const auto& trackA : camA.tracks) {
            cv::Point2d pA_ud = undistortPixel(
                trackA.bb_center_x,
                trackA.bb_center_y,
                K2, dist2
            );

            double num = std::abs(a * pA_ud.x + b * pA_ud.y + c);
            double distance = num / denom;

            bool allowed = (distance < 30.0); // Increased threshold
            binary_mask[trackB.track_id].push_back(
                std::make_pair(trackA.track_id, allowed)
            );
        }
    }

    return binary_mask;
}

// Wrapper function for backwards compatibility
std::unordered_map<int, std::vector<std::pair<int, bool>>> EpipolarGating(const FrameData& frame) {
    return EpipolarGatingDebug(frame, false);
}

// Helper function to print test results with better formatting
void printTestResults(const int& test_name, const std::unordered_map<int, std::vector<std::pair<int, bool>>>& binary_mask) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    
    if (binary_mask.empty()) {
        std::cout << "No track associations found." << std::endl;
        return;
    }
    
    // Count allowed and total associations
    int total_associations = 0;
    int allowed_associations = 0;
    
    for (const auto& entry : binary_mask) {
        for (const auto& pair : entry.second) {
            total_associations++;
            if (pair.second) {
                allowed_associations++;
                std::cout << "Track " << entry.first << " <-> Track " 
                          << pair.first << ": ALLOWED" << std::endl;
            }
        }
    }
    
    std::cout << "Allowed associations: " << allowed_associations << "/" << total_associations << std::endl;
}

// Enhanced helper function to create a track with proper classification constraints
TrackInfo createTrack(int id, int x, int y, 
                     float armed_ratio = 0.5f,     // 0.0 = unarmed, 1.0 = armed
                     float military_ratio = 0.3f,  // 0.0 = civilian, 1.0 = military  
                     float surrender_ratio = 0.1f, // 0.0 = no_surrender, 1.0 = surrender
                     int width = 80, int height = 120) {
    
    // Enforce mutual exclusivity constraints - one must be near 0, the other near 1
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

// Validation function to check classification constraints
bool validateTrackClassifications(const TrackInfo& track) {
    const float tolerance = 0.001f;
    
    bool armed_valid = std::abs((track.armed_confidence + track.unarmed_confidence) - 1.0f) < tolerance;
    bool military_valid = std::abs((track.military_confidence + track.civilian_confidence) - 1.0f) < tolerance;
    bool surrender_valid = std::abs((track.surrender_confidence + track.no_surrender_confidence) - 1.0f) < tolerance;
    
    return armed_valid && military_valid && surrender_valid;
}

// Helper function to generate realistic track dimensions based on object type and depth
std::pair<int, int> generateRealisticDimensions(const std::string& object_type, double depth_meters) {
    double base_width_pixels, base_height_pixels;
    
    // Human dimensions at reference distance (5m)
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

// Helper function to calculate expected disparity for stereo cameras
// Given a 3D point depth, calculate expected pixel shift due to baseline
double calculateExpectedDisparity(double depth_meters, double baseline_meters = 0.125) {
    // Using simple stereo disparity: disparity = (baseline * focal_length) / depth
    // Average focal length from both cameras
    double avg_focal_length = (690.25849569 + 691.76303387) / 2.0;
    return (baseline_meters * avg_focal_length) / depth_meters;
}

// Generate realistic stereo corresponding points
std::pair<cv::Point2d, cv::Point2d> generateStereoCorrespondence(
    const cv::Point2d& left_point, 
    double depth_meters,
    double baseline_meters = 0.125) {
    
    // Calculate expected disparity (mainly in x-direction for horizontal baseline)
    double disparity = calculateExpectedDisparity(depth_meters, baseline_meters);
    
    // Add some realistic noise/uncertainty
    double noise_x = (rand() % 3 - 1) * 0.5; // ±0.5 pixel noise
    double noise_y = (rand() % 3 - 1) * 0.2; // ±0.2 pixel noise in y

    double cy_diff = K3.at<double>(1, 2) - K2.at<double>(1, 2); // cy3 - cy2
    
    cv::Point2d right_point;
    right_point.x = left_point.x - disparity + noise_x; // Negative because right camera sees object shifted left
    right_point.y = left_point.y + cy_diff + noise_y;
    
    return std::make_pair(left_point, right_point);
}

// Test case for debugging epipolar distances with realistic stereo data
FrameData createRealisticDebugTestCase() {
    FrameData frame;
    frame.frame_id = 0;

    CameraTracks cam1, cam2;
    cam1.camera_id = 2;
    cam2.camera_id = 3;
    
    // Test objects at different depths
    struct TestObject {
        cv::Point2d point;
        double depth;
        int track_id;
    };
    
    std::vector<TestObject> objects = {
        {{400, 300}, 2.0, 1},  // Close object (2m) - large disparity
        {{600, 200}, 5.0, 2},  // Medium distance (5m) - medium disparity  
        {{200, 400}, 10.0, 3}, // Far object (10m) - small disparity
        {{800, 350}, 20.0, 4}  // Very far object (20m) - very small disparity
    };
    
    for (const auto& obj : objects) {
        auto correspondence = generateStereoCorrespondence(obj.point, obj.depth);
        
        // Add to left camera (cam1)
        cam1.tracks.push_back(createTrack(obj.track_id, 
                                        (int)correspondence.first.x, 
                                        (int)correspondence.first.y));
        
        // Add corresponding point to right camera (cam2)
        cam2.tracks.push_back(createTrack(obj.track_id + 10, 
                                        (int)correspondence.second.x, 
                                        (int)correspondence.second.y));
        
        std::cout << "Object " << obj.track_id << " at depth " << obj.depth << "m: "
                  << "Left(" << correspondence.first.x << "," << correspondence.first.y << ") -> "
                  << "Right(" << correspondence.second.x << "," << correspondence.second.y << ") "
                  << "Disparity: " << (correspondence.first.x - correspondence.second.x) << "px" << std::endl;
    }
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

double computeEpipolarDistanceTracks(
    const TrackInfo& trackA,
    const TrackInfo& trackB,
    const cv::Mat& F_2to3)
{
    // Undistort both centers
    cv::Point2d pA_ud = undistortPixel(
        trackA.bb_center_x,
        trackA.bb_center_y,
        K2, dist2
    );

    cv::Point2d pB_ud = undistortPixel(
        trackB.bb_center_x,
        trackB.bb_center_y,
        K3, dist3
    );

    // Build homogeneous point in cam2
    cv::Mat xA = (cv::Mat_<double>(3,1) << pA_ud.x, pA_ud.y, 1.0);

    // Epipolar line in cam3
    cv::Mat lB = F_2to3 * xA;

    double a = lB.at<double>(0);
    double b = lB.at<double>(1);
    double c = lB.at<double>(2);

    double denom = std::sqrt(a*a + b*b) + 1e-12;
    double num   = std::abs(a * pB_ud.x + b * pB_ud.y + c);

    return num / denom;  // distance in pixels
}

double epipolarDistanceToCost(double distance)
{
    // Gaussian soft gate
    const double sigma   = 2.0;    // tune this
    const double d_hard  = 30.0;   // same as the gate

    if (distance > d_hard) {
        return 1e6; // "infinite" cost
    }

    double gate = std::exp(-0.5 * (distance * distance) / (sigma * sigma));
    double cost = 1.0 - gate;  // 0..1

    return cost;
}

double computeWidthHeightRatioCost(const TrackInfo& trackA, const TrackInfo& trackB)
{
    // Calculate aspect ratios for both tracks
    double ratioA = (double)trackA.width / (trackA.height + 1e-6); // Avoid division by zero
    double ratioB = (double)trackB.width / (trackB.height + 1e-6);
    
    // Calculate relative difference (scale-invariant)
    double ratio_diff = std::abs(ratioA - ratioB) / std::max(ratioA, ratioB);
    
    // Normalize to [0,1] range using sigmoid-like function
    // Objects with similar aspect ratios should have low cost
    double max_expected_diff = 0.5; // Tune this threshold
    double normalized_cost = std::min(1.0, ratio_diff / max_expected_diff);
    
    return normalized_cost;
}

double computeSubclassConfidenceCost(const TrackInfo& trackA, const TrackInfo& trackB)
{
    // Compare confidence distributions using weighted L1 distance
    // Focus on the most discriminative confidence pairs
    
    double armed_diff = std::abs(trackA.armed_confidence - trackB.armed_confidence);
    double military_diff = std::abs(trackA.military_confidence - trackB.military_confidence);
    double surrender_diff = std::abs(trackA.surrender_confidence - trackB.surrender_confidence);
    
    // Weight armed/unarmed and surrender more heavily as they're more discriminative
    double weighted_diff = 0.4 * armed_diff + 0.3 * military_diff + 0.3 * surrender_diff;
    
    // Normalize to [0,1] range
    // Since confidences are in [0,1], max possible diff is 1.0
    return std::min(1.0, weighted_diff);
}

double computeFinalCost(double epipolar_cost, double ratio_cost, double conf_cost)
{
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

// Stereo triangulation function
Point3D triangulatePoints(const TrackInfo& track_cam2, const TrackInfo& track_cam3) {
    // Undistort points
    cv::Point2d p2_ud = undistortPixel(track_cam2.bb_center_x, track_cam2.bb_center_y, K2, dist2);
    cv::Point2d p3_ud = undistortPixel(track_cam3.bb_center_x, track_cam3.bb_center_y, K3, dist3);
    
    // Build projection matrices P1 = K2[I|0], P2 = K3[R|t]  
    cv::Mat P1 = K2 * (cv::Mat_<double>(3,4) << 
        1, 0, 0, 0,
        0, 1, 0, 0, 
        0, 0, 1, 0);
    
    cv::Mat Rt = (cv::Mat_<double>(3,4) << 
        1, 0, 0, 0.125,  // R = I, t = [0.125, 0, 0]
        0, 1, 0, 0,
        0, 0, 1, 0);
    
    cv::Mat P2 = K3 * Rt;
    
    // Triangulate using DLT
    cv::Mat A = (cv::Mat_<double>(4,4) <<
        p2_ud.x * P1.at<double>(2,0) - P1.at<double>(0,0), 
        p2_ud.x * P1.at<double>(2,1) - P1.at<double>(0,1),
        p2_ud.x * P1.at<double>(2,2) - P1.at<double>(0,2),
        p2_ud.x * P1.at<double>(2,3) - P1.at<double>(0,3),
        
        p2_ud.y * P1.at<double>(2,0) - P1.at<double>(1,0),
        p2_ud.y * P1.at<double>(2,1) - P1.at<double>(1,1), 
        p2_ud.y * P1.at<double>(2,2) - P1.at<double>(1,2),
        p2_ud.y * P1.at<double>(2,3) - P1.at<double>(1,3),
        
        p3_ud.x * P2.at<double>(2,0) - P2.at<double>(0,0),
        p3_ud.x * P2.at<double>(2,1) - P2.at<double>(0,1),
        p3_ud.x * P2.at<double>(2,2) - P2.at<double>(0,2), 
        p3_ud.x * P2.at<double>(2,3) - P2.at<double>(0,3),
        
        p3_ud.y * P2.at<double>(2,0) - P2.at<double>(1,0),
        p3_ud.y * P2.at<double>(2,1) - P2.at<double>(1,1),
        p3_ud.y * P2.at<double>(2,2) - P2.at<double>(1,2),
        p3_ud.y * P2.at<double>(2,3) - P2.at<double>(1,3));
    
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

// Simple Hungarian Algorithm implementation
class HungarianAssignment {
private:
    std::vector<std::vector<double>> cost_matrix;
    std::vector<int> assignment;
    std::vector<int> row_track_ids, col_track_ids;
    int n, m;
    
public:
    HungarianAssignment(const std::vector<int>& row_ids, const std::vector<int>& col_ids)
        : row_track_ids(row_ids), col_track_ids(col_ids) {
        n = row_ids.size();
        m = col_ids.size();
        
        // Make it a square matrix by padding with high costs
        int max_dim = std::max(n, m);
        cost_matrix.assign(max_dim, std::vector<double>(max_dim, 1e6));
        assignment.assign(max_dim, -1);
    }
    
    void setCost(int row_idx, int col_idx, double cost) {
        if (row_idx < n && col_idx < m) {
            cost_matrix[row_idx][col_idx] = cost;
        }
    }
    
    // Simplified Hungarian algorithm using greedy approach for now
    // For production use, consider a proper Hungarian implementation
    std::vector<TrackAssignment> solve(double max_cost_threshold = 1.0) {
        std::vector<TrackAssignment> assignments;
        std::vector<bool> row_assigned(n, false);
        std::vector<bool> col_assigned(m, false);
        
        // Create pairs sorted by cost
        std::vector<std::tuple<double, int, int>> cost_pairs;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (cost_matrix[i][j] < 1e5) { // Only consider valid costs
                    cost_pairs.push_back(std::make_tuple(cost_matrix[i][j], i, j));
                }
            }
        }
        
        // Sort by cost (ascending)
        std::sort(cost_pairs.begin(), cost_pairs.end());
        
        // Greedy assignment
        for (const auto& pair : cost_pairs) {
            double cost = std::get<0>(pair);
            int row = std::get<1>(pair);
            int col = std::get<2>(pair);
            
            if (cost > max_cost_threshold) {
                break; // Stop if cost too high
            }
            
            if (!row_assigned[row] && !col_assigned[col]) {
                assignments.push_back({
                    row_track_ids[row],
                    col_track_ids[col],
                    static_cast<float>(cost),
                    true
                });
                
                row_assigned[row] = true;
                col_assigned[col] = true;
            }
        }
        
        return assignments;
    }
    
    std::vector<int> getUnassignedRows() const {
        std::vector<int> unassigned;
        std::vector<bool> row_assigned(n, false);
        
        // Mark assigned rows (this is a simplified check)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (assignment[i] == j) {
                    row_assigned[i] = true;
                    break;
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            if (!row_assigned[i]) {
                unassigned.push_back(row_track_ids[i]);
            }
        }
        
        return unassigned;
    }
    
    std::vector<int> getUnassignedCols() const {
        std::vector<int> unassigned;
        std::vector<bool> col_assigned(m, false);
        
        // Mark assigned columns
        for (int i = 0; i < n; i++) {
            if (assignment[i] >= 0 && assignment[i] < m) {
                col_assigned[assignment[i]] = true;
            }
        }
        
        for (int j = 0; j < m; j++) {
            if (!col_assigned[j]) {
                unassigned.push_back(col_track_ids[j]);
            }
        }
        
        return unassigned;
    }
};

// Function to solve track assignment using Hungarian algorithm
AssignmentResult solveTrackAssignment(
    const FrameData& frame,
    const std::unordered_map<int, std::unordered_map<int, float>>& final_cost_matrix) {
    
    AssignmentResult result;
    result.frame_id = frame.frame_id;
    
    if (frame.camera_tracks.size() < 2) {
        return result;
    }
    
    // Extract track IDs for both cameras
    std::vector<int> cam1_track_ids, cam2_track_ids;
    
    for (const auto& track : frame.camera_tracks[0].tracks) {
        cam1_track_ids.push_back(track.track_id);
    }
    
    for (const auto& track : frame.camera_tracks[1].tracks) {
        cam2_track_ids.push_back(track.track_id);
    }
    
    if (cam1_track_ids.empty() || cam2_track_ids.empty()) {
        result.unassigned_camera1_tracks = cam1_track_ids;
        result.unassigned_camera2_tracks = cam2_track_ids;
        return result;
    }
    
    // Create Hungarian assignment solver
    HungarianAssignment hungarian(cam1_track_ids, cam2_track_ids);
    
    // Populate cost matrix
    for (int i = 0; i < cam1_track_ids.size(); i++) {
        int track_id1 = cam1_track_ids[i];
        
        if (final_cost_matrix.find(track_id1) != final_cost_matrix.end()) {
            for (int j = 0; j < cam2_track_ids.size(); j++) {
                int track_id2 = cam2_track_ids[j];
                
                auto it = final_cost_matrix.at(track_id1).find(track_id2);
                if (it != final_cost_matrix.at(track_id1).end()) {
                    hungarian.setCost(i, j, it->second);
                }
            }
        }
    }
    
    // Solve assignment
    result.assignments = hungarian.solve(1.0); // Max cost threshold
    
    // Find unassigned tracks
    std::set<int> assigned_cam1, assigned_cam2;
    for (const auto& assignment : result.assignments) {
        assigned_cam1.insert(assignment.camera1_track_id);
        assigned_cam2.insert(assignment.camera2_track_id);
    }
    
    for (int id : cam1_track_ids) {
        if (assigned_cam1.find(id) == assigned_cam1.end()) {
            result.unassigned_camera1_tracks.push_back(id);
        }
    }
    
    for (int id : cam2_track_ids) {
        if (assigned_cam2.find(id) == assigned_cam2.end()) {
            result.unassigned_camera2_tracks.push_back(id);
        }
    }
    
    return result;
}


// Test case 1: Equal number of tracks with realistic stereo geometry
FrameData createTestCase1() {
    FrameData frame;
    frame.frame_id = 1;

    CameraTracks cam1, cam2;
    cam1.camera_id = 2;
    cam2.camera_id = 3;
    
    // Define realistic test scenarios with proper depth
    struct ScenarioObject {
        cv::Point2d point;
        double depth;
        int left_id, right_id;
    };
    
    std::vector<ScenarioObject> objects = {
        {{150, 300}, 3.0, 1, 5},   // Left side, medium depth
        {{500, 300}, 5.0, 2, 6},   // Center, far depth
        {{850, 300}, 2.5, 3, 7},   // Right side, close depth
        {{350, 150}, 8.0, 4, 8}    // Upper area, very far
    };
    
    for (const auto& obj : objects) {
        auto correspondence = generateStereoCorrespondence(obj.point, obj.depth);
        
        cam1.tracks.push_back(createTrack(obj.left_id, 
                                        (int)correspondence.first.x, 
                                        (int)correspondence.first.y));
        cam2.tracks.push_back(createTrack(obj.right_id, 
                                        (int)correspondence.second.x, 
                                        (int)correspondence.second.y));
    }
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 2: Camera 1 has more tracks than Camera 2
FrameData createTestCase2() {
    FrameData frame;
    frame.frame_id = 22;

    CameraTracks cam1;
    cam1.camera_id = 2;
    cam1.tracks.push_back(createTrack(10, 200, 200));
    cam1.tracks.push_back(createTrack(11, 400, 200));
    cam1.tracks.push_back(createTrack(12, 600, 200));
    cam1.tracks.push_back(createTrack(13, 800, 200));
    cam1.tracks.push_back(createTrack(14, 300, 400));
    cam1.tracks.push_back(createTrack(15, 700, 400));
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    // Only 3 tracks in camera 2
    cam2.tracks.push_back(createTrack(20, 195, 195));
    cam2.tracks.push_back(createTrack(21, 395, 195));
    cam2.tracks.push_back(createTrack(22, 295, 395));
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 3: Camera 2 has more tracks than Camera 1
FrameData createTestCase3() {
    FrameData frame;
    frame.frame_id = 3;

    CameraTracks cam1;
    cam1.camera_id = 2;
    // Only 2 tracks in camera 1
    cam1.tracks.push_back(createTrack(30, 350, 250));
    cam1.tracks.push_back(createTrack(31, 750, 350));
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    // 5 tracks in camera 2
    cam2.tracks.push_back(createTrack(40, 345, 245));
    cam2.tracks.push_back(createTrack(41, 745, 345));
    cam2.tracks.push_back(createTrack(42, 100, 100));
    cam2.tracks.push_back(createTrack(43, 900, 500));
    cam2.tracks.push_back(createTrack(44, 640, 360)); // Center of image
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 4: Edge positions (corners and edges of image)
FrameData createTestCase4() {
    FrameData frame;
    frame.frame_id = 4;

    CameraTracks cam1;
    cam1.camera_id = 2;
    cam1.tracks.push_back(createTrack(50, 10, 10));     // Top-left corner
    cam1.tracks.push_back(createTrack(51, 1270, 10));   // Top-right corner
    cam1.tracks.push_back(createTrack(52, 10, 710));    // Bottom-left corner
    cam1.tracks.push_back(createTrack(53, 1270, 710));  // Bottom-right corner
    cam1.tracks.push_back(createTrack(54, 640, 10));    // Top center
    cam1.tracks.push_back(createTrack(55, 640, 710));   // Bottom center
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    // Corresponding positions with slight shifts
    cam2.tracks.push_back(createTrack(60, 5, 5));
    cam2.tracks.push_back(createTrack(61, 1265, 5));
    cam2.tracks.push_back(createTrack(62, 5, 705));
    cam2.tracks.push_back(createTrack(63, 1265, 705));
    cam2.tracks.push_back(createTrack(64, 635, 5));
    cam2.tracks.push_back(createTrack(65, 635, 705));
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 5: One camera has no tracks
FrameData createTestCase5() {
    FrameData frame;
    frame.frame_id = 5;

    CameraTracks cam1;
    cam1.camera_id = 2;
    cam1.tracks.push_back(createTrack(70, 400, 300));
    cam1.tracks.push_back(createTrack(71, 800, 400));
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    // No tracks in camera 2
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 6: Both cameras have no tracks
FrameData createTestCase6() {
    FrameData frame;
    frame.frame_id = 6;

    CameraTracks cam1;
    cam1.camera_id = 2;
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 7: Single camera (edge case)
FrameData createTestCase7() {
    FrameData frame;
    frame.frame_id = 7;

    CameraTracks cam1;
    cam1.camera_id = 2;
    cam1.tracks.push_back(createTrack(80, 400, 300));
    cam1.tracks.push_back(createTrack(81, 600, 400));
    
    frame.camera_tracks.push_back(cam1);
    // Only one camera
    return frame;
}

// Test case 8: Medium tracks stress test
FrameData createTestCase8() {
    FrameData frame;
    frame.frame_id = 8;

    CameraTracks cam1;
    cam1.camera_id = 2;
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    
    // Create a smaller grid of tracks (3x2 = 6 tracks each)
    int track_id = 100;
    for (int y = 200; y <= 400; y += 200) {
        for (int x = 200; x <= 800; x += 300) {
            cam1.tracks.push_back(createTrack(track_id++, x, y));
            // Corresponding track in cam2 with slight offset
            cam2.tracks.push_back(createTrack(track_id++, x - 5, y - 5));
        }
    }
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 9: Very close tracks (potential false positives)
FrameData createTestCase9() {
    FrameData frame;
    frame.frame_id = 9;

    CameraTracks cam1;
    cam1.camera_id = 2;
    cam1.tracks.push_back(createTrack(200, 400, 300));
    cam1.tracks.push_back(createTrack(201, 405, 305)); // Very close to track 200
    
    CameraTracks cam2;
    cam2.camera_id = 3;
    cam2.tracks.push_back(createTrack(210, 395, 295));
    cam2.tracks.push_back(createTrack(211, 400, 300)); // Very close to track 210
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Validate fundamental matrix calculation
void validateFundamentalMatrix() {
    std::cout << "\n=== FUNDAMENTAL MATRIX VALIDATION ===" << std::endl;
    
    cv::Mat F = computeFundamental_2to3();
    std::cout << "Computed Fundamental Matrix:" << std::endl;
    std::cout << F << std::endl;
    
    // Check if F has rank 2 (should be rank-deficient)
    cv::Mat w, u, vt;
    cv::SVD::compute(F, w, u, vt);
    std::cout << "Singular values: " << w.t() << std::endl;
    std::cout << "Rank should be 2, smallest singular value should be ~0: " << w.at<double>(2) << std::endl;
    
    // Test epipolar constraint for a known correspondence
    // For horizontally aligned cameras with baseline, a point at (x,y) in left image
    // should have epipolar line roughly horizontal in right image
    cv::Point2d test_point_left(400, 300);
    cv::Mat x_left = (cv::Mat_<double>(3,1) << test_point_left.x, test_point_left.y, 1.0);
    cv::Mat epipolar_line = F * x_left;
    
    double a = epipolar_line.at<double>(0);
    double b = epipolar_line.at<double>(1);
    double c = epipolar_line.at<double>(2);
    
    std::cout << "Test point (" << test_point_left.x << "," << test_point_left.y 
              << ") in left image" << std::endl;
    std::cout << "Epipolar line in right image: " << a << "x + " << b << "y + " << c << " = 0" << std::endl;
    std::cout << "Line slope: " << -a/b << " (should be close to 0 for horizontal baseline)" << std::endl;
}

std::unordered_map<int, std::unordered_map<int, TrackInfo>> look_up_converter(const FrameData& frame) {
    // Make it easily accessible to track by camera id and track id
    std::unordered_map<int, std::unordered_map<int, TrackInfo>> look_up;
    for (const auto& cam_tracks : frame.camera_tracks) {
        // Get the camera id for this camera
        int cam_id = cam_tracks.camera_id;

        // Fill in the look up for this key
        for (const auto& track : cam_tracks.tracks) {
            look_up[cam_id][track.track_id] = track;
        }
    }

    return look_up;
};

// Enhanced stereo track association with depth calculation
EnhancedAssignmentResult processFrameWithDepth(int frame_id, 
                                               const std::vector<TrackInfo>& tracks_cam2,
                                               const std::vector<TrackInfo>& tracks_cam3) {
    
    // Validate input tracks
    for (const auto& track : tracks_cam2) {
        if (!validateTrackClassifications(track)) {
            std::cerr << "Warning: Track " << track.track_id << " from cam2 has invalid classifications!" << std::endl;
        }
    }
    for (const auto& track : tracks_cam3) {
        if (!validateTrackClassifications(track)) {
            std::cerr << "Warning: Track " << track.track_id << " from cam3 has invalid classifications!" << std::endl;
        }
    }
    
    std::cout << "\n=== Enhanced Processing Frame " << frame_id << " ===" << std::endl;
    std::cout << "Camera 2 tracks: " << tracks_cam2.size() 
              << ", Camera 3 tracks: " << tracks_cam3.size() << std::endl;

    // 1. Epipolar gating using existing system
    FrameData frame_data;
    frame_data.frame_id = frame_id;
    
    CameraTracks cam2_data;
    cam2_data.camera_id = 2;
    cam2_data.tracks = tracks_cam2;
    
    CameraTracks cam3_data;
    cam3_data.camera_id = 3;
    cam3_data.tracks = tracks_cam3;
    
    frame_data.camera_tracks.push_back(cam2_data);
    frame_data.camera_tracks.push_back(cam3_data);
    
    auto binary_mask = EpipolarGating(frame_data);
    auto look_up = look_up_converter(frame_data);
    
    if (binary_mask.empty()) {
        std::cout << "No tracks passed epipolar gating." << std::endl;
        EnhancedAssignmentResult result;
        result.frame_id = frame_id;
        for (const auto& t : tracks_cam2) result.unassigned_camera1_tracks.push_back(t.track_id);
        for (const auto& t : tracks_cam3) result.unassigned_camera2_tracks.push_back(t.track_id);
        result.average_depth = 0.0;
        result.valid_triangulations = 0;
        result.average_reprojection_error = 0.0;
        return result;
    }

    // 2. Build cost matrix using existing system
    std::unordered_map<int, std::unordered_map<int, float>> final_cost_matrix;
    std::set<std::pair<int, int>> processed_pairs;
    static const cv::Mat F_2to3 = computeFundamental_2to3();

    for (const auto& entry : binary_mask) {
        int idA = entry.first;
        
        if (look_up[2].find(idA) == look_up[2].end()) {
            continue;
        }
        
        for (const auto& pair : entry.second) {
            if (pair.second) { // If allowed by epipolar gating
                int idB = pair.first;
                
                if (look_up[3].find(idB) == look_up[3].end()) {
                    continue;
                }
                
                std::pair<int, int> track_pair = std::make_pair(std::min(idA, idB), std::max(idA, idB));
                if (processed_pairs.find(track_pair) != processed_pairs.end()) {
                    continue;
                }
                processed_pairs.insert(track_pair);

                const TrackInfo& trackA = look_up[2][idA];
                const TrackInfo& trackB = look_up[3][idB];

                double d_epi = computeEpipolarDistanceTracks(trackA, trackB, F_2to3);
                double c_epi = epipolarDistanceToCost(d_epi);
                double c_ratio = computeWidthHeightRatioCost(trackA, trackB);
                double c_conf = computeSubclassConfidenceCost(trackA, trackB);
                double c_final = computeFinalCost(c_epi, c_ratio, c_conf);

                final_cost_matrix[idA][idB] = static_cast<float>(c_final);
            }
        }
    }
    
    // 3. Solve assignment using existing system
    AssignmentResult basic_result = solveTrackAssignment(frame_data, final_cost_matrix);
    
    // 4. Build enhanced result with 3D coordinates
    EnhancedAssignmentResult result;
    result.frame_id = frame_id;
    
    double total_depth = 0.0;
    int valid_count = 0;
    double total_reproj_error = 0.0;
    
    for (const auto& assignment : basic_result.assignments) {
        // Find the corresponding tracks
        TrackInfo* trackA_ptr = nullptr;
        TrackInfo* trackB_ptr = nullptr;
        
        for (const auto& track : tracks_cam2) {
            if (track.track_id == assignment.camera1_track_id) {
                trackA_ptr = const_cast<TrackInfo*>(&track);
                break;
            }
        }
        
        for (const auto& track : tracks_cam3) {
            if (track.track_id == assignment.camera2_track_id) {
                trackB_ptr = const_cast<TrackInfo*>(&track);
                break;
            }
        }
        
        if (trackA_ptr && trackB_ptr) {
            // Calculate 3D position for this assignment
            Point3D world_pos = triangulatePoints(*trackA_ptr, *trackB_ptr);
            
            TrackAssignmentWithDepth enhanced_assignment;
            enhanced_assignment.camera1_track_id = assignment.camera1_track_id;
            enhanced_assignment.camera2_track_id = assignment.camera2_track_id;
            enhanced_assignment.assignment_cost = assignment.assignment_cost;
            enhanced_assignment.world_position = world_pos;
            enhanced_assignment.depth_meters = world_pos.depth;
            enhanced_assignment.is_valid = world_pos.is_valid;
            enhanced_assignment.epipolar_distance = 0.0; // Could be extracted from cost matrix
            
            result.assignments.push_back(enhanced_assignment);
            
            if (world_pos.is_valid) {
                total_depth += world_pos.depth;
                total_reproj_error += world_pos.reprojection_error;
                valid_count++;
            }
            
            std::cout << "Assigned: Cam2[" << enhanced_assignment.camera1_track_id 
                      << "] <-> Cam3[" << enhanced_assignment.camera2_track_id 
                      << "], Cost: " << std::fixed << std::setprecision(3) << enhanced_assignment.assignment_cost
                      << ", Depth: " << enhanced_assignment.depth_meters << "m"
                      << ", Valid: " << (enhanced_assignment.is_valid ? "Yes" : "No") << std::endl;
        }
    }
    
    // Copy unassigned tracks from basic result
    result.unassigned_camera1_tracks = basic_result.unassigned_camera1_tracks;
    result.unassigned_camera2_tracks = basic_result.unassigned_camera2_tracks;
    
    // Calculate summary statistics
    result.average_depth = (valid_count > 0) ? total_depth / valid_count : 0.0;
    result.valid_triangulations = valid_count;
    result.average_reprojection_error = (valid_count > 0) ? total_reproj_error / valid_count : 0.0;
    
    std::cout << "Summary: " << result.assignments.size() << " assignments, "
              << valid_count << " valid triangulations, "
              << "Avg depth: " << std::fixed << std::setprecision(2) << result.average_depth << "m" << std::endl;
    
    return result;
}

int main() {
    std::cout << "=== Comprehensive Epipolar Gating Test Suite ===" << std::endl;
    
    // Validate fundamental matrix first
    validateFundamentalMatrix();
    
    // First, run a debug test to see actual epipolar distances
    std::cout << "\n=== DEBUGGING EPIPOLAR DISTANCES ===" << std::endl;
    FrameData debug_frame = createRealisticDebugTestCase();
    auto debug_mask = EpipolarGatingDebug(debug_frame, true);
    
    std::cout << "\n=== Testing different thresholds on realistic data ===" << std::endl;
    
    // Test with different thresholds to understand false negatives
    std::vector<double> test_thresholds = {1.0, 5.0, 10.0, 20.0, 50.0};
    
    for (double threshold : test_thresholds) {
        std::cout << "\n--- Testing with threshold: " << threshold << " pixels ---" << std::endl;
        
        // Modify the epipolar threshold temporarily
        FrameData test_frame = createTestCase1(); // Use equal tracks test
        
        // Count associations manually with the new threshold
        const CameraTracks& camA = test_frame.camera_tracks[0];
        const CameraTracks& camB = test_frame.camera_tracks[1];
        const cv::Mat F_2to3 = computeFundamental_2to3();
        
        int total_pairs = 0;
        int allowed_pairs = 0;
        
        for (const auto& trackA : camA.tracks) {
            cv::Point2d pA_ud = undistortPixel(trackA.bb_center_x, trackA.bb_center_y, K2, dist2);
            cv::Mat xA = (cv::Mat_<double>(3,1) << pA_ud.x, pA_ud.y, 1.0);
            cv::Mat lB = F_2to3 * xA;
            
            double a = lB.at<double>(0);
            double b = lB.at<double>(1); 
            double c = lB.at<double>(2);
            double denom = std::sqrt(a*a + b*b) + 1e-12;
            
            for (const auto& trackB : camB.tracks) {
                cv::Point2d pB_ud = undistortPixel(trackB.bb_center_x, trackB.bb_center_y, K3, dist3);
                double num = std::abs(a * pB_ud.x + b * pB_ud.y + c);
                double distance = num / denom;
                
                total_pairs++;
                if (distance < threshold) {
                    allowed_pairs++;
                    std::cout << "PASS: Track " << trackA.track_id << " <-> " << trackB.track_id 
                              << " (distance: " << distance << "px)" << std::endl;
                }
            }
        }
        
        std::cout << "Results: " << allowed_pairs << "/" << total_pairs 
                  << " (" << (100.0 * allowed_pairs / total_pairs) << "%)" << std::endl;
    }
    
    // Run original test cases
    std::cout << "\n=== ORIGINAL TEST CASES ===" << std::endl;
    std::vector<FrameData> test_cases = {
        createTestCase1(), // Equal tracks
        createTestCase2(), // More tracks in cam1
        createTestCase3(), // More tracks in cam2
        createTestCase4(), // Edge positions
        createTestCase5(), // Empty cam2
        createTestCase6(), // Both empty
        createTestCase7(), // Single camera
        createTestCase8(), // Medium tracks
        createTestCase9()  // Close tracks
    };
    
    for (const auto& frame : test_cases) {
        auto binary_mask = EpipolarGating(frame);
        printTestResults(frame.frame_id, binary_mask);
        // look_up = look_up_converter(frame);
        
        // Print detailed summary statistics
        int total_associations = 0;
        int allowed_associations = 0;
        
        for (const auto& entry : binary_mask) {
            for (const auto& pair : entry.second) {
                total_associations++;
                if (pair.second) allowed_associations++;
            }
        }
        
        if (frame.camera_tracks.size() >= 2) {
            std::cout << "Camera stats: Cam1=" << frame.camera_tracks[0].tracks.size() 
                      << " tracks, Cam2=" << frame.camera_tracks[1].tracks.size() 
                      << " tracks" << std::endl;
        } else if (frame.camera_tracks.size() == 1) {
            std::cout << "Single camera with " << frame.camera_tracks[0].tracks.size() 
                      << " tracks" << std::endl;
        } else {
            std::cout << "No cameras in frame" << std::endl;
        }
        
        std::cout << "Overall success rate: " 
                  << (total_associations > 0 ? 
                      (100.0 * allowed_associations / total_associations) : 0.0) 
                  << "%" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
    }

    // Build the cost matrix here
    // Run original test cases
    std::cout << "\n=== ORIGINAL TEST CASES (COST MATRIX) ===" << std::endl;
    std::vector<FrameData> cost_matrix_test_cases = {
        createTestCase1(), // Equal tracks
        createTestCase2(), // More tracks in cam1
        createTestCase3(), // More tracks in cam2
        createTestCase4(), // Edge positions
        createTestCase5(), // Empty cam2
        createTestCase6(), // Both empty
        createTestCase7(), // Single camera
        createTestCase8(), // Medium tracks
        createTestCase9()  // Close tracks
    };

    for (const auto& frame : cost_matrix_test_cases) {
        // For each frame, build a cost matrix for:
        // - Soft epipolar distance comparison
        // - Width/Height Ratios
        // - Subclassification confidence differences

        // Get the binary mask from epipolar gating
        std::unordered_map<int, std::vector<std::pair<int, bool>>> binary_mask = EpipolarGating(frame);
        std::unordered_map<int, std::unordered_map<int, TrackInfo>> look_up = look_up_converter(frame);
        
        // At this point, we have the binary mask and the look up table
        // We will want to only build costs for allowed pairs
        
        std::cout << "\n--- Building Cost Matrices for Frame " << frame.frame_id << " ---" << std::endl;

        // Skip if insufficient cameras
        if (frame.camera_tracks.size() < 2) {
            std::cout << "Insufficient cameras for cost matrix building." << std::endl;
            continue;
        }
        
        // Initialize cost matrices
        std::unordered_map<int, std::unordered_map<int, float>> ep_cost_matrix;
        std::unordered_map<int, std::unordered_map<int, float>> wh_ratio_cost_matrix;
        std::unordered_map<int, std::unordered_map<int, float>> subclass_conf_cost_matrix;
        std::unordered_map<int, std::unordered_map<int, float>> final_cost_matrix;
        
        // Use set to track processed pairs and avoid duplicates
        std::set<std::pair<int, int>> processed_pairs;
        
        static const cv::Mat F_2to3 = computeFundamental_2to3();

        // Iterate through the binary mask to find allowed pairs
        for (const auto& entry : binary_mask) {
            int idA = entry.first;
            
            // Check if track exists in camera 2 (skip tracks from camera 3)
            if (look_up[2].find(idA) == look_up[2].end()) {
                continue;
            }
            
            for (const auto& pair : entry.second) {
                if (pair.second) { // If allowed by epipolar gating
                    int idB = pair.first;
                    
                    // Check if track exists in camera 3
                    if (look_up[3].find(idB) == look_up[3].end()) {
                        continue;
                    }
                    
                    // Avoid duplicate processing (A->B and B->A)
                    std::pair<int, int> track_pair = std::make_pair(std::min(idA, idB), std::max(idA, idB));
                    if (processed_pairs.find(track_pair) != processed_pairs.end()) {
                        continue;
                    }
                    processed_pairs.insert(track_pair);

                    const TrackInfo& trackA = look_up[2][idA]; // cam 2
                    const TrackInfo& trackB = look_up[3][idB]; // cam 3

                    // Compute all three cost components
                    double d_epi = computeEpipolarDistanceTracks(trackA, trackB, F_2to3);
                    double c_epi = epipolarDistanceToCost(d_epi);
                    double c_ratio = computeWidthHeightRatioCost(trackA, trackB);
                    double c_conf = computeSubclassConfidenceCost(trackA, trackB);
                    double c_final = computeFinalCost(c_epi, c_ratio, c_conf);

                    // Store in cost matrices
                    ep_cost_matrix[idA][idB] = static_cast<float>(c_epi);
                    wh_ratio_cost_matrix[idA][idB] = static_cast<float>(c_ratio);
                    subclass_conf_cost_matrix[idA][idB] = static_cast<float>(c_conf);
                    final_cost_matrix[idA][idB] = static_cast<float>(c_final);

                    std::cout << "Track " << idA << " <-> Track " << idB << std::endl;
                    std::cout << "  Epipolar: d=" << d_epi << "px, c=" << c_epi << std::endl;
                    std::cout << "  W/H Ratio: c=" << c_ratio << std::endl;
                    std::cout << "  Confidence: c=" << c_conf << std::endl;
                    std::cout << "  Final Cost: " << c_final << std::endl;
                    std::cout << std::endl;
                }
            }
        }
        
        // Print summary statistics
        std::cout << "Cost matrix summary:" << std::endl;
        std::cout << "  Total valid pairs: " << processed_pairs.size() << std::endl;
        
        if (!final_cost_matrix.empty()) {
            // Find min/max costs for analysis
            float min_cost = 1e6, max_cost = 0.0;
            for (const auto& row : final_cost_matrix) {
                for (const auto& col : row.second) {
                    min_cost = std::min(min_cost, col.second);
                    max_cost = std::max(max_cost, col.second);
                }
            }
            std::cout << "  Cost range: [" << min_cost << ", " << max_cost << "]" << std::endl;
        }
        
        // Solve the assignment problem using Hungarian algorithm
        std::cout << "\n--- Solving Track Assignment for Frame " << frame.frame_id << " ---" << std::endl;
        AssignmentResult assignment_result = solveTrackAssignment(frame, final_cost_matrix);
        
        std::cout << "Assignment Results:" << std::endl;
        std::cout << "  Successful assignments: " << assignment_result.assignments.size() << std::endl;
        
        for (const auto& assignment : assignment_result.assignments) {
            std::cout << "    Track " << assignment.camera1_track_id << " (Cam2) <-> Track " 
                      << assignment.camera2_track_id << " (Cam3), Cost: " << assignment.assignment_cost << std::endl;
        }
        
        std::cout << "  Unassigned Camera 2 tracks: ";
        for (int id : assignment_result.unassigned_camera1_tracks) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Unassigned Camera 3 tracks: ";
        for (int id : assignment_result.unassigned_camera2_tracks) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

    }
    
    std::cout << "\n=== ASSIGNMENT SUMMARY ===" << std::endl;
    std::cout << "Multi-modal track assignment system completed successfully!" << std::endl;
    std::cout << "Ready for temporal smoothing and triangulation integration." << std::endl;

    // =========================================================================
    // COMPREHENSIVE REALISTIC TEST SCENARIOS WITH 3D DEPTH CALCULATION
    // =========================================================================
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPREHENSIVE REALISTIC TEST SCENARIOS WITH STEREO DEPTH" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Test Scenario 1: Military Personnel at Various Depths
    {
        std::cout << "\n--- Test 1: Military Personnel Scenario ---" << std::endl;
        
        auto dims_5m = generateRealisticDimensions("person_standing", 5.0);
        auto dims_8m = generateRealisticDimensions("person_standing", 8.0);
        auto dims_12m = generateRealisticDimensions("person_crouching", 12.0);
        
        std::vector<TrackInfo> cam2_tracks = {
            createTrack(101, 400, 300, 0.95f, 0.85f, 0.15f, dims_5m.first, dims_5m.second),    // Armed military standing
            createTrack(102, 600, 250, 0.90f, 0.80f, 0.20f, dims_8m.first, dims_8m.second),    // Armed military standing  
            createTrack(103, 800, 350, 0.85f, 0.75f, 0.10f, dims_12m.first, dims_12m.second)   // Armed military crouching
        };
        
        std::vector<TrackInfo> cam3_tracks = {
            createTrack(201, 350, 305, 0.93f, 0.82f, 0.18f, dims_5m.first, dims_5m.second),    // Armed military standing
            createTrack(202, 540, 245, 0.88f, 0.78f, 0.25f, dims_8m.first, dims_8m.second),    // Armed military standing
            createTrack(203, 720, 355, 0.87f, 0.73f, 0.12f, dims_12m.first, dims_12m.second)   // Armed military crouching
        };
        
        auto result1 = processFrameWithDepth(1, cam2_tracks, cam3_tracks);
        
        std::cout << "Results: " << result1.assignments.size() << " assignments";
        std::cout << ", Avg depth: " << std::fixed << std::setprecision(1) << result1.average_depth << "m" << std::endl;
    }
    
    // Test Scenario 2: Mixed Civilian and Military
    {
        std::cout << "\n--- Test 2: Mixed Civilian/Military Scenario ---" << std::endl;
        
        auto dims_6m_stand = generateRealisticDimensions("person_standing", 6.0);
        auto dims_10m_crouch = generateRealisticDimensions("person_crouching", 10.0);
        auto dims_15m_stand = generateRealisticDimensions("person_standing", 15.0);
        
        std::vector<TrackInfo> cam2_tracks = {
            createTrack(111, 300, 280, 0.15f, 0.25f, 0.05f, dims_6m_stand.first, dims_6m_stand.second),     // Unarmed civilian
            createTrack(112, 500, 320, 0.80f, 0.70f, 0.30f, dims_10m_crouch.first, dims_10m_crouch.second), // Armed military 
            createTrack(113, 750, 290, 0.10f, 0.20f, 0.90f, dims_15m_stand.first, dims_15m_stand.second)    // Unarmed civilian surrendering
        };
        
        std::vector<TrackInfo> cam3_tracks = {
            createTrack(211, 250, 285, 0.12f, 0.22f, 0.08f, dims_6m_stand.first, dims_6m_stand.second),     // Unarmed civilian
            createTrack(212, 440, 315, 0.85f, 0.75f, 0.25f, dims_10m_crouch.first, dims_10m_crouch.second), // Armed military
            createTrack(213, 670, 295, 0.08f, 0.18f, 0.85f, dims_15m_stand.first, dims_15m_stand.second)    // Unarmed civilian surrendering
        };
        
        auto result2 = processFrameWithDepth(2, cam2_tracks, cam3_tracks);
        
        std::cout << "Results: " << result2.assignments.size() << " assignments";
        std::cout << ", Valid triangulations: " << result2.valid_triangulations << std::endl;
    }
    
    // Test Scenario 3: Close Range High Precision
    {
        std::cout << "\n--- Test 3: Close Range Precision Test ---" << std::endl;
        
        auto dims_3m = generateRealisticDimensions("person_standing", 3.0);
        auto dims_4m = generateRealisticDimensions("person_standing", 4.0);
        
        std::vector<TrackInfo> cam2_tracks = {
            createTrack(121, 450, 250, 0.70f, 0.60f, 0.40f, dims_3m.first, dims_3m.second),    // Armed military
            createTrack(122, 550, 280, 0.25f, 0.30f, 0.10f, dims_4m.first, dims_4m.second)     // Unarmed civilian
        };
        
        std::vector<TrackInfo> cam3_tracks = {
            createTrack(221, 395, 255, 0.75f, 0.65f, 0.35f, dims_3m.first, dims_3m.second),    // Armed military 
            createTrack(222, 485, 275, 0.20f, 0.28f, 0.15f, dims_4m.first, dims_4m.second)     // Unarmed civilian
        };
        
        auto result3 = processFrameWithDepth(3, cam2_tracks, cam3_tracks);
        
        std::cout << "Results: " << result3.assignments.size() << " assignments";
        std::cout << ", Avg depth: " << std::fixed << std::setprecision(1) << result3.average_depth << "m" << std::endl;
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPREHENSIVE TESTING COMPLETE - READY FOR 3D PROCESSING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << "\nEnhanced Features Implemented:" << std::endl;
    std::cout << "- ✓ Proper classification constraints (sum to 1.0)" << std::endl;
    std::cout << "- ✓ Realistic track dimensions based on depth" << std::endl;
    std::cout << "- ✓ Stereo triangulation with 3D coordinates" << std::endl;
    std::cout << "- ✓ Depth calculation for each assignment" << std::endl;
    std::cout << "- ✓ Enhanced validation and error checking" << std::endl;


    // Camera 2:
    /**
        Camera Matrix:
        [[690.25849569   0.         614.21523163]
        [  0.         689.6248818  338.94900124]
        [  0.           0.           1.        ]]
        Distortion Coeff:
        [[-3.58152939e-01  1.85211072e-01  7.39179696e-05  1.70375285e-04
        -5.82790603e-02]]
        Reproj Error (pixels): 0.1660
    */

    // Camera 3:
    /**
        Camera Matrix:
        [[691.76303387   0.         631.403727  ]
        [  0.         691.18295277 362.73688706]
        [  0.           0.           1.        ]]
        Distortion Coeff:
        [[-3.57820636e-01  1.84797837e-01  2.20893356e-04  9.55935431e-05
        -5.91141748e-02]]
        Reproj Error (pixels): 0.1505
    */

    // Camera Extrinsics:
    /**
        R = [1, 0, 0, 
            0, 1, 0, 
            0, 0, 1]
            
        T = [0.125, 0, 0]       // Baseline of 125 mm between the two cameras
    */



    return 0;
}