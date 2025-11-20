#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>
#include <opencv2/opencv.hpp>
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
    std::string camera_id;
    std::vector<TrackInfo> tracks;
};

struct FrameData {
    std::string frame_id;
    std::vector<CameraTracks> camera_tracks;
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
void printTestResults(const std::string& test_name, const std::unordered_map<int, std::vector<std::pair<int, bool>>>& binary_mask) {
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

// Helper function to create a track
TrackInfo createTrack(int id, int x, int y, float armed_conf = 0.5f, float unarmed_conf = 0.3f) {
    return {id, 1280, 720, x, y, armed_conf, unarmed_conf, 0.7f, 0.8f, 0.1f, 0.9f};
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
    
    cv::Point2d right_point;
    right_point.x = left_point.x - disparity + noise_x; // Negative because right camera sees object shifted left
    right_point.y = left_point.y + noise_y;
    
    return std::make_pair(left_point, right_point);
}

// Test case for debugging epipolar distances with realistic stereo data
FrameData createRealisticDebugTestCase() {
    FrameData frame;
    frame.frame_id = "realistic_debug_epipolar_distances";

    CameraTracks cam1, cam2;
    cam1.camera_id = "camera_2";
    cam2.camera_id = "camera_3";
    
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

// Test case 1: Equal number of tracks with realistic stereo geometry
FrameData createTestCase1() {
    FrameData frame;
    frame.frame_id = "test_case_1_equal_tracks_realistic";

    CameraTracks cam1, cam2;
    cam1.camera_id = "camera_2";
    cam2.camera_id = "camera_3";
    
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
    frame.frame_id = "test_case_2_unequal_more_cam1";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    cam1.tracks.push_back(createTrack(10, 200, 200));
    cam1.tracks.push_back(createTrack(11, 400, 200));
    cam1.tracks.push_back(createTrack(12, 600, 200));
    cam1.tracks.push_back(createTrack(13, 800, 200));
    cam1.tracks.push_back(createTrack(14, 300, 400));
    cam1.tracks.push_back(createTrack(15, 700, 400));
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
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
    frame.frame_id = "test_case_3_unequal_more_cam2";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    // Only 2 tracks in camera 1
    cam1.tracks.push_back(createTrack(30, 350, 250));
    cam1.tracks.push_back(createTrack(31, 750, 350));
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
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
    frame.frame_id = "test_case_4_edge_positions";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    cam1.tracks.push_back(createTrack(50, 10, 10));     // Top-left corner
    cam1.tracks.push_back(createTrack(51, 1270, 10));   // Top-right corner
    cam1.tracks.push_back(createTrack(52, 10, 710));    // Bottom-left corner
    cam1.tracks.push_back(createTrack(53, 1270, 710));  // Bottom-right corner
    cam1.tracks.push_back(createTrack(54, 640, 10));    // Top center
    cam1.tracks.push_back(createTrack(55, 640, 710));   // Bottom center
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
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
    frame.frame_id = "test_case_5_empty_camera";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    cam1.tracks.push_back(createTrack(70, 400, 300));
    cam1.tracks.push_back(createTrack(71, 800, 400));
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
    // No tracks in camera 2
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 6: Both cameras have no tracks
FrameData createTestCase6() {
    FrameData frame;
    frame.frame_id = "test_case_6_both_empty";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
    
    frame.camera_tracks.push_back(cam1);
    frame.camera_tracks.push_back(cam2);
    return frame;
}

// Test case 7: Single camera (edge case)
FrameData createTestCase7() {
    FrameData frame;
    frame.frame_id = "test_case_7_single_camera";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    cam1.tracks.push_back(createTrack(80, 400, 300));
    cam1.tracks.push_back(createTrack(81, 600, 400));
    
    frame.camera_tracks.push_back(cam1);
    // Only one camera
    return frame;
}

// Test case 8: Medium tracks stress test
FrameData createTestCase8() {
    FrameData frame;
    frame.frame_id = "test_case_8_medium_tracks";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
    
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
    frame.frame_id = "test_case_9_close_tracks";

    CameraTracks cam1;
    cam1.camera_id = "camera_2";
    cam1.tracks.push_back(createTrack(200, 400, 300));
    cam1.tracks.push_back(createTrack(201, 405, 305)); // Very close to track 200
    
    CameraTracks cam2;
    cam2.camera_id = "camera_3";
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
        int cam_id = std::stoi(cam_tracks.camera_id);

        // Fill in the look up for this key
        for (const auto& track : cam_tracks.tracks) {
            look_up[cam_id][track.track_id] = track;
        }
    }

    return look_up;
};

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
        // - IoU of bounding boxes
        // - Width/Height Ratios
        // - Subclassification confidence differences

        // Get the binary mask from epipolar gating
        std::unordered_map<int, std::unordered_map<int, bool>> binary_mask = EpipolarGating(frame);
        std::unordered_map<int, std::unordered_map<int, TrackInfo>> look_up = look_up_converter(frame);
        
        // At this point, we have the binary mask and the look up table
        // We will want to only build costs for allowed pairs

        // Iterate through the binary mask
        for (const auto& entry : binary_mask) {
            // Make an IoU cost matrix

            // Make a width/height ratio cost matrix
            
            // Make a subclassification confidence difference cost matrix
        
            // Put all these together into a final cost matrix
        }

    }
        

    // Solve the assignment problem via Hungarian here

    // Do temporal smoothing here

    // Triangulate and attach depth/distances to each uniform track message to be sent to detection module


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