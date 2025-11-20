#include <iostream>
#include <vector>
#include <string>

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

int main() {
    // Expect to feed in an unknown of tracks from each camera

    // Generate some sort of dummy data for testing
    FrameData frame;
    frame.frame_id = "frame_001";

    CameraTracks cam1;
    cam1.camera_id = "camera_01";
    TrackInfo track1 = {1, 1280, 720, 100, 300, 0.9f, 0.0f, 0.8f, 0.0f, 0.0f, 0.95f};
    TrackInfo track2 = {2, 1280, 720, 480, 270, 0.0f, 0.8f, 0.0f, 0.9f, 0.9f, 0.0f};
    cam1.tracks.push_back(track1);
    cam1.tracks.push_back(track2);
    CameraTracks cam2;
    cam2.camera_id = "camera_02";
    TrackInfo track3 = {3, 1280, 720, 500, 300, 0.0f, 0.85f, 0.0f, 0.95f, 0.8f, 0.0f};
    TrackInfo track4 = {4, 1280, 720, 95, 290, 0.8f, 0.0f, 0.9f, 0.0f, 0.0f, 0.9f};
    cam2.tracks.push_back(track3);
    cam2.tracks.push_back(track4);
    
    frame.camera_tracks.push_back(cam1);    
    frame.camera_tracks.push_back(cam2);
    // Output the dummy data
    std::cout << "Frame ID: " << frame.frame_id << std::endl;
    for (const auto& camera : frame.camera_tracks) {
        std::cout << " Camera ID: " << camera.camera_id << std::endl;
        for (const auto& track : camera.tracks) {
            std::cout << "  Track ID: " << track.track_id
                        << ", Width: " << track.width
                        << ", Height: " << track.height
                        << ", BB Center: (" << track.bb_center_x << ", " << track.bb_center_y << ")"
                        << ", Armed Confidence: " << track.armed_confidence
                        << ", Unarmed Confidence: " << track.unarmed_confidence
                        << ", Military Confidence: " << track.military_confidence
                        << ", Civilian Confidence: " << track.civilian_confidence
                        << ", Surrender Confidence: " << track.surrender_confidence
                        << ", No Surrender Confidence: " << track.no_surrender_confidence
                        << std::endl;
        }
    }

    // Do the Epipolar Gating here

    // Build the cost matrix here

    // Solve the assignment problem via Hungarian here

    // Do temporal smoothing here

    // Triangulate and attach depth/distances to each uniform track message to be sent to detection module

    return 0;
}