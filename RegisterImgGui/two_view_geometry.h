#pragma once
#include "types.h"
#include "camera.h"
#include "image.h"
struct TwoViewGeometry {
    // The configuration of the two-view geometry.
    enum ConfigurationType {
        UNDEFINED = 0,
        // Degenerate configuration (e.g., no overlap or not enough inliers).
        DEGENERATE = 1,
        // Essential matrix.
        CALIBRATED = 2,
        // Fundamental matrix.
        UNCALIBRATED = 3,
        // Homography, planar scene with baseline.
        PLANAR = 4,
        // Homography, pure rotation without baseline.
        PANORAMIC = 5,
        // Homography, planar or panoramic.
        PLANAR_OR_PANORAMIC = 6,
        // Watermark, pure 2D translation in image borders.
        WATERMARK = 7,
        // Multi-model configuration, i.e. the inlier matches result from multiple
        // individual, non-degenerate configurations.
        MULTIPLE = 8,
    };

    // One of `ConfigurationType`.
    int config = ConfigurationType::UNDEFINED;

    // Essential matrix.
    Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
    // Fundamental matrix.
    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    // Homography matrix.
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();

    // Relative pose.
    Rigid3d cam2_from_cam1;

    // Median triangulation angle.
    double tri_angle = -1;

    // Invert the geometry to match swapped cameras.
    void Invert();
};

TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const struct Camera& camera1,
    const class Image& img1,
    const struct Camera& camera2,
    const class Image& img2);
bool EstimateTwoViewGeometryPose(
    const struct Camera& camera1,
    const class Image& img1,
    const struct Camera& camera2,
    const class Image& img2,
    TwoViewGeometry* geometry);