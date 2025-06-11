#include <string>
#include <map>
#include <vector>
#include "two_view_geometry.h"
#include "camera.h"
#include "scene.h"
#include "image.h"

void TwoViewGeometry::Invert() {
    F.transposeInPlace();
    E.transposeInPlace();
    H = H.inverse().eval();
    cam2_from_cam1 = Inverse(cam2_from_cam1);
}
TwoViewGeometry EstimateUncalibratedTwoViewGeometry(
    const Camera& camera1,
    const Image& img1,
    const Camera& camera2,
    const Image& img2) {
    TwoViewGeometry geometry;

    const size_t min_num_inliers = 8;
    //if (matches.size() < static_cast<size_t>(min_num_inliers)) {
    //    geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
    //    return geometry;
    //}
    std::vector<int>matches;// ----
    // Extract corresponding points.
    std::vector<Eigen::Vector2d> matched_points1(matches.size());
    std::vector<Eigen::Vector2d> matched_points2(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        matched_points1[i] = img1.Points2D()[matches[i]].xy;// ----
        matched_points2[i] = img2.Points2D()[matches[i]].xy;// ----
    }

    // Estimate epipolar model.

    LORANSAC<FundamentalMatrixSevenPointEstimator,
        FundamentalMatrixEightPointEstimator>
        F_ransac(options.ransac_options);
    const auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
    geometry.F = F_report.model;

    // Estimate planar or panoramic model.

    LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator> H_ransac(
        options.ransac_options);
    const auto H_report = H_ransac.Estimate(matched_points1, matched_points2);
    geometry.H = H_report.model;

    if ((!F_report.success && !H_report.success) ||
        (F_report.support.num_inliers < min_num_inliers &&
            H_report.support.num_inliers < min_num_inliers)) {
        geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
        return geometry;
    }

    // Determine inlier ratios of different models.

    const double H_F_inlier_ratio =
        static_cast<double>(H_report.support.num_inliers) /
        F_report.support.num_inliers;

    const std::vector<char>* best_inlier_mask = &F_report.inlier_mask;
    int num_inliers = F_report.support.num_inliers;
    //if (H_F_inlier_ratio > options.max_H_inlier_ratio)
    //{
    //  geometry.config = TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC;
    //  if (H_report.support.num_inliers >= F_report.support.num_inliers) {
    //    num_inliers = H_report.support.num_inliers;
    //    best_inlier_mask = &H_report.inlier_mask;
    //  }
    //} 
    //else 
    {
        geometry.config = TwoViewGeometry::ConfigurationType::UNCALIBRATED;
    }

    geometry.inlier_matches =
        ExtractInlierMatches(matches, num_inliers, *best_inlier_mask);

    //if (options.detect_watermark && DetectWatermark(camera1,
    //                                                matched_points1,
    //                                                camera2,
    //                                                matched_points2,
    //                                                num_inliers,
    //                                                *best_inlier_mask,
    //                                                options)) {
    //  geometry.config = TwoViewGeometry::ConfigurationType::WATERMARK;
    //}

    if (options.compute_relative_pose) {
        EstimateTwoViewGeometryPose(camera1, points1, camera2, points2, &geometry);
    }

    return geometry;
}

int test_geometry()
{
    //auto prev_camera_= Camera::CreateFromModelId(0,CameraModelId::kPinhole,100,400,300);
    std::map<Camera, std::vector<Image>> dataset = loadImageData("D:/repo/colmapThird/data", ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList; std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    return 0;
}