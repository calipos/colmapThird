#include <list>
#include <string>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include "two_view_geometry.h"
#include "estimate.h"
#include "camera.h"
#include "scene.h"
#include "image.h"
int Combination1(const int&n, const  const int& m) {
    if (n<=0 ||m<=0)
    {
        return 0;
    }
    int res = 1;
    for (int i = 1; i <= m; ++i) {
        res = res * (n - m + i) / i;        // ÏÈ³Ëºó³ý
    }
    return res;
}
void TwoViewGeometry::Invert() {
    F.transposeInPlace();
    E.transposeInPlace();
    H = H.inverse().eval();
    cam2_from_cam1 = Inverse(cam2_from_cam1);
}

TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1,
    const Image& img1,
    const Camera& camera2,
    const Image& img2)
{
    TwoViewGeometry geometry;
    std::vector<point2D_t>matchesPointId;
    matchesPointId.reserve(std::min(img1.featPts.size(), img2.featPts.size()));
    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = img1.featPts.begin(); iter != img1.featPts.end(); iter++) 
    {
        if (img2.featPts.count(iter->first) != 0)
        {
            matchesPointId.emplace_back(iter->first);
        }
    }
    const size_t min_num_inliers = std::min(EssentialMatrixFivePointEstimator::kMinNumSamples, FundamentalMatrixSevenPointEstimator::kMinNumSamples);
    if (matchesPointId.size() < static_cast<size_t>(min_num_inliers)) {
        geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
        return geometry;
    }
    // Extract corresponding points.
    std::vector<Eigen::Vector2d> matched_points1(matchesPointId.size());
    std::vector<Eigen::Vector2d> matched_points2(matchesPointId.size());
    std::vector<Eigen::Vector2d> matched_points1_normalized(matchesPointId.size());
    std::vector<Eigen::Vector2d> matched_points2_normalized(matchesPointId.size());
    for (size_t i = 0; i < matchesPointId.size(); ++i) {
        const auto& featId = matchesPointId[i];
        matched_points1[i] = img1.Points2D()[featId];
        matched_points2[i] = img2.Points2D()[featId];
        matched_points1_normalized[i] = camera1.CamFromImg(matched_points1[i]);
        matched_points2_normalized[i] = camera2.CamFromImg(matched_points2[i]);
    }


    // Estimate epipolar models.
    RANSACOptions ransac_options;
    ransac_options.max_error = 40;
    LORANSAC<EssentialMatrixFivePointEstimator, EssentialMatrixFivePointEstimator>
        E_ransac(ransac_options);
    auto E_report =
        E_ransac.Estimate(matched_points1_normalized, matched_points2_normalized);
    geometry.E = E_report.model;

    LORANSAC<FundamentalMatrixSevenPointEstimator,FundamentalMatrixEightPointEstimator>
        F_ransac(ransac_options);
    auto F_report = F_ransac.Estimate(matched_points1, matched_points2);
    geometry.F = F_report.model;


    // No valid model was found
    if (F_report.support.num_inliers < min_num_inliers) {
        F_report.success = false;
    }
    else
    {
        F_report.success = true;
    }
    geometry.config = TwoViewGeometry::ConfigurationType::UNCALIBRATED;
    geometry.F = F_report.model;

    return geometry;
}
bool EstimateTwoViewGeometryPose(
    const Camera& camera1,
    const Image& img1,
    const Camera& camera2,
    const Image& img2,
    TwoViewGeometry* geometry) {
    // We need a valid epopolar geometry to estimate the relative pose.
    if (geometry->config != TwoViewGeometry::ConfigurationType::CALIBRATED &&
        geometry->config != TwoViewGeometry::ConfigurationType::UNCALIBRATED &&
        geometry->config != TwoViewGeometry::ConfigurationType::PLANAR &&
        geometry->config != TwoViewGeometry::ConfigurationType::PANORAMIC &&
        geometry->config !=
        TwoViewGeometry::ConfigurationType::PLANAR_OR_PANORAMIC) {
        return false;
    }

    // Extract normalized inlier points.
    std::vector<Eigen::Vector2d> inlier_points1_normalized;
    inlier_points1_normalized.reserve(geometry->inlier_matches.size());
    std::vector<Eigen::Vector2d> inlier_points2_normalized;
    inlier_points2_normalized.reserve(geometry->inlier_matches.size());
    for (const auto& match : geometry->inlier_matches) {
        inlier_points1_normalized.push_back(
            camera1.CamFromImg(points1[match.point2D_idx1]));
        inlier_points2_normalized.push_back(
            camera2.CamFromImg(points2[match.point2D_idx2]));
    }

    std::vector<Eigen::Vector3d> points3D;

    if (geometry->config == TwoViewGeometry::ConfigurationType::CALIBRATED ||
        geometry->config == TwoViewGeometry::ConfigurationType::UNCALIBRATED) {
        // Try to recover relative pose for calibrated and uncalibrated
        // configurations. In the uncalibrated case, this most likely leads to a
        // ill-defined reconstruction, but sometimes it succeeds anyways after e.g.
        // subsequent bundle-adjustment etc.
        PoseFromEssentialMatrix(geometry->E,
            inlier_points1_normalized,
            inlier_points2_normalized,
            &geometry->cam2_from_cam1,
            &points3D);
    }
    else {
        return false;
    }

    if (points3D.empty()) {
        geometry->tri_angle = 0;
    }
    else {
        const Eigen::Vector3d proj_center1 = Eigen::Vector3d::Zero();
        const Eigen::Vector3d proj_center2 = geometry->cam2_from_cam1.rotation *
            -geometry->cam2_from_cam1.translation;
        geometry->tri_angle = Median(
            CalculateTriangulationAngles(proj_center1, proj_center2, points3D));
    }
    return true;
}

int test_geometry()
{
    //auto prev_camera_= Camera::CreateFromModelId(0,CameraModelId::kPinhole,100,400,300);
    std::map<Camera, std::vector<Image>> dataset = loadImageData("D:/repo/colmapThird/data", ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList; std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);

    image_t pickedA = 3;
    image_t pickedB = 31;
    const Image& img1 = imageList[pickedA];
    const Image& img2 = imageList[pickedB];
    const Camera& camera1 = cameraList[img1.CameraId()];
    const Camera& camera2 = cameraList[img2.CameraId()];
    TwoViewGeometry two_view_geometry=EstimateCalibratedTwoViewGeometry(camera1, img1, camera2, img2);
    if (!EstimateTwoViewGeometryPose(camera1, img1, camera2, img2,))
    {

    }
    return 0;
}