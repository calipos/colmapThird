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
TwoViewGeometry EstimateUncalibratedTwoViewGeometry(
    const Camera& camera1,
    const Image& img1,
    const Camera& camera2,
    const Image& img2) {
    TwoViewGeometry geometry;
    std::vector<point2D_t>matches;
    matches.reserve(std::min(img1.featPts.size(), img2.featPts.size()));
    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = img1.featPts.begin(); iter != img1.featPts.end(); iter++) {
        if (img2.featPts.count(iter->first)!=0)
        {
            matches.emplace_back(iter->first);
        }
    }
    const size_t min_num_inliers = 8;
    if (matches.size() < static_cast<size_t>(min_num_inliers)) {
        geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
        return geometry;
    }
    std::vector<Eigen::Vector2d> totalPoints1(matches.size());
    std::vector<Eigen::Vector2d> totalPoints2(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        totalPoints1[i] = img1.Points2D()[matches[i]];
        totalPoints2[i] = img2.Points2D()[matches[i]];
    }
    InlierSupportMeasurer support_measurer;
    typename InlierSupportMeasurer::Support best_support;    
    typename FundamentalMatrixSevenPointEstimator::M_t best_model;
    bool best_model_is_local = false;
    std::vector<typename FundamentalMatrixEightPointEstimator::X_t> X_inlier;
    std::vector<typename FundamentalMatrixEightPointEstimator::Y_t> Y_inlier;

    const double max_residual = 40.;
    FundamentalMatrixSevenPointEstimator estimator;
    FundamentalMatrixEightPointEstimator local_estimator;

    int maxTrialsNum = std::min(Combination1(matches.size(), 7), 200); 
    for (int num_trials = 0; num_trials < maxTrialsNum; num_trials++)
    {
        std::shuffle(matches.begin(), matches.end(), std::default_random_engine(std::time(0)));
        std::vector<Eigen::Vector2d> sevenPoints1(7);
        std::vector<Eigen::Vector2d> sevenPoints2(7);
        for (size_t i = 0; i < 7; ++i) {
            sevenPoints1[i] = img1.Points2D()[matches[i]];
            sevenPoints2[i] = img2.Points2D()[matches[i]];
        }
        std::vector<double> residuals;
        std::vector<double> best_local_residuals;
        std::vector<typename FundamentalMatrixSevenPointEstimator::M_t> sample_models;
        std::vector<typename FundamentalMatrixEightPointEstimator::M_t> local_models;

        estimator.Estimate(sevenPoints1, sevenPoints2, &sample_models);
        for (const auto& sample_model : sample_models)
        {
            estimator.Residuals(totalPoints1, totalPoints2, sample_model, &residuals);
            const auto support = support_measurer.Evaluate(residuals, max_residual);
            // Do local optimization if better than all previous subsets.
            if (support_measurer.IsLeftBetter(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;
                // Estimate locally optimized model from inliers.
                if (support.num_inliers > FundamentalMatrixSevenPointEstimator::kMinNumSamples &&
                    support.num_inliers >= FundamentalMatrixEightPointEstimator::kMinNumSamples) 
                {
                    // Recursive local optimization to expand inlier set.

                    X_inlier.clear();
                    Y_inlier.clear();
                    X_inlier.reserve(matches.size());
                    Y_inlier.reserve(matches.size());
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(totalPoints1[i]);
                            Y_inlier.push_back(totalPoints2[i]);
                        }
                    }

                    local_estimator.Estimate(X_inlier, Y_inlier, &local_models);

                    const size_t prev_best_num_inliers = best_support.num_inliers;

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(totalPoints1, totalPoints2, local_model, &residuals);
                        const auto local_support = support_measurer.Evaluate(residuals, max_residual);

                        // Check if locally optimized model is better.
                        if (support_measurer.IsLeftBetter(local_support, best_support)) {
                            best_support = local_support;
                            best_model = local_model;
                            best_model_is_local = true;
                            std::swap(residuals, best_local_residuals);
                        }
                    }

                    // Only continue recursive local optimization, if the inlier set
                    // size increased and we thus have a chance to further improve.
                    if (best_support.num_inliers <= prev_best_num_inliers) {
                        break;
                    }
                    // Swap back the residuals, so we can extract the best inlier
                    // set in the next recursion of local optimization.
                    std::swap(residuals, best_local_residuals);
                }
            }
        }
    }
   
    typename Report<FundamentalMatrixSevenPointEstimator, InlierSupportMeasurer>::Report F_report;
    F_report.support = best_support;
    F_report.model = best_model;

    // No valid model was found
    if (F_report.support.num_inliers < FundamentalMatrixSevenPointEstimator::kMinNumSamples) {
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
TwoViewGeometry EstimateCalibratedTwoViewGeometry(
    const Camera& camera1,
    const Image& img1,
    const Camera& camera2,
    const Image& img2) {
    TwoViewGeometry geometry;
    std::vector<point2D_t>matches;
    matches.reserve(std::min(img1.featPts.size(), img2.featPts.size()));
    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = img1.featPts.begin(); iter != img1.featPts.end(); iter++) 
    {
        if (img2.featPts.count(iter->first) != 0)
        {
            matches.emplace_back(iter->first);
        }
    }
    const size_t min_num_inliers = 8;
    if (matches.size() < static_cast<size_t>(min_num_inliers)) {
        geometry.config = TwoViewGeometry::ConfigurationType::DEGENERATE;
        return geometry;
    }
    // Extract corresponding points.
    std::vector<Eigen::Vector2d> matched_points1(matches.size());
    std::vector<Eigen::Vector2d> matched_points2(matches.size());
    std::vector<Eigen::Vector2d> matched_points1_normalized(matches.size());
    std::vector<Eigen::Vector2d> matched_points2_normalized(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& featId = matches[i];
        matched_points1[i] = img1.Points2D()[featId];
        matched_points2[i] = img2.Points2D()[featId];
        matched_points1_normalized[i] = camera1.CamFromImg(matched_points1[i]);
        matched_points2_normalized[i] = camera2.CamFromImg(matched_points2[i]);
    }


    // Estimate epipolar models.
    EssentialMatrixFivePointEstimator essentialMatrix;


    /// <summary>
    /// ////////////////****************************
    std::vector<Eigen::Vector2d> totalPoints1(matches.size());
    std::vector<Eigen::Vector2d> totalPoints2(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        totalPoints1[i] = img1.Points2D()[matches[i]];
        totalPoints2[i] = img2.Points2D()[matches[i]];
    }
    InlierSupportMeasurer support_measurer;
    typename InlierSupportMeasurer::Support best_support;
    typename FundamentalMatrixSevenPointEstimator::M_t best_model;
    bool best_model_is_local = false;
    std::vector<typename FundamentalMatrixEightPointEstimator::X_t> X_inlier;
    std::vector<typename FundamentalMatrixEightPointEstimator::Y_t> Y_inlier;

    const double max_residual = 40.;
    FundamentalMatrixSevenPointEstimator estimator;
    FundamentalMatrixEightPointEstimator local_estimator;

    int maxTrialsNum = std::min(Combination1(matches.size(), 7), 200);
    for (int num_trials = 0; num_trials < maxTrialsNum; num_trials++)
    {
        std::shuffle(matches.begin(), matches.end(), std::default_random_engine(std::time(0)));
        std::vector<Eigen::Vector2d> sevenPoints1(7);
        std::vector<Eigen::Vector2d> sevenPoints2(7);
        for (size_t i = 0; i < 7; ++i) {
            sevenPoints1[i] = img1.Points2D()[matches[i]];
            sevenPoints2[i] = img2.Points2D()[matches[i]];
        }
        std::vector<double> residuals;
        std::vector<double> best_local_residuals;
        std::vector<typename FundamentalMatrixSevenPointEstimator::M_t> sample_models;
        std::vector<typename FundamentalMatrixEightPointEstimator::M_t> local_models;

        estimator.Estimate(sevenPoints1, sevenPoints2, &sample_models);
        for (const auto& sample_model : sample_models)
        {
            estimator.Residuals(totalPoints1, totalPoints2, sample_model, &residuals);
            const auto support = support_measurer.Evaluate(residuals, max_residual);
            // Do local optimization if better than all previous subsets.
            if (support_measurer.IsLeftBetter(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;
                // Estimate locally optimized model from inliers.
                if (support.num_inliers > FundamentalMatrixSevenPointEstimator::kMinNumSamples &&
                    support.num_inliers >= FundamentalMatrixEightPointEstimator::kMinNumSamples)
                {
                    // Recursive local optimization to expand inlier set.

                    X_inlier.clear();
                    Y_inlier.clear();
                    X_inlier.reserve(matches.size());
                    Y_inlier.reserve(matches.size());
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(totalPoints1[i]);
                            Y_inlier.push_back(totalPoints2[i]);
                        }
                    }

                    local_estimator.Estimate(X_inlier, Y_inlier, &local_models);

                    const size_t prev_best_num_inliers = best_support.num_inliers;

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(totalPoints1, totalPoints2, local_model, &residuals);
                        const auto local_support = support_measurer.Evaluate(residuals, max_residual);

                        // Check if locally optimized model is better.
                        if (support_measurer.IsLeftBetter(local_support, best_support)) {
                            best_support = local_support;
                            best_model = local_model;
                            best_model_is_local = true;
                            std::swap(residuals, best_local_residuals);
                        }
                    }

                    // Only continue recursive local optimization, if the inlier set
                    // size increased and we thus have a chance to further improve.
                    if (best_support.num_inliers <= prev_best_num_inliers) {
                        break;
                    }
                    // Swap back the residuals, so we can extract the best inlier
                    // set in the next recursion of local optimization.
                    std::swap(residuals, best_local_residuals);
                }
            }
        }
    }

    typename Report<FundamentalMatrixSevenPointEstimator, InlierSupportMeasurer>::Report F_report;
    F_report.support = best_support;
    F_report.model = best_model;

    // No valid model was found
    if (F_report.support.num_inliers < FundamentalMatrixSevenPointEstimator::kMinNumSamples) {
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
    EstimateUncalibratedTwoViewGeometry(camera1, img1, camera2, img2);

    return 0;
}