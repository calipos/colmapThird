#include <list>
#include <string>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include "colmath.h"
#include "rigid3.h"
#include "two_view_geometry.h"
#include "estimate.h"
#include "camera.h"
#include "scene.h"
#include "image.h"
#include "essential_matrix.h"
#include "triangulation.h"
#include "bundle_adjustment.h"
#include "two_view_geometry.h"
#include "bitmap.h"
#include "undistortion.h"
#include "opencv2/opencv.hpp"
#include "registerFrame.h"
int test_bitmap()
{
    std::filesystem::path dataPath = "../data";
    std::map<Camera, std::vector<Image>> dataset = loadImageData(dataPath, ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList;
    std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    const Image& image1 = imageList[0];
    const Camera& camera1 = cameraList[image1.CameraId()];
    Bitmap distorted_bitmap;
    distorted_bitmap.Read(image1.Name());
    UndistortCameraOptions undistortion_options;
    Camera  undistorted_camera;
    Bitmap  undistorted_bitmap;// = distorted_bitmap.Clone();
    UndistortImage(undistortion_options,
        distorted_bitmap,
        camera1,
        &undistorted_bitmap,
        &undistorted_camera);
    undistorted_bitmap.Write(image1.Name() + ".jpg");
    return 0;
}
namespace utils
{
    cv::Mat intrConvert(const Eigen::Matrix3d& intr)
    {
        cv::Mat intrMat(3, 3, CV_64FC1);
        for (int i = 0; i < 9; i++)
        {
            int r = i / 3;
            int c = i % 3;
            intrMat.ptr<double>(r)[c] = intr(r, c);
        }
        return intrMat;
    }
    bool convertRt(const Eigen::Matrix3x4d&Rt, cv::Mat&rvec, cv::Mat& tvec)
    {
        cv::Mat R(3, 3, CV_64FC1);
        for (int i = 0; i < 9; i++)
        {
            int r = i / 3;
            int c = i % 3;
            R.ptr<double>(r)[c] = Rt(r, c);
        }
        cv::Rodrigues(R,rvec);
        tvec = cv::Mat(3,1,CV_64FC1);
        for (int r = 0; r < 3; r++)
        {
            tvec.ptr<double>(r)[0] = Rt(r, 3);
        }
        return true;
    }
}
int register_incremental(const std::string& folder)
{
    //std::shuffle(incrementalImages.begin(), incrementalImages.end(), std::default_random_engine(0));
    //std::vector<image_t>incrementalImages = { 0,1,2,3,4 };
    std::filesystem::path dataPath = folder;
    std::map<Camera, std::vector<Image>> dataset = loadImageData(dataPath, ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList;
    std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    std::vector<image_t> incrementalImages(imageList.size());
    std::iota(incrementalImages.begin(), incrementalImages.end(), 0);
    std::unordered_map<point3D_t, Eigen::Vector3d>objPts;
    std::unordered_map < image_t, struct Rigid3d>poses;
    for (int i = 1; i < incrementalImages.size(); i++)
    {
        LOG_OUT << "\n====================================  " << i << " ====================================";
        image_t picked2 = incrementalImages[i];
        {
            //figure new frame pose from prev Image
            image_t picked1 = incrementalImages[i - 1];
            Image& image1 = imageList[picked1];
            Image& image2 = imageList[picked2];
            Camera& camera1 = cameraList[image1.CameraId()];
            Camera& camera2 = cameraList[image2.CameraId()];
            TwoViewGeometry two_view_geometry = EstimateCalibratedTwoViewGeometry(camera1, image1, camera2, image2);
            bool EstimateRet = EstimateTwoViewGeometryPose(camera1, image1, camera2, image2, &two_view_geometry);
            if (!EstimateRet)
            {
                LOG_ERR_OUT << "EstimateTwoViewGeometryPose failed!";
                return -1;
            }

            if (poses.count(picked1) == 0)
            {
                poses[picked1] = Rigid3d();
                image1.SetCamFromWorld(poses[picked1]);
            }
            poses[picked2] = two_view_geometry.cam2_from_cam1 * poses[picked1];
            image2.SetCamFromWorld(poses[picked2]);

            {

                std::vector<cv::Point2d>imgPtsPnp;
                std::vector<cv::Point3d>objPtsPnp;
                imgPtsPnp.reserve(image2.featPts.size());
                objPtsPnp.reserve(image2.featPts.size());
                for (const auto& [ptId, imgPt] : image2.featPts)
                {
                    if (objPts.count(ptId) > 0)
                    {
                        imgPtsPnp.emplace_back(imgPt[0], imgPt[1]);
                        objPtsPnp.emplace_back(objPts[ptId][0], objPts[ptId][1], objPts[ptId][2]);
                    }
                }
                if (objPtsPnp.size() >= 6)
                {
                    Eigen::Matrix3x4d Rt = image1.CamFromWorld().ToMatrix();
                    cv::Mat rvec, tvec;
                    utils::convertRt(Rt, rvec, tvec);

                    //Eigen::AngleAxisd eulerAngle(cv::norm(rvec), Eigen::Vector3d(rvec.ptr<double>(0)[0] / cv::norm(rvec), rvec.ptr<double>(1)[0] / cv::norm(rvec), rvec.ptr<double>(2)[0] / cv::norm(rvec)));
                    //Eigen::Quaterniond q2(eulerAngle);
                    //LOG_OUT << q2;


                    cv::Mat intrMat = utils::intrConvert(camera2.CalibrationMatrix());
                    cv::solvePnP(objPtsPnp, imgPtsPnp, intrMat, cv::Mat(), rvec, tvec, true);
                    Eigen::AngleAxisd eulerAngle(cv::norm(rvec), Eigen::Vector3d(rvec.ptr<double>(0)[0] / cv::norm(rvec), rvec.ptr<double>(1)[0] / cv::norm(rvec), rvec.ptr<double>(2)[0] / cv::norm(rvec)));
                    Rigid3d pnpRt(Eigen::Quaterniond(eulerAngle), Eigen::Vector3d(tvec.ptr<double>(0)[0], tvec.ptr<double>(1)[0], rvec.ptr<double>(2)[0]));
                    LOG_OUT << "before pnp: " << image2.CamFromWorld().ToMatrix();
                    //if (i!=9)
                    //{
                    image2.SetCamFromWorld(pnpRt);
                    LOG_OUT << "after  pnp: " << image2.CamFromWorld().ToMatrix();
                    //}

                } 

            }
        }
        {
            //updata pose3d from total prev
            for (int j = 0; j < i; j++)
            {
                image_t picked1 = incrementalImages[j];
                Image& image1 = imageList[picked1];
                Image& image2 = imageList[picked2];
                Camera& camera1 = cameraList[image1.CameraId()];
                Camera& camera2 = cameraList[image2.CameraId()];
                const Eigen::Matrix3x4d cam_from_world1 = image1.CamFromWorld().ToMatrix();
                const Eigen::Matrix3x4d cam_from_world2 = image2.CamFromWorld().ToMatrix();
                const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
                const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();
                // Update Reconstruction
                std::vector<point2D_t>matchesPointId;
                matchesPointId.reserve(std::min(image1.featPts.size(), image2.featPts.size()));
                for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = image1.featPts.begin(); iter != image1.featPts.end(); iter++)
                {
                    if (image2.featPts.count(iter->first) != 0 && objPts.count(iter->first) == 0)
                    {
                        matchesPointId.emplace_back(iter->first);
                    }
                }
                for (const auto& ptId : matchesPointId)
                {
                    //if (j==8)
                    //{
                    //    continue;
                    //}
                    const Eigen::Vector2d point2D1 = camera1.CamFromImg(image1.featPts[ptId]);
                    const Eigen::Vector2d point2D2 = camera2.CamFromImg(image2.featPts[ptId]);
                    Eigen::Vector3d xyz;
                    bool triangulatePointRet = TriangulatePoint(cam_from_world1, cam_from_world2, point2D1, point2D2, &xyz);
                    if (triangulatePointRet)
                    {
                        objPts[ptId] = xyz;
                        std::pair<bool, Eigen::Vector2d>imgPt1 = image1.ProjectPoint(xyz);
                        std::pair<bool, Eigen::Vector2d>imgPt2 = image2.ProjectPoint(xyz);
                    }
                }
            }
        }
        if (i != 1)// only two images need not ba.
        {
            //ba
            BundleAdjustmentOptions ba_options;
            BundleAdjustmentConfig ba_config;
            for (int j = 0; j <= i; j++)
            {
                ba_config.AddImage(incrementalImages[j]);
                //LOG_OUT << incrementalImages[j];
            }
            for (const auto& d : objPts) ba_config.AddVariablePoint(d.first);
            std::unique_ptr<BundleAdjuster> bundle_adjuster;
            ba_config.SetConstantCamPose(incrementalImages[0]);  // 1st image
            bundle_adjuster = CreateDefaultBundleAdjuster(std::move(ba_options), std::move(ba_config), cameraList, imageList, objPts);

            auto solverRet = bundle_adjuster->Solve();
            //for (const auto& d : objPts) LOG_OUT << "objPts : " << d.second[0] << "  " << d.second[1] << "  " << d.second[2];
            LOG_OUT << "after  ba: " << imageList[picked2].CamFromWorld().ToMatrix();
            for (int j = 0; j < cameraList.size(); j++) LOG_OUT << cameraList[j];
            if (solverRet.termination_type != ceres::CONVERGENCE)
            {
                LOG_ERR_OUT << "not convergence! incremental at " << imageList[picked2].Name();
                return -1;
            }
            {
                //std::filesystem::create_directories(dataPath / ("result"+std::to_string(i)));
                //writeResult(dataPath / ("result" + std::to_string(i)), cameraList, imageList, objPts, poses);
            }
        }
    }

    for (auto& d : poses)
    {
        d.second = imageList[d.first].CamFromWorld();
    }
    writeResult(dataPath / "result", cameraList, imageList, objPts, poses);

    return 0;
}
double reprojectTotal(const std::set<int>&pickedImgs,const std::vector<Camera>&cameraList,const std::vector<Image>& imageList,const std::unordered_map<point3D_t, Eigen::Vector3d>&objPts)
{

    std::vector<double>errs;
    errs.reserve(pickedImgs.size()* objPts.size());
    for (const auto&imgId: pickedImgs)
    {
        const Image& img = imageList[imgId];
        const Camera& camera = cameraList[img.CameraId()];
        const Eigen::Matrix3x4d& Rt = img.CamFromWorld().ToMatrix();
        const Eigen::Matrix3d& K = camera.CalibrationMatrix();
        for (const auto& feat2d:img.featPts)
        {
            const auto& ptId = feat2d.first;
            const Eigen::Vector2d& pt2d = feat2d.second;
            if (objPts.count(ptId)>0)
            {
                Eigen::Vector4d pt3d;
                pt3d[0] = objPts.at(ptId)[0];
                pt3d[1] = objPts.at(ptId)[1];
                pt3d[2] = objPts.at(ptId)[2];
                pt3d[3] = 1;
                Eigen::Vector3d repojectPt = K* Rt* pt3d;
                double diffX = repojectPt[0] / repojectPt[2] - pt2d[0];
                double diffY = repojectPt[1] / repojectPt[2] - pt2d[1];
                errs.emplace_back(std::sqrt(diffX * diffX + diffY* diffY));
            }
        }
    }
    if (errs.size()==0)
    {
        return 1e20;
    }
    return std::accumulate(errs.begin(), errs.end(),0.)/ errs.size();
}
int countSharedPtsCount(const Image& img1, const Image& img2)
{
    std::vector<point2D_t>matchesPointId;
    matchesPointId.reserve(std::min(img1.featPts.size(), img2.featPts.size()));
    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = img1.featPts.begin(); iter != img1.featPts.end(); iter++)
    {
        if (img2.featPts.count(iter->first) != 0)
        {
            matchesPointId.emplace_back(iter->first);
        }
    }
    return matchesPointId.size();
}
int register_incremental_loop(const std::string& folder)
{
    //std::shuffle(incrementalImages.begin(), incrementalImages.end(), std::default_random_engine(0));
    //std::vector<image_t>incrementalImages = { 0,1,2,3,4 };
    std::filesystem::path dataPath = folder;
    std::map<Camera, std::vector<Image>> dataset = loadImageData(dataPath, ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList;
    std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    std::vector<image_t> incrementalImages(imageList.size());
    std::iota(incrementalImages.begin(), incrementalImages.end(), 0);
    std::unordered_map<point3D_t, Eigen::Vector3d>objPts;
    std::unordered_map < image_t, struct Rigid3d>poses;
    {
        std::set<int>pickedImgs;
        pickedImgs.insert(0);
        {
            poses[0] = Rigid3d();
            imageList[0].SetCamFromWorld(poses[0]);
        }
        int prevPickedImgsSize = 1;
        std::map<std::string, TwoViewGeometry> TwoViewGeometryRecode;
        while (true)
        {

            int bestTarget = -1;
            int bestSource = -1;
            int largestAngleRadx100 = -100;
            int sharedPtsCnt = 0;
            //pick incremental instance
            for (int k = 0; k < incrementalImages.size(); k++)
            {
                if (pickedImgs.count(k) != 0)
                {
                    continue;
                }
                Image& image2 = imageList[k];
                Camera& camera2 = cameraList[image2.CameraId()];
                int largestAngleRadxTotalThisIncremental = 0;
                int largestAngleRadxThisIncremental = -100;
                int largestAngleRadxThisIncrementalTargetId = -1;
                for (const auto& targetImgIdx : pickedImgs)
                {
                    Image& image1 = imageList[targetImgIdx];
                    Camera& camera1 = cameraList[image1.CameraId()];
                    std::string recodeKeyStr1 = std::to_string(targetImgIdx) + "_" + std::to_string(k);
                    std::string recodeKeyStr2 = std::to_string(k) + "_" + std::to_string(targetImgIdx);
                    int sharedPtsCnt_ = countSharedPtsCount(image1, image2);
                    if (sharedPtsCnt_ > 5)
                    {
                        TwoViewGeometry two_view_geometry;
                        if (TwoViewGeometryRecode.count(recodeKeyStr1) != 0)
                        {
                            two_view_geometry = TwoViewGeometryRecode[recodeKeyStr1];
                        }
                        else if (TwoViewGeometryRecode.count(recodeKeyStr2) != 0)
                        {
                            two_view_geometry = TwoViewGeometryRecode[recodeKeyStr2];
                        }
                        else
                        {
                            two_view_geometry = EstimateCalibratedTwoViewGeometry(camera1, image1, camera2, image2);
                            bool EstimateRet = EstimateTwoViewGeometryPose(camera1, image1, camera2, image2, &two_view_geometry);
                            TwoViewGeometryRecode[recodeKeyStr1] = two_view_geometry;
                            TwoViewGeometryRecode[recodeKeyStr2] = two_view_geometry;
                            if (!EstimateRet)
                            {
                                LOG_ERR_OUT << "cannot be here.fatal.";
                                continue;
                            }
                        }
                        Eigen::AngleAxisd aa(two_view_geometry.cam2_from_cam1.rotation);
                        int angle_radx100 = abs(aa.angle()) * 100;
                        const int& angle_threshold = 3.1415926 * 50;//90deg x 100
                        //if (angle_radx100 > angle_threshold) angle_radx100 = angle_threshold;//30deg x 100                           
                        double baseLine = two_view_geometry.cam2_from_cam1.translation.norm();
                        largestAngleRadxTotalThisIncremental += angle_radx100;
                        if (angle_radx100> largestAngleRadxThisIncremental)
                        {
                            largestAngleRadxThisIncremental = angle_radx100;
                            largestAngleRadxThisIncrementalTargetId = targetImgIdx;
                        }
                    }                     
                }
                if (largestAngleRadxTotalThisIncremental==0)
                {
                    continue;
                    LOG_ERR_OUT << "cannot find incremental match: " << image2.Name();
                }
                if (largestAngleRadxTotalThisIncremental >= largestAngleRadx100)
                {
                    largestAngleRadx100 = largestAngleRadxTotalThisIncremental;
                    bestTarget = largestAngleRadxThisIncrementalTargetId;
                    bestSource = k;
                }
            }


            LOG_OUT << "\n====================================  " << bestTarget<<" & "<< bestSource << " ====================================";
            Image& image1 = imageList[bestTarget];
            Image& image2 = imageList[bestSource];
            Camera& camera1 = cameraList[image1.CameraId()];
            Camera& camera2 = cameraList[image2.CameraId()];
            int img2FeatSharedCnt = 0;
            for (const auto&sourceFeat: image2.featPts)
            {
                if (objPts.count(sourceFeat.first)>0)
                {
                    img2FeatSharedCnt += 1;
                }
            }
            if (img2FeatSharedCnt>5)
            {
                //figure initial rt from pnp
                std::vector<cv::Point2d>imgPtsPnp;
                std::vector<cv::Point3d>objPtsPnp;
                imgPtsPnp.reserve(image2.featPts.size());
                objPtsPnp.reserve(image2.featPts.size());
                for (const auto& [ptId, imgPt] : image2.featPts)
                {
                    if (objPts.count(ptId) > 0)
                    {
                        imgPtsPnp.emplace_back(imgPt[0], imgPt[1]);
                        objPtsPnp.emplace_back(objPts[ptId][0], objPts[ptId][1], objPts[ptId][2]);
                    }
                }
                {
                    cv::Mat rvec, tvec;
                    cv::Mat intrMat = utils::intrConvert(camera2.CalibrationMatrix());
                    cv::solvePnP(objPtsPnp, imgPtsPnp, intrMat, cv::Mat(), rvec, tvec, false);
                    Eigen::AngleAxisd eulerAngle(cv::norm(rvec), Eigen::Vector3d(rvec.ptr<double>(0)[0] / cv::norm(rvec), rvec.ptr<double>(1)[0] / cv::norm(rvec), rvec.ptr<double>(2)[0] / cv::norm(rvec)));
                    Rigid3d pnpRt(Eigen::Quaterniond(eulerAngle), Eigen::Vector3d(tvec.ptr<double>(0)[0], tvec.ptr<double>(1)[0], rvec.ptr<double>(2)[0]));

                    poses[bestSource] = pnpRt;
                    image2.SetCamFromWorld(pnpRt);
                }

            }
            else
            {
                //figure initial rt from EstimateTwoViewGeometryPose 
                std::string recodeKeyStr1 = std::to_string(bestTarget) + "_" + std::to_string(bestSource);
                if (TwoViewGeometryRecode.count(recodeKeyStr1)==0)
                {
                    LOG_ERR_OUT << "cannot be here..";
                }
                TwoViewGeometry& two_view_geometry = TwoViewGeometryRecode[recodeKeyStr1];
                bool EstimateRet = EstimateTwoViewGeometryPose(camera1, image1, camera2, image2, &two_view_geometry);
                if (!EstimateRet)
                {
                    LOG_ERR_OUT << "EstimateTwoViewGeometryPose failed!";
                    return -1;
                }
                poses[bestSource] = two_view_geometry.cam2_from_cam1 * poses[bestTarget];
                image2.SetCamFromWorld(poses[bestSource]);
            }

            {
                //triganlePoints
                std::unordered_map<point2D_t, std::vector<Eigen::Vector3d>>potentialPts3d;
                for (const auto&prevImgId: pickedImgs)
                {
                    const Image& image0 = imageList[prevImgId];
                    const Camera& camera0 = cameraList[image0.CameraId()];
                    const Eigen::Matrix3x4d cam_from_world0 = image0.CamFromWorld().ToMatrix();
                    const Eigen::Matrix3x4d cam_from_world2 = image2.CamFromWorld().ToMatrix();
                    const Eigen::Vector3d proj_center0 = image0.ProjectionCenter();
                    const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();
                    // Update Reconstruction
                    std::vector<point2D_t>matchesPointId;
                    matchesPointId.reserve(std::min(image0.featPts.size(), image2.featPts.size()));
                    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = image0.featPts.begin(); iter != image0.featPts.end(); iter++)
                    {
                        if (image2.featPts.count(iter->first) != 0 && objPts.count(iter->first) == 0)
                        {
                            matchesPointId.emplace_back(iter->first);
                        }
                    }
                    for (const auto& ptId : matchesPointId)
                    {
                        const Eigen::Vector2d point2D0 = camera0.CamFromImg(image0.featPts.at(ptId));
                        const Eigen::Vector2d point2D2 = camera2.CamFromImg(image2.featPts.at(ptId));
                        Eigen::Vector3d xyz;
                        bool triangulatePointRet = TriangulatePoint(cam_from_world0, cam_from_world2, point2D0, point2D2, &xyz);
                        if (triangulatePointRet)
                        {
                            if (potentialPts3d.count(ptId)==0)
                            {
                                potentialPts3d[ptId].reserve(pickedImgs.size());
                            }
                            potentialPts3d[ptId].emplace_back(xyz);                            
                            std::pair<bool, Eigen::Vector2d>imgPt1 = image1.ProjectPoint(xyz);
                            std::pair<bool, Eigen::Vector2d>imgPt2 = image2.ProjectPoint(xyz);
                            LOG_OUT << imgPt1.second.transpose() << " " << imgPt2.second.transpose();
                        }
                    }
                }
                for (const auto &potentialIncrementalPt3d: potentialPts3d)
                {
                    auto ptId = potentialIncrementalPt3d.first;
                    if (objPts.count(ptId)==0)
                    {
                        objPts[ptId] = Eigen::Vector3d::Zero();
                        for (const auto&d: potentialIncrementalPt3d.second)
                        {
                            objPts[ptId][0] += d[0];
                            objPts[ptId][1] += d[1];
                            objPts[ptId][2] += d[2];
                        }
                        objPts[ptId][0] /= potentialIncrementalPt3d.second.size();
                        objPts[ptId][1] /= potentialIncrementalPt3d.second.size();
                        objPts[ptId][2] /= potentialIncrementalPt3d.second.size();
                    }
                }
            }
            pickedImgs.insert(bestSource);
            {
                //ba
                BundleAdjustmentOptions ba_options;
                ba_options.solver_options.max_num_iterations = 5000;
                //ba_options.solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
                //ba_options.solver_options.minimizer_progress_to_stdout = true;
                BundleAdjustmentConfig ba_config;
                for (const auto& d : pickedImgs)
                {
                    ba_config.AddImage(incrementalImages[d]);
                }
                for (const auto& d : objPts) ba_config.AddVariablePoint(d.first);
                std::unique_ptr<BundleAdjuster> bundle_adjuster;
                ba_config.SetConstantCamPose(incrementalImages[0]);  // 1st image
                bundle_adjuster = CreateDefaultBundleAdjuster(std::move(ba_options), std::move(ba_config), cameraList, imageList, objPts);


                std::map<int, Eigen::Quaterniond>qs;
                std::map<int, Eigen::RowVector3d>ts;
                for (const auto& d : pickedImgs)
                {
                    const Image& imag = imageList[d];
                    qs[d] = imag.CamFromWorld().rotation;
                    ts[d] = imag.CamFromWorld().translation.transpose();
                }


                auto solverRet = bundle_adjuster->Solve();
                for (int j = 0; j < cameraList.size(); j++) LOG_OUT << cameraList[j];
                if (solverRet.termination_type != ceres::CONVERGENCE)
                {
                    LOG_ERR_OUT << "not convergence! incremental at " << imageList[bestSource].Name();
                    //return -1;
                    //break;
                }
                //else
                {
                    double final_cost = reprojectTotal(pickedImgs, cameraList, imageList, objPts);
                    LOG_OUT << "final_cost = " << final_cost;
                    if (final_cost > 5)
                    {
                        LOG_ERR_OUT << "final_cost>5 at " << imageList[bestSource].Name();
                        return -1;
                        break;
                    }
                    for (const auto& d : pickedImgs)
                    {
                        const Image& imag = imageList[d];
                        LOG_OUT << d << "qt" << qs[d] << ", " << ts[d] << "    " << imag.CamFromWorld().rotation << ", " << imag.CamFromWorld().translation.transpose();
                    }
                }
            }
            
            if (pickedImgs.size() == incrementalImages.size())
            {
                break;
            }
            if (prevPickedImgsSize== pickedImgs.size())
            {
                for (int k = 0; k < incrementalImages.size(); k++)
                {
                    if (pickedImgs.count(k) != 0)
                        LOG_OUT << imageList[k].Name();
                }
                for (int k = 0; k < incrementalImages.size(); k++)
                {
                    if (pickedImgs.count(k) == 0)
                    {
                        LOG_OUT << imageList[k].Name() << " not find a pair";
                        for (const auto& l : pickedImgs)
                        {
                            std::string recodeKeyStr = std::to_string(k) + "_" + std::to_string(l);
                            TwoViewGeometry two_view_geometry;
                            if (TwoViewGeometryRecode.count(recodeKeyStr) != 0)
                            {
                                two_view_geometry = TwoViewGeometryRecode[recodeKeyStr];
                            }
                            else
                            {
                                LOG_ERR_OUT << "cannot be here.";
                                return -1;
                            }
                            Eigen::AngleAxisd aa(two_view_geometry.cam2_from_cam1.rotation);
                            Image& image1 = imageList[l];
                            Image& image2 = imageList[k];
                            int sharedPtsCnt_ = countSharedPtsCount(image1, image2);
                            if (sharedPtsCnt_ > 5)
                            {
                                LOG_OUT << "\t\t" << imageList[l].Name() << " : deg=" << aa.angle() * 180 / 3.1415926 << "; sharedPtsCnt=" << sharedPtsCnt_;
                            }
                        }


                    }
                }
                LOG_ERR_OUT << "annotation need fixs.";
                return -1;
            }
            prevPickedImgsSize = pickedImgs.size();
        }
    }
    for (auto& d : poses)
    {
        d.second = imageList[d.first].CamFromWorld();
    }
    writeResult(dataPath / "result", cameraList, imageList, objPts, poses);

    return 0;
}
int test_incremental()
{
    register_incremental("../data/a");

    return 0;
}
