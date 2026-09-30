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
#include "labelme.h"
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
double reprojectTotal(const std::set<image_t>&pickedImgs,const std::vector<Camera>&cameraList,const std::vector<Image>& imageList,const std::unordered_map<point3D_t, Eigen::Vector3d>&objPts)
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
    std::set<image_t>pickedImgs;
    std::unordered_map<std::uint32_t, std::set<std::uint32_t>>blacklist;
    {
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
            float projectError = std::numeric_limits<float>::max();
            //pick incremental instance
            for (int k = 0; k < incrementalImages.size(); k++)
            {
                if (pickedImgs.count(k) != 0)
                {
                    continue;
                }
                Image& image2 = imageList[k];
                Camera& camera2 = cameraList[image2.CameraId()];
                int largestAngleRadxThisIncremental = -100;
                int largestAngleRadxThisIncrementalTargetId = -1;
                for (const auto& targetImgIdx : pickedImgs)
                {
                    if (blacklist.count(targetImgIdx) > 0 && blacklist.at(targetImgIdx).count(k)>0)
                    {
                        continue;
                    }
                    Image& image1 = imageList[targetImgIdx];
                    Camera& camera1 = cameraList[image1.CameraId()];
                    std::string recodeKeyStr1 = std::to_string(targetImgIdx) + "_" + std::to_string(k);
                    int sharedPtsCnt_ = countSharedPtsCount(image1, image2);
                    if (sharedPtsCnt_ > 5)
                    {
                        TwoViewGeometry two_view_geometry;
                        if (TwoViewGeometryRecode.count(recodeKeyStr1) != 0)
                        {
                            two_view_geometry = TwoViewGeometryRecode[recodeKeyStr1];
                        }
                        else
                        {
                            two_view_geometry = EstimateCalibratedTwoViewGeometry(camera1, image1, camera2, image2);   
                            bool EstimateRet = EstimateTwoViewGeometryPose(camera1, image1, camera2, image2, &two_view_geometry);
                            TwoViewGeometryRecode[recodeKeyStr1] = two_view_geometry;
                            two_view_geometry.Invert();
                            std::string recodeKeyStr2 = std::to_string(k) + "_" + std::to_string(targetImgIdx);
                            TwoViewGeometryRecode[recodeKeyStr2] = two_view_geometry;
                            if (!EstimateRet)
                            {
                                LOG_ERR_OUT << "cannot be here.fatal.";
                                continue;
                            }
                        }
                        float projectErrorCurr = -1;
                        if (true)
                        {
                            auto guessPose1 = Rigid3d();
                            auto guessPose2 = TwoViewGeometryRecode[recodeKeyStr1].cam2_from_cam1;
                            std::vector<point2D_t>matchesPointId;
                            matchesPointId.reserve(std::min(image1.featPts.size(), image2.featPts.size()));
                            for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = image1.featPts.begin(); iter != image1.featPts.end(); iter++)
                            {
                                if (image2.featPts.count(iter->first) != 0)
                                {
                                    matchesPointId.emplace_back(iter->first);
                                }
                            }
                            for (const auto& ptId : matchesPointId)
                            {
                                const Eigen::Vector2d point2D1 = camera1.CamFromImg(image1.featPts.at(ptId));
                                const Eigen::Vector2d point2D2 = camera2.CamFromImg(image2.featPts.at(ptId));
                                Eigen::Vector3d xyz;
                                bool triangulatePointRet = TriangulatePoint(guessPose1.ToMatrix(), guessPose2.ToMatrix(), point2D1, point2D2, &xyz);
                                if (triangulatePointRet)
                                {
                                    const Eigen::Vector3d point3D_in_cam1 = guessPose1 * xyz;
                                    const Eigen::Vector3d point3D_in_cam2 = guessPose2 * xyz;
                                    if (point3D_in_cam1.z() > std::numeric_limits<double>::epsilon()&& point3D_in_cam2.z() > std::numeric_limits<double>::epsilon()) {
                                        Eigen::Vector2d  imgPt1 = camera1.ImgFromCam(point3D_in_cam1.hnormalized());
                                        Eigen::Vector2d  imgPt2 = camera2.ImgFromCam(point3D_in_cam2.hnormalized());
                                        float projectError_ = (std::max)((image1.featPts.at(ptId) - imgPt1).norm(), (image2.featPts.at(ptId) - imgPt2).norm());
                                        if (projectErrorCurr<0)
                                        {
                                            projectErrorCurr = projectError_;
                                        }
                                        else if (projectErrorCurr < projectError_)
                                        {
                                            projectErrorCurr = projectError_;
                                        } 
                                    }
                                    else
                                    {
                                        projectErrorCurr = -1;
                                        break;
                                    }
                                }
                            }
                        }
                        if (projectErrorCurr>0)
                        {
                            Eigen::AngleAxisd aa(TwoViewGeometryRecode[recodeKeyStr1].cam2_from_cam1.rotation);
                            double baseLine = TwoViewGeometryRecode[recodeKeyStr1].cam2_from_cam1.translation.norm();
                            int angle_radx100 = abs(aa.angle()) * 180 / 3.1415926 * 100;
                            const int& angle_threshold = 20'00;
                            //LOG_OUT << targetImgIdx << ":" << k << "   angle=" << angle_radx100 << "  projErr=" << projectErrorCurr;
                            //LOG_OUT << TwoViewGeometryRecode[recodeKeyStr1].cam2_from_cam1;
                            if (angle_radx100 > angle_threshold && projectErrorCurr< projectError)
                            {
                                projectError = projectErrorCurr;
                                bestTarget = targetImgIdx;
                                bestSource = k;
                            }
                        }  
                    }                     
                } 
            }
            if (bestTarget<0 || bestSource<0)
            {
                for (int k = 0; k < incrementalImages.size(); k++)
                {
                    if (pickedImgs.count(k) == 0)
                    {
                        LOG_OUT << imageList[k].Name() << " not find a pair";
                    }
                }
                return -1;
            }

            LOG_OUT << "\n====================================  " << bestTarget<<" & "<< bestSource << " ====================================";
            Image& image1 = imageList[bestTarget];
            Image& image2 = imageList[bestSource];
            Camera& camera1 = cameraList[image1.CameraId()];
            Camera& camera2 = cameraList[image2.CameraId()];
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
                float matchesPointIdEvalue = 0;
                for (const auto&prevImgId: pickedImgs)
                {
                    const Image& image0 = imageList[prevImgId];
                    const Camera& camera0 = cameraList[image0.CameraId()];
                    //LOG_OUT << image0.CamFromWorld(); 
                    //LOG_OUT << image2.CamFromWorld();
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
                            std::pair<bool, Eigen::Vector2d>imgPt0 = image0.ProjectPoint(xyz);
                            std::pair<bool, Eigen::Vector2d>imgPt2 = image2.ProjectPoint(xyz);
                            //LOG_OUT << imgPt0.first << " - " << imgPt0.second.transpose() << " , " << image0.featPts.at(ptId).transpose();
                            //LOG_OUT << imgPt2.first << " - " << imgPt2.second.transpose() << " , " << image2.featPts.at(ptId).transpose();
                             
                        }
                    }
                }
                for (const auto &potentialIncrementalPt3d: potentialPts3d)
                {
                    auto ptId = potentialIncrementalPt3d.first;
                    if (objPts.count(ptId)==0)
                    {
                        objPts[ptId] = potentialIncrementalPt3d.second[0];
                        //objPts[ptId] = Eigen::Vector3d::Zero();
                        //for (const auto&d: potentialIncrementalPt3d.second)
                        //{
                        //    objPts[ptId][0] += d[0];
                        //    objPts[ptId][1] += d[1];
                        //    objPts[ptId][2] += d[2];
                        //}
                        //objPts[ptId][0] /= potentialIncrementalPt3d.second.size();
                        //objPts[ptId][1] /= potentialIncrementalPt3d.second.size();
                        //objPts[ptId][2] /= potentialIncrementalPt3d.second.size();
                    }
                }
            }
            if (pickedImgs.size() < 2)
            {
                pickedImgs.insert(bestSource);
            }
            else
            {
                //ba
                BundleAdjustmentOptions ba_options;
                ba_options.solver_options.max_num_iterations = 1000;
                ba_options.solver_options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
                ba_options.solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
                ba_options.solver_options.minimizer_progress_to_stdout = true;
                BundleAdjustmentConfig ba_config;
                for (const auto& d : pickedImgs)
                {
                    ba_config.AddImage(incrementalImages[d]);
                }
                ba_config.AddImage(bestSource);//try
                for (const auto& d : objPts) ba_config.AddVariablePoint(d.first);
                std::unique_ptr<BundleAdjuster> bundle_adjuster;
                ba_config.SetConstantCamPose(incrementalImages[0]);  // 1st image
                ba_config.SetConstantCamIntrinsics(0); // 1st camera fxfxcxcy
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
                    blacklist[bestTarget].insert(bestSource); 
                    //return -1;
                    //break;
                }
                else
                {
                    double final_cost = reprojectTotal(pickedImgs, cameraList, imageList, objPts);
                    LOG_OUT << "final_cost = " << final_cost;
                    if (final_cost > 10)
                    {
                        LOG_ERR_OUT << "final_cost>5 at " << imageList[bestSource].Name();
                        blacklist[bestTarget].insert(bestSource);
                        //return -1;
                        //break;
                    }
                    else
                    {
                        pickedImgs.insert(bestSource);
                        for (const auto& d : pickedImgs)
                        {
                            const Image& imag = imageList[d];
                            LOG_OUT << d << "qt" << qs[d] << ", " << ts[d] << "    " << imag.CamFromWorld().rotation << ", " << imag.CamFromWorld().translation.transpose();
                        }
                    }
                }
            }
            
            if (pickedImgs.size() == incrementalImages.size())
            {
                break;
            }
            //if (prevPickedImgsSize== pickedImgs.size())
            //{
            //    for (int k = 0; k < incrementalImages.size(); k++)
            //    {
            //        if (pickedImgs.count(k) != 0)
            //            LOG_OUT << imageList[k].Name();
            //    }
            //    for (int k = 0; k < incrementalImages.size(); k++)
            //    {
            //        if (pickedImgs.count(k) == 0)
            //        {
            //            LOG_OUT << imageList[k].Name() << " not find a pair";
            //            for (const auto& l : pickedImgs)
            //            {
            //                std::string recodeKeyStr = std::to_string(k) + "_" + std::to_string(l);
            //                TwoViewGeometry two_view_geometry;
            //                if (TwoViewGeometryRecode.count(recodeKeyStr) != 0)
            //                {
            //                    two_view_geometry = TwoViewGeometryRecode[recodeKeyStr];
            //                }
            //                else
            //                {
            //                    LOG_ERR_OUT << "cannot be here.";
            //                    return -1;
            //                }
            //                Eigen::AngleAxisd aa(two_view_geometry.cam2_from_cam1.rotation);
            //                Image& image1 = imageList[l];
            //                Image& image2 = imageList[k];
            //                int sharedPtsCnt_ = countSharedPtsCount(image1, image2);
            //                if (sharedPtsCnt_ > 5)
            //                {
            //                    LOG_OUT << "\t\t" << imageList[l].Name() << " : deg=" << aa.angle() * 180 / 3.1415926 << "; sharedPtsCnt=" << sharedPtsCnt_;
            //                }
            //            }


            //        }
            //    }
            //    LOG_ERR_OUT << "annotation need fixs.";
            //    return -1;
            //}
            //prevPickedImgsSize = pickedImgs.size();
        }
    }
    for (auto& d : poses)
    {
        d.second = imageList[d.first].CamFromWorld();
    }

    for (auto&cam: cameraList)
    {
        auto camera_id = cam.camera_id;
        auto focal_length = cam.FocalLength();
        auto width = cam.width;
        auto height = cam.height;
        cam =  Camera::CreateFromModelId(camera_id, CameraModelId::kSimpleRadial, focal_length, width, height);
    }
    for (auto&img:imageList)
    { 
        img.SetCameraPtr(&cameraList[img.CameraId()]);
    } 
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
            LOG_ERR_OUT << "not convergence! incremental at total";
            return -1;
        }
        //else
        {
            double final_cost = reprojectTotal(pickedImgs, cameraList, imageList, objPts);
            LOG_OUT << "final_cost = " << final_cost;
            if (final_cost > 50)
            {
                LOG_ERR_OUT << "final_cost>5 at total";
                return -1;
            }
            for (const auto& d : pickedImgs)
            {
                const Image& imag = imageList[d];
                LOG_OUT << d << "qt" << qs[d] << ", " << ts[d] << "    " << imag.CamFromWorld().rotation << ", " << imag.CamFromWorld().translation.transpose();
            }
        }
    }



    writeResult(dataPath / "result", cameraList, imageList, objPts, poses);

    return 0;
}
int register_incremental_base_hint(const std::string& folder, const std::vector<std::string>&initViewName, const std::vector<std::string>&discardViewName, const bool& optimCameraIntr)
{ 
    std::map<Camera, std::vector<Image>> dataset = loadImageData("D:/repo/colmapThird/data/c", ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList;
    std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    image_t seed0 = -1, seed1 = -1, seed2 = -1;
    if (initViewName.size()!=3)
    {
        LOG_ERR_OUT << "initViewName.size()!=3";
        return -1;
    }
    try
    {
        seed0 = Image::picNameToIndx.at(initViewName[0]);
        seed1 = Image::picNameToIndx.at(initViewName[1]);
        seed2 = Image::picNameToIndx.at(initViewName[2]);
        if (seed0 == seed1 || seed0 == seed2 || seed1 == seed2)
        {
            LOG_ERR_OUT << "seed0 == seed1 || seed0 == seed2 || seed1 == seed2";
            return -1;
        }
    }
    catch (const std::exception&)
    {
        LOG_ERR_OUT << "not found seed";
        return -1;
    }
    LOG_OUT << "seed0 = " << seed0;
    LOG_OUT << "seed1 = " << seed1;
    LOG_OUT << "seed2 = " << seed2;

    std::vector<int>discardIds;
    discardIds.reserve(discardViewName.size());
    for (size_t i = 0; i < discardViewName.size(); i++)
    {
        try
        {
            int discardId = Image::picNameToIndx.at(discardViewName[i]); 
            discardIds.emplace_back(discardId);
        }
        catch (const std::exception&)
        {
            LOG_ERR_OUT << "not found discardId = "<< discardViewName[i];
            return -1;
        }
    }
    std::set<image_t>seedImgs;
    seedImgs.insert(seed0);
    seedImgs.insert(seed1);
    seedImgs.insert(seed2);
    Image& image0 = imageList[seed0];
    Camera& camera0 = cameraList[image0.CameraId()];
    Image& image1 = imageList[seed1];
    Camera& camera1 = cameraList[image1.CameraId()];

    TwoViewGeometry  two_view_geometry = EstimateCalibratedTwoViewGeometry(camera0, image0, camera1, image1);
    if (two_view_geometry.config == TwoViewGeometry::ConfigurationType::DEGENERATE)
    {
        LOG_ERR_OUT << "not enough match.";
        return -1;
    }
    bool EstimateRet = EstimateTwoViewGeometryPose(camera0, image0, camera1, image1, &two_view_geometry);
    image0.SetCamFromWorld(Rigid3d()); 
    image1.SetCamFromWorld(two_view_geometry.cam2_from_cam1 * Rigid3d());

    const Eigen::Matrix3x4d cam_from_world0 = image0.CamFromWorld().ToMatrix();
    const Eigen::Matrix3x4d cam_from_world1 = image1.CamFromWorld().ToMatrix(); 
    // Update Reconstruction
    std::vector<point2D_t>matchesPointId;
    matchesPointId.reserve(std::min(image0.featPts.size(), image1.featPts.size()));
    for (std::map<point2D_t, Eigen::Vector2d>::const_iterator iter = image0.featPts.begin(); iter != image0.featPts.end(); iter++)
    {
        if (image1.featPts.count(iter->first) != 0  )
        {
            matchesPointId.emplace_back(iter->first);
        }
    }
    std::unordered_map<point2D_t, Eigen::Vector3d>objPts;
    for (const auto& ptId : matchesPointId)
    {
        const Eigen::Vector2d point2D0 = camera0.CamFromImg(image0.featPts.at(ptId));
        const Eigen::Vector2d point2D1 = camera1.CamFromImg(image1.featPts.at(ptId));
        Eigen::Vector3d xyz;
        bool triangulatePointRet = TriangulatePoint(cam_from_world0, cam_from_world1, point2D0, point2D1, &xyz);
        if (triangulatePointRet)
        {  
            std::pair<bool, Eigen::Vector2d>imgPt0 = image0.ProjectPoint(xyz);
            std::pair<bool, Eigen::Vector2d>imgPt1 = image1.ProjectPoint(xyz);
            LOG_OUT << imgPt0.first << " - " << imgPt0.second.transpose() << " , " << image0.featPts.at(ptId).transpose();
            LOG_OUT << imgPt1.first << " - " << imgPt1.second.transpose() << " , " << image1.featPts.at(ptId).transpose();
            if (imgPt0.first == false || imgPt1.first == false)
            {
                LOG_ERR_OUT << "TriangulatePoint error.";
                return -1;
            }
            objPts[ptId] = xyz;
        }
    }
    {
        Image& image2 = imageList[seed2];
        Camera& camera2 = cameraList[image2.CameraId()];
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
        if (objPtsPnp.size() < 6)
        {
            LOG_ERR_OUT << "not enough match.";
            for (const auto& d : objPts)
            {

                LOG_OUT << "label = " << Image::keypointIndexToName[d.first];
            }
            return -1;
        }
        cv::Mat intrMat = utils::intrConvert(camera0.CalibrationMatrix());
        cv::Mat rvec, tvec;
        cv::solvePnP(objPtsPnp, imgPtsPnp, intrMat, cv::Mat(), rvec, tvec);
        Eigen::AngleAxisd eulerAngle(cv::norm(rvec), Eigen::Vector3d(rvec.ptr<double>(0)[0] / cv::norm(rvec), rvec.ptr<double>(1)[0] / cv::norm(rvec), rvec.ptr<double>(2)[0] / cv::norm(rvec)));
        Rigid3d pnpRt(Eigen::Quaterniond(eulerAngle), Eigen::Vector3d(tvec.ptr<double>(0)[0], tvec.ptr<double>(1)[0], rvec.ptr<double>(2)[0]));
        image2.SetCamFromWorld(pnpRt);
    }
    std::set<image_t>pickedImgs;
    pickedImgs.insert(seed0);
    pickedImgs.insert(seed1);
    pickedImgs.insert(seed2);
    const auto&baFun=[&](const bool& refine_focal_length=false, const bool& refine_principal_point = false)->double
    {
        //ba
        BundleAdjustmentOptions ba_options;
        ba_options.refine_focal_length = refine_focal_length;
        ba_options.refine_principal_point = refine_principal_point;
        ba_options.solver_options.max_num_iterations = 10000;
        //ba_options.solver_options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
        //ba_options.solver_options.minimizer_progress_to_stdout = true;
        BundleAdjustmentConfig ba_config;
        for (const auto& d : pickedImgs)
        {
            ba_config.AddImage(d);
        }
        for (const auto& d : objPts) ba_config.AddVariablePoint(d.first);
        std::unique_ptr<BundleAdjuster> bundle_adjuster;
        ba_config.SetConstantCamPose(seed0);  // 1st image
        //ba_config.SetConstantCamIntrinsics(0);
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
            LOG_ERR_OUT << "not convergence! incremental at total";
            return -1;
        }
        //else
        {
            double final_cost = reprojectTotal(pickedImgs, cameraList, imageList, objPts);
            LOG_OUT << "final_cost = " << final_cost;
            if (final_cost > 6)
            {
                LOG_ERR_OUT << "final_cost>6 at total @ ";
                return final_cost;
            }
            for (const auto& d : pickedImgs)
            {
                const Image& imag = imageList[d];
                LOG_OUT << d << "qt" << qs[d] << ", " << ts[d] << "    " << imag.CamFromWorld().rotation << ", " << imag.CamFromWorld().translation.transpose();
            }
        }
        return 0;
    };
    baFun();
    const auto&refigureAfterBa = [&](const std::set<image_t>&pickedImgs)
    {
        std::unordered_map<point2D_t, std::list<image_t>>cnts;
        for (const auto&imgId: pickedImgs)
        {
            for (const auto& [featId, _] : imageList[imgId].featPts) {
                cnts.try_emplace(featId, std::list<image_t>()).first->second.emplace_back(imgId);
            }
        } 
        std::unordered_map<point2D_t, Eigen::Vector3d>newObjPts;
        for (const auto& [featId, cnt] : cnts)
        {
            if (cnt.size()>=3)
            {
                if (objPts.count(featId))
                {
                    newObjPts[featId] = objPts[featId];
                }
                else
                {
                    //std::vector<Eigen::Matrix3x4d> cams_from_world;
                    //std::vector<Eigen::Vector2d> points;
                    //cams_from_world.reserve(cnt.size());
                    //points.reserve(cnt.size());
                    //for (const auto&d: cnt)
                    //{
                    //    cams_from_world.emplace_back(imageList[d].CamFromWorld().ToMatrix());
                    //    points.emplace_back(imageList[d].featPts.at(featId));
                    //}
                    //Eigen::Vector3d xyz;
                    //bool triangulatePointRet = TriangulateMultiViewPoint(cams_from_world, points, &xyz);
                    //if (triangulatePointRet)
                    //{
                    //    for (const auto& d : cnt)
                    //    {
                    //        std::pair<bool, Eigen::Vector2d>imgPt_ = imageList[d].ProjectPoint(xyz);
                    //        LOG_OUT << imgPt_.first << " - " << imgPt_.second.transpose() << " , " << imageList[d].featPts.at(featId).transpose();
                    //        if (imgPt_.first == false  )
                    //        {
                    //            LOG_ERR_OUT << "TriangulatePoint error.";
                    //            return -1;
                    //        }
                    //    } 
                    //    newObjPts[featId] = xyz;
                    //} 
                }
            }
            else if (cnt.size() ==2)
            {
                const auto&img0Id = *cnt.begin();
                const auto&img1Id = cnt.back();
                const Image& image0 = imageList[img0Id];
                const Camera& camera0 = cameraList[image0.CameraId()];
                const Image& image1 = imageList[img1Id];
                const Camera& camera1 = cameraList[image1.CameraId()];
                const Eigen::Vector2d point2D0 = camera0.CamFromImg(image0.featPts.at(featId));
                const Eigen::Vector2d point2D1 = camera1.CamFromImg(image1.featPts.at(featId));
                Eigen::Vector3d xyz;
                bool triangulatePointRet = TriangulatePoint(image0.CamFromWorld().ToMatrix(), image1.CamFromWorld().ToMatrix(), point2D0, point2D1, &xyz);
                if (triangulatePointRet)
                {
                    std::pair<bool, Eigen::Vector2d>imgPt0 = image0.ProjectPoint(xyz);
                    std::pair<bool, Eigen::Vector2d>imgPt1 = image1.ProjectPoint(xyz);
                    LOG_OUT << imgPt0.first << " - " << imgPt0.second.transpose() << " , " << image0.featPts.at(featId).transpose();
                    LOG_OUT << imgPt1.first << " - " << imgPt1.second.transpose() << " , " << image1.featPts.at(featId).transpose();
                    if (imgPt0.first == false || imgPt1.first == false)
                    {
                        LOG_ERR_OUT << "TriangulatePoint error.";
                        return -1;
                    }
                    newObjPts[featId] = xyz;
                }
            }
        }
        objPts = newObjPts;
        for (const auto& [ptId, pt3d] : objPts)
        {
            LOG_OUT << ptId << "  " << pt3d.transpose();;
        }
        return 0;
    };
    int refigureRet = refigureAfterBa(pickedImgs);
    if (refigureRet!=0)
    {
        return -1;
    }
    const auto& resortImgDistOrder = [&](const std::set<image_t>&hasRegisterd)
    {
        std::vector<std::uint32_t>dists(imageList.size(), imageList.size());
        for (const auto&img:imageList)
        {
            const auto& thisImgId = img.ImageId();
            if (seedImgs.count(thisImgId)!=0 || hasRegisterd.count(thisImgId)!=0)
            {
                continue;
            }            
            else
            { 
                for (const auto&d: seedImgs)
                {
                    int dist = (d > thisImgId) ? (d - thisImgId) : (thisImgId - d);
                    if (dists[thisImgId]> dist)
                    {
                        dists[thisImgId] = dist;
                    }
                }
            }
        }
        return dists;
    };

    const auto& resortImgDistOrder2 = [&](const std::set<image_t>& hasRegisterd)
    {
        std::unordered_map<point2D_t, std::list<image_t>>cnts;
        for (const auto& imgId : hasRegisterd)
        {
            for (const auto& [featId, _] : imageList[imgId].featPts) {
                cnts.try_emplace(featId, std::list<image_t>()).first->second.emplace_back(imgId);
            }
        }
        std::vector<std::uint32_t>scores(imageList.size(), 0);
        for (const auto& img : imageList)
        {
            const auto& thisImgId = img.ImageId();
            if (seedImgs.count(thisImgId) != 0 || hasRegisterd.count(thisImgId) != 0)
            {
                continue;
            }
            else
            {
                for (const auto& [featId, _] : img.featPts)
                {
                    const std::uint32_t& score = cnts[featId].size();
                    if (score >=2)
                    {
                        scores[thisImgId] += score*10;
                    }
                } 
                int neighberScore = imageList.size();
                for (const auto& d : hasRegisterd)
                {
                    int dist = (d > thisImgId) ? (d - thisImgId) : (thisImgId - d);
                    if (neighberScore > dist)
                    {
                        neighberScore = dist;
                    }
                }
                scores[thisImgId] += neighberScore;
            }
        }
        return scores;
    };
    while (true)
    {
        //std::vector<std::uint32_t> imgDistOrder = resortImgDistOrder(pickedImgs);
        std::vector<std::uint32_t> imgDistOrder = resortImgDistOrder2(pickedImgs);
        for (const auto&d: discardIds)
        {
            imgDistOrder[d] = 0;
        }
        int addImgId= -1;
        
        auto minIter = std::max_element(imgDistOrder.begin(), imgDistOrder.end());
        {
            image_t nextImgId = std::distance(imgDistOrder.begin(), minIter);;
            Image& image2 = imageList[nextImgId];
            Camera& camera2 = cameraList[image2.CameraId()];
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
            if (objPtsPnp.size() < 6)
            {
                LOG_ERR_OUT << "not enough match.@"<< nextImgId;
                for (const auto& d : objPts)
                {
                    LOG_OUT << "label = " << Image::keypointIndexToName[d.first];
                }
                return -1;
            }
            cv::Mat intrMat = utils::intrConvert(camera0.CalibrationMatrix());
            cv::Mat rvec, tvec;
            bool useNeighbor = false;
            //if (pickedImgs.count(nextImgId + 1) > 0)
            //{
            //    useNeighbor = true;
            //    tvec = cv::Mat(3, 1, CV_64FC1);
            //    rvec = cv::Mat(3, 1, CV_64FC1);
            //    const Rigid3d& neighborRt = imageList[nextImgId + 1].CamFromWorld();
            //    Eigen::Quaterniond rotation = neighborRt.rotation;
            //    Eigen::Vector3d translation = neighborRt.translation;
            //    Eigen::AngleAxisd aa(rotation);
            //    Eigen::Vector3d rotatervec = aa.axis() * aa.angle();
            //    rvec.ptr<double>(0)[0] = rotatervec[0];
            //    rvec.ptr<double>(1)[0] = rotatervec[1];
            //    rvec.ptr<double>(2)[0] = rotatervec[2];
            //    tvec.ptr<double>(0)[0] = translation[0];
            //    tvec.ptr<double>(1)[0] = translation[1];
            //    tvec.ptr<double>(2)[0] = translation[2];
            //}
            //else if (pickedImgs.count(nextImgId - 1) > 0)
            //{
            //    useNeighbor = true;
            //    tvec = cv::Mat(3, 1, CV_64FC1);
            //    rvec = cv::Mat(3, 1, CV_64FC1);
            //    const Rigid3d& neighborRt = imageList[nextImgId - 1].CamFromWorld();
            //    Eigen::Quaterniond rotation = neighborRt.rotation;
            //    Eigen::Vector3d translation = neighborRt.translation;
            //    Eigen::AngleAxisd aa(rotation);
            //    Eigen::Vector3d rotatervec = aa.axis() * aa.angle();
            //    rvec.ptr<double>(0)[0] = rotatervec[0];
            //    rvec.ptr<double>(1)[0] = rotatervec[1];
            //    rvec.ptr<double>(2)[0] = rotatervec[2];
            //    tvec.ptr<double>(0)[0] = translation[0];
            //    tvec.ptr<double>(1)[0] = translation[1];
            //    tvec.ptr<double>(2)[0] = translation[2];
            //}
            cv::solvePnP(objPtsPnp, imgPtsPnp, intrMat, cv::Mat(), rvec, tvec, useNeighbor);
            Eigen::AngleAxisd eulerAngle(cv::norm(rvec), Eigen::Vector3d(rvec.ptr<double>(0)[0] / cv::norm(rvec), rvec.ptr<double>(1)[0] / cv::norm(rvec), rvec.ptr<double>(2)[0] / cv::norm(rvec)));
            Rigid3d pnpRt(Eigen::Quaterniond(eulerAngle), Eigen::Vector3d(tvec.ptr<double>(0)[0], tvec.ptr<double>(1)[0], rvec.ptr<double>(2)[0]));
            image2.SetCamFromWorld(pnpRt);
            addImgId = nextImgId;
            pickedImgs.insert(nextImgId);
        }
        
        if (addImgId<0)
        {
            LOG_ERR_OUT << "size_t tryI = 0; tryI < seedImgs.size() * 2; tryI++";
            return -1;
        }
        int baRet = baFun();
        if (baRet<0)
        { 
            LOG_ERR_OUT << "not convergence. " << addImgId;
            return -1;
        }
        else if (baRet > 6)
        {
            LOG_ERR_OUT << "baRet > 6. @" << Image::picIndexTopicName[addImgId];
            return -1;
        }
        if (imageList.size()== pickedImgs.size()+ discardIds.size())
        {
            LOG_OUT << "ok";
            break;
        }
        int refigureRet = refigureAfterBa(pickedImgs);
        if (refigureRet != 0)
        {
            return -1;
        }
    }
    if (imageList.size() == pickedImgs.size() + discardIds.size() && optimCameraIntr)
    {
        baFun(true,true);
    }
    return 0;
}
int test_incremental()
{
    return 0;
}