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

int test_incremental()
{


    //test_bitmap();
    std::vector<image_t>incrementalImages = { 0,1,2,3,4 };
    std::filesystem::path dataPath = "../data2";
    std::map<Camera, std::vector<Image>> dataset = loadImageData(dataPath, ImageIntrType::SHARED_ALL);
    std::vector<Camera>cameraList;
    std::vector<Image> imageList;
    convertDataset(dataset, cameraList, imageList);
    std::unordered_map<point3D_t, Eigen::Vector3d>objPts;
    std::unordered_map < image_t, struct Rigid3d>poses;
    for (int i = 1; i < incrementalImages.size(); i++)
    {
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
                return -1;
            }

            if (poses.count(picked1) == 0)
            {
                poses[picked1] = Rigid3d();
                image1.SetCamFromWorld(poses[picked1]);
            }
            poses[picked2] = two_view_geometry.cam2_from_cam1 * poses[picked1];
            image2.SetCamFromWorld(poses[picked2]);
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
                    const Eigen::Vector2d point2D1 = camera1.CamFromImg(image1.featPts[ptId]);
                    const Eigen::Vector2d point2D2 = camera2.CamFromImg(image2.featPts[ptId]);
                    Eigen::Vector3d xyz;
                    bool triangulatePointRet = TriangulatePoint(cam_from_world1, cam_from_world2, point2D1, point2D2, &xyz);
                    if (triangulatePointRet)
                    {
                        objPts[ptId] = xyz;
                    }
                }
            }
        }
        {
            //ba
            BundleAdjustmentOptions ba_options;
            BundleAdjustmentConfig ba_config;
            for (int j = 0; j <= i; j++) ba_config.AddImage(incrementalImages[j]);
            for (const auto& d : objPts) ba_config.AddVariablePoint(d.first);
            std::unique_ptr<BundleAdjuster> bundle_adjuster;
            ba_config.SetConstantCamPose(incrementalImages[0]);  // 1st image
            bundle_adjuster = CreateDefaultBundleAdjuster(std::move(ba_options), std::move(ba_config), cameraList, imageList, objPts);

            //for (int j = 0; j <= i; j++) LOG_OUT << imageList[incrementalImages[j]].CamFromWorld();
            auto solverRet = bundle_adjuster->Solve();
            for (int j = 0; j < cameraList.size(); j++) LOG_OUT << cameraList[j];
            if (solverRet.termination_type != ceres::CONVERGENCE)
            {
                LOG_ERR_OUT << "not convergence!";
            }
        }
    }

    writeResult(dataPath/"result", cameraList, imageList, objPts, poses);

    return 0;
}
