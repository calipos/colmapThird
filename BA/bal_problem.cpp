#include "bal_problem.h"
#include <map>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "ceres/rotation.h"
#include "glog/logging.h"
#include "BALog.h"
namespace ba {
        using VectorRef = Eigen::Map<Eigen::VectorXd>;
        using ConstVectorRef = Eigen::Map<const Eigen::VectorXd>;

        void PerturbPoint3(std::function<double()> dist, double* point) {
            for (int i = 0; i < 3; ++i) {
                point[i] += dist();
            }
        }

        double Median(std::vector<double>* data) {
            auto mid_point = data->begin() + data->size() / 2;
            std::nth_element(data->begin(), mid_point, data->end());
            return *mid_point;
        }


    BALProblem::BALProblem(
        const ba::OptiModel& optiModel,
        const std::map<int, std::array<double, 3>>& colmapObjPts,
        const std::vector<col::ImgPt>& colmapImgPts,
        const std::map<int, col::Camera>& cameras, bool use_quaternions) 
    {
        optiModel_ = optiModel;
        num_cameras_ = cameras.size();
        num_points_ = colmapObjPts.size();
        img_Cnt_ = colmapImgPts.size();
        num_observations_ = colmapImgPts.size() * num_points_;
        std::map<int, int>cameraIdToIdx;
        int eachCameraParamCnt = getEachCameraParamCnt(optiModel_);
        num_parameters_ = eachCameraParamCnt * num_cameras_ + 6 * colmapImgPts.size() + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        int idx = 0;
        for (const auto&d: cameras)
        {
            switch (optiModel_)
            {
            case ba::OptiModel::fk1k2:
                parameters_[idx * eachCameraParamCnt] = d.second.fx;
                parameters_[idx * eachCameraParamCnt + 1] = 0;
                parameters_[idx * eachCameraParamCnt + 2] = 0;
                if (d.second.distoCoeff.size()==1)
                {
                    parameters_[idx * eachCameraParamCnt + 1] = d.second.distoCoeff[0];
                }
                break;
            case ba::OptiModel::fk1:
                parameters_[idx * eachCameraParamCnt] = d.second.fx;
                parameters_[idx * eachCameraParamCnt + 1] = 0;
                if (d.second.distoCoeff.size() == 1)
                {
                    parameters_[idx * eachCameraParamCnt + 1] = d.second.distoCoeff[0];
                }
                break;
            case ba::OptiModel::fcxcyk1:
                parameters_[idx * eachCameraParamCnt] = d.second.fx;
                parameters_[idx * eachCameraParamCnt + 1] = d.second.cx;
                parameters_[idx * eachCameraParamCnt + 2] = d.second.cy;
                parameters_[idx * eachCameraParamCnt + 3] = 0;
                if (d.second.distoCoeff.size() == 1)
                {
                    parameters_[idx * eachCameraParamCnt + 3] = d.second.distoCoeff[0];
                }
                break;
            default:
                assert(false);
                break;
            }
            cameraIdToIdx[d.first] = idx++;
        }

        LOG_OUT << "Header: " << num_cameras_ << " " << num_points_ << " "
            << num_observations_;

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];


        for (int imgIdx = 0; imgIdx < colmapImgPts.size(); imgIdx++)
        {
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 0] = colmapImgPts[imgIdx].r[0];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 1] = colmapImgPts[imgIdx].r[1];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 2] = colmapImgPts[imgIdx].r[2];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 3] = colmapImgPts[imgIdx].t[0];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 4] = colmapImgPts[imgIdx].t[1];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * imgIdx + 5] = colmapImgPts[imgIdx].t[2];
            for (int imgPtIdx = 0; imgPtIdx < num_points_; imgPtIdx++)
            {
                int i = num_points_ * imgIdx + imgPtIdx;
                camera_index_[i] = cameraIdToIdx[colmapImgPts[imgIdx].cameraId];
                point_index_[i] = imgPtIdx;
                observations_[2 * i + 0] = colmapImgPts[imgIdx].imgPts[imgPtIdx][0];
                observations_[2 * i + 1] = colmapImgPts[imgIdx].imgPts[imgPtIdx][1];
                if (optiModel_ == ba::OptiModel::fk1k2 || optiModel_ == ba::OptiModel::fk1)
                {
                    const col::Camera& theCamera = cameras.at(colmapImgPts[imgIdx].cameraId);
                    observations_[2 * i + 0] -= (theCamera.width * 0.5);
                    observations_[2 * i + 1] -= (theCamera.height * 0.5);;
                }
            }
        }
        for (int imgPtIdx = 0; imgPtIdx < num_points_; imgPtIdx++)
        { 
            CHECK(colmapObjPts.count(imgPtIdx) > 0); 
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * colmapImgPts.size() + 3 * imgPtIdx] = colmapObjPts.at(imgPtIdx)[0];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * colmapImgPts.size() + 3 * imgPtIdx + 1] = colmapObjPts.at(imgPtIdx)[1];
            parameters_[eachCameraParamCnt * num_cameras_ + 6 * colmapImgPts.size() + 3 * imgPtIdx + 2] = colmapObjPts.at(imgPtIdx)[2];
        }
 

        use_quaternions_ = use_quaternions;
        if (use_quaternions) {
            // Switch the angle-axis rotations to quaternions.
            int num_parameters_ = eachCameraParamCnt * num_cameras_ + 7 * colmapImgPts.size() + 3 * num_points_;;
            auto* quaternion_parameters = new double[num_parameters_];
            memcpy(quaternion_parameters, parameters_, eachCameraParamCnt * num_cameras_*sizeof(double));
            
            for (int i = 0; i < colmapImgPts.size(); ++i) {
                double* original_cursor = parameters_ + eachCameraParamCnt * num_cameras_ + 6 * i;
                double* quaternion_cursor = quaternion_parameters + eachCameraParamCnt * num_cameras_ + 7 * i;
                ceres::AngleAxisToQuaternion(original_cursor, quaternion_cursor);
                quaternion_cursor += 4;
                original_cursor += 3;
                quaternion_cursor[0] = original_cursor[0];
                quaternion_cursor[1] = original_cursor[1];
                quaternion_cursor[2] = original_cursor[2];
            }

            memcpy(quaternion_parameters + eachCameraParamCnt * num_cameras_ + 7 * colmapImgPts.size(),
                parameters_ + eachCameraParamCnt * num_cameras_ + 6 * colmapImgPts.size(), 
                3 * num_points_ * sizeof(double));
            delete[] parameters_;
            parameters_ = quaternion_parameters;
        }
    }



    void BALProblem::Perturb(const double rotation_sigma,
        const double translation_sigma,
        const double point_sigma) {
        CHECK_GE(point_sigma, 0.0);
        CHECK_GE(rotation_sigma, 0.0);
        CHECK_GE(translation_sigma, 0.0);
        std::mt19937 prng;
        std::normal_distribution<double> point_noise_distribution(0.0, point_sigma);
        double* points = mutable_points();
        if (point_sigma > 0) {
            for (int i = 0; i < num_points_; ++i) {
                PerturbPoint3(std::bind(point_noise_distribution, std::ref(prng)),
                    points + 3 * i);
            }
        }

        std::normal_distribution<double> rotation_noise_distribution(0.0,
            point_sigma);
        std::normal_distribution<double> translation_noise_distribution(
            0.0, translation_sigma);
        for (int i = 0; i < num_cameras_; ++i) {
            double* camera = mutable_cameras() + getEachCameraParamCnt(optiModel_) * i;

            double angle_axis[3];
            double center[3];
            //CameraToAngleAxisAndCenter(camera, angle_axis, center);
            //if (rotation_sigma > 0.0) {
            //    PerturbPoint3(std::bind(rotation_noise_distribution, std::ref(prng)),
            //        angle_axis);
            //}
            //AngleAxisAndCenterToCamera(angle_axis, center, camera);
            //if (translation_sigma > 0.0) {
            //    PerturbPoint3(std::bind(translation_noise_distribution, std::ref(prng)),
            //        camera + camera_block_size() - 6);
            //}
        }
    }

    BALProblem::~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
}
