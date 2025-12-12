#pragma once
#include <filesystem>
#include <map>
#include <random>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
namespace bfm
{
    class Bfm2019
    {
    public:
        Bfm2019();
        Bfm2019(const std::filesystem::path& bfmFacePath);
        int shapeDim{ 0 };
        int expressionDim{ 0 };
        int colorDim{ 0 };
        Eigen::MatrixX3f points;
        Eigen::MatrixX3i F;
        std::map<std::string, cv::Point3f>landmarks;
        std::map<std::string, int>landmarkIdx;
        Eigen::VectorXf shape_mean, expression_mean, color_mean;
        Eigen::VectorXf shape_pcaStandardDeviation, expression_pcaStandardDeviation, color_pcaStandardDeviation;
        Eigen::MatrixXf shape_pcaBasis;
        Eigen::MatrixXf expression_pcaBasis;
        Eigen::MatrixXf color_pcaBasis;
        void generateRandomFace(Eigen::MatrixX3f& V, Eigen::MatrixX3f& C)const;
        void capture(const Eigen::MatrixX3f& V, const Eigen::MatrixX3f& C)const;
        void saveObj(const std::filesystem::path& path, const Eigen::MatrixX3f& V, const Eigen::MatrixX3f& C)const;
    };

}