#ifndef _OPENCV_TOOLS_H_
#define _OPENCV_TOOLS_H_
#include <string>
#include "opencv2/opencv.hpp"
#include "Eigen/Core"
namespace tools
{
	bool saveMask(const std::string& path, const cv::Mat& mask);
	cv::Mat loadMask(const std::string& path);
	bool saveColMajorPts3d(const std::string& path, const Eigen::MatrixXf& pts);
}

#endif // !_OPENCV_TOOLS_H_
