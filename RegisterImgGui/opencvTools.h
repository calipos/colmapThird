#ifndef _OPENCV_TOOLS_H_
#define _OPENCV_TOOLS_H_
#include <string>
#include "opencv2/opencv.hpp"
namespace tools
{
	bool saveMask(const std::string& path, const cv::Mat& mask);
	cv::Mat loadMask(const std::string& path);
}

#endif // !_OPENCV_TOOLS_H_
