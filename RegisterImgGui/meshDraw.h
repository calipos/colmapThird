#pragma once
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
namespace meshdraw
{
	int meshOrthoDraw(const Eigen::MatrixX3f& V, const Eigen::MatrixX3i& F, const Eigen::MatrixX3f& C, const int& anchorIdx, const int& tarImgSize, const float& additionalScale = 1.f);
}