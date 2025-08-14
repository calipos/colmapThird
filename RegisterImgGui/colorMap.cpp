#include "imgui_tools.h"
#include <filesystem>
#include "opencv2/opencv.hpp"


struct ColorMap
{
public:
	static cv::Vec3b getColor()
	{
		ColorMap*ins =  getInstance();
		if (ins != nullptr)
		{
			double thisAngle = 2 * CV_PI * idx * phi;
			int x = radius * cos(thisAngle) + centerX;
			int y = radius * sin(thisAngle) + centerY;
			
			idx++;

			return ColorMap::colorMat.at<cv::Vec3b>(y, x);

		}
		else
		{
			return cv::Vec3b(255, 255, 255);
		}
	}
private:
	static ColorMap* getInstance()
	{
		if (ColorMap::instance != nullptr)
		{
			return ColorMap::instance;
		}
		else
		{
			if (std::filesystem::exists("../models/cubediagonal.png"))
			{
				ColorMap::instance = new ColorMap();
				ColorMap::colorMat = cv::imread("../models/cubediagonal.png");

				ColorMap::centerX = instance->colorMat.cols * 0.5;
				ColorMap::centerY = instance->colorMat.rows * 0.5;
				ColorMap::radius = (centerX > centerY ? centerY : centerX) - 2;
			}
			else
			{
				ColorMap::instance = nullptr;
			}
			return ColorMap::instance;
		}
	}
	ColorMap(){}
	static cv::Mat colorMat;
	static ColorMap* instance;
	static double phi;
	static int idx;
	static int centerX;
	static int centerY;
	static int radius;
};
ColorMap* ColorMap::instance = nullptr;
int ColorMap::idx = 0;
int ColorMap::centerX = 0;
int ColorMap::centerY = 0;
int ColorMap::radius = 0;
double ColorMap::phi = 0.6180339887498948482045868343656;
cv::Mat ColorMap::colorMat;
ImU32 getImguiColor()
{
	cv::Vec3b rgb_opencv = getColor();

	ImU32 alpha = 255<<24;
	ImU32 b = rgb_opencv[0]<<16;;
	ImU32 g = rgb_opencv[1]<<8;;
	ImU32 r = rgb_opencv[2];


	return alpha + b + g + r;
}
cv::Vec3b getColor()
{
	return ColorMap::getColor();
}