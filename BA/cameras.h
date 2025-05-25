#ifndef _CAMERA_H_
#define _CAMERA_H_
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>
#include <map>

namespace col
{ 
	struct Camera
	{
		int height;
		int width;
		double fx, fy, cx, cy;
		std::vector<double>distoCoeff;
		enum cameraModel
		{
			err = -1,
			SIMPLE_RADIAL = 1, // f cx cy k
			SIMPLE_PINHOLE = 2,//f cx cy 
		};
		cameraModel camera_model_;
		static std::string cameraModelStr(const cameraModel& model);
		static cameraModel str2CameraModel(const std::string& str);
		Camera() {};
		Camera(const cameraModel&camera_model,const double& fx_ = 0, const double& fy_ = 0, const double& cx_ = 0, const double& cy_ = 0, const int& height_ = 0, const int& width_ = 0, const std::vector<double>* distoCoeff_ = nullptr);
	};

}
std::map<int, col::Camera>readCamerasFromTXT(const std::filesystem::path& cameraTXT);
#endif // !_CAMERA_H_
