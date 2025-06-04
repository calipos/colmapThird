#include "cameras.h"
#include "utils.h"
#include "BALog.h"
namespace col
{
	Camera::Camera(const cameraModel& camera_model, const double& fx_, const double& fy_, const double& cx_, const double& cy_, const int& height_, const int& width_, const std::vector<double>* distoCoeff_)
	{
		camera_model_ = camera_model;
		fx = fx_;
		fy = fy_;
		cx = cx_;
		cy = cy_;
		height = height_;
		width = width_;
		if (distoCoeff_)
		{
			distoCoeff.clear();
			distoCoeff.insert(distoCoeff.end(), distoCoeff_->begin(), distoCoeff_->end());
		}
	}
	std::string Camera::cameraModelStr(const Camera::cameraModel& model)
	{
		if (model == Camera::cameraModel::SIMPLE_RADIAL)
		{
			return "SIMPLE_RADIAL";
		}
		else if (model == Camera::cameraModel::SIMPLE_PINHOLE)
		{
			return "SIMPLE_PINHOLE";
		}
	}
	Camera::cameraModel Camera::str2CameraModel(const std::string& str)
	{
		if (str.compare("SIMPLE_RADIAL")==0)
		{
			return Camera::cameraModel::SIMPLE_RADIAL;
		}
		else if (str.compare("SIMPLE_PINHOLE") == 0)
		{
			return Camera::cameraModel::SIMPLE_PINHOLE;
		}
		else
		{
			LOG_ERR_OUT << "not support yet.";
			return Camera::cameraModel::err;
		}
	} 

}

std::map<int, col::Camera>readCamerasFromTXT(const std::filesystem::path& cameraTXT)
{
	std::map<int, col::Camera>ret;
	std::fstream fin(cameraTXT, std::ios::in);
	std::string aline;
	while (std::getline(fin, aline))
	{
		if (aline[0] == '#')
		{
			continue;
		}
		int cameraId = -1;
		col::Camera::cameraModel thisCameraModel = col::Camera::cameraModel::err;
		int height = -1;
		int width = -1;
		std::vector<double>param;
		std::vector<double>disto;
		param.reserve(8);
		disto.reserve(8);
		std::vector<std::string>segs = utils::splitString(aline, " ");
		if (segs.size() >= 4)
		{
			if (!utils::stringToNumber<int>(segs[0], cameraId))
			{
				continue;
			}
			thisCameraModel = col::Camera::str2CameraModel(segs[1]);
			if (thisCameraModel == col::Camera::cameraModel::err)
			{
				continue;
			}
			if (!utils::stringToNumber<int>(segs[2], width))
			{
				continue;
			}
			if (!utils::stringToNumber<int>(segs[3], height))
			{
				continue;
			}
			if (thisCameraModel == col::Camera::cameraModel::SIMPLE_RADIAL)
			{
				double f = 0, cx = 0, cy = 0, k = 0;
				if (!utils::stringToNumber<double>(segs[4], f))
				{
					continue;
				}
				if (!utils::stringToNumber<double>(segs[5], cx))
				{
					continue;
				}
				if (!utils::stringToNumber<double>(segs[6], cy))
				{
					continue;
				}
				if (!utils::stringToNumber<double>(segs[7], k))
				{
					continue;
				}
				param.emplace_back(f);
				param.emplace_back(cx);
				param.emplace_back(cy);
				disto.emplace_back(k);
			}
			ret[cameraId] = col::Camera(thisCameraModel, param[0], param[0], param[1], param[2], height, width, &disto);
		}
	}
	return ret;
}