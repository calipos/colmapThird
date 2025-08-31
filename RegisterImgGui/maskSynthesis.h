#ifndef _MASK_SYNTHESIS_H_
#define _MASK_SYNTHESIS_H_
#include <numeric>
#include <cstdint>
#include <iostream>
#include <vector>
#include "Eigen/Core"
#include "log.h"
namespace sdf
{
	struct VolumeDat
	{
		VolumeDat(const std::uint64_t& indexMax, const double& startX, const double& startY, const double& startZ, const double& endX, const double& endY, const double& endZ);
		std::uint64_t maxIndex;
		double unit;
		double resolution;
		double x_start;
		double y_start;
		double z_start;
		double x_end;
		double y_end;
		double z_end;
		std::uint64_t x_size;
		std::uint64_t y_size;
		std::uint64_t z_size; 
		std::vector<bool>gridFlag;
		Eigen::Matrix4Xf gridCenter;
	};
}
#endif // _MASK_SYNTHESIS_H_
