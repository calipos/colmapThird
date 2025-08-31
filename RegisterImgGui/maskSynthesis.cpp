#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include "Eigen/Core"
#include "maskSynthesis.h"
#include "opencvTools.h"
namespace sdf
{
	VolumeDat::VolumeDat(const std::uint64_t& indexMax, const double& startX, const double& startY, const double& startZ, const double& endX, const double& endY, const double& endZ)
	{
		maxIndex = indexMax;
		double maxIndexDouble = static_cast<double>(maxIndex);
		double volume = abs(endX - startX) * abs(endY - startY) * abs(endZ - startZ);
		do
		{
			maxIndexDouble *= 0.99;
			unit = std::pow(volume / maxIndexDouble, 0.3333);
			resolution = abs(endX - startX) / unit;
			x_size = static_cast<int>(abs(endX - startX) / unit);
			y_size = static_cast<int>(abs(endY - startY) / unit);
			z_size = static_cast<int>(abs(endZ - startZ) / unit);
		} while (x_size > 512 || y_size > 512 || z_size > 512);

		std::uint64_t totalCnt = x_size * y_size * z_size;
		int yx_size = x_size * y_size; 
		gridFlag = std::vector<bool>(x_size * y_size * z_size, 0);
		gridCenter = Eigen::Matrix4Xf(4, gridFlag.size());
		for (std::uint64_t i = 0; i < gridFlag.size(); i++)
		{
			std::uint64_t i_z = i / yx_size;
			std::uint64_t i_y = i % yx_size;
			std::uint64_t i_x = i_y % x_size;
			i_y = i_y/x_size;
			gridCenter(0, i) = startX + i_z * unit;
			gridCenter(1, i) = startY + i_y * unit;
			gridCenter(2, i) = startZ + i_x * unit;
			gridCenter(3, i) = 1;
		}
	}
}

Eigen::MatrixX3d readPoint3d(const std::filesystem::path& path)
{
	if (!std::filesystem::exists(path))
	{
		LOG_ERR_OUT << "not found : "<<path;
		return Eigen::MatrixX3d();
	}
	std::fstream fin(path,std::ios::in);
	std::string aline;
	std::list<Eigen::Vector3d>ptsList;
	while (std::getline(fin,aline))
	{
		std::stringstream ss;
		ss << aline;
		double x = 0, y = 0, z = 0;
		ss >> x >> y >> z;
		ptsList.emplace_back(x, y, z);
	}
	fin.close();
	Eigen::MatrixX3d pts(ptsList.size(), 3);
	int idx = 0;
	for (const auto&d:ptsList)
	{
		pts(idx, 0) = d[0];
		pts(idx, 1) = d[1];
		pts(idx, 2) = d[2];
		idx++;
	}
	return pts;
}
int test_sdf()
{
	Eigen::MatrixX3d pts =  readPoint3d("../data/a/result/pts.txt");
	double x_strat = pts.col(0).minCoeff();
	double y_strat = pts.col(1).minCoeff();
	double z_strat = pts.col(2).minCoeff();
	double x_end = pts.col(0).maxCoeff();
	double y_end = pts.col(1).maxCoeff();
	double z_end = pts.col(2).maxCoeff();


	sdf::VolumeDat a(static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()), x_strat, y_strat, z_strat, x_end, y_end, z_end);


	for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{ "../data/a/result" })
	{
		const auto& thisFilename = dir_entry.path();
		if (thisFilename.has_extension())
		{
			const auto& ext = thisFilename.extension().string();
			if (ext.compare(".json") == 0 )
			{
					std::string stem = thisFilename.filename().stem().string();
					auto maskPath = thisFilename.parent_path() / ("mask_" + stem + ".dat");
					if (std::filesystem::exists(maskPath))
					{
						cv::Mat mask = tools::loadMask(maskPath.string());
						if (!mask.empty())
						{

						}
					}

			}
		}
	}

	return 0;
}