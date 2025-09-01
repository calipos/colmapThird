#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include "Eigen/Core"
#include "maskSynthesis.h"
#include "opencvTools.h"
#include "labelme.h"
#include "marchCube.h"
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
			x_size = static_cast<int>(abs(endX - startX) / unit)+1;
			y_size = static_cast<int>(abs(endY - startY) / unit)+1;
			z_size = static_cast<int>(abs(endZ - startZ) / unit)+1;
		} while (x_size > 512 || y_size > 512 || z_size > 512);

		std::uint64_t totalCnt = x_size * y_size * z_size;
		int yx_size = x_size * y_size;
		gridCenterHitValue = std::vector<float>(x_size * y_size * z_size, 1.0f);
		grid = Eigen::Matrix4Xf(4, gridCenterHitValue.size());
		for (int i = 0; i < gridCenterHitValue.size(); i++)
		{
			int i_z = i / yx_size;
			int i_y = i % yx_size;
			int i_x = i_y % x_size;
			i_y = i_y / x_size;
			grid(0, i) = startX + i_x * unit;
			grid(1, i) = startY + i_y * unit;
			grid(2, i) = startZ + i_z * unit;
			grid(3, i) = 1;
		}
		LOG_OUT << grid.row(0).maxCoeff() << " " << grid.row(0).minCoeff();;
		LOG_OUT << grid.row(1).maxCoeff() << " " << grid.row(1).minCoeff();;
		LOG_OUT << grid.row(2).maxCoeff() << " " << grid.row(2).minCoeff();;
	}
	bool VolumeDat::getCloud(const std::filesystem::path& path, const int& thre)
	{
		int flagCnt = this->gridCenterHitValue.size();
		int gridCnt = grid.cols();
		if (flagCnt == 0 || flagCnt != gridCnt)
		{
			LOG_ERR_OUT << "grid not initialized!";
			return false;
		}
		int ptsCnt = 0;
		{
			std::fstream fout(path, std::ios::out);
			fout << "ply" << std::endl;
			fout << "format binary_little_endian 1.0" << std::endl;
			fout << "element vertex                                 " << std::endl;//可以容纳32位数
			fout << "property float x" << std::endl;
			fout << "property float y" << std::endl;
			fout << "property float z" << std::endl;
			fout << "end_header" << std::endl;
		}
		{
			std::fstream fout(path, std::ios::app | std::ios::binary);
			for (int i = 0; i < gridCnt; i++)
			{
				if (gridCenterHitValue[i] > thre)
				{
					ptsCnt++;
					fout.write((const char*)&(this->grid(0, i)), sizeof(float));
					fout.write((const char*)&(this->grid(1, i)), sizeof(float));
					fout.write((const char*)&(this->grid(2, i)), sizeof(float));
				}
			}
			fout.close();
		}
		{
			std::ofstream fout(path, std::ios::ate | std::ios::in);
			fout.seekp(53, std::ios::beg); //基地址为文件头，偏移量为0，于是定位在文件头
			std::string pointsNumber = std::to_string(ptsCnt);
			if (pointsNumber.length() <= 32)
			{
				for (int i = 0; i < 32 - pointsNumber.length(); i++)
				{
					pointsNumber += " ";
				}
			}
			fout << pointsNumber;
			fout.close();
		}


		return false;
	}


	Eigen::MatrixX3d readPoint3d(const std::filesystem::path& path)
	{
		if (!std::filesystem::exists(path))
		{
			LOG_ERR_OUT << "not found : " << path;
			return Eigen::MatrixX3d();
		}
		std::fstream fin(path, std::ios::in);
		std::string aline;
		std::list<Eigen::Vector3d>ptsList;
		while (std::getline(fin, aline))
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
		for (const auto& d : ptsList)
		{
			pts(idx, 0) = d[0];
			pts(idx, 1) = d[1];
			pts(idx, 2) = d[2];
			idx++;
		}
		return pts;
	}
}
int test_sdf()
{
	Eigen::MatrixX3d pts = sdf::readPoint3d("../data/a/result/pts.txt");
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
			if (ext.compare(".json") == 0)
			{
				std::string stem = thisFilename.filename().stem().string();
				auto maskPath = thisFilename.parent_path() / ("mask_" + stem + ".dat");
				if (std::filesystem::exists(maskPath))
				{
					cv::Mat mask = tools::loadMask(maskPath.string());
					if (!mask.empty())
					{
						Eigen::Matrix4d cameraMatrix, Rt;
						bool readRet = labelme::readCmaeraFromRegisterJson(thisFilename, cameraMatrix, Rt);
						if (!readRet)
						{
							LOG_WARN_OUT << "read fail : " << thisFilename;
							continue;
						}
						Eigen::Matrix4d  p = cameraMatrix * Rt;
						Eigen::Matrix4Xf gridInCamera = p.cast<float>() * a.grid;
						int gridCnt = gridInCamera.cols();
						for (size_t i = 0; i < gridCnt; i++)
						{
							int u = static_cast<int>(gridInCamera(0, i) / gridInCamera(2, i));
							int v = static_cast<int>(gridInCamera(1, i) / gridInCamera(2, i));
							if (u >= 0 && v >= 0 && u < mask.cols && v < mask.rows)
							{
								if (mask.ptr<uchar>(v)[u] == 0)
								{
									a.gridCenterHitValue[i] = -1;
								}
								else
								{
									a.gridCenterHitValue[i] *= 1;
								}
							}
						}
					}
				}

			}
		}
	}
	mc::marchcube(a.grid, a.gridCenterHitValue, a.x_size, a.y_size, a.z_size, a.unit, 0);
	return 0;
}