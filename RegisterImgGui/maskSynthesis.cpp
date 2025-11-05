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
#include "camera.h"
#include "warp.h"
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
		int numX_1 = x_size - 1;
		int numY_1 = y_size - 1;
		int numZ_1 = z_size - 1;
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
			if (i_x == numX_1 || i_y == numY_1 || i_z == numZ_1 || i_x == 0 || i_y == 0 || i_z == 0)
			{
				gridCenterHitValue[i]=-1;//border
			}
		}
		LOG_OUT << grid.row(0).maxCoeff() << " " << grid.row(0).minCoeff();;
		LOG_OUT << grid.row(1).maxCoeff() << " " << grid.row(1).minCoeff();;
		LOG_OUT << grid.row(2).maxCoeff() << " " << grid.row(2).minCoeff();;
	}
	bool saveCloud(const std::filesystem::path& path, const Eigen::MatrixXf& pts)
	{
		{
			std::fstream fout(path, std::ios::out);
			fout << "ply" << std::endl;
			fout << "format binary_little_endian 1.0" << std::endl;
			fout << "element vertex " << pts.cols() << std::endl;//可以容纳32位数
			fout << "property float x" << std::endl;
			fout << "property float y" << std::endl;
			fout << "property float z" << std::endl;	
			fout << "end_header" << std::endl;
			fout.close();
		}
		{
			std::fstream fout(path, std::ios::app | std::ios::binary);
			for (int i = 0; i < pts.cols(); i++)
			{
				fout.write((const char*)&(pts(0, i)), sizeof(float));
				fout.write((const char*)&(pts(1, i)), sizeof(float));
				fout.write((const char*)&(pts(2, i)), sizeof(float));

			}
			fout.close();
		}
		return true;
	}
	bool saveCloud(const std::filesystem::path& path, const Eigen::MatrixXf& pts,const std::vector<std::uint8_t>&colors)
	{
		if (colors.size()!=3*pts.cols())
		{
			LOG_ERR_OUT << "colors.size()!=3*pts.cols()";
			return false;
		}
		{
			std::fstream fout(path, std::ios::out);
			fout << "ply" << std::endl;
			fout << "format binary_little_endian 1.0" << std::endl;
			fout << "element vertex " << pts.cols() << std::endl;//可以容纳32位数
			fout << "property float x" << std::endl;
			fout << "property float y" << std::endl;
			fout << "property float z" << std::endl;
			fout << "property uchar red" << std::endl;
			fout << "property uchar green" << std::endl;
			fout << "property uchar blue" << std::endl;
			fout << "end_header" << std::endl;
			fout.close();
		}
		{
			std::fstream fout(path, std::ios::app | std::ios::binary);
			for (int i = 0; i < pts.cols(); i++)
			{
				int i3 = 3 * i;
				fout.write((const char*)&(pts(0, i)), sizeof(float));
				fout.write((const char*)&(pts(1, i)), sizeof(float));
				fout.write((const char*)&(pts(2, i)), sizeof(float));
				const char* r = (const char*)&colors[i3];
				const char* g = (const char*)&colors[i3 + 1];
				const char* b = (const char*)&colors[i3+2];
				fout.write(r, sizeof(char));
				fout.write(g, sizeof(char));
				fout.write(b, sizeof(char));
			}
			fout.close();
		}
		return true;
	}
	Eigen::Matrix3Xf VolumeDat::getCloud(const int& thre)const
	{
		int flagCnt = this->gridCenterHitValue.size();
		int gridCnt = grid.cols();
		if (flagCnt == 0 || flagCnt != gridCnt)
		{
			LOG_ERR_OUT << "grid not initialized!";
			return Eigen::Matrix3Xf();
		}
		std::list<Eigen::Vector3f>ptsList;
		{
			int numX_1 = x_size - 1;
			int numY_1 = y_size - 1;
			int numZ_1 = z_size - 1;
			const int yx_size = x_size * y_size;
			for (int i = 0; i < gridCnt; i++)
			{
				if (gridCenterHitValue[i] > thre)
				{
					ptsList.emplace_back(this->grid(0, i), this->grid(1, i), this->grid(2, i));
				}
			}
		}
		Eigen::Matrix3Xf ret(3, ptsList.size());
		int idx = 0;
		for (const auto&d: ptsList)
		{
			ret(0, idx) = d[0];
			ret(1, idx) = d[1];
			ret(2, idx) = d[2];
			idx++;
		}
		return ret;
	}
	bool VolumeDat::emptyShellPts(const int& thre)
	{
		int flagCnt = this->gridCenterHitValue.size();
		int gridCnt = grid.cols();
		if (flagCnt == 0 || flagCnt != gridCnt)
		{
			LOG_ERR_OUT << "grid not initialized!";
			return false;
		}
		std::list<Eigen::Vector3f>shell_xyzs;
		std::list<float>hitValue;
		int ptsCnt = 0;
		{
			int numX_1 = x_size - 1;
			int numY_1 = y_size - 1;
			int numZ_1 = z_size - 1;
			const int yx_size = x_size * y_size;
			std::vector<int>idxShift = { -1 - static_cast<int>(x_size) ,-static_cast<int>(x_size) ,1 - static_cast<int>(x_size) , -1  ,0 ,1  , -1 + static_cast<int>(x_size) ,static_cast<int>(x_size) ,1 + static_cast<int>(x_size) };
			std::vector<int>idxShift2(27);
			for (int i = 0; i < 27; i++)
			{
				if (i < 9)
				{
					idxShift2[i] = idxShift[i % 9] - yx_size;
				}
				else if (i < 18)
				{

				}
				else
				{
					idxShift2[i] = idxShift[i % 9] + yx_size;
				}
			}
			for (int i = 0; i < gridCnt; i++)
			{
				int i_z = i / yx_size;
				int i_y = i % yx_size;
				int i_x = i_y % x_size;
				i_y = i_y / x_size;
				if (i_x == numX_1 || i_y == numY_1 || i_z == numZ_1 || i_x == 0 || i_y == 0 || i_z == 0)
				{
					continue;
				}
				bool jump = true;
				for (size_t j = 0; j < 27; j++)
				{
					if (gridCenterHitValue[i + idxShift2[j]] <= thre)
					{
						jump = false;
						break;
					}
				}
				if (!jump && gridCenterHitValue[i] > thre)
				{
					ptsCnt++;
					shell_xyzs.emplace_back(this->grid(0, i), this->grid(1, i), this->grid(2, i));
					hitValue.emplace_back(this->gridCenterHitValue[i]);
				}
			}
		}
		this->grid = Eigen::Matrix4Xf::Ones(4, ptsCnt);
		this->gridCenterHitValue.resize(ptsCnt);
		ptsCnt = 0;
		auto iter = hitValue.begin();
		for (const auto&d: shell_xyzs)
		{
			grid(0, ptsCnt) = d[0];
			grid(1, ptsCnt) = d[1];
			grid(2, ptsCnt) = d[2];
			this->gridCenterHitValue[ptsCnt] = *iter;
			iter++;
			ptsCnt++;
		}
		return true;
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


int test_undistortFoxImg()
{
	Bitmap distorted_bitmap;
	distorted_bitmap.Read("D:/repo/Instant-NGP-for-GTX-1000/data/nerf/fox/images/0001.jpg");
	Bitmap  undistorted_bitmap;
	Camera distorted_camera;
	distorted_camera.camera_id = 0;
	distorted_camera.model_id = CameraModelId::kOpenCV;
	distorted_camera.width = 1080;
	distorted_camera.height= 1920;
	distorted_camera.params.resize(8);
	distorted_camera.params[0] = 1375.52;//f1
	distorted_camera.params[1] = 1374.49;//f2
	distorted_camera.params[2] = 554.558;//c1
	distorted_camera.params[3] = 965.268;//c2
	distorted_camera.params[4] = 0.0578421 ;//k1
	distorted_camera.params[5] = -0.0805099 ;//k2
	distorted_camera.params[6] = -0.000980296 ;//p1
	distorted_camera.params[7] = 0.00015575 ;//p2

	Camera undistorted_camera;
	undistorted_camera.camera_id = 0;
	undistorted_camera.model_id = CameraModelId::kOpenCV;
	undistorted_camera.width = 1080;
	undistorted_camera.height = 1920;
	undistorted_camera.params.resize(8,0);
	undistorted_camera.params[0] = 1375.52;//f1
	undistorted_camera.params[1] = 1374.49;//f2
	undistorted_camera.params[2] = 554.558;//c1
	undistorted_camera.params[3] = 965.268;//c2
	WarpImageBetweenCameras(distorted_camera,
		undistorted_camera,
		distorted_bitmap,
		&undistorted_bitmap);
	undistorted_bitmap.Write("D:/repo/Instant-NGP-for-GTX-1000/data/nerf/fox/images/0001a.jpg");
	return 0;
}

int test_sdf()
{
	//return test_undistortFoxImg();
	Eigen::MatrixX3d landmarkPts = sdf::readPoint3d("../data/a/result/pts.txt");
	double x_strat = landmarkPts.col(0).minCoeff();
	double y_strat = landmarkPts.col(1).minCoeff();
	double z_strat = landmarkPts.col(2).minCoeff();
	double x_end = landmarkPts.col(0).maxCoeff();
	double y_end = landmarkPts.col(1).maxCoeff();
	double z_end = landmarkPts.col(2).maxCoeff();


	sdf::VolumeDat a(static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()*0.0001), x_strat, y_strat, z_strat, x_end, y_end, z_end);

	std::vector<std::filesystem::path> maskPaths;
	std::vector<std::filesystem::path> imgPaths;
	std::vector<std::filesystem::path> jsonPaths;
	std::vector<Eigen::Matrix4d> cameraPs;
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
						std::string imgPath;
						if (!labelme::readJsonStringElement(thisFilename, "imagePath", imgPath))
						{
							LOG_WARN_OUT << "not found : " << imgPath;
							continue;
						};
						if (!std::filesystem::exists(imgPath))
						{
							LOG_WARN_OUT << "not found : " << imgPath;
						}
						maskPaths.emplace_back(maskPath);
						imgPaths.emplace_back(imgPath);
						jsonPaths.emplace_back(thisFilename);
						Eigen::Matrix4d  p = cameraMatrix * Rt;
						cameraPs.emplace_back(p);
						LOG_OUT << imgPath;
						LOG_OUT << Rt;
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

	a.emptyShellPts(0);
	Eigen::Matrix3Xf pts = a.getCloud(0);
	std::vector<std::uint8_t> colors = sdf::fuseColor(pts, imgPaths, cameraPs);
	sdf::saveCloud("c.ply", pts, colors);
	//mc::marchcube(a.grid, a.gridCenterHitValue, a.x_size, a.y_size, a.z_size, a.unit, 0);
	return 0;
}