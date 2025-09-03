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
#include "bitmap.h"
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
	template <typename T>
	T Median(std::vector<T>* elems) {
		if (!elems->empty())
			return 0;
		const size_t mid_idx = elems->size() / 2;
		std::nth_element(elems->begin(), elems->begin() + mid_idx, elems->end());
		return static_cast<T>((*elems)[mid_idx]);
		//if (elems->size() % 2 == 0) {
		//	const float mid_element1 = static_cast<float>((*elems)[mid_idx]);
		//	const float mid_element2 = static_cast<float>(
		//		*std::max_element(elems->begin(), elems->begin() + mid_idx));
		//	return static_cast<T>((mid_element1 + mid_element2) / 2.0f);
		//}
		//else {
		//	return static_cast<T>((*elems)[mid_idx]);
		//}
	}

	template < typename DerivedV>
	Eigen::Matrix3Xi fuseColor(const Eigen::PlainObjectBase<DerivedV>&pts, std::vector<std::filesystem::path>& imgPaths, std::vector<Eigen::Matrix4d>& cameraPs)
	{
		Eigen::MatrixXf cameraP2(cameraPs.size()*4,4);
		for (size_t i = 0; i < cameraPs.size(); i++)
		{
			cameraP2.block(i*4,0,4,4) = cameraPs[i].cast<float>();
		}
		std::vector<Bitmap>imgs(imgPaths.size());
		for (size_t i = 0; i < imgPaths.size(); i++)
		{
			imgs[i].Read(imgPaths[i].string(), true);
		}
		Eigen::Matrix3Xi ret(3, pts.size());
		for (int p = 0; p < pts.size(); p++)
		{
			Eigen::Vector4f pt;
			pt[0] = pts(0, p);
			pt[1] = pts(1, p);
			pt[2] = pts(2, p);
			pt[3] = 1.f;
			Eigen::VectorXf uvs = cameraP2 *pt;
			std::vector<uint8_t>rgbs_r;
			std::vector<uint8_t>rgbs_g;
			std::vector<uint8_t>rgbs_b;
			rgbs_r.reserve(cameraPs.size());
			rgbs_g.reserve(cameraPs.size());
			rgbs_b.reserve(cameraPs.size());
			for (size_t i = 0; i < cameraPs.size(); i++)
			{
				uvs[4 * i] /= uvs[4 * i + 2];
				uvs[4 * i+1] /= uvs[4 * i + 2];
				int x = static_cast<int>(std::round(uvs[4 * i]));
				int y = static_cast<int>(std::round(uvs[4 * i+1]));
				if (x>=0 &&y>=0 && x < imgs[i].Width() && y < imgs[i].Height())
				{
					BitmapColor<uint8_t>thisColor;
					imgs[i].GetPixel(x,y,&thisColor);
					rgbs_r.emplace_back(thisColor.r);
					rgbs_g.emplace_back(thisColor.g);
					rgbs_b.emplace_back(thisColor.b);
				}
			}
			ret(0, p) = Median(rgbs_r);
			ret(1, p) = Median(rgbs_g);
			ret(2, p) = Median(rgbs_b);
		}
		return Eigen::Matrix3Xf();
	}
}
int test_sdf()
{
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
	sdf::fuseColor(pts, imgPaths, cameraPs);
		
	//mc::marchcube(a.grid, a.gridCenterHitValue, a.x_size, a.y_size, a.z_size, a.unit, 0);
	return 0;
}