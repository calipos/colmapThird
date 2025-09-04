#ifndef _MASK_SYNTHESIS_H_
#define _MASK_SYNTHESIS_H_
#include <numeric>
#include <cstdint>
#include <iostream>
#include <vector>
#include "Eigen/Core"
#include "log.h"
#include "bitmap.h"
namespace sdf
{
	template <typename T>
	T Median(std::vector<T>& elems) {
		if (elems.size() == 0)
			return 0;
		const size_t mid_idx = elems.size() / 2;
		std::nth_element(elems.begin(), elems.begin() + mid_idx, elems.end());
		return elems[mid_idx];
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
	Eigen::MatrixX3d readPoint3d(const std::filesystem::path& path);
	struct VolumeDat
	{
		VolumeDat(const std::uint64_t& indexMax, const double& startX, const double& startY, const double& startZ, const double& endX, const double& endY, const double& endZ);
		Eigen::Matrix3Xf getCloud(const int& thre)const;
		bool emptyShellPts(const int& thre = 1);
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
		std::vector<float>gridCenterHitValue;
		Eigen::Matrix4Xf grid;
	};
	template < typename DerivedV>
	std::vector<std::uint8_t> fuseColor(const Eigen::PlainObjectBase<DerivedV>& pts, std::vector<std::filesystem::path>& imgPaths, std::vector<Eigen::Matrix4d>& cameraPs)
	{
		Eigen::MatrixXf cameraP2(cameraPs.size() * 4, 4);
		for (size_t i = 0; i < cameraPs.size(); i++)
		{
			cameraP2.block(i * 4, 0, 4, 4) = cameraPs[i].cast<float>();
		}
		std::vector<Bitmap>imgs(imgPaths.size());
		for (size_t i = 0; i < imgPaths.size(); i++)
		{
			imgs[i].Read(imgPaths[i].string(), true);
		}
		std::vector<std::uint8_t> ret(3 * pts.cols(), 0);
		for (int p = 0; p < pts.cols(); p++)
		{
			Eigen::Vector4f pt;
			pt[0] = pts(0, p);
			pt[1] = pts(1, p);
			pt[2] = pts(2, p);
			pt[3] = 1.f;
			Eigen::VectorXf uvs = cameraP2 * pt;
			std::vector<std::uint8_t>rgbs_r;
			std::vector<std::uint8_t>rgbs_g;
			std::vector<std::uint8_t>rgbs_b;
			rgbs_r.reserve(cameraPs.size());
			rgbs_g.reserve(cameraPs.size());
			rgbs_b.reserve(cameraPs.size());
			for (size_t i = 0; i < cameraPs.size(); i++)
			{
				uvs[4 * i] /= uvs[4 * i + 2];
				uvs[4 * i + 1] /= uvs[4 * i + 2];
				int x = static_cast<int>(std::round(uvs[4 * i]));
				int y = static_cast<int>(std::round(uvs[4 * i + 1]));
				if (x >= 0 && y >= 0 && x < imgs[i].Width() && y < imgs[i].Height())
				{
					BitmapColor<std::uint8_t>thisColor;
					imgs[i].GetPixel(x, y, &thisColor);
					rgbs_r.emplace_back(thisColor.r);
					rgbs_g.emplace_back(thisColor.g);
					rgbs_b.emplace_back(thisColor.b);
				}
			}
			std::uint8_t r = Median<std::uint8_t>(rgbs_r);
			std::uint8_t g = Median<std::uint8_t>(rgbs_g);
			std::uint8_t b = Median<std::uint8_t>(rgbs_b);
			int i3 = 3 * p;
			ret[i3] = r;
			ret[i3 + 1] = g;
			ret[i3 + 2] = b;
		}
		return ret;
	}
	bool saveCloud(const std::filesystem::path& path, const Eigen::MatrixXf& pts, const std::vector<std::uint8_t>& colors);
}
#endif // _MASK_SYNTHESIS_H_
